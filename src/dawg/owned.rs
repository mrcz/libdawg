use std::collections::HashSet as StdHashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

use hashbrown::HashMap;
use smallvec::SmallVec;

use super::builder::{is_comment, BuilderCore, BuilderError, IntoWord};
use super::char_trait::DawgChar;
use super::node::DawgNode;
use super::node_arena::NodeArena;

/// Internal state for DAWG mutation operations.
///
/// Lazily initialized on first `add_word`/`remove_word` call.
struct MutationState<C: DawgChar + 'static> {
    /// Register of all canonical nodes, keyed by structural hash/equality,
    /// with incoming edge count as the value. Root has refcount 1 (from self.root).
    register: HashMap<&'static DawgNode<'static, C>, usize>,
    /// Reusable arena slots from freed nodes.
    free_list: Vec<*mut DawgNode<'static, C>>,
}

/// A self-contained DAWG that owns its arena internally.
///
/// Unlike the arena-based API where the caller manages an external arena,
/// `OwnedDawg` handles allocation internally.
/// The returned DAWG can be freely moved, stored, and owned.
///
/// After construction, words can be added and removed with [`add_word`](OwnedDawg::add_word)
/// and [`remove_word`](OwnedDawg::remove_word). The DAWG remains minimal after each operation.
///
/// # Examples
///
/// ```
/// use libdawg::dawg::owned::build_owned_dawg;
///
/// let mut dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE", "LAKE", "MAKE"]).unwrap();
/// let root = dawg.root();
///
/// let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
/// assert!(is_word("CAKE"));
/// assert!(!is_word("AKE"));
///
/// dawg.add_word("SAKE");
/// assert!(dawg.contains("SAKE"));
///
/// dawg.remove_word("BAKE");
/// assert!(!dawg.contains("BAKE"));
/// ```
pub struct OwnedDawg<C: DawgChar + 'static> {
    // SAFETY: `root` points into `arena`'s heap-allocated chunks.
    // The arena's chunks never move, so the pointer remains valid
    // as long as the arena exists. Since both live in the same struct
    // and we only expose &'a-bounded references via root(), this is safe.
    arena: NodeArena<DawgNode<'static, C>>,
    root: *const DawgNode<'static, C>,
    /// Lazily initialized on first mutation. None for read-only usage.
    mutation_state: Option<MutationState<C>>,
}

// SAFETY: Access is controlled via Rust's borrow system: shared reads via `&self`,
// exclusive writes via `&mut self`. The raw pointer is derived from arena-allocated
// data. The arena's heap chunks never move.
unsafe impl<C: DawgChar + 'static> Send for OwnedDawg<C> {}
unsafe impl<C: DawgChar + 'static> Sync for OwnedDawg<C> {}

impl<C: DawgChar + 'static> OwnedDawg<C> {
    /// Returns a reference to the root node.
    ///
    /// The returned reference has the same API as [`DawgNode`] — use
    /// [`get()`](DawgNode::get), [`is_word()`](DawgNode::is_word),
    /// [`children()`](DawgNode::children), etc. to traverse the graph.
    pub fn root(&self) -> &DawgNode<'_, C> {
        // SAFETY: root points into arena's heap chunks, which remain valid.
        // DawgNode is covariant in 'w, so 'static shortens to 'a.
        unsafe { &*self.root }
    }

    /// Returns the number of unique nodes in the DAWG.
    pub fn node_count(&self) -> usize {
        self.arena.len()
    }
}

// --- Mutation support ---

impl<C: DawgChar + 'static> MutationState<C> {
    /// Build the register by traversing the entire DAWG.
    /// Each node is inserted with its incoming edge count as the value.
    fn initialize(root: *const DawgNode<'static, C>) -> Self {
        let root_ref = unsafe { &*root };
        let mut register: HashMap<&'static DawgNode<'static, C>, usize> = HashMap::new();
        let mut visited = StdHashSet::new();
        let mut stack = vec![root_ref];

        // DFS to collect all nodes and count incoming edges.
        while let Some(node) = stack.pop() {
            let ptr = node as *const DawgNode<'static, C>;
            if !visited.insert(ptr) {
                continue;
            }
            register.entry(node).or_insert(0);
            for (_, child) in node.children() {
                *register.entry(child).or_insert(0) += 1;
                stack.push(child);
            }
        }

        // Root has refcount 1 (from OwnedDawg::root pointer).
        *register.entry(root_ref).or_insert(0) += 1;

        MutationState {
            register,
            free_list: Vec::new(),
        }
    }
}

impl<C: DawgChar + 'static> OwnedDawg<C> {
    /// Ensures the mutation state is initialized.
    fn ensure_mutation_state(&mut self) {
        if self.mutation_state.is_none() {
            self.mutation_state = Some(MutationState::initialize(self.root));
        }
    }

    /// Allocates a node from the free-list or arena.
    fn alloc_node(
        arena: &NodeArena<DawgNode<'static, C>>,
        free_list: &mut Vec<*mut DawgNode<'static, C>>,
        mut node: DawgNode<'static, C>,
    ) -> &'static DawgNode<'static, C> {
        node.set_canonical();
        if let Some(slot) = free_list.pop() {
            // SAFETY: slot points into arena's heap chunks which never move.
            // The old value has been overwritten with a sentinel (DawgNode::new(false))
            // before being pushed onto the free-list. We overwrite it with the new node.
            unsafe {
                std::ptr::write(slot, node);
                &*slot
            }
        } else {
            // SAFETY: arena ref transmuted to 'static — same pattern as make_builder.
            // The arena lives as long as OwnedDawg, and we only expose references
            // bounded by &self.
            let arena_ref: &'static NodeArena<DawgNode<'static, C>> =
                unsafe { &*(arena as *const NodeArena<DawgNode<'static, C>>) };
            arena_ref.alloc(node)
        }
    }

    /// Canonicalizes a node: returns an existing equivalent from the register,
    /// or allocates and registers a new one.
    ///
    /// When a new node is allocated, its children's refcounts are incremented
    /// (since the new node creates edges to them).
    fn canonicalize_node(
        arena: &NodeArena<DawgNode<'static, C>>,
        state: &mut MutationState<C>,
        node: DawgNode<'static, C>,
    ) -> &'static DawgNode<'static, C> {
        if let Some((&existing, _)) = state.register.get_key_value(&node) {
            existing
        } else {
            // Collect child references before moving node.
            // We use children_ref().get() instead of children() because
            // children() requires &'static self which we can't provide for a local.
            let child_refs: SmallVec<[&'static DawgNode<'static, C>; 8]> =
                (0..node.child_count())
                    .map(|i| {
                        let (_, child) = node.children_ref().get(i).unwrap();
                        child
                    })
                    .collect();

            let allocated = Self::alloc_node(arena, &mut state.free_list, node);
            state.register.insert(allocated, 0);

            // Increment refcounts for all children (new edges from this node).
            for child in child_refs {
                *state.register.get_mut(child).expect("child not in register") += 1;
            }

            allocated
        }
    }

    /// Decrements a node's refcount. If it reaches 0, the node is freed:
    /// unregistered, children's refcounts recursively decremented, and the
    /// slot added to the free-list.
    fn decrement_refcount_cascade(state: &mut MutationState<C>, node_ptr: *const DawgNode<'static, C>) {
        let node = unsafe { &*node_ptr };
        let rc = state.register.get_mut(node).expect("node not in register");
        *rc -= 1;
        if *rc == 0 {
            // Collect children before modifying state to avoid borrow issues.
            let children: SmallVec<[*const DawgNode<'static, C>; 8]> = node
                .children()
                .map(|(_, child)| child as *const _)
                .collect();

            // Remove from register.
            state.register.remove(node);

            // Recursively decrement children.
            for child_ptr in children {
                Self::decrement_refcount_cascade(state, child_ptr);
            }

            // Overwrite with sentinel value and add to free-list.
            // SAFETY: node_ptr points into arena. No live references exist
            // (refcount was 0, meaning no parent edges point here, and `node`
            // is no longer used after register.remove).
            let slot = node_ptr as *mut DawgNode<'static, C>;
            unsafe {
                std::ptr::write(slot, DawgNode::new(false));
            }
            state.free_list.push(slot);
        }
    }

    /// Adds a word to the DAWG.
    ///
    /// Returns `true` if the word was added, `false` if it was already present.
    /// The DAWG remains minimal (suffix sharing is maintained).
    ///
    /// # Examples
    ///
    /// ```
    /// use libdawg::dawg::owned::build_owned_dawg;
    ///
    /// let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
    /// assert!(dawg.add_word("FAKE"));
    /// assert!(!dawg.add_word("FAKE")); // already present
    /// assert!(dawg.contains("FAKE"));
    /// ```
    pub fn add_word(&mut self, word: impl IntoWord<C>) -> bool {
        let word = word.collect_word();
        if word.is_empty() {
            return false;
        }

        self.ensure_mutation_state();

        let root = unsafe { &*self.root };

        // Phase 1: Walk the DAWG following the word, collecting path nodes.
        // path_nodes[0] = root, path_nodes[i] follows word[0..i].
        let mut path_nodes: SmallVec<[*const DawgNode<'static, C>; 32]> = SmallVec::new();
        path_nodes.push(self.root);
        let mut current = root;
        let mut prefix_len = 0;

        for &ch in word.iter() {
            if let Some(child) = current.get(ch) {
                path_nodes.push(child as *const _);
                current = child;
                prefix_len += 1;
            } else {
                break;
            }
        }

        // If the full word path exists, check if it's already a word.
        if prefix_len == word.len() && current.is_word() {
            return false; // Already exists
        }

        // Destructure self to avoid borrow conflicts.
        let Self {
            arena,
            mutation_state,
            ..
        } = self;
        let state = mutation_state.as_mut().unwrap();

        // Phase 2: Build the updated child, starting from the bottom.
        let mut updated_child: &'static DawgNode<'static, C>;

        if prefix_len == word.len() {
            // The path exists but the terminal isn't marked as a word.
            // Clone it with word=true.
            let terminal = unsafe { &*path_nodes[prefix_len] };
            let new_terminal =
                DawgNode::with_children(true, terminal.children_ref().clone());
            updated_child = Self::canonicalize_node(arena, state, new_terminal);
        } else {
            // Create new suffix nodes for word[prefix_len..].
            // Start with the leaf (last character).
            let leaf = DawgNode::new(true);
            updated_child = Self::canonicalize_node(arena, state, leaf);

            // Build intermediate nodes bottom-up for word[prefix_len+1..word.len()-1].
            for i in (prefix_len + 1..word.len()).rev() {
                let mut intermediate = DawgNode::new(false);
                intermediate.insert(word[i], updated_child);
                updated_child = Self::canonicalize_node(arena, state, intermediate);
            }
        }

        // Phase 3: Walk the path bottom-up, cloning each ancestor with the updated child.
        // When prefix_len < word.len(): at level prefix_len we ADD a new child edge,
        //   at ancestors above we REPLACE.
        // When prefix_len == word.len(): the terminal was already handled in Phase 2,
        //   so we start one level up and only REPLACE.
        let start_level = if prefix_len < word.len() {
            prefix_len
        } else {
            prefix_len - 1
        };
        for level in (0..=start_level).rev() {
            let old_node = unsafe { &*path_nodes[level] };
            let ch = word[level];

            let new_children = if level == prefix_len && prefix_len < word.len() {
                // Divergence point: add new child edge.
                old_node.children_ref().with_added_child(ch, updated_child)
            } else {
                // Ancestor (or prefix case): replace existing child edge.
                old_node.children_ref().with_replaced_child(ch, updated_child)
            };

            let new_node = DawgNode::with_children(old_node.is_word(), new_children);
            updated_child = Self::canonicalize_node(arena, state, new_node);
        }

        // Phase 4: Update root.
        let old_root = self.root;
        self.root = updated_child as *const _;
        let state = self.mutation_state.as_mut().unwrap();
        *state.register.get_mut(updated_child).expect("new root not in register") += 1;
        Self::decrement_refcount_cascade(state, old_root);

        true
    }

    /// Removes a word from the DAWG.
    ///
    /// Returns `true` if the word was removed, `false` if it was not present.
    /// Nodes that become unreachable are placed on a free-list for reuse by
    /// future [`add_word`](OwnedDawg::add_word) calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use libdawg::dawg::owned::build_owned_dawg;
    ///
    /// let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
    /// assert!(dawg.remove_word("BAKE"));
    /// assert!(!dawg.remove_word("BAKE")); // already removed
    /// assert!(!dawg.contains("BAKE"));
    /// assert!(dawg.contains("CAKE"));
    /// ```
    pub fn remove_word(&mut self, word: impl IntoWord<C>) -> bool {
        let word = word.collect_word();
        if word.is_empty() {
            return false;
        }

        self.ensure_mutation_state();

        let root = unsafe { &*self.root };

        // Phase 1: Walk the full path. If word not found, return false.
        let mut path_nodes: SmallVec<[*const DawgNode<'static, C>; 32]> = SmallVec::new();
        path_nodes.push(self.root);
        let mut current = root;

        for &ch in word.iter() {
            if let Some(child) = current.get(ch) {
                path_nodes.push(child as *const _);
                current = child;
            } else {
                return false; // Path doesn't exist
            }
        }

        if !current.is_word() {
            return false; // Path exists but isn't a word
        }

        // Destructure self to avoid borrow conflicts.
        let Self {
            arena,
            mutation_state,
            ..
        } = self;
        let state = mutation_state.as_mut().unwrap();

        // Phase 2: Clone terminal with word=false.
        let terminal = unsafe { &*path_nodes[word.len()] };
        let new_terminal = DawgNode::with_children(false, terminal.children_ref().clone());

        // Check if terminal becomes dead (no children and not a word).
        let mut updated_child: Option<&'static DawgNode<'static, C>> =
            if new_terminal.child_count() == 0 {
                None // Pruned
            } else {
                Some(Self::canonicalize_node(arena, state, new_terminal))
            };

        // Phase 3: Walk path bottom-up.
        for level in (0..word.len()).rev() {
            let old_node = unsafe { &*path_nodes[level] };
            let ch = word[level];

            let new_node = match updated_child {
                None => {
                    // Child was pruned: remove the edge.
                    let new_children = old_node.children_ref().without_child(ch);
                    DawgNode::with_children(old_node.is_word(), new_children)
                }
                Some(child) => {
                    // Replace the child edge.
                    let new_children =
                        old_node.children_ref().with_replaced_child(ch, child);
                    DawgNode::with_children(old_node.is_word(), new_children)
                }
            };

            // Check if this node is now dead (no children and not a word).
            if new_node.child_count() == 0 && !new_node.is_word() {
                updated_child = None; // Prune this node too
            } else {
                updated_child = Some(Self::canonicalize_node(arena, state, new_node));
            }
        }

        // Phase 4: Update root.
        let old_root = self.root;
        let new_root_ref = match updated_child {
            Some(node) => node,
            None => {
                // Entire DAWG pruned — create empty root.
                let empty = DawgNode::new(false);
                Self::canonicalize_node(arena, state, empty)
            }
        };
        self.root = new_root_ref as *const _;
        let state = self.mutation_state.as_mut().unwrap();
        *state.register.get_mut(new_root_ref).expect("new root not in register") += 1;
        Self::decrement_refcount_cascade(state, old_root);

        true
    }

    /// Returns `true` if the given word is in the DAWG.
    ///
    /// # Examples
    ///
    /// ```
    /// use libdawg::dawg::owned::build_owned_dawg;
    ///
    /// let dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
    /// assert!(dawg.contains("CAKE"));
    /// assert!(!dawg.contains("FAKE"));
    /// ```
    pub fn contains(&self, word: impl IntoWord<C>) -> bool {
        let word = word.collect_word();
        let root = self.root();
        let mut current = root;
        for &ch in word.iter() {
            match current.get(ch) {
                Some(child) => current = child,
                None => return false,
            }
        }
        current.is_word()
    }
}

impl<C: DawgChar + 'static> std::fmt::Debug for OwnedDawg<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedDawg")
            .field("node_count", &self.node_count())
            .finish()
    }
}

/// Creates a builder with a `'static` lifetime from a `NodeArena`.
///
/// # Safety
///
/// The returned builder holds a reference into `arena` that is transmuted
/// to `'static`. The caller must ensure:
/// - The builder is consumed (via `build()`) before returning
/// - The arena is moved into an `OwnedDawg` alongside the root pointer
unsafe fn make_builder<C: DawgChar + 'static>(
    arena: &NodeArena<DawgNode<'static, C>>,
) -> BuilderCore<'static, C, NodeArena<DawgNode<'static, C>>> {
    let arena_ref: &'static NodeArena<DawgNode<'static, C>> =
        &*(arena as *const NodeArena<DawgNode<'static, C>>);
    BuilderCore::new(arena_ref)
}

/// Builds an owned DAWG from an iterator of words.
///
/// Each word must implement [`IntoWord`]. Words **must** be in sorted order.
///
/// # Examples
///
/// ```
/// use libdawg::dawg::owned::build_owned_dawg;
///
/// let dawg = build_owned_dawg(["APPLE", "BANANA", "CHERRY"]).unwrap();
/// let root = dawg.root();
///
/// let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
/// assert!(is_word("BANANA"));
/// assert!(!is_word("APRICOT"));
/// ```
pub fn build_owned_dawg<C, W>(
    words: impl IntoIterator<Item = W>,
) -> Result<OwnedDawg<C>, BuilderError<C>>
where
    C: DawgChar + 'static,
    W: IntoWord<C>,
{
    let arena: NodeArena<DawgNode<'static, C>> = NodeArena::new();

    // SAFETY: builder is consumed within this function. The root pointer
    // is stored alongside the arena in OwnedDawg.
    let root = unsafe {
        let mut builder = make_builder(&arena);
        for word in words {
            builder.add_word(word)?;
        }
        builder.build() as *const DawgNode<'static, C>
    };

    Ok(OwnedDawg {
        arena,
        root,
        mutation_state: None,
    })
}

/// Builds an owned DAWG from a dictionary file.
///
/// Reads words from a text file (one word per line) and constructs a DAWG.
/// Words must be in sorted order. Lines starting with '#' are treated as
/// comments and ignored.
///
/// # Examples
///
/// ```no_run
/// use libdawg::dawg::owned::build_owned_dawg_from_file;
///
/// let dawg = build_owned_dawg_from_file("dictionary.txt").unwrap();
/// ```
pub fn build_owned_dawg_from_file(
    filename: &str,
) -> Result<OwnedDawg<char>, Box<dyn Error>> {
    let arena: NodeArena<DawgNode<'static, char>> = NodeArena::new();

    // SAFETY: same as build_owned_dawg — builder is consumed here,
    // root pointer stored alongside arena.
    let root = unsafe {
        let mut builder = make_builder(&arena);
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let mut buf = String::with_capacity(80);
        loop {
            let bytes_read = reader.read_line(&mut buf);
            match bytes_read {
                Ok(0) => break,
                Err(e) => return Err(e.into()),
                _ => {}
            }
            let word = buf.trim_end();
            if !word.is_empty() && !is_comment(word) {
                builder.add_word(word)?;
            }
            buf.clear();
        }
        builder.build() as *const DawgNode<'static, char>
    };

    Ok(OwnedDawg {
        arena,
        root,
        mutation_state: None,
    })
}

#[cfg(test)]
mod test {
    use super::*;

    fn is_word(root: &DawgNode<'_, char>, word: &str) -> bool {
        word.chars()
            .try_fold(root, |n, ch| n.get(ch))
            .is_some_and(|n| n.is_word())
    }

    #[test]
    fn basic_word_lookup() {
        let dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE", "LAKE"]).unwrap();
        let root = dawg.root();
        assert!(is_word(root, "BAKE"));
        assert!(is_word(root, "CAKE"));
        assert!(!is_word(root, "MAKE"));
        assert!(!is_word(root, "BAK"));
    }

    #[test]
    fn sorted_input_required() {
        let res = build_owned_dawg(["ZULU", "ALFA"]);
        assert!(res.is_err());
    }

    #[test]
    fn suffix_sharing() {
        let dawg = build_owned_dawg([
            "ASUFFIX",
            "BSUFFIX",
            "CDESUFFIX",
            "FFFFFFFSUFFIX",
            "INBETWEEN",
            "JSUFFIX",
            "XXSUFFIX",
        ])
        .unwrap();
        let root = dawg.root();
        let suffix_node = root.get('A').unwrap().get('S').unwrap();
        for prefix_char in ['B', 'J'] {
            let node = root.get(prefix_char).unwrap().get('S').unwrap();
            assert!(std::ptr::addr_eq(node, suffix_node));
        }
    }

    #[test]
    fn generic_u8() {
        let dawg: OwnedDawg<u8> =
            build_owned_dawg([vec![1, 2, 3], vec![1, 2, 4], vec![2, 3, 4]]).unwrap();
        let root = dawg.root();
        assert!(root
            .get(1)
            .and_then(|n| n.get(2))
            .and_then(|n| n.get(3))
            .is_some_and(|n| n.is_word()));
        assert!(root
            .get(1)
            .and_then(|n| n.get(2))
            .and_then(|n| n.get(5))
            .is_none());
    }

    #[test]
    fn node_count() {
        let dawg = build_owned_dawg(["ABC", "ABD"]).unwrap();
        // root + A + B + C + D = 5, but C and D may share if same structure
        assert!(dawg.node_count() > 0);
    }

    #[test]
    fn owned_dawg_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OwnedDawg<char>>();
    }

    // --- Mutation tests ---

    #[test]
    fn contains_basic() {
        let dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE"]).unwrap();
        assert!(dawg.contains("BAKE"));
        assert!(dawg.contains("CAKE"));
        assert!(!dawg.contains("MAKE"));
        assert!(!dawg.contains("BAK"));
    }

    #[test]
    fn add_word_to_empty() {
        let mut dawg = build_owned_dawg::<char, &str>([]).unwrap();
        assert!(dawg.add_word("HELLO"));
        assert!(dawg.contains("HELLO"));
        assert!(!dawg.contains("HELL"));
    }

    #[test]
    fn add_word_returns_false_for_duplicate() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
        assert!(!dawg.add_word("BAKE"));
        assert!(!dawg.add_word("CAKE"));
    }

    #[test]
    fn add_word_preserves_existing() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
        dawg.add_word("FAKE");
        assert!(dawg.contains("BAKE"));
        assert!(dawg.contains("CAKE"));
        assert!(dawg.contains("FAKE"));
    }

    #[test]
    fn add_prefix_of_existing() {
        let mut dawg = build_owned_dawg(["CART"]).unwrap();
        assert!(dawg.add_word("CAR"));
        assert!(dawg.contains("CAR"));
        assert!(dawg.contains("CART"));
    }

    #[test]
    fn add_extension_of_existing() {
        let mut dawg = build_owned_dawg(["CAR"]).unwrap();
        assert!(dawg.add_word("CART"));
        assert!(dawg.contains("CAR"));
        assert!(dawg.contains("CART"));
    }

    #[test]
    fn add_multiple_words() {
        let mut dawg = build_owned_dawg::<char, &str>([]).unwrap();
        for word in ["FAKE", "CAKE", "BAKE", "LAKE", "MAKE"] {
            assert!(dawg.add_word(word));
        }
        for word in ["FAKE", "CAKE", "BAKE", "LAKE", "MAKE"] {
            assert!(dawg.contains(word));
        }
        assert!(!dawg.contains("SAKE"));
    }

    #[test]
    fn remove_word_basic() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE"]).unwrap();
        assert!(dawg.remove_word("BAKE"));
        assert!(!dawg.contains("BAKE"));
        assert!(dawg.contains("CAKE"));
        assert!(dawg.contains("FAKE"));
    }

    #[test]
    fn remove_word_returns_false_for_missing() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
        assert!(!dawg.remove_word("FAKE"));
        assert!(!dawg.remove_word("BAK"));
    }

    #[test]
    fn remove_word_returns_false_for_already_removed() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
        assert!(dawg.remove_word("BAKE"));
        assert!(!dawg.remove_word("BAKE"));
    }

    #[test]
    fn remove_preserves_existing() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE"]).unwrap();
        dawg.remove_word("CAKE");
        assert!(dawg.contains("BAKE"));
        assert!(!dawg.contains("CAKE"));
        assert!(dawg.contains("FAKE"));
    }

    #[test]
    fn remove_last_word() {
        let mut dawg = build_owned_dawg(["HELLO"]).unwrap();
        assert!(dawg.remove_word("HELLO"));
        assert!(!dawg.contains("HELLO"));
        // DAWG should be empty but root should still exist.
        assert!(!dawg.root().is_word());
        assert_eq!(dawg.root().child_count(), 0);
    }

    #[test]
    fn remove_all_words() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE"]).unwrap();
        assert!(dawg.remove_word("BAKE"));
        assert!(dawg.remove_word("CAKE"));
        assert!(dawg.remove_word("FAKE"));
        assert!(!dawg.contains("BAKE"));
        assert!(!dawg.contains("CAKE"));
        assert!(!dawg.contains("FAKE"));
        assert_eq!(dawg.root().child_count(), 0);
    }

    #[test]
    fn remove_prefix_keeps_extension() {
        let mut dawg = build_owned_dawg(["CAR", "CART"]).unwrap();
        assert!(dawg.remove_word("CAR"));
        assert!(!dawg.contains("CAR"));
        assert!(dawg.contains("CART"));
    }

    #[test]
    fn remove_extension_keeps_prefix() {
        let mut dawg = build_owned_dawg(["CAR", "CART"]).unwrap();
        assert!(dawg.remove_word("CART"));
        assert!(dawg.contains("CAR"));
        assert!(!dawg.contains("CART"));
    }

    #[test]
    fn interleaved_add_remove() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
        dawg.add_word("FAKE");
        dawg.remove_word("BAKE");
        dawg.add_word("LAKE");
        dawg.add_word("MAKE");
        dawg.remove_word("CAKE");

        assert!(!dawg.contains("BAKE"));
        assert!(!dawg.contains("CAKE"));
        assert!(dawg.contains("FAKE"));
        assert!(dawg.contains("LAKE"));
        assert!(dawg.contains("MAKE"));
    }

    #[test]
    fn add_maintains_suffix_sharing() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE"]).unwrap();
        dawg.add_word("FAKE");

        let root = dawg.root();
        // All three words share the "AKE" suffix.
        let bake_a = root.get('B').unwrap().get('A').unwrap();
        let cake_a = root.get('C').unwrap().get('A').unwrap();
        let fake_a = root.get('F').unwrap().get('A').unwrap();
        assert!(std::ptr::addr_eq(bake_a, cake_a));
        assert!(std::ptr::addr_eq(cake_a, fake_a));
    }

    #[test]
    fn remove_does_not_break_sharing() {
        let mut dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE"]).unwrap();
        dawg.remove_word("FAKE");

        let root = dawg.root();
        let bake_a = root.get('B').unwrap().get('A').unwrap();
        let cake_a = root.get('C').unwrap().get('A').unwrap();
        assert!(std::ptr::addr_eq(bake_a, cake_a));
    }

    #[test]
    fn add_word_to_prebuilt_dawg_shares_suffix() {
        let mut dawg = build_owned_dawg([
            "ASUFFIX",
            "BSUFFIX",
        ])
        .unwrap();
        dawg.add_word("CSUFFIX");

        let root = dawg.root();
        let a_suffix = root.get('A').unwrap().get('S').unwrap();
        let b_suffix = root.get('B').unwrap().get('S').unwrap();
        let c_suffix = root.get('C').unwrap().get('S').unwrap();
        assert!(std::ptr::addr_eq(a_suffix, b_suffix));
        assert!(std::ptr::addr_eq(b_suffix, c_suffix));
    }

    #[test]
    fn minimality_matches_fresh_build() {
        // Build DAWG incrementally.
        let mut dawg_inc = build_owned_dawg::<char, &str>([]).unwrap();
        let words = ["BAKE", "BAKED", "CAKE", "CAKED", "FAKE", "FAKED"];
        for word in words {
            dawg_inc.add_word(word);
        }

        // Build DAWG from scratch with same words.
        let dawg_fresh = build_owned_dawg(words).unwrap();

        // Both should have the same number of nodes in the register.
        let inc_state = dawg_inc.mutation_state.as_ref().unwrap();
        let fresh_register = {
            let mut dawg = dawg_fresh;
            dawg.ensure_mutation_state();
            dawg.mutation_state.unwrap().register.len()
        };
        assert_eq!(inc_state.register.len(), fresh_register);
    }

    #[test]
    fn add_then_remove_returns_to_empty() {
        let mut dawg = build_owned_dawg::<char, &str>([]).unwrap();
        dawg.add_word("HELLO");
        dawg.add_word("WORLD");
        dawg.remove_word("HELLO");
        dawg.remove_word("WORLD");

        assert!(!dawg.contains("HELLO"));
        assert!(!dawg.contains("WORLD"));
        assert_eq!(dawg.root().child_count(), 0);
    }

    #[test]
    fn generic_u8_add_remove() {
        let mut dawg: OwnedDawg<u8> =
            build_owned_dawg([vec![1, 2, 3], vec![1, 2, 4]]).unwrap();
        dawg.add_word(vec![2, 3, 4]);
        assert!(dawg.contains([2u8, 3, 4].as_slice()));
        assert!(dawg.contains([1u8, 2, 3].as_slice()));

        dawg.remove_word([1u8, 2, 3].as_slice());
        assert!(!dawg.contains([1u8, 2, 3].as_slice()));
        assert!(dawg.contains([1u8, 2, 4].as_slice()));
    }

    // --- Free-list reuse tests ---

    #[test]
    fn remove_populates_free_list() {
        // "HELLO" has 6 unique nodes (root + H + E + L + L + O).
        // Removing it should free the non-shared ones.
        let mut dawg = build_owned_dawg(["HELLO"]).unwrap();
        dawg.remove_word("HELLO");

        let state = dawg.mutation_state.as_ref().unwrap();
        assert!(
            !state.free_list.is_empty(),
            "free-list should have entries after removing the only word"
        );
    }

    #[test]
    fn add_after_remove_consumes_free_list() {
        let mut dawg = build_owned_dawg(["HELLO"]).unwrap();
        dawg.remove_word("HELLO");

        let free_before = dawg.mutation_state.as_ref().unwrap().free_list.len();
        assert!(free_before > 0);

        dawg.add_word("WORLD");

        let free_after = dawg.mutation_state.as_ref().unwrap().free_list.len();
        assert!(
            free_after < free_before,
            "free-list should shrink after add (was {free_before}, now {free_after})"
        );
        assert!(dawg.contains("WORLD"));
    }

    #[test]
    fn arena_does_not_grow_when_free_list_has_nodes() {
        // Build a DAWG, remove a long word to fill the free-list, then add
        // a short word that fits entirely within the free-list.
        let mut dawg = build_owned_dawg(["ABCDEFGH", "XY"]).unwrap();
        dawg.remove_word("ABCDEFGH");

        let free_count = dawg.mutation_state.as_ref().unwrap().free_list.len();
        let arena_before = dawg.node_count();

        // "ZW" needs 3 nodes (root-replacement + Z + W). The free-list
        // from removing "ABCDEFGH" should have enough entries.
        assert!(
            free_count >= 3,
            "need at least 3 free slots, got {free_count}"
        );

        dawg.add_word("ZW");
        assert!(dawg.contains("ZW"));
        assert!(dawg.contains("XY"));

        let arena_after = dawg.node_count();
        assert_eq!(
            arena_before, arena_after,
            "arena should not grow when free-list provides all nodes \
             (before={arena_before}, after={arena_after})"
        );
    }

    #[test]
    fn free_list_reuse_produces_correct_dawg() {
        // Cycle through remove/add several times and verify correctness.
        let mut dawg = build_owned_dawg(["ALPHA", "BRAVO", "CHARLIE"]).unwrap();

        // Remove all, filling the free-list.
        dawg.remove_word("ALPHA");
        dawg.remove_word("BRAVO");
        dawg.remove_word("CHARLIE");
        let free_after_remove = dawg.mutation_state.as_ref().unwrap().free_list.len();
        assert!(free_after_remove > 0);

        // Add new words — these should reuse freed slots.
        dawg.add_word("DELTA");
        dawg.add_word("ECHO");
        dawg.add_word("FOXTROT");

        assert!(dawg.contains("DELTA"));
        assert!(dawg.contains("ECHO"));
        assert!(dawg.contains("FOXTROT"));
        assert!(!dawg.contains("ALPHA"));
        assert!(!dawg.contains("BRAVO"));
        assert!(!dawg.contains("CHARLIE"));

        // Free-list should have been partially or fully consumed.
        let free_after_add = dawg.mutation_state.as_ref().unwrap().free_list.len();
        assert!(
            free_after_add < free_after_remove,
            "free-list should shrink after adding words (was {free_after_remove}, now {free_after_add})"
        );
    }

    #[test]
    fn repeated_add_remove_cycles_reuse_nodes() {
        let mut dawg = build_owned_dawg::<char, &str>([]).unwrap();

        // Do several add/remove cycles and check that the arena doesn't
        // grow unboundedly — freed nodes should be reused.
        for _ in 0..5 {
            dawg.add_word("TESTING");
            dawg.remove_word("TESTING");
        }

        let arena_after_cycles = dawg.node_count();

        // Do 5 more cycles.
        for _ in 0..5 {
            dawg.add_word("TESTING");
            dawg.remove_word("TESTING");
        }

        let arena_after_more_cycles = dawg.node_count();

        // Arena should not have grown — all nodes are reused from the free-list.
        assert_eq!(
            arena_after_cycles, arena_after_more_cycles,
            "arena should not grow when repeatedly adding/removing the same word \
             (after 5 cycles: {arena_after_cycles}, after 10: {arena_after_more_cycles})"
        );
    }
}
