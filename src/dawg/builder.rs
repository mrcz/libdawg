use hashbrown::HashSet;
use mark_last::MarkLastIterator;
use smallvec::SmallVec;

use std::error::Error;
#[cfg(feature = "arena")]
use std::fs::File;
#[cfg(feature = "arena")]
use std::io::{BufRead, BufReader};

#[cfg(feature = "arena")]
use typed_arena::Arena;

use super::char_trait::DawgChar;
use super::children::DawgNode;

/// Trait for types that can be used as a word when building a DAWG.
///
/// Implemented for common string and sequence types so that [`Builder::add_word`]
/// and [`build_dawg`] accept them directly without manual conversion.
pub trait IntoWord<C: DawgChar> {
    /// Collects this word into a character buffer.
    fn collect_word(self) -> SmallVec<[C; 32]>;
}

// String types → char

impl IntoWord<char> for &str {
    fn collect_word(self) -> SmallVec<[char; 32]> {
        self.chars().collect()
    }
}

impl IntoWord<char> for &&str {
    fn collect_word(self) -> SmallVec<[char; 32]> {
        self.chars().collect()
    }
}

impl IntoWord<char> for String {
    fn collect_word(self) -> SmallVec<[char; 32]> {
        self.chars().collect()
    }
}

impl IntoWord<char> for &String {
    fn collect_word(self) -> SmallVec<[char; 32]> {
        self.chars().collect()
    }
}

// Generic sequence types → C

impl<C: DawgChar> IntoWord<C> for &[C] {
    fn collect_word(self) -> SmallVec<[C; 32]> {
        self.iter().copied().collect()
    }
}

impl<C: DawgChar> IntoWord<C> for Vec<C> {
    fn collect_word(self) -> SmallVec<[C; 32]> {
        self.into_iter().collect()
    }
}

impl<C: DawgChar> IntoWord<C> for &Vec<C> {
    fn collect_word(self) -> SmallVec<[C; 32]> {
        self.iter().copied().collect()
    }
}

impl<C: DawgChar, const N: usize> IntoWord<C> for [C; N] {
    fn collect_word(self) -> SmallVec<[C; 32]> {
        self.into_iter().collect()
    }
}

impl<C: DawgChar, const N: usize> IntoWord<C> for &[C; N] {
    fn collect_word(self) -> SmallVec<[C; 32]> {
        self.iter().copied().collect()
    }
}

/// Trait for arena-like allocators that can allocate DAWG nodes.
pub(crate) trait NodeAlloc<'w, C: DawgChar> {
    /// Allocates a node and returns a reference with the arena's lifetime.
    fn alloc_node(&'w self, node: DawgNode<'w, C>) -> &'w DawgNode<'w, C>;
}

#[cfg(feature = "arena")]
impl<'w, C: DawgChar> NodeAlloc<'w, C> for Arena<DawgNode<'w, C>> {
    fn alloc_node(&'w self, node: DawgNode<'w, C>) -> &'w DawgNode<'w, C> {
        self.alloc(node)
    }
}

/// The core DAWG builder, generic over the allocator type.
///
/// Words must be added in lexicographically sorted order. Uses hash-based
/// node deduplication to minimize the graph size.
pub(crate) struct BuilderCore<'arena, C: DawgChar, A: NodeAlloc<'arena, C>> {
    arena: &'arena A,
    build_state: Vec<BuildState<'arena, C>>,
    classes: HashSet<&'arena DawgNode<'arena, C>>,
}

impl<'arena, C: DawgChar, A: NodeAlloc<'arena, C>> BuilderCore<'arena, C, A> {
    /// Creates a new builder using the provided allocator.
    pub(crate) fn new(arena: &'arena A) -> Self {
        BuilderCore {
            arena,
            build_state: vec![BuildState {
                ch: C::default(),
                node: DawgNode::new(false),
            }],
            classes: HashSet::default(),
        }
    }

    /// Adds a word to the DAWG being constructed.
    pub(crate) fn add_word(&mut self, word: impl IntoWord<C>) -> Result<(), BuilderError<C>> {
        let word = word.collect_word();
        self.add_word_slice(&word)
    }

    fn add_word_slice(&mut self, word: &[C]) -> Result<(), BuilderError<C>> {
        let prefix_length = self.prefix_length(word)?;
        self.canonicalize_suffix(prefix_length);
        self.build_state
            .extend(
                word[prefix_length..]
                    .iter()
                    .copied()
                    .mark_last()
                    .map(|(last, ch)| BuildState {
                        ch,
                        node: DawgNode::new(last),
                    }),
            );
        Ok(())
    }

    fn prefix_length(&self, word: &[C]) -> Result<usize, BuilderError<C>> {
        let mut prefix_len = 0;
        for (i, &ch) in word.iter().enumerate() {
            let is_last = i == word.len() - 1;
            if let Some(prev_state) = self.build_state.get(prefix_len + 1) {
                if ch > prev_state.ch {
                    break;
                }
                if ch < prev_state.ch || is_last {
                    return Err(BuilderError::Order(self.previous_word(), word.to_vec()));
                }
                prefix_len += 1;
            } else {
                break;
            }
        }
        Ok(prefix_len)
    }

    fn previous_word(&self) -> Vec<C> {
        self.build_state[1..].iter().map(|e| e.ch).collect()
    }

    fn canonicalize_suffix(&mut self, target_length: usize) {
        assert!(self.build_state.len() > target_length);
        let target_length = target_length
            .checked_add(1)
            .expect("target_length overflow");
        while self.build_state.len() > target_length {
            let state = self.pop_build_state();
            let child = self.canonicalize(state.node);
            self.add_build_state_child(state.ch, child);
        }
    }

    fn pop_build_state(&mut self) -> BuildState<'arena, C> {
        self.build_state
            .pop()
            .expect("Build state will always have at least one entry")
    }

    fn add_build_state_child(&mut self, ch: C, node: &'arena DawgNode<'arena, C>) {
        self.build_state
            .last_mut()
            .expect("Build state will always have at least one entry")
            .node
            .insert(ch, node)
    }

    fn canonicalize(&mut self, mut node: DawgNode<'arena, C>) -> &'arena DawgNode<'arena, C> {
        debug_assert!(
            node.children().all(|(_, ch)| self.classes.contains(ch)),
            "Cannot canonicalize unless all children are canonical"
        );

        if let Some(&val) = self.classes.get(&node) {
            val
        } else {
            node.set_canonical();
            let val = self.arena.alloc_node(node);
            self.classes.insert(val);
            val
        }
    }

    /// Finalizes the DAWG construction and returns the root node.
    pub(crate) fn build(mut self) -> &'arena DawgNode<'arena, C> {
        self.canonicalize_suffix(0);
        let root_node = self.pop_build_state().node;
        self.canonicalize(root_node)
    }
}

/// A builder for constructing DAWG wordlists incrementally.
///
/// This builder constructs a minimal DAWG (Directed Acyclic Word Graph) by adding words
/// one at a time. Words must be added in lexicographically sorted order.
///
/// The builder uses an arena allocator for efficient memory management and hash-based
/// node deduplication to minimize the graph size.
#[cfg(feature = "arena")]
pub struct Builder<'arena, C: DawgChar>(BuilderCore<'arena, C, Arena<DawgNode<'arena, C>>>);

#[cfg(feature = "arena")]
impl<'arena, C: DawgChar> Builder<'arena, C> {
    /// Creates a new DAWG builder using the provided arena for node allocation.
    pub fn new(arena: &'arena Arena<DawgNode<'arena, C>>) -> Self {
        Builder(BuilderCore::new(arena))
    }

    /// Adds a word to the DAWG being constructed.
    ///
    /// The word can be any type that implements [`IntoWord`], including `&str`,
    /// `String`, `&[u8]`, `Vec<u8>`, or fixed-size arrays like `[u8; 3]`.
    ///
    /// # Errors
    ///
    /// Returns `BuilderError::Order` if the word is not in lexicographically sorted order
    /// relative to the previously added word.
    ///
    /// # Panics
    ///
    /// Panics if the word is empty.
    pub fn add_word(&mut self, word: impl IntoWord<C>) -> Result<(), BuilderError<C>> {
        self.0.add_word(word)
    }

    /// Finalizes the DAWG construction and returns the root node.
    ///
    /// This method consumes the builder and performs final canonicalization
    /// to produce a minimal DAWG structure.
    pub fn build(self) -> &'arena DawgNode<'arena, C> {
        self.0.build()
    }
}


/// Errors that can occur when building a DAWG wordlist.
#[derive(Debug, PartialEq)]
pub enum BuilderError<C: DawgChar> {
    /// Words were not provided in lexicographically sorted order.
    ///
    /// Contains the two words that were out of order (previous word, current word).
    Order(Vec<C>, Vec<C>),
}

impl<C: DawgChar> std::fmt::Display for BuilderError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuilderError::Order(s1, s2) => write!(f, "OrderError - {s1:?} came before {s2:?}"),
        }
    }
}

impl<C: DawgChar> Error for BuilderError<C> {}

struct BuildState<'arena, C: DawgChar> {
    ch: C,
    node: DawgNode<'arena, C>,
}

/// Builds a DAWG from an iterator of words and returns the root node.
///
/// Each word must implement [`IntoWord`], allowing this function to accept
/// `&str`, `String`, slices, vectors, arrays, or any other supported word type.
///
/// Words **must** be provided in lexicographically sorted order, or this function will
/// return an error. This requirement allows the builder to construct a minimal DAWG
/// efficiently in a single pass.
///
/// # Examples
///
/// Building from byte sequences:
///
/// ```
/// use libdawg::dawg::builder::build_dawg;
/// use libdawg::dawg::Arena;
///
/// let arena = Arena::new();
/// let words: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![1, 2, 4], vec![2, 3, 4]];
/// let root = build_dawg(&arena, words).unwrap();
///
/// let contains = |seq: &[u8]| seq.iter().try_fold(root, |n, &ch| n.get(ch)).is_some_and(|n| n.is_word());
/// assert!(contains(&[1, 2, 3]));
/// assert!(!contains(&[1, 2, 5]));
/// ```
///
/// Building from strings:
///
/// ```
/// use libdawg::dawg::builder::build_dawg;
/// use libdawg::dawg::Arena;
///
/// let arena = Arena::new();
/// let root = build_dawg(&arena, ["APPLE", "BANANA", "CHERRY"]).unwrap();
///
/// let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
/// assert!(is_word("BANANA"));
/// assert!(!is_word("APRICOT"));
/// ```
#[cfg(feature = "arena")]
pub fn build_dawg<'arena, C, W>(
    arena: &'arena Arena<DawgNode<'arena, C>>,
    words: impl IntoIterator<Item = W>,
) -> Result<&'arena DawgNode<'arena, C>, BuilderError<C>>
where
    C: DawgChar,
    W: IntoWord<C>,
{
    let mut builder = Builder::new(arena);
    for word in words {
        builder.add_word(word)?;
    }
    Ok(builder.build())
}

/// Builds a DAWG from a dictionary file and returns the root node.
///
/// Reads words from a text file (one word per line) and constructs a DAWG. Words must
/// be in sorted order. Lines starting with '#' are treated as comments and ignored.
/// Empty lines are skipped.
///
/// # Examples
///
/// ```no_run
/// use libdawg::dawg::builder::build_dawg_from_file;
/// use libdawg::dawg::Arena;
///
/// let arena = Arena::new();
/// let root = build_dawg_from_file(&arena, "dictionary.txt").unwrap();
/// ```
#[cfg(feature = "arena")]
pub fn build_dawg_from_file<'arena>(
    arena: &'arena Arena<DawgNode<'arena, char>>,
    filename: &str,
) -> Result<&'arena DawgNode<'arena, char>, Box<dyn Error>> {
    let mut builder = Builder::new(arena);
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);

    // Instead of using BufReader::readlines() we optimize by calling read_line repeatedly which
    // allows us to reuse the same string instead of allocating a new string for every line.
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
    Ok(builder.build())
}

/// Returns true if this line is a comment.
pub(crate) fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

#[cfg(test)]
mod test {
    use super::*;

    #[cfg(feature = "arena")]
    fn order_err(a: &str, b: &str) -> BuilderError<char> {
        BuilderError::Order(a.chars().collect(), b.chars().collect())
    }

    #[cfg(feature = "arena")]
    #[test]
    fn graph_shares_nodes() {
        let arena1 = Arena::new();
        let _ = build_dawg(&arena1, ["ABCDEF"]).unwrap();
        assert_eq!(arena1.len(), "ABCDEF".len() + 1);

        let arena2 = Arena::new();
        let _ =
            build_dawg(&arena2, ["ABCDEF", "ABDEF", "ABEF", "AF"]).unwrap();
        assert_eq!(arena1.len(), arena2.len());
    }

    #[cfg(feature = "arena")]
    #[test]
    fn graph_shares_nodes_unicode() {
        let arena1 = Arena::new();
        build_dawg(&arena1, ["授人以鱼不如授人以渔"]).unwrap();

        let arena2 = Arena::new();
        build_dawg(&arena2, ["授人以渔", "授人以鱼不如授人以渔"]).unwrap();
        assert_eq!(arena1.len(), arena2.len());
    }

    #[cfg(feature = "arena")]
    #[test]
    fn sorted_input_words_gives_no_error() {
        let arena = Arena::new();
        let res = build_dawg(&arena, ["ALFA", "BRAVO", "CHARLIE", "DELTA"]);
        assert!(res.is_ok());
    }

    #[cfg(feature = "arena")]
    #[test]
    fn unsorted_input_words_gives_error() {
        use itertools::Itertools;
        const SORTED_WORDS: [&str; 8] = [
            "ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL",
        ];
        let arena = Arena::new();
        let mut sorted_count = 0;
        // Go through all possible permutations and see that each permutation except the sorted one
        // returns an error.
        let permutations = SORTED_WORDS
            .iter()
            .cloned()
            .permutations(SORTED_WORDS.len());
        for wordlist in permutations {
            let is_sorted = wordlist == SORTED_WORDS;
            let res = build_dawg(&arena, &wordlist);
            assert_eq!(res.is_ok(), is_sorted);
            sorted_count += is_sorted as i32;
        }

        assert_eq!(sorted_count, 1);
    }

    #[cfg(feature = "arena")]
    #[test]
    fn same_word_twice_in_input_words_gives_error() {
        let arena = Arena::new();
        let res = build_dawg(&arena, ["ALFA", "BRAVO", "CHARLIE", "CHARLIE"]);
        assert_eq!(res.unwrap_err(), order_err("CHARLIE", "CHARLIE"));
    }

    #[cfg(feature = "arena")]
    #[test]
    fn unsorted_input_words_gives_unsorted_words_in_error() {
        let arena = Arena::new();
        let res = build_dawg(
            &arena,
            [
                "ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "GOLF", "FOXTROT", "HOTEL",
            ],
        );
        assert_eq!(res.unwrap_err(), order_err("GOLF", "FOXTROT"));

        let arena = Arena::new();
        let res = build_dawg(&arena, ["ZULU", "ALFA", "BRAVO", "CHARLIE"]);
        assert_eq!(res.unwrap_err(), order_err("ZULU", "ALFA"));
    }

    #[test]
    fn comment_that_starts_with_pound() {
        let comment = is_comment("# This is a comment");
        assert!(comment)
    }

    #[test]
    fn comment_with_whitespace_before_pound() {
        let comment = is_comment("        # This is a comment with whitespace");
        assert!(comment)
    }

    #[test]
    fn non_comment() {
        let not_comment = is_comment("REVERBERATE");
        assert!(!not_comment)
    }

    #[test]
    fn non_comment_whitespace() {
        let not_comment = is_comment(" REVERBERATE");
        assert!(!not_comment)
    }

    #[cfg(feature = "arena")]
    #[test]
    fn suffixes_are_shared() {
        let testdata = [
            "ASUFFIX",
            "BSUFFIX",
            "CDESUFFIX",
            "FFFFFFFSUFFIX",
            "INBETWEEN",
            "JSUFFIX",
            "XXSUFFIX",
        ];

        let arena = Arena::new();
        let root = build_dawg(&arena, testdata).unwrap();
        let suffix_node = root.get('A').unwrap().get('S').unwrap();
        for word in testdata {
            if word.ends_with("SUFFIX") {
                let prefix_len = word.len() - "SUFFIX".len();
                let prefix = &word[..prefix_len + 1];
                let node = prefix
                    .chars()
                    .fold(root, |node, ch| node.get(ch).unwrap());
                assert_eq!(node, suffix_node);
                assert!(std::ptr::addr_eq(node, suffix_node));
            }
        }
    }

    #[cfg(feature = "arena")]
    fn contains<C: DawgChar>(root: &DawgNode<C>, word: impl IntoIterator<Item = C>) -> bool {
        word.into_iter()
            .try_fold(root, |n, ch| n.get(ch))
            .is_some_and(|n| n.is_word())
    }

    #[cfg(feature = "arena")]
    #[test]
    fn generic_dawg_with_u8() {
        let arena = Arena::new();
        let mut builder = Builder::<u8>::new(&arena);
        builder.add_word([1, 2, 3]).unwrap();
        builder.add_word([1, 2, 4]).unwrap();
        builder.add_word([2, 3, 4]).unwrap();
        let root = builder.build();
        assert!(contains(root, [1, 2, 3]));
        assert!(contains(root, [1, 2, 4]));
        assert!(contains(root, [2, 3, 4]));
        assert!(!contains(root, [1, 2, 5]));
        assert!(!contains(root, [1, 2]));
    }

    #[cfg(feature = "arena")]
    #[test]
    fn generic_dawg_with_build_dawg() {
        let arena = Arena::new();
        let words: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![1, 2, 4], vec![2, 3, 4]];
        let root = build_dawg(&arena, words).unwrap();
        assert!(contains(root, [1, 2, 3]));
        assert!(!contains(root, [1, 2, 5]));
    }
}
