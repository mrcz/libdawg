use std::cmp::Ordering;
use std::hash;

use super::char_trait::DawgChar;

/// A compact representation of the children of a DawgNode that doesn't allocate until
/// there are at least three children.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Children<'w, C: DawgChar> {
    /// No children.
    None,
    /// Exactly one child (letter, node).
    One((C, &'w DawgNode<'w, C>)),
    /// Exactly two children (letter1, node1, letter2, node2).
    Two((C, &'w DawgNode<'w, C>, C, &'w DawgNode<'w, C>)),
    /// Three or more children stored in a vector.
    Many(Vec<(C, &'w DawgNode<'w, C>)>),
}

impl<'w, C: DawgChar> Children<'w, C> {
    /// Gets the child at the specified index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<(C, &'w DawgNode<'w, C>)> {
        match &self {
            Children::None => None,
            Children::One(child) => match index {
                0 => Some(*child),
                _ => None,
            },
            Children::Two((c1, n1, c2, n2)) => match index {
                0 => Some((*c1, *n1)),
                1 => Some((*c2, *n2)),
                _ => None,
            },
            Children::Many(children) => children.get(index).cloned(),
        }
    }

    /// Returns new `Children` with the edge labeled `letter` pointing to `new_child`.
    ///
    /// Panics if `letter` is not present.
    pub(crate) fn with_replaced_child(
        &self,
        letter: C,
        new_child: &'w DawgNode<'w, C>,
    ) -> Children<'w, C> {
        match self {
            Children::None => panic!("with_replaced_child: letter not found"),
            Children::One((c, _)) => {
                assert!(*c == letter, "with_replaced_child: letter not found");
                Children::One((letter, new_child))
            }
            Children::Two((c1, n1, c2, n2)) => {
                if *c1 == letter {
                    Children::Two((letter, new_child, *c2, *n2))
                } else if *c2 == letter {
                    Children::Two((*c1, *n1, letter, new_child))
                } else {
                    panic!("with_replaced_child: letter not found")
                }
            }
            Children::Many(children) => {
                let new_children: Vec<_> = children
                    .iter()
                    .map(|&(c, n)| {
                        if c == letter {
                            (c, new_child)
                        } else {
                            (c, n)
                        }
                    })
                    .collect();
                debug_assert!(new_children.iter().any(|&(c, n)| c == letter && std::ptr::eq(n, new_child)),
                    "with_replaced_child: letter not found");
                Children::Many(new_children)
            }
        }
    }

    /// Returns new `Children` with an additional edge inserted in sorted position.
    ///
    /// Panics if `letter` already exists.
    pub(crate) fn with_added_child(
        &self,
        letter: C,
        child: &'w DawgNode<'w, C>,
    ) -> Children<'w, C> {
        match self {
            Children::None => Children::One((letter, child)),
            Children::One((c1, n1)) => {
                debug_assert!(*c1 != letter, "with_added_child: letter already exists");
                if letter < *c1 {
                    Children::Two((letter, child, *c1, *n1))
                } else {
                    Children::Two((*c1, *n1, letter, child))
                }
            }
            Children::Two((c1, n1, c2, n2)) => {
                debug_assert!(
                    *c1 != letter && *c2 != letter,
                    "with_added_child: letter already exists"
                );
                let mut v = vec![(*c1, *n1), (*c2, *n2), (letter, child)];
                v.sort_by_key(|&(c, _)| c);
                Children::Many(v)
            }
            Children::Many(children) => {
                debug_assert!(
                    children.iter().all(|&(c, _)| c != letter),
                    "with_added_child: letter already exists"
                );
                let pos = children.partition_point(|&(c, _)| c < letter);
                let mut new_children = children.clone();
                new_children.insert(pos, (letter, child));
                Children::Many(new_children)
            }
        }
    }

    /// Returns new `Children` with the edge labeled `letter` removed.
    ///
    /// Panics if `letter` is not present.
    pub(crate) fn without_child(&self, letter: C) -> Children<'w, C> {
        match self {
            Children::None => panic!("without_child: letter not found"),
            Children::One((c, _)) => {
                assert!(*c == letter, "without_child: letter not found");
                Children::None
            }
            Children::Two((c1, n1, c2, n2)) => {
                if *c1 == letter {
                    Children::One((*c2, *n2))
                } else if *c2 == letter {
                    Children::One((*c1, *n1))
                } else {
                    panic!("without_child: letter not found")
                }
            }
            Children::Many(children) => {
                let new_children: Vec<_> =
                    children.iter().filter(|&&(c, _)| c != letter).cloned().collect();
                debug_assert!(
                    new_children.len() == children.len() - 1,
                    "without_child: letter not found"
                );
                match new_children.len() {
                    0 => Children::None,
                    1 => Children::One(new_children[0]),
                    2 => Children::Two((
                        new_children[0].0,
                        new_children[0].1,
                        new_children[1].0,
                        new_children[1].1,
                    )),
                    _ => Children::Many(new_children),
                }
            }
        }
    }
}

/// An iterator over the children of a DawgNode.
#[derive(Clone)]
pub struct ChildIter<'w, C: DawgChar> {
    node: &'w DawgNode<'w, C>,
    index: Option<usize>,
}

impl<'w, C: DawgChar> Iterator for ChildIter<'w, C> {
    type Item = (C, &'w DawgNode<'w, C>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index?;
        let next_child = self.node.children.get(index);
        self.index = if next_child.is_some() {
            index.checked_add(1)
        } else {
            None
        };
        next_child
    }

    /// Since we know the exact size, we can do better than the default implementation.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = match self.index {
            Some(i) => self.node.child_count().saturating_sub(i),
            None => 0,
        };
        (remaining, Some(remaining))
    }
}

impl<C: DawgChar> ExactSizeIterator for ChildIter<'_, C> {}

/// A node in the directed acyclic word graph.
#[derive(Clone, Debug, Eq)]
pub struct DawgNode<'w, C: DawgChar> {
    children: Children<'w, C>,
    word: bool,
    #[cfg(debug_assertions)]
    canonical: bool,
}

impl<'w, C: DawgChar> DawgNode<'w, C> {
    /// Creates a new DAWG node.
    ///
    /// # Arguments
    ///
    /// * `word` - Whether this node represents the end of a valid word
    #[cfg(debug_assertions)]
    pub fn new(word: bool) -> Self {
        DawgNode {
            children: Children::None,
            word,
            canonical: false,
        }
    }

    /// Creates a new DAWG node.
    ///
    /// # Arguments
    ///
    /// * `word` - Whether this node represents the end of a valid word
    #[cfg(not(debug_assertions))]
    pub fn new(word: bool) -> Self {
        DawgNode {
            children: Children::None,
            word,
        }
    }

    /// Returns the node that letter's edge leads to, or None if no such edge exists.
    #[inline]
    pub fn get(&'w self, letter: C) -> Option<&'w DawgNode<'w, C>> {
        match &self.children {
            Children::None => None,
            Children::One((ch, node)) => (*ch == letter).then_some(*node),
            Children::Two((c1, n1, c2, n2)) => {
                if letter == *c1 {
                    Some(*n1)
                } else if letter == *c2 {
                    Some(*n2)
                } else {
                    None
                }
            }
            Children::Many(children) => {
                // Unrolling by 2 exposes load-level parallelism (multiple loads per cycle)
                // and is faster than both binary search and a scalar loop. Chunk sizes of
                // 2 and 3 benchmark equally; 4 is slower due to wasted work on small nodes.
                let chunks = children.chunks_exact(2);
                let remainder = chunks.remainder();
                for chunk in chunks {
                    if chunk[0].0 == letter {
                        return Some(chunk[0].1);
                    }
                    if chunk[1].0 == letter {
                        return Some(chunk[1].1);
                    }
                }
                for &(ch, node) in remainder {
                    if ch == letter {
                        return Some(node);
                    }
                }
                None
            }
        }
    }

    /// True if this node corresponds to the end of a word.
    #[inline]
    pub fn is_word(&self) -> bool {
        self.word
    }

    /// Returns true if this node has the given suffix as the path to a valid word.
    #[inline]
    pub fn has_suffix<I: Iterator<Item = C>>(&self, suffix: &mut I) -> bool {
        suffix
            .try_fold(self, |no, ch| no.get(ch))
            .is_some_and(|n| n.is_word())
    }

    /// Inserts a child node.
    pub fn insert(&mut self, letter: C, value: &'w DawgNode<'w, C>) {
        debug_assert!(self.children().all(|(ch, _)| ch != letter));
        let c = (letter, value);
        match &mut self.children {
            Children::None => self.children = Children::One(c),
            Children::One((c1, n1)) => self.children = Children::Two((*c1, *n1, c.0, c.1)),
            Children::Two((c1, n1, c2, n2)) => {
                self.children = Children::Many(vec![(*c1, *n1), (*c2, *n2), c])
            }
            Children::Many(children) => children.push(c),
        };
    }

    /// Returns an iterator over all children of this node.
    #[inline]
    pub fn children(&'w self) -> ChildIter<'w, C> {
        ChildIter {
            node: self,
            index: Some(0),
        }
    }

    /// Returns the number of children.
    #[inline]
    pub fn child_count(&self) -> usize {
        match &self.children {
            Children::None => 0,
            Children::One(_) => 1,
            Children::Two(_) => 2,
            Children::Many(children) => children.len(),
        }
    }

    /// Marks this node as canonical (finalized and deduplicated).
    ///
    /// In debug builds, this sets an internal flag used for assertions.
    /// In release builds, this is a no-op.
    #[cfg(debug_assertions)]
    pub fn set_canonical(&mut self) {
        self.canonical = true;
    }

    /// Marks this node as canonical (finalized and deduplicated).
    ///
    /// In debug builds, this sets an internal flag used for assertions.
    /// In release builds, this is a no-op.
    #[cfg(not(debug_assertions))]
    pub fn set_canonical(&mut self) {}

    /// Returns a reference to the children of this node.
    pub(crate) fn children_ref(&self) -> &Children<'w, C> {
        &self.children
    }

    /// Creates a new node with the given word flag and children.
    #[cfg(debug_assertions)]
    pub(crate) fn with_children(word: bool, children: Children<'w, C>) -> Self {
        DawgNode {
            children,
            word,
            canonical: false,
        }
    }

    /// Creates a new node with the given word flag and children.
    #[cfg(not(debug_assertions))]
    pub(crate) fn with_children(word: bool, children: Children<'w, C>) -> Self {
        DawgNode { children, word }
    }

    /// Returns true if all children are canonical.
    #[cfg(debug_assertions)]
    fn canonical_children(&self) -> bool {
        self.children().all(|(_, child)| child.canonical)
    }

    #[cfg(not(debug_assertions))]
    fn canonical_children(&self) -> bool {
        true
    }
}

// Instead of using derive(PartialEq), we optimize eq by making it work on raw pointers,
// since we know that we only have to go one level down, because nodes are normalized
// bottom-up, so the children will always be normalized already.
impl<'w, C: DawgChar> PartialEq for DawgNode<'w, C> {
    fn eq(&self, rhs: &Self) -> bool {
        debug_assert!(self.canonical_children());
        debug_assert!(rhs.canonical_children());
        debug_assert!(self.children().map(|ch_node| ch_node.0).is_sorted());
        self.word == rhs.word
            && self.child_count() == rhs.child_count()
            && self
                .children()
                .zip(rhs.children())
                .all(|((ch1, n1), (ch2, n2))| {
                    let p1 = n1 as *const DawgNode<C>;
                    let p2 = n2 as *const DawgNode<C>;
                    (ch1, p1) == (ch2, p2)
                })
    }
}

// Just as for PartialEq, we use the knowledge that the children are always canonicalized;
// only hash their raw pointers instead of recursing all the way to the leaf nodes.
impl<'w, C: DawgChar> hash::Hash for DawgNode<'w, C> {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        debug_assert!(self.canonical_children());
        self.word.hash(state);
        for (letter, node) in self.children() {
            let p = node as *const DawgNode<C>;
            (letter, p).hash(state);
        }
    }
}

impl<C: DawgChar> Ord for DawgNode<'_, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert!(self.canonical_children());
        let ord = self
            .word
            .cmp(&other.word)
            .then(self.child_count().cmp(&other.child_count()));
        self.children()
            .zip(other.children())
            .fold(ord, |ord, ((c1, n1), (c2, n2))| {
                ord.then_with(|| {
                    let p1 = n1 as *const DawgNode<C>;
                    let p2 = n2 as *const DawgNode<C>;
                    c1.cmp(&c2).then(p1.cmp(&p2))
                })
            })
    }
}

impl<C: DawgChar> PartialOrd for DawgNode<'_, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn no_children() {
        let n = DawgNode::<char>::new(false);
        assert_eq!(n.children().next(), None);
        assert_eq!(n.child_count(), 0);
    }

    #[test]
    fn one_children() {
        let mut n = DawgNode::new(false);
        let c = DawgNode::new(false);
        n.insert('a', &c);
        let mut children = n.children();
        assert_eq!(children.next(), Some(('a', &c)));
        assert_eq!(children.next(), None);
        assert_eq!(n.child_count(), 1);
    }

    #[test]
    fn two_children() {
        let mut n = DawgNode::new(false);
        let c1 = DawgNode::new(false);
        let c2 = DawgNode::new(false);
        n.insert('a', &c1);
        n.insert('b', &c2);
        let mut children = n.children();
        assert_eq!(children.next(), Some(('a', &c1)));
        assert_eq!(children.next(), Some(('b', &c2)));
        assert_eq!(children.next(), None);
        assert_eq!(n.child_count(), 2);
    }

    #[test]
    fn three_children() {
        let mut n = DawgNode::new(false);
        let c1 = DawgNode::new(false);
        let c2 = DawgNode::new(false);
        let c3 = DawgNode::new(false);
        n.insert('a', &c1);
        n.insert('b', &c2);
        n.insert('c', &c3);
        let mut children = n.children();
        assert_eq!(children.next(), Some(('a', &c1)));
        assert_eq!(children.next(), Some(('b', &c2)));
        assert_eq!(children.next(), Some(('c', &c3)));
        assert_eq!(children.next(), None);
        assert_eq!(n.child_count(), 3);
    }

    #[cfg(feature = "arena")]
    #[test]
    fn a_thousand_children() {
        use typed_arena::Arena;
        let arena = Arena::new();
        let mut n = DawgNode::new(false);
        let letters = (0..).filter_map(std::char::from_u32).take(1000);
        for ch in letters.clone() {
            n.insert(ch, arena.alloc(DawgNode::new(false)));
        }
        let mut children = n.children();
        let cmp_child = DawgNode::new(false);
        for ch in letters {
            assert_eq!(children.next(), Some((ch, &cmp_child)));
        }
        assert_eq!(children.next(), None);
        assert_eq!(n.child_count(), 1000);
    }
}
