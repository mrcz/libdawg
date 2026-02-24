/// DAWG builder module for constructing wordlists from sorted word lists.
pub mod builder;
/// Trait for types that can serve as DAWG edge labels.
pub mod char_trait;
/// Core graph node and edge data structures.
pub mod node;
/// Internal chunked arena allocator.
pub(crate) mod node_arena;
/// Owned DAWG that manages its own arena internally.
pub mod owned;

pub use builder::IntoWord;
pub use char_trait::DawgChar;
pub use node::DawgNode;

/// Re-export `typed_arena::Arena` for use with the arena-based API.
#[cfg(feature = "arena")]
pub use typed_arena::Arena;

#[cfg(test)]
#[cfg(feature = "arena")]
mod test {
    use super::builder::build_dawg;
    use typed_arena::Arena;

    #[test]
    fn add_word() {
        let words = ["TEST", "TESTER", "WTEST"];
        let arena = Arena::new();
        let root = build_dawg(&arena, words).unwrap();

        let n = root.get('T').unwrap();
        assert!(!n.is_word());

        let n = n.get('E').unwrap();
        assert!(!n.is_word());

        let n = n.get('S').unwrap();
        assert!(!n.is_word());

        let n = n.get('T').unwrap();
        assert!(n.is_word());

        let n = n.get('E').unwrap();
        assert!(!n.is_word());

        let n = n.get('R').unwrap();
        assert!(n.is_word());

        let n = n.get('T');
        assert_eq!(n, None);
    }
}
