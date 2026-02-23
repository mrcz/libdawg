/// DAWG builder module for constructing wordlists from sorted word lists.
pub mod builder;
/// Trait for types that can serve as DAWG edge labels.
pub mod char_trait;
/// DAWG node and children module containing the core graph data structures.
pub mod children;
/// Internal chunked arena allocator.
pub(crate) mod node_arena;
/// Owned DAWG that manages its own arena internally.
pub mod owned;

pub use builder::IntoWord;
pub use char_trait::DawgChar;
pub use children::DawgNode;

/// Re-export `typed_arena::Arena` for use with the arena-based API.
#[cfg(feature = "arena")]
pub use typed_arena::Arena;

#[cfg(test)]
#[cfg(feature = "arena")]
mod test {
    use super::builder::{build_dawg, build_dawg_from_file};
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use typed_arena::Arena;

    fn is_word<'w>(root: &'w super::DawgNode<'w, char>, word: &str) -> bool {
        word.chars()
            .try_fold(root, |n, ch| n.get(ch))
            .is_some_and(|n| n.is_word())
    }

    #[test]
    fn all_words() {
        let dict_filename = "../dict-sv.txt";
        let arena = Arena::new();
        let root = build_dawg_from_file(&arena, dict_filename).unwrap();
        let file = File::open(dict_filename).unwrap();
        for line in BufReader::new(file).lines() {
            let word = &line.unwrap();
            if let Some(first_char) = word.chars().next() {
                if first_char.is_alphabetic() {
                    assert!(is_word(root, word), "{}", word);
                }
            }
        }
        // test some non-words
        assert!(!is_word(root, "URSINN"));
        assert!(!is_word(root, "URSINNESS"));
        assert!(!is_word(root, "ÅTMINSTON"));
        assert!(!is_word(root, "ÅTMINSTONDE"));
    }

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
