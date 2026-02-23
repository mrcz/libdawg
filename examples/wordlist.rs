//! Example: building a Wordlist wrapper around DawgNode.
//!
//! This shows how to create a convenient high-level API on top of the raw DAWG
//! node interface. The `Wordlist` struct wraps a root `DawgNode` and provides
//! word lookup, prefix checking, and iteration.
//!
//! Run with: cargo run --example wordlist

use libdawg::dawg::builder::build_dawg;
use libdawg::dawg::{Arena, DawgNode};

/// A convenient wrapper around a DAWG root node for word validation.
struct Wordlist<'w> {
    root: &'w DawgNode<'w, char>,
}

impl<'w> Wordlist<'w> {
    fn new(root: &'w DawgNode<'w, char>) -> Self {
        Wordlist { root }
    }

    /// Returns true if the word is in the wordlist.
    fn is_word(&self, word: &str) -> bool {
        word.chars()
            .try_fold(self.root, |node, ch| node.get(ch))
            .is_some_and(|n| n.is_word())
    }

    /// Returns true if any word in the wordlist starts with the given prefix.
    fn has_prefix(&self, prefix: &str) -> bool {
        prefix
            .chars()
            .try_fold(self.root, |node, ch| node.get(ch))
            .is_some()
    }

    /// Returns all words in the wordlist.
    fn all_words(&self) -> Vec<String> {
        let mut words = Vec::new();
        let mut stack = Vec::new();
        Self::collect_words(self.root, &mut stack, &mut words);
        words
    }

    fn collect_words(
        node: &DawgNode<'_, char>,
        prefix: &mut Vec<char>,
        words: &mut Vec<String>,
    ) {
        if node.is_word() {
            words.push(prefix.iter().collect());
        }
        for (ch, child) in node.children() {
            prefix.push(ch);
            Self::collect_words(child, prefix, words);
            prefix.pop();
        }
    }
}

fn main() {
    let arena = Arena::new();
    let words = ["BAKE", "BAKED", "BAKER", "CAKE", "CAKED", "FAKE", "LAKE"];
    let root = build_dawg(&arena, words).unwrap();
    let wordlist = Wordlist::new(root);

    // Word lookup
    println!("Word lookup:");
    for word in ["BAKE", "BAKER", "BAKES", "CAKE", "LAKE", "MAKE"] {
        println!("  {word}: {}", if wordlist.is_word(word) { "yes" } else { "no" });
    }

    // Prefix checking
    println!("\nPrefix checking:");
    for prefix in ["BA", "CAK", "MA", "FAK"] {
        println!("  {prefix}*: {}", if wordlist.has_prefix(prefix) { "yes" } else { "no" });
    }

    // List all words
    println!("\nAll words: {:?}", wordlist.all_words());
}
