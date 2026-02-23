//! # libdawg
//!
//! A fast, memory-efficient [DAWG](https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton)
//! (Directed Acyclic Word Graph) library for Rust.
//!
//! A DAWG is a minimal acyclic finite-state automaton — essentially a trie with shared
//! suffixes — that provides compact storage and O(word length) lookups. This implementation
//! is based on the algorithm described in [Daciuk et al. (2000)](https://arxiv.org/abs/cs/0007009v1).
//!
//! ## Features
//!
//! - **Generic over character type**: works with `char`, `u8`, `u16`, or any type implementing
//!   [`DawgChar`](dawg::DawgChar)
//! - **Compact**: suffix sharing minimizes memory usage
//! - **Fast**: O(word length) lookups with arena-allocated nodes
//! - **Thread-safe**: [`DawgNode`](dawg::DawgNode) uses only immutable arena references
//!
//! ## Quick Start
//!
//! The simplest way to build a DAWG is with [`OwnedDawg`](dawg::owned::OwnedDawg),
//! which manages allocation internally:
//!
//! ```
//! use libdawg::dawg::owned::build_owned_dawg;
//!
//! let dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE", "LAKE", "MAKE"]).unwrap();
//! let root = dawg.root();
//!
//! let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
//! assert!(is_word("CAKE"));
//! assert!(!is_word("AKE"));
//! ```
//!
//! For explicit control over allocation (requires the `arena` feature, enabled by default):
//!
//! ```
//! # #[cfg(feature = "arena")] {
//! use libdawg::dawg::builder::build_dawg;
//! use libdawg::dawg::Arena;
//!
//! let arena = Arena::new();
//! let root = build_dawg(&arena, ["BAKE", "CAKE", "FAKE", "LAKE", "MAKE"]).unwrap();
//!
//! let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
//! assert!(is_word("CAKE"));
//! assert!(!is_word("AKE"));
//! # }
//! ```
//!
//! ## Generic Usage
//!
//! The DAWG is generic over the edge label type:
//!
//! ```
//! # #[cfg(feature = "arena")] {
//! use libdawg::dawg::builder::build_dawg;
//! use libdawg::dawg::Arena;
//!
//! let arena = Arena::new();
//! let words: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![1, 2, 4], vec![2, 3, 4]];
//! let root = build_dawg(&arena, words).unwrap();
//!
//! let contains = |seq: &[u8]| seq.iter().try_fold(root, |n, &ch| n.get(ch)).is_some_and(|n| n.is_word());
//! assert!(contains(&[1, 2, 3]));
//! assert!(!contains(&[1, 2, 5]));
//! # }
//! ```

#![warn(missing_docs)]

/// Core DAWG data structure: node types, builder, and character trait.
pub mod dawg;
