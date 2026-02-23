use std::fmt::Debug;
use std::hash::Hash;

/// Trait for types that can serve as edge labels in a DAWG.
///
/// This trait is automatically implemented for any type satisfying all the
/// required bounds (`char`, `u8`, `u16`, `u32`, etc.).
///
/// - `Copy`: edges store labels by value
/// - `Eq + Ord`: comparing and ordering edge labels
/// - `Hash`: node deduplication during DAWG construction
/// - `Debug`: debug printing of nodes
/// - `Default`: sentinel value for the builder's root node
pub trait DawgChar: Copy + Eq + Ord + Hash + Debug + Default {}

impl<T: Copy + Eq + Ord + Hash + Debug + Default> DawgChar for T {}
