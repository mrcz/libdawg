//! A simple chunked arena allocator for DAWG nodes.
//!
//! Provides stable-address allocation without depending on `typed-arena`.
//! Used internally by [`OwnedDawg`](super::owned::OwnedDawg).

use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;

use super::builder::NodeAlloc;
use super::char_trait::DawgChar;
use super::node::DawgNode;

const CHUNK_CAP: usize = 64;

/// A chunked arena that allocates values with stable heap addresses.
///
/// Each chunk is a `Box<[MaybeUninit<T>; CHUNK_CAP]>`, so individual
/// elements never move even when new chunks are added.
pub(crate) struct NodeArena<T> {
    chunks: RefCell<Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>>,
    len: Cell<usize>,
}

impl<T> NodeArena<T> {
    /// Creates an empty arena.
    pub fn new() -> Self {
        NodeArena {
            chunks: RefCell::new(Vec::new()),
            len: Cell::new(0),
        }
    }

    /// Allocates a value in the arena and returns a reference to it.
    ///
    /// The returned reference is valid for the lifetime of the arena.
    /// Internally, each chunk is heap-allocated via `Box`, so elements
    /// have stable addresses even as new chunks are added.
    pub fn alloc(&self, value: T) -> &T {
        let len = self.len.get();
        let chunk_idx = len / CHUNK_CAP;
        let offset = len % CHUNK_CAP;

        let mut chunks = self.chunks.borrow_mut();
        if chunk_idx == chunks.len() {
            chunks.push(Box::new([const { MaybeUninit::uninit() }; CHUNK_CAP]));
        }

        // Get a raw pointer to the slot. The pointer remains valid because
        // the Box's heap allocation never moves.
        let slot_ptr: *mut MaybeUninit<T> = &mut chunks[chunk_idx][offset];

        self.len.set(len + 1);

        // SAFETY: slot_ptr points into a Box<[MaybeUninit<T>; CHUNK_CAP]>.
        // The Box is heap-allocated and its address is stable. We never
        // mutate or move existing elements — only append to new slots.
        // The RefCell borrow is released, but the pointer targets heap
        // memory owned by the Box (not the Vec's buffer).
        unsafe {
            (*slot_ptr).write(value);
            (*slot_ptr).assume_init_ref()
        }
    }

    /// Returns the number of values allocated in this arena.
    pub fn len(&self) -> usize {
        self.len.get()
    }
}

impl<T> Drop for NodeArena<T> {
    fn drop(&mut self) {
        let chunks = self.chunks.get_mut();
        let len = *self.len.get_mut();

        for i in 0..len {
            let chunk_idx = i / CHUNK_CAP;
            let offset = i % CHUNK_CAP;
            // SAFETY: elements 0..len were initialized via alloc().
            unsafe {
                chunks[chunk_idx][offset].assume_init_drop();
            }
        }
    }
}

impl<'w, C: DawgChar> NodeAlloc<'w, C> for NodeArena<DawgNode<'w, C>> {
    fn alloc_node(&'w self, node: DawgNode<'w, C>) -> &'w DawgNode<'w, C> {
        // SAFETY: The returned reference points into heap memory owned by
        // the arena. The caller (OwnedDawg) ensures the arena outlives all
        // references by owning both in the same struct. The 'w lifetime is
        // transmuted to 'static by the caller via raw pointer casts — same
        // pattern used previously with typed_arena.
        unsafe { &*(self.alloc(node) as *const DawgNode<'w, C>) }
    }
}
