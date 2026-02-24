# libdawg

A fast, memory-efficient [DAWG](https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton)
(Directed Acyclic Word Graph) library for Rust.

A DAWG is a minimal acyclic finite-state automaton — essentially a trie with shared
suffixes — providing compact dictionary storage and O(word length) lookups. Based on
[Daciuk et al. (2000)](https://arxiv.org/abs/cs/0007009v1).

## Terminology

The API uses the terms "word" and "character" but is not limited to text. A
"character" is any type that implements `DawgChar` (`Copy + Eq + Ord + Hash +
Debug + Default`) — for example `char`, `u8`, `u16`, or a custom enum. A "word"
is simply a sequence of such characters. This means the DAWG can store any set
of sorted sequences, not just strings.

## Features

- **Generic over character type** — works with `char`, `u8`, `u16`, or any type
  implementing `DawgChar`
- **Compact** — suffix sharing minimizes memory usage
- **Fast** — O(word length) lookups with arena-allocated nodes
- **Mutable** — `OwnedDawg` supports adding and removing words after construction
- **Thread-safe** — `DawgNode` uses only immutable arena references

## Two APIs

libdawg provides two ways to build a DAWG:

|   | `OwnedDawg` | Arena-based (`build_dawg`) |
| --- | --- | --- |
| Allocation | Internal | Caller-managed `Arena` |
| Mutation | `add_word` / `remove_word` | Read-only after construction |
| Input order | Sorted (for initial build) | Sorted |
| Use when | You need to modify the DAWG after building, or want a simpler API | You want explicit lifetime control or only need a read-only DAWG |

Both produce a minimal DAWG with full suffix sharing. `OwnedDawg` preserves
minimality across mutations using clone-on-write path updates and a free-list
for arena slot reuse.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
libdawg = "1"
```

The default `arena` feature re-exports `typed_arena::Arena` as `libdawg::dawg::Arena`.
If you only need `OwnedDawg`, you can disable it to drop the `typed-arena` dependency:

```toml
[dependencies]
libdawg = { version = "1", default-features = false }
```

### Owned DAWG (no arena management)

The simplest API uses `OwnedDawg`, which manages allocation internally:

```rust
use libdawg::dawg::owned::build_owned_dawg;

fn is_word(dawg: &OwnedDawg<char>, word: &str) -> bool {
    // Start at the root of the graph
    let mut node = dawg.root();

    for ch in word.chars() {
        match node.get(ch) {
            // Follow the edge labeled `ch`
            Some(child) => node = child,
            // No edge for this character — word is not in the dictionary
            None => return false,
        }
    }

    // All characters matched; check if this node is a word endpoint
    node.is_word()
}

let dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE", "LAKE", "MAKE"]).unwrap();
assert!(is_word(&dawg, "CAKE"));
assert!(!is_word(&dawg, "AKE"));
```

### Adding and removing words

`OwnedDawg` supports mutation after construction. Words can be added in any
order (unlike the initial sorted build), and the DAWG stays minimal:

```rust
use libdawg::dawg::owned::build_owned_dawg;

let mut dawg = build_owned_dawg(["BAKE", "CAKE", "FAKE"]).unwrap();

// Add words (returns true if the word was new)
assert!(dawg.add_word("SAKE"));
assert!(!dawg.add_word("CAKE")); // already present

assert!(dawg.contains("SAKE"));
assert!(dawg.contains("BAKE"));

// Remove words (returns true if the word existed)
assert!(dawg.remove_word("BAKE"));
assert!(!dawg.remove_word("BAKE")); // already removed

assert!(!dawg.contains("BAKE"));
assert!(dawg.contains("CAKE"));
```

Removed nodes are placed on a free-list and reused by future `add_word` calls
before allocating from the arena.

### Arena-based DAWG (read-only)

For explicit control over allocation, pass your own arena. Words must be
added in **lexicographic (sorted) order**:

```rust
use libdawg::dawg::builder::build_dawg;
use libdawg::dawg::Arena;

let arena = Arena::new();
let root = build_dawg(&arena, ["BAKE", "CAKE", "FAKE", "LAKE", "MAKE"]).unwrap();

// Check word containment by traversing from the root
let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
assert!(is_word("CAKE"));
assert!(is_word("BAKE"));
assert!(!is_word("AKE"));
```

### Building from a file

```rust,no_run
use libdawg::dawg::builder::build_dawg_from_file;
use libdawg::dawg::Arena;

let arena = Arena::new();
let root = build_dawg_from_file(&arena, "dictionary.txt").unwrap();

let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
println!("Contains 'HELLO': {}", is_word("HELLO"));
```

The file should have one word per line, sorted alphabetically. Lines starting
with `#` are treated as comments.

### Generic usage (non-char types)

The DAWG is generic over the edge label type — `build_dawg` accepts any
iterator of words where each word implements `IntoWord<C>`:

```rust
use libdawg::dawg::builder::build_dawg;
use libdawg::dawg::Arena;

let arena = Arena::new();
let words: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![1, 2, 4], vec![2, 3, 4]];
let root = build_dawg(&arena, words).unwrap();

let contains = |seq: &[u8]| seq.iter().try_fold(root, |n, &ch| n.get(ch)).is_some_and(|n| n.is_word());
assert!(contains(&[1, 2, 3]));
assert!(!contains(&[1, 2, 5]));
```

### Incremental construction

For more control, use `Builder` directly:

```rust
use libdawg::dawg::builder::Builder;
use libdawg::dawg::Arena;

let arena = Arena::new();
let mut builder = Builder::new(&arena);

// Words must be added in sorted order
builder.add_word("JUMPING").unwrap();
builder.add_word("PLAYING").unwrap();
builder.add_word("RUNNING").unwrap();
builder.add_word("WALKING").unwrap();

let root = builder.build();

let is_word = |w: &str| w.chars().try_fold(root, |n, ch| n.get(ch)).is_some_and(|n| n.is_word());
assert!(is_word("RUNNING"));
```

### Walking the graph

You can traverse the DAWG directly via `DawgNode`:

```rust
use libdawg::dawg::builder::build_dawg;
use libdawg::dawg::Arena;

let arena = Arena::new();
let root = build_dawg(&arena, ["BAKE", "BAKED", "CAKE", "CAKED"]).unwrap();

// Two starting letters: 'B' and 'C'
assert_eq!(root.child_count(), 2);

let b_node = root.get('B').unwrap();
let a_node = b_node.get('A').unwrap();
let k_node = a_node.get('K').unwrap();
let e_node = k_node.get('E').unwrap();

// 'BAKE' is a word, and has child 'D' for 'BAKED'
assert!(e_node.is_word());
assert_eq!(e_node.child_count(), 1);
```

## How it works

The DAWG is constructed in a single pass over sorted input. As words are added,
the builder identifies shared suffixes and deduplicates nodes, producing a
minimal graph. All nodes are arena-allocated, so the entire graph is freed at once
when the arena is dropped.

**Mutation** (`OwnedDawg` only): Because nodes are deduplicated by structure,
modifying a shared node in-place would corrupt other words. Instead, `add_word`
and `remove_word` clone the path from the affected node to the root and
re-canonicalize each clone against the register. Reference counting tracks when
nodes become unreachable; freed slots go to a free-list and are reused by future
allocations.

## License

MIT
