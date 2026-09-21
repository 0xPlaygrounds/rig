//! Portable streaming types re-exported for classic runtime users.
//!
//! ```
//! let accumulator = rig_agent::streaming::BlockAccumulator::new();
//! assert!(accumulator.snapshot().is_empty());
//! ```

pub use rig_core::streaming::*;
