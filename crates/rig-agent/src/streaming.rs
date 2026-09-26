//! Portable streaming types re-exported for classic runtime users.
//!
//! ```
//! let fold = rig_agent::streaming::CompletionFold::default();
//! assert!(fold.snapshot().is_empty());
//! ```

pub use rig_core::streaming::*;
