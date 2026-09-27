//! Portable streaming types re-exported for classic runtime users.
//!
//! ```
//! use rig_agent::streaming::{BlockId, MintKind, StreamEvent};
//!
//! let event = StreamEvent::text(BlockId::minted(MintKind::Text, 0), "hi");
//! assert_eq!(event.name(), "BlockDelta");
//! ```

pub use rig_core::streaming::*;
