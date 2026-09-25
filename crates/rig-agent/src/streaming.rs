//! Portable streaming types re-exported for classic runtime users.
//!
//! ```
//! use rig_agent::streaming::{StreamEvent, StreamFinal};
//!
//! let terminal = StreamEvent::Final(StreamFinal::new("mock", Default::default(), serde_json::Value::Null));
//! assert!(matches!(terminal, StreamEvent::Final(_)));
//! ```

pub use rig_core::streaming::*;
