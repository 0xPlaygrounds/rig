#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Recording and replay for Rig's effects and provider HTTP exchanges.
//!
//! [`effect_log`] is available without optional features and supports WASM.
//! The `agent` feature enables the classic-agent integration; it does not
//! enable the native HTTP engine or change JSON map/float semantics.
//! The `http` feature enables the native provider cassette engine, including
//! ordered JSON maps and round-trip float parsing. `bedrock` extends it with
//! AWS event-stream support. [`journal`] keeps an agent session's logs,
//! image blobs and effect log in a store. `jsonl` adds [`effect_log::jsonl`],
//! effect logs appended to and read from JSON-lines files, and a session store
//! of such files. No optional feature is enabled by default.
//!
//! ```
//! let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
//! let snapshot = recorder.log();
//! assert!(snapshot.is_empty());
//! ```

pub mod effect_log;
pub mod journal;

#[cfg(feature = "agent")]
pub mod agent;

#[cfg(feature = "http")]
pub mod http;
