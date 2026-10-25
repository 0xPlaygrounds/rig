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
//! AWS event-stream support. `traffic` adds [`traffic`], every HTTP exchange
//! through a transport as a scrubbed JSON line (native), with the scrubber
//! cassettes use; `http` includes it. `jsonl` adds [`effect_log::jsonl`], effect
//! logs appended to and read from JSON-lines files, also as a file of a
//! session's store. No optional feature is enabled by default.
//!
//! ```
//! let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
//! let snapshot = recorder.log();
//! assert!(snapshot.is_empty());
//! ```

pub mod effect_log;

#[cfg(feature = "agent")]
pub mod agent;

#[cfg(feature = "http")]
pub mod http;

// The traffic log writes on a thread of its own; say so on wasm before an
// unbuildable recorder fails at run time.
#[cfg(all(feature = "traffic", target_family = "wasm"))]
compile_error!(
    "rig-cassette's `traffic` feature (and `http`) records on a native thread; wasm has none."
);

#[cfg(feature = "traffic")]
mod scrub;

#[cfg(feature = "traffic")]
pub mod traffic;
