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
//! The `agent` and `ecs` features independently enable runtime integrations;
//! neither enables the native HTTP engine or changes JSON map/float semantics.
//! The `http` feature enables the native provider cassette engine, including
//! ordered JSON maps and round-trip float parsing. `bedrock` extends it with
//! AWS event-stream support. [`migrate`] upgrades persisted effect logs,
//! rig-ecs checkpoints and rig-agent runs offline; the `migrate` feature builds
//! its `rig-migrate` command. No optional feature is enabled by default.
//!
//! ```
//! let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
//! let snapshot = recorder.log();
//! assert!(snapshot.is_empty());
//! ```

pub mod effect_log;
pub mod migrate;

#[cfg(feature = "agent")]
pub mod agent;

#[cfg(feature = "ecs")]
pub mod ecs;

#[cfg(feature = "http")]
pub mod http;
