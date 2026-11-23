//! Runtime scenarios run once, against real replies. The agent loop, tool
//! lifecycle, turn endings, resume, memory and the effect bus are the same
//! for every provider, so each scenario here runs once (or once per
//! reply shape that matters to it) over a transport serving replies from the
//! reply bank (`crates/rig-cassette/fixtures/bank/`, built by `cargo xtask
//! cassette bank` from the cassette corpus). Each reply still passes through
//! its provider's real decoder. The per-provider copies of these scenarios
//! under `tests/providers/` repeat the same runtime over each provider's own
//! cassette.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "runtime/cells.rs"]
mod cells;
#[path = "runtime/decode.rs"]
mod decode;
#[path = "runtime/families.rs"]
mod families;
#[path = "runtime/faults.rs"]
mod faults;
#[path = "runtime/lifecycle.rs"]
mod lifecycle;
#[path = "runtime/sessions.rs"]
mod sessions;
#[path = "runtime/termination.rs"]
mod termination;
#[path = "runtime/typed_options.rs"]
mod typed_options;
#[path = "runtime/wires.rs"]
mod wires;
