//! Bus integration suites with shared support compiled once.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    dead_code,
    reason = "test suites assert directly and share support each uses part of"
)]

#[path = "../bus_custom_persistence.rs"]
mod bus_custom_persistence;
#[path = "../bus_delivery.rs"]
mod bus_delivery;
#[path = "../bus_effects.rs"]
mod bus_effects;
#[path = "../bus_scale.rs"]
mod bus_scale;
#[path = "../bus_scene.rs"]
mod bus_scene;
#[path = "../bus_stream_delivery.rs"]
mod bus_stream_delivery;
#[path = "../bus_successors.rs"]
mod bus_successors;
#[path = "../bus_support/mod.rs"]
mod bus_support;
#[path = "../bus_tool_replay.rs"]
mod bus_tool_replay;
#[path = "../bus_witness.rs"]
mod bus_witness;
#[path = "../bus_world.rs"]
mod bus_world;
#[path = "../run_support/mod.rs"]
mod run_support;
