//! The published log and classic replay regressions without the HTTP engine.
#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_cassette::agent::replay::register_all;
use rig_cassette::effect_log::{self, *};

#[path = "../../src/agent/replay/tests.rs"]
mod agent_replay;
#[path = "../../src/effect_log/tests.rs"]
mod log;
#[path = "../../src/effect_log/log/stable_hash_tests.rs"]
mod stable_hash_tests;
