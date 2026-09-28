#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_test_support::cache_conformance;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
use rig_test_support::ecs_agent;
use rig_test_support::goldens;
use rig_test_support::history_survival;
use rig_test_support::reasoning;
use rig_test_support::stream_faults;
use rig_test_support::support;

#[path = "providers/gemini/mod.rs"]
mod gemini;

#[allow(
    dead_code,
    reason = "each provider exercises its own subset of matrix cells"
)]
#[path = "common/ecs_matrix.rs"]
mod ecs_matrix;

use rig_test_support::matrix;
