#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_test_support::ecs_agent;
#[path = "common/ecs_extractor.rs"]
mod ecs_extractor;

use rig_test_support::raw_capture;
use rig_test_support::support;

use rig_test_support::cache_conformance;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;

#[path = "providers/doubleword/mod.rs"]
mod doubleword;

use rig_test_support::goldens;

#[allow(dead_code)]
#[path = "common/ecs_termination.rs"]
mod ecs_termination;

use rig_test_support::stream_faults;

#[allow(
    dead_code,
    reason = "each provider exercises its own subset of matrix cells"
)]
#[path = "common/ecs_matrix.rs"]
mod ecs_matrix;

use rig_test_support::history_survival;
use rig_test_support::matrix;
