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
#[path = "common/ecs_extractor.rs"]
mod ecs_extractor;
#[path = "common/ecs_lifecycle.rs"]
mod ecs_lifecycle;
#[path = "common/ecs_observation.rs"]
mod ecs_observation;
#[path = "common/ecs_termination.rs"]
mod ecs_termination;
use rig_test_support::goldens;
use rig_test_support::raw_capture;
use rig_test_support::reasoning;
use rig_test_support::stream_faults;
use rig_test_support::support;

#[path = "providers/openai/mod.rs"]
mod openai;

#[path = "common/ecs_cache.rs"]
mod ecs_cache;

#[allow(
    dead_code,
    reason = "each provider exercises its own subset of matrix cells"
)]
#[path = "common/ecs_matrix.rs"]
mod ecs_matrix;

use rig_test_support::history_survival;
use rig_test_support::matrix;

#[path = "common/image_inputs.rs"]
mod image_inputs;

#[path = "common/request_identity.rs"]
mod request_identity;
