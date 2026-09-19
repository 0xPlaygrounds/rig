#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use rig_test_support::ecs_agent;
#[path = "common/ecs_extractor.rs"]
mod ecs_extractor;

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
use rig_test_support::raw_capture;
use rig_test_support::reasoning;
use rig_test_support::support;

#[path = "providers/copilot/mod.rs"]
mod copilot;
