#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use rig_test_support::ecs_agent;

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
use rig_test_support::history_survival;
use rig_test_support::matrix;
use rig_test_support::raw_capture;
use rig_test_support::reasoning;
use rig_test_support::support;

#[path = "providers/ollama/mod.rs"]
mod ollama;
