#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

// Unconditional, both of them: the cassette safety guard parses this file
// structurally and fails if either `mod` is missing or `#[cfg]`-gated.
use rig_test_support::ecs_agent;
#[path = "common/ecs_extractor.rs"]
mod ecs_extractor;

use rig_test_support::cache_conformance;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
use rig_test_support::raw_capture;
use rig_test_support::support;

#[path = "providers/llamacpp/mod.rs"]
mod llamacpp;
