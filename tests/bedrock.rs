#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[cfg(feature = "bedrock")]
use rig_test_support::ecs_agent;

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
#[cfg(feature = "bedrock")]
use rig_test_support::raw_capture;
#[cfg(feature = "bedrock")]
use rig_test_support::support;

#[cfg(feature = "bedrock")]
#[path = "providers/bedrock/mod.rs"]
mod bedrock;
