#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "common/cache_conformance.rs"]
mod cache_conformance;
#[path = "common/cache_prefix.rs"]
mod cache_prefix;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[path = "common/cassettes.rs"]
mod cassettes;
#[path = "common/ecs_agent.rs"]
mod ecs_agent;
#[path = "common/ecs_goldens.rs"]
mod ecs_goldens;
#[path = "common/ecs_lifecycle.rs"]
mod ecs_lifecycle;
#[path = "common/ecs_observation.rs"]
mod ecs_observation;
#[path = "common/ecs_termination.rs"]
mod ecs_termination;
#[path = "common/goldens.rs"]
mod goldens;
#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "providers/anthropic/mod.rs"]
mod anthropic;

#[path = "common/ecs_cache.rs"]
mod ecs_cache;
