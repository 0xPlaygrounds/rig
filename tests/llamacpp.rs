#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

// Unconditional, both of them: the cassette safety guard parses this file
// structurally and fails if either `mod` is missing or `#[cfg]`-gated.
#[allow(dead_code)]
#[path = "common/ecs_agent.rs"]
mod ecs_agent;
#[path = "common/ecs_extractor.rs"]
mod ecs_extractor;

#[path = "common/cache_conformance.rs"]
mod cache_conformance;
#[path = "common/cache_prefix.rs"]
mod cache_prefix;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[path = "common/cassettes.rs"]
mod cassettes;
#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[allow(dead_code)]
#[path = "common/ecs_observation.rs"]
mod ecs_observation;
#[allow(dead_code)]
#[path = "common/ecs_session.rs"]
mod ecs_session;

#[path = "providers/llamacpp/mod.rs"]
mod llamacpp;
