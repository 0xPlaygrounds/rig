#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[allow(dead_code)]
#[path = "common/ecs_agent.rs"]
mod ecs_agent;
#[path = "common/ecs_extractor.rs"]
mod ecs_extractor;

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[path = "common/cassettes.rs"]
mod cassettes;
#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "providers/copilot/mod.rs"]
mod copilot;
