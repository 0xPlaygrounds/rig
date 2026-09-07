#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[allow(dead_code)]
#[path = "common/ecs_agent.rs"]
mod ecs_agent;
#[allow(dead_code)]
#[path = "common/ecs_observation.rs"]
mod ecs_observation;
#[allow(dead_code)]
#[path = "common/ecs_session.rs"]
mod ecs_session;

#[path = "providers/mira/mod.rs"]
mod mira;
