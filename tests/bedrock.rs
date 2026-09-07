#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[cfg(feature = "bedrock")]
#[allow(dead_code)]
#[path = "common/ecs_agent.rs"]
mod ecs_agent;

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[path = "common/cassettes.rs"]
mod cassettes;
#[path = "common/support.rs"]
mod support;

#[cfg(feature = "bedrock")]
#[allow(dead_code)]
#[path = "common/ecs_observation.rs"]
mod ecs_observation;
#[cfg(feature = "bedrock")]
#[allow(dead_code)]
#[path = "common/ecs_session.rs"]
mod ecs_session;

#[path = "providers/bedrock/mod.rs"]
mod bedrock;
