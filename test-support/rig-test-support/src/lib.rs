//! Shared implementation of repository test drivers, fixtures and matrix registration.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

extern crate self as rig_test_support;

pub mod cache_conformance;
pub mod cache_prefix;
pub mod cassettes;
pub mod recording;
pub use rig_cassette_inventory::cassette_inventory;
pub use rig_cassette_macros::cassette;
pub mod ecs_agent;
pub mod ecs_goldens;
pub mod goldens;
pub mod matrix;
pub mod matrix_registry;
pub mod raw_capture;
pub mod reasoning;
pub mod scenario_registry;
pub mod stream_faults;
pub mod support;
