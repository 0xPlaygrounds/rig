//! Shared implementation of repository test drivers, fixtures and matrix registration.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

pub mod cache_conformance;
pub mod cache_prefix;
pub mod cassettes;
pub mod comparison_guard;
pub mod ecs_agent;
pub mod goldens;
pub mod history_survival;
pub mod matrix;
pub mod matrix_registry;
pub mod raw_capture;
pub mod reasoning;
pub mod scenario_registry;
pub mod stream_faults;
pub mod support;
