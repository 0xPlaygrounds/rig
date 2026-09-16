//! Run integration suites with shared support compiled once.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    dead_code,
    reason = "test suites assert directly and share support each uses part of"
)]

#[path = "../bus_support/mod.rs"]
mod bus_support;
#[path = "../memory_graph.rs"]
mod memory_graph;
#[path = "../memory_resume.rs"]
mod memory_resume;
#[path = "../run_binding.rs"]
mod run_binding;
#[path = "../run_checkpoint.rs"]
mod run_checkpoint;
#[path = "../run_commands.rs"]
mod run_commands;
#[path = "../run_content_binary.rs"]
mod run_content_binary;
#[path = "../run_content_edits.rs"]
mod run_content_edits;
#[path = "../run_content_parts.rs"]
mod run_content_parts;
#[path = "../run_content_scene.rs"]
mod run_content_scene;
#[path = "../run_delivery.rs"]
mod run_delivery;
#[path = "../run_fork.rs"]
mod run_fork;
#[path = "../run_graph.rs"]
mod run_graph;
#[path = "../run_identity.rs"]
mod run_identity;
#[path = "../run_lifetime.rs"]
mod run_lifetime;
#[path = "../run_missing_model.rs"]
mod run_missing_model;
#[path = "../run_output_tool_config.rs"]
mod run_output_tool_config;
#[path = "../run_provider_retry.rs"]
mod run_provider_retry;
#[path = "../run_replay_metadata.rs"]
mod run_replay_metadata;
#[path = "../run_replay_policy.rs"]
mod run_replay_policy;
#[path = "../run_scene.rs"]
mod run_scene;
#[path = "../run_scene_extensions.rs"]
mod run_scene_extensions;
#[path = "../run_stream_boundary.rs"]
mod run_stream_boundary;
#[path = "../run_support/mod.rs"]
mod run_support;
#[path = "../run_tool_access.rs"]
mod run_tool_access;
#[path = "../run_witness.rs"]
mod run_witness;
#[path = "../steer_hooks.rs"]
mod steer_hooks;
#[path = "../tool_batch.rs"]
mod tool_batch;
#[path = "../tool_result_limit.rs"]
mod tool_result_limit;
#[path = "../tool_result_status.rs"]
mod tool_result_status;
