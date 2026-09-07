//! Rig repository paths for the reusable cassette engine.
#![allow(dead_code, unused_imports)]

#[path = "../consumer/registry.rs"]
pub(crate) mod consumer_registry;

pub(crate) use rig_cassette::*;
use std::path::PathBuf;

pub(crate) fn cassette_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes")
}

pub(crate) fn cassette_path(provider: &str, scenario: &str) -> PathBuf {
    rig_cassette::cassette_path(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_interaction_bodies(provider: &str, scenario: &str) -> Vec<(String, String)> {
    rig_cassette::recorded_interaction_bodies(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_json_request(provider: &str, scenario: &str) -> serde_json::Value {
    rig_cassette::recorded_json_request(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_json_response(provider: &str, scenario: &str) -> serde_json::Value {
    rig_cassette::recorded_json_response(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_request_header_pairs(
    provider: &str,
    scenario: &str,
) -> Vec<Vec<(String, String)>> {
    rig_cassette::recorded_request_header_pairs(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_request_paths(provider: &str, scenario: &str) -> Vec<String> {
    rig_cassette::recorded_request_paths(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_statuses_and_bodies(provider: &str, scenario: &str) -> Vec<(u16, String)> {
    rig_cassette::recorded_statuses_and_bodies(&cassette_root(), provider, scenario)
}

pub(crate) fn recorded_sse_json_frames(provider: &str, scenario: &str) -> Vec<serde_json::Value> {
    rig_cassette::recorded_sse_json_frames(&cassette_root(), provider, scenario)
}
