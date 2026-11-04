//! The integration tests of rig-vertexai, one module per suite, linked into one
//! binary. `support` is the local endpoint the SDK suites share.

mod history_conformance;
mod raw_api;
mod runtime_ownership;
mod sdk_boundary;
#[path = "../support/mod.rs"]
mod support;
