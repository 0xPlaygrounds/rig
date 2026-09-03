//! The root package's own tests: guards that scan the source tree and the
//! fixture runners, which need the repository root. Behaviour of the bus and
//! the agent over it is verified in `crates/rig-verify`; provider behaviour
//! in `tests/providers`; anything needing crate-private types stays a unit
//! test in its crate.

mod agent_run_stepper;
mod bevy_bus_host;
mod dependency_graph;
#[cfg(feature = "derive")]
mod embed_macro;
mod loaders;
mod name_keyed_serializers;
mod nightly_paths_registry;
mod one_erasure;
mod prompt_response_messages;
mod provider_layout;
mod reasoning_stream_stats;
mod stream_ids;
mod streaming_conformance;
mod streaming_conformance_registry;
mod streaming_conformance_suites;
#[cfg(feature = "derive")]
mod tool_macro;
