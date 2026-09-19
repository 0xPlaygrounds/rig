//! The root package's own tests: guards that scan the source tree and the
//! fixture runners, which need the repository root. Behaviour of the bus and
//! the agent over it is verified in `crates/rig-cassette`; provider behaviour
//! in `tests/providers`; anything needing crate-private types stays a unit
//! test in its crate.

mod agent_run_stepper;
mod dependency_graph;
#[cfg(feature = "derive")]
mod embed_macro;
mod fixtures_hold_no_key;
mod golden_causal;
mod golden_delta;
mod golden_endings;
mod golden_hooks;
mod golden_invalid;
mod golden_layers;
mod golden_leftovers;
mod golden_memory;
mod golden_oracle;
mod golden_outcome;
mod golden_output;
mod golden_pairing;
mod golden_recovery;
mod loaders;
mod no_random_ids;
mod prompt_response_messages;
mod reasoning_stream_stats;
#[cfg(feature = "derive")]
mod rig_tool_facade;
mod streaming_conformance;
mod streaming_conformance_registry;
mod streaming_conformance_suites;
#[allow(dead_code)]
#[path = "../../xtask/src/verify/checks.rs"]
mod verification_checks;

mod pull_parser;
