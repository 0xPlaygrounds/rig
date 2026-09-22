//! Behavioral verification with one shared corpus implementation.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[macro_use]
#[path = "../corpus/mod.rs"]
mod corpus;
#[path = "../corpus_breadth.rs"]
mod corpus_breadth;
#[path = "../corpus_causal.rs"]
mod corpus_causal;
#[path = "../corpus_checkpoint.rs"]
mod corpus_checkpoint;
#[path = "../corpus_delta.rs"]
mod corpus_delta;
#[path = "../corpus_endings.rs"]
mod corpus_endings;
#[path = "../corpus_header.rs"]
mod corpus_header;
#[path = "../corpus_hooks.rs"]
mod corpus_hooks;
#[path = "../corpus_host.rs"]
mod corpus_host;
#[path = "../corpus_invalid.rs"]
mod corpus_invalid;
#[path = "../corpus_layers.rs"]
mod corpus_layers;
#[path = "../corpus_leftovers.rs"]
mod corpus_leftovers;
#[path = "../corpus_memory.rs"]
mod corpus_memory;
#[path = "../corpus_oracle.rs"]
mod corpus_oracle;
#[path = "../corpus_outcome.rs"]
mod corpus_outcome;
#[path = "../corpus_output.rs"]
mod corpus_output;
#[path = "../corpus_request_shape.rs"]
mod corpus_request_shape;
#[path = "../corpus_resume.rs"]
mod corpus_resume;
#[path = "../corpus_retrieval.rs"]
mod corpus_retrieval;
#[path = "../corpus_serving.rs"]
mod corpus_serving;
#[path = "../corpus_shaping.rs"]
mod corpus_shaping;
#[path = "../durable_execution.rs"]
mod durable_execution;
#[path = "../golden_refusal.rs"]
mod golden_refusal;
#[path = "../golden_replay.rs"]
mod golden_replay;
#[path = "../interpreters_agree.rs"]
mod interpreters_agree;
#[path = "../log_header.rs"]
mod log_header;
#[path = "../record_replay.rs"]
mod record_replay;
