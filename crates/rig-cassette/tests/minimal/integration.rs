//! The minimal-dependency runner, linked into one binary: the shared
//! effect-bus verification suite `verify` and the published log and classic
//! replay regressions.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[macro_use]
#[allow(dead_code, reason = "each suite uses a different subset of the corpus")]
#[path = "../corpus/mod.rs"]
mod corpus;
mod effect_log;
#[path = "../verify/main.rs"]
mod verify;
