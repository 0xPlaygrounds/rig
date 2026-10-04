//! The setup cells of the Doubleword wire (`Qwen/Qwen3.5-397B-A17B-FP8`, `reasoning_effort: none`): the failure rows of
//! `tests/common/ecs_matrix/faults.rs` with this wire's recorded facts (the
//! model it refuses, the recorded status, the body's own code), which the
//! producer rows in `corpus_faults.rs` run.

use crate::ecs_matrix::{
    cells::Cell,
    corpus::Program,
    faults::{self, Fault},
};

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
pub(super) const SETUP_UNARY: Cell = Cell {
    program: Program {
        prompt: "Reply with error-probe.",
        max_tokens: Some(8),
        ..faults::SETUP_UNARY.program
    },
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("model_not_found"),
    }),
    ..faults::SETUP_UNARY
};
pub(super) const SETUP_STREAMED: Cell = Cell {
    program: Program {
        streamed: true,
        ..SETUP_UNARY.program
    },
    name: faults::SETUP_STREAMED.name,
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("model_not_found"),
    }),
    ..SETUP_UNARY
};
