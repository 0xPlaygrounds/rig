//! The setup cells of the OpenAI Responses wire (`gpt-5-mini`): the failure rows of
//! `tests/common/ecs_matrix/faults.rs` with this wire's recorded facts (the
//! model it refuses, the recorded status, the body's own code), which the
//! producer rows in `corpus_faults_responses.rs` run.

use crate::ecs_matrix::{
    cells::Cell,
    corpus::Program,
    faults::{self, Fault},
};

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
/// The unary reply the wire gives today (404) differs from the streamed
/// recording's (400, recorded earlier): the wire's own two replies.
pub(super) const SETUP_UNARY: Cell = Cell {
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
        status: 400,
        code: Some("model_not_found"),
    }),
    ..SETUP_UNARY
};
