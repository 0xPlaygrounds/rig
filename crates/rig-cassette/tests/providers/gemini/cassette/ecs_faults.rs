//! The setup cells of the Gemini REST wire (`gemini-3-flash-preview`): the failure rows of
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
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("NOT_FOUND"),
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
        code: Some("NOT_FOUND"),
    }),
    ..SETUP_UNARY
};
