//! Run-scoped toolset factories for stateful shells, REPLs, sandboxes, and connections.

use thiserror::Error;

use crate::{
    agent::RunContext,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::ToolSet;

/// Run-tool lifecycle failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolFactoryError {
    /// Factory setup failed.
    #[error("run tool factory setup failed: {0}")]
    Setup(String),
    /// Deterministic cleanup failed.
    #[error("run tool factory cleanup failed: {0}")]
    Cleanup(String),
}

/// Creates a toolset exactly once for a run and deterministically closes its
/// associated resources when the shared driver settles.
pub trait RunToolsetFactory: WasmCompatSend + WasmCompatSync {
    /// Create the run-local tool overlay.
    fn create<'a>(
        &'a self,
        context: &'a RunContext,
    ) -> WasmBoxedFuture<'a, Result<ToolSet, ToolFactoryError>>;

    /// Close state created for this run. Called on success, cancellation, and
    /// ordinary driver errors. Implementations must make cleanup idempotent.
    fn close<'a>(
        &'a self,
        _context: &'a RunContext,
    ) -> WasmBoxedFuture<'a, Result<(), ToolFactoryError>> {
        Box::pin(async { Ok(()) })
    }
}
