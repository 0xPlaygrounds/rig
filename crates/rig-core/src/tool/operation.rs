//! Host-selected operation backends and coding-tool safety primitives.
//!
//! Core defines the contract; local process, SSH, container, and sandbox
//! implementations belong in hosts or optional companion crates.

use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
};

use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

use super::ToolFailure;

/// Backend family selected by a host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationBackendKind {
    /// Local machine execution.
    Local,
    /// Remote SSH execution.
    Ssh,
    /// Container execution.
    Container,
    /// Sandboxed execution.
    Sandbox,
}

/// A backend-neutral operation request.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperationRequest {
    /// Program or operation name.
    pub program: String,
    /// Argument vector; no shell interpolation is implied.
    pub args: Vec<String>,
    /// Optional working directory selected after host path policy.
    pub working_directory: Option<String>,
    /// Maximum model-visible output bytes.
    pub output_limit: Option<usize>,
}

/// Reference to complete output retained outside the model transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OutputArtifact {
    /// Host-defined stable identifier or URI.
    pub reference: String,
    /// Complete output byte length.
    pub byte_len: usize,
}

/// Structured backend output with explicit truncation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperationOutput {
    /// Model-visible bounded output.
    pub output: String,
    /// Process/operation status when meaningful.
    pub status: Option<i32>,
    /// Complete-output reference when `output` was truncated.
    pub artifact: Option<OutputArtifact>,
}

/// Progress update for a long-running operation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperationProgress {
    /// Human-readable progress message.
    pub message: String,
    /// Optional normalized completion fraction.
    pub fraction: Option<f32>,
}

/// Receives progress without coupling the backend to a UI.
pub trait ProgressSink: WasmCompatSend + WasmCompatSync {
    /// Publish one progress update.
    fn publish(&self, progress: OperationProgress);
}

/// Pluggable execution backend for coding tools.
pub trait ToolOperationBackend: WasmCompatSend + WasmCompatSync {
    /// Backend family for policy and telemetry.
    fn kind(&self) -> OperationBackendKind;

    /// Execute an operation while observing run cancellation through the
    /// automatically injected [`RunContext`](crate::agent::RunContext).
    fn execute<'a>(
        &'a self,
        request: OperationRequest,
        context: &'a crate::agent::RunContext,
        progress: Option<&'a dyn ProgressSink>,
    ) -> WasmBoxedFuture<'a, Result<OperationOutput, ToolFailure>>;
}

/// Host policy evaluated independently from execution isolation.
pub trait OperationPolicy: WasmCompatSend + WasmCompatSync {
    /// Approve or reject a request before a backend sees it.
    fn authorize(
        &self,
        request: &OperationRequest,
        context: &crate::agent::RunContext,
    ) -> Result<(), ToolFailure>;
}

/// Keyed mutation serializer for files or other resources.
///
/// Equal normalized resource keys execute one-at-a-time; unrelated resources
/// remain concurrent. Hosts are responsible for canonicalizing keys according
/// to project trust and path policy before calling [`run`](Self::run).
#[derive(Clone, Default)]
pub struct ResourceMutationQueue {
    locks: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl ResourceMutationQueue {
    /// Construct an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run one mutation under the resource's FIFO Tokio mutex.
    pub async fn run<F, Fut, T>(&self, resource: impl Into<String>, operation: F) -> T
    where
        F: FnOnce() -> Fut + WasmCompatSend,
        Fut: Future<Output = T> + WasmCompatSend,
        T: WasmCompatSend,
    {
        let resource = resource.into();
        let lock = {
            let mut locks = self.locks.lock().unwrap_or_else(|error| error.into_inner());
            locks.entry(resource).or_default().clone()
        };
        let _guard = lock.lock().await;
        operation().await
    }
}
