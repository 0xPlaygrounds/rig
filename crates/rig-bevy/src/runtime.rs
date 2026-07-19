//! Local and hosted drivers over the same authoritative ECS schedule.

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard},
};

use bevy_ecs::{prelude::*, schedule::Schedule};
use futures::StreamExt;
use rig_core::{
    client::CompletionClient,
    completion::{CompletionModel, CompletionRequest, Message},
    memory::ConversationMemory,
    message::ToolChoice,
    streaming::StreamedAssistantContent,
    tool::{
        IntoToolOutput, PortableDynamicTool, PortableTool, ToolExecutionError, ToolOutput,
        portable_tool_definition,
    },
};
use tokio::{
    sync::{Notify, OwnedSemaphorePermit, Semaphore, broadcast, mpsc, watch},
    task::{AbortHandle, JoinSet},
};
use tracing::Instrument;

use crate::effects::{
    EffectCompletion, EffectIntent, ErasedRawFinal, MemoryEffectError, ModelEffectError,
};
use crate::{
    AgentId, AgentNode, AgentSpec, BindingIdentity, CapabilityId, CapabilityKind, CapabilityNode,
    EffectHeader, EffectIngress, Generation, GrantId, GrantNode, HostedProviderDiagnostic,
    InvalidToolPolicy, MemoryId, ModelId, OperationId, ProvisionalDelta, ResponseRetryPolicy,
    RunAccounting, RunEvent, RunId, RuntimeConfig, RuntimeError, RuntimeId, StreamingMode,
    StructuredOutputPolicy, TenantId,
    components::{
        AcceptedDeltas, ActiveOperations, CancellationRequest, CanonicalTranscript,
        CapabilitiesToDrop, CapabilityReferences, EffectQueueWait, InvalidToolRetryState,
        MemoryProgress, PendingEffects, PendingIngress, RawFinalRecord, RecoveryFeedback,
        RejectionLog, ResponseRetryState, RunEvents, RunNode, RunPhase, RunProgress,
        RunTelemetrySpan, RuntimeIdentity, RuntimeTick, StructuredOutputState, TerminalCause,
        TerminalDiagnostic, TerminalObservation, TerminalReason, TerminalState, TopologyIndex,
    },
};

type NativeFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

struct EffectCapacityGuard(watch::Sender<u64>);

impl Drop for EffectCapacityGuard {
    fn drop(&mut self) {
        self.0.send_modify(|epoch| *epoch = epoch.saturating_add(1));
    }
}

fn try_acquire_permits(
    global: &Arc<Semaphore>,
    secondary: Option<&Arc<Semaphore>>,
) -> Option<(OwnedSemaphorePermit, Option<OwnedSemaphorePermit>)> {
    let global_permit = Arc::clone(global).try_acquire_owned().ok()?;
    match secondary {
        None => Some((global_permit, None)),
        // The global permit drops here when the kind-specific limit is
        // exhausted, preserving acquire-global-then-kind ordering.
        Some(limit) => match Arc::clone(limit).try_acquire_owned() {
            Ok(permit) => Some((global_permit, Some(permit))),
            Err(_) => None,
        },
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[derive(Clone)]
pub(crate) struct IngressSender {
    sender: mpsc::Sender<EffectIngress>,
    run_notifies: Arc<Mutex<HashMap<RunId, Arc<Notify>>>>,
    ingress_progress_tx: watch::Sender<u64>,
}

impl IngressSender {
    fn run_notifies(&self) -> MutexGuard<'_, HashMap<RunId, Arc<Notify>>> {
        lock_unpoisoned(&self.run_notifies)
    }

    async fn send(
        &self,
        ingress: EffectIngress,
    ) -> Result<(), mpsc::error::SendError<EffectIngress>> {
        let run_id = ingress.run_id();
        self.sender.send(ingress).await?;
        self.ingress_progress_tx
            .send_modify(|epoch| *epoch = epoch.saturating_add(1));
        if let Some(notify) = self.run_notifies().get(&run_id) {
            notify.notify_one();
        }
        Ok(())
    }

    fn try_send(&self, ingress: EffectIngress) -> Result<(), RuntimeError> {
        let run_id = ingress.run_id();
        self.sender.try_send(ingress).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => RuntimeError::IngressFull,
            mpsc::error::TrySendError::Closed(_) => RuntimeError::IngressClosed,
        })?;
        self.ingress_progress_tx
            .send_modify(|epoch| *epoch = epoch.saturating_add(1));
        if let Some(notify) = self.run_notifies().get(&run_id) {
            notify.notify_one();
        }
        Ok(())
    }
}

pub(crate) trait ModelBinding: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> NativeFuture<Result<crate::effects::ModelEffectOutput, ModelEffectError>>;

    fn stream(
        &self,
        request: CompletionRequest,
        header: EffectHeader,
        ingress: IngressSender,
    ) -> NativeFuture<Result<crate::effects::ModelEffectOutput, ModelEffectError>>;

    fn composes_native_output_with_tools(&self) -> bool;
}

pub(crate) struct TypedModelBinding<M>(pub(crate) M);

impl<M> ModelBinding for TypedModelBinding<M>
where
    M: CompletionModel + Send + Sync + 'static,
    M::Response: Any + Send + Sync,
    M::StreamingResponse: Any + Send + Sync,
{
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> NativeFuture<Result<crate::effects::ModelEffectOutput, ModelEffectError>> {
        let model = self.0.clone();
        Box::pin(async move {
            let response = model
                .completion(request)
                .await
                .map_err(ModelEffectError::Provider)?;
            Ok(crate::effects::ModelEffectOutput {
                choice: response.choice.into_iter().collect(),
                usage: response.usage,
                message_id: response.message_id,
                raw_final: Some(ErasedRawFinal::new(response.raw_response)),
            })
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
        header: EffectHeader,
        ingress: IngressSender,
    ) -> NativeFuture<Result<crate::effects::ModelEffectOutput, ModelEffectError>> {
        let model = self.0.clone();
        Box::pin(async move {
            let mut response = model
                .stream(request)
                .await
                .map_err(ModelEffectError::Provider)?;
            let mut raw_final = None;
            let mut sequence = 0_u64;
            while let Some(item) = response.next().await {
                let item = item.map_err(ModelEffectError::Provider)?;
                let delta = match item {
                    StreamedAssistantContent::Text(text) => Some(ProvisionalDelta::Text(text.text)),
                    StreamedAssistantContent::ToolCall { tool_call, .. } => {
                        Some(ProvisionalDelta::ToolCall(tool_call))
                    }
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    } => Some(ProvisionalDelta::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    }),
                    StreamedAssistantContent::Reasoning(reasoning) => {
                        Some(ProvisionalDelta::Reasoning(reasoning))
                    }
                    StreamedAssistantContent::ReasoningDelta { id, reasoning } => {
                        Some(ProvisionalDelta::ReasoningDelta { id, reasoning })
                    }
                    StreamedAssistantContent::Unknown(value) => {
                        Some(ProvisionalDelta::Unknown(value))
                    }
                    StreamedAssistantContent::Final(final_response) => {
                        raw_final = Some(ErasedRawFinal::new(final_response));
                        None
                    }
                };
                if let Some(delta) = delta {
                    ingress
                        .send(EffectIngress::Delta {
                            header,
                            sequence,
                            delta,
                        })
                        .await
                        .map_err(|_| ModelEffectError::IngressClosed)?;
                    sequence = sequence.saturating_add(1);
                }
            }
            let usage = response.usage();
            Ok(crate::effects::ModelEffectOutput {
                choice: response.choice.into_iter().collect(),
                usage,
                message_id: response.message_id,
                raw_final,
            })
        })
    }

    fn composes_native_output_with_tools(&self) -> bool {
        self.0.composes_native_output_with_tools()
    }
}

pub(crate) trait ToolBinding: Send + Sync {
    fn definition(&self) -> rig_core::completion::ToolDefinition;

    fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> NativeFuture<Result<ToolOutput, ToolExecutionError>>;
}

pub(crate) struct TypedToolBinding<T>(pub(crate) Arc<T>);

impl<T> ToolBinding for TypedToolBinding<T>
where
    T: PortableTool + Send + Sync + 'static,
    T::Args: Send + Sync + 'static,
    T::Output: Send + 'static,
{
    fn definition(&self) -> rig_core::completion::ToolDefinition {
        portable_tool_definition(self.0.as_ref())
    }

    fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> NativeFuture<Result<ToolOutput, ToolExecutionError>> {
        let tool = Arc::clone(&self.0);
        Box::pin(async move {
            let arguments = serde_json::from_value(arguments).map_err(|error| {
                ToolExecutionError::invalid_args(error.to_string()).with_source(error)
            })?;
            let output = tool
                .call(arguments)
                .await
                .map_err(|error| tool.map_error(error))?;
            output.into_tool_output()
        })
    }
}

pub(crate) struct DynamicToolBinding(pub(crate) PortableDynamicTool);

impl ToolBinding for DynamicToolBinding {
    fn definition(&self) -> rig_core::completion::ToolDefinition {
        self.0.definition()
    }

    fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> NativeFuture<Result<ToolOutput, ToolExecutionError>> {
        let tool = self.0.clone();
        Box::pin(async move { tool.execute(arguments).await })
    }
}

pub(crate) trait MemoryBinding: Send + Sync {
    fn load(
        &self,
        conversation_id: String,
    ) -> NativeFuture<Result<Vec<Message>, MemoryEffectError>>;

    fn append(
        &self,
        conversation_id: String,
        messages: Vec<Message>,
    ) -> NativeFuture<Result<(), MemoryEffectError>>;
}

pub(crate) struct TypedMemoryBinding<M>(pub(crate) Arc<M>);

impl<M> MemoryBinding for TypedMemoryBinding<M>
where
    M: ConversationMemory + Send + Sync + 'static,
{
    fn load(
        &self,
        conversation_id: String,
    ) -> NativeFuture<Result<Vec<Message>, MemoryEffectError>> {
        let memory = Arc::clone(&self.0);
        Box::pin(async move {
            memory
                .load(&conversation_id)
                .await
                .map_err(MemoryEffectError::Backend)
        })
    }

    fn append(
        &self,
        conversation_id: String,
        messages: Vec<Message>,
    ) -> NativeFuture<Result<(), MemoryEffectError>> {
        let memory = Arc::clone(&self.0);
        Box::pin(async move {
            memory
                .append(&conversation_id, messages)
                .await
                .map_err(MemoryEffectError::Backend)
        })
    }
}

/// Built agent definition that keeps the concrete model until runtime registration.
pub struct BevyAgentDefinition<M> {
    pub(crate) model: M,
    pub(crate) spec: AgentSpec,
    pub(crate) binding_identity: Option<BindingIdentity>,
}

impl<M> BevyAgentDefinition<M> {
    /// Stable model identity that will be registered with this definition.
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        self.spec.model_id
    }
}

/// Builder for an ECS-native agent specification.
pub struct BevyAgentBuilder<M> {
    model: M,
    spec: AgentSpec,
    binding_identity: Option<BindingIdentity>,
}

impl<M> BevyAgentBuilder<M>
where
    M: CompletionModel,
{
    /// Construct a builder around one concrete completion model.
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            spec: AgentSpec::new(ModelId::new(), TenantId::new()),
            binding_identity: None,
        }
    }

    /// Set the tenant security boundary.
    #[must_use]
    pub fn tenant(mut self, tenant_id: TenantId) -> Self {
        self.spec.tenant_id = tenant_id;
        self
    }

    /// Set an optional diagnostic agent name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.spec = self.spec.name(name);
        self
    }

    /// Set a system preamble.
    #[must_use]
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.spec = self.spec.preamble(preamble);
        self
    }

    /// Set the total model-call budget.
    #[must_use]
    pub fn max_model_calls(mut self, value: usize) -> Self {
        self.spec = self.spec.max_model_calls(value);
        self
    }

    /// Set invalid-tool recovery policy.
    #[must_use]
    pub fn invalid_tool_policy(mut self, value: InvalidToolPolicy) -> Self {
        self.spec = self.spec.invalid_tool_policy(value);
        self
    }

    /// Set tool-free response retry policy.
    #[must_use]
    pub fn response_retry_policy(mut self, value: ResponseRetryPolicy) -> Self {
        self.spec = self.spec.response_retry_policy(value);
        self
    }

    /// Configure typed structured output.
    #[must_use]
    pub fn structured_output<T>(mut self, policy: StructuredOutputPolicy) -> Self
    where
        T: schemars::JsonSchema,
    {
        self.spec = self.spec.structured_output::<T>(policy);
        self
    }

    /// Configure a raw structured-output schema.
    #[must_use]
    pub fn structured_output_raw(
        mut self,
        schema: schemars::Schema,
        policy: StructuredOutputPolicy,
    ) -> Self {
        self.spec = self.spec.structured_output_raw(schema, policy);
        self
    }

    /// Configure conversation memory.
    #[must_use]
    pub fn memory(mut self, memory_id: MemoryId, conversation_id: impl Into<String>) -> Self {
        self.spec = self.spec.memory(memory_id, conversation_id);
        self
    }

    /// Select provider tool-choice policy.
    #[must_use]
    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.spec = self.spec.tool_choice(value);
        self
    }

    /// Set the sampling temperature used for model requests.
    #[must_use]
    pub fn temperature(mut self, value: f64) -> Self {
        self.spec = self.spec.temperature(value);
        self
    }

    /// Set the maximum generated-token count used for model requests.
    #[must_use]
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.spec = self.spec.max_tokens(value);
        self
    }

    /// Set provider-specific request parameters.
    #[must_use]
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.spec = self.spec.additional_params(value);
        self
    }

    /// Opt in to sensitive provider telemetry content.
    #[must_use]
    pub fn record_telemetry_content(mut self, value: bool) -> Self {
        self.spec = self.spec.record_telemetry_content(value);
        self
    }

    /// Select provider streaming for new runs.
    #[must_use]
    pub fn streaming(mut self, value: StreamingMode) -> Self {
        self.spec = self.spec.streaming(value);
        self
    }

    /// Assign the stable implementation/configuration identity required by snapshots.
    #[must_use]
    pub fn binding_identity(mut self, identity: BindingIdentity) -> Self {
        self.binding_identity = Some(identity);
        self
    }

    /// Finish the immutable agent definition.
    #[must_use]
    pub fn build(self) -> BevyAgentDefinition<M> {
        BevyAgentDefinition {
            model: self.model,
            spec: self.spec,
            binding_identity: self.binding_identity,
        }
    }
}

/// Distinct construction extension for completion provider clients.
pub trait BevyClientExt: CompletionClient {
    /// Construct an ECS agent builder without colliding with classic `.agent()`.
    fn bevy_agent(&self, model: impl Into<String>) -> BevyAgentBuilder<Self::CompletionModel> {
        BevyAgentBuilder::new(self.completion_model(model))
    }
}

impl<C> BevyClientExt for C where C: CompletionClient {}

/// Distinct construction extension for concrete completion models.
pub trait BevyModelExt: CompletionModel + Sized {
    /// Convert this model into an ECS agent builder.
    fn into_bevy_agent_builder(self) -> BevyAgentBuilder<Self> {
        BevyAgentBuilder::new(self)
    }
}

impl<M> BevyModelExt for M where M: CompletionModel {}

/// Stable handle to one run in one runtime generation.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct RunHandle {
    /// Runtime world identity.
    pub runtime_id: RuntimeId,
    /// Stable run identity.
    pub run_id: RunId,
    /// Generation captured when the handle was created.
    pub generation: Generation,
    /// Tenant that owns the run.
    pub tenant_id: TenantId,
}

impl fmt::Debug for RunHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunHandle")
            .field("runtime_id", &self.runtime_id)
            .field("run_id", &self.run_id)
            .field("generation", &self.generation)
            .field("tenant_id", &"<redacted>")
            .finish()
    }
}

/// Exact capability version and grant returned during portable-tool installation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolGrant {
    /// Installed capability version.
    pub capability_id: CapabilityId,
    /// Grant authorizing one agent to advertise it.
    pub grant_id: GrantId,
    /// Capability revision captured by new turns.
    pub revision: u64,
}

/// Whether a single bounded schedule pass is quiescent, progressing, or terminal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunStepStatus {
    /// No effect or policy transition is currently ready.
    Quiescent,
    /// At least one transition or dispatch occurred.
    Progressed,
    /// An owned effect delivered external ingress for the target run.
    ///
    /// Drivers reset their consecutive schedule-pass bound at this boundary;
    /// a valid long stream is not a scheduler livelock.
    EffectProgressed,
    /// The run reached terminal state.
    Terminal,
}

/// Canonical local terminal result with a typed raw-final side-channel accessor.
#[derive(Clone)]
pub struct LocalRunResult {
    /// Stable run identity.
    pub run_id: RunId,
    /// Final canonical assistant text, if present.
    pub text: Option<String>,
    /// Full canonical transcript including loaded history.
    pub transcript: Vec<Message>,
    /// Aggregated provider usage.
    pub usage: rig_core::completion::Usage,
    /// Exactly-once model call records.
    pub model_calls: Vec<crate::ModelCallRecord>,
    /// Observable terminal reason.
    pub terminal_reason: TerminalReason,
    /// Validated structured value when configured and produced.
    pub structured_output: Option<serde_json::Value>,
    /// Bounded retained tail of runtime events available before cleanup.
    pub events: Vec<RunEvent>,
    /// Operation that owns the retained provider final.
    pub raw_final_operation: Option<OperationId>,
    /// Operator-facing non-persisted diagnostic for a failed terminal run.
    pub failure_diagnostic: Option<String>,
    raw_final: Option<ErasedRawFinal>,
    failure_cause: Option<TerminalCause>,
}

impl fmt::Debug for LocalRunResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalRunResult")
            .field("run_id", &self.run_id)
            .field("text_available", &self.text.is_some())
            .field("transcript_messages", &self.transcript.len())
            .field("usage", &self.usage)
            .field("model_calls", &self.model_calls)
            .field("terminal_reason", &self.terminal_reason)
            .field(
                "structured_output_available",
                &self.structured_output.is_some(),
            )
            .field("event_count", &self.events.len())
            .field("raw_final_operation", &self.raw_final_operation)
            .field("raw_final_type", &self.raw_final_type())
            .field(
                "failure_diagnostic_available",
                &self.failure_diagnostic.is_some(),
            )
            .finish()
    }
}

impl LocalRunResult {
    /// Downcast the non-persisted local provider final to its concrete type.
    #[must_use]
    pub fn raw_final<T>(&self) -> Option<Arc<T>>
    where
        T: Any + Send + Sync,
    {
        self.raw_final.as_ref()?.downcast::<T>()
    }

    /// Return the concrete raw-final type name without exposing its content.
    #[must_use]
    pub fn raw_final_type(&self) -> Option<&'static str> {
        self.raw_final.as_ref().map(ErasedRawFinal::type_name)
    }

    /// Return the non-persisted typed model failure, when one caused termination.
    #[must_use]
    pub fn model_error(&self) -> Option<&ModelEffectError> {
        match self.failure_cause.as_ref()? {
            TerminalCause::Model(error) => Some(error),
            TerminalCause::Memory(_) => None,
        }
    }

    /// Return the non-persisted typed memory failure, when one occurred.
    #[must_use]
    pub fn memory_error(&self) -> Option<&MemoryEffectError> {
        match self.failure_cause.as_ref()? {
            TerminalCause::Memory(error) => Some(error),
            TerminalCause::Model(_) => None,
        }
    }
}

/// One streaming lifecycle event with a concrete typed provider-final variant.
#[derive(Clone)]
#[non_exhaustive]
pub enum StreamingRunEvent<R> {
    /// Runtime event other than the matching provider final.
    Runtime(Box<RunEvent>),
    /// Concrete provider final emitted only after the entire stream succeeded.
    ProviderFinal {
        /// Model operation that produced the final.
        operation_id: OperationId,
        /// Concrete provider response.
        response: Arc<R>,
    },
}

impl<R> fmt::Debug for StreamingRunEvent<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(event) => formatter.debug_tuple("Runtime").field(event).finish(),
            Self::ProviderFinal { operation_id, .. } => formatter
                .debug_struct("ProviderFinal")
                .field("operation_id", operation_id)
                .field("response_type", &std::any::type_name::<R>())
                .field("response", &"<redacted>")
                .finish(),
        }
    }
}

/// Collected streaming run outcome and typed event sequence.
#[derive(Clone)]
pub struct StreamingRunResult<R> {
    /// Canonical terminal result.
    pub result: LocalRunResult,
    /// Provisional deltas and, when supplied by the provider, its accepted concrete final.
    pub events: Vec<StreamingRunEvent<R>>,
}

impl<R> fmt::Debug for StreamingRunResult<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingRunResult")
            .field("result", &self.result)
            .field("event_count", &self.events.len())
            .finish()
    }
}

/// Live typed streaming run over the same authoritative local schedule.
pub struct LocalStreamingRun<'runtime, R> {
    runtime: &'runtime mut LocalRuntime,
    handle: RunHandle,
    receiver: broadcast::Receiver<RunEvent>,
    terminal_seen: bool,
    closed: bool,
    response: PhantomData<fn() -> R>,
}

impl<R> LocalStreamingRun<'_, R>
where
    R: Any + Send + Sync + 'static,
{
    /// Stable handle for cancellation, diagnostics, or hosted coordination.
    #[must_use]
    pub const fn handle(&self) -> RunHandle {
        self.handle
    }

    /// Cancel an active stream, abort its owned effect, and observe its retained result.
    ///
    /// If the run already reached terminal state, this only observes and returns that
    /// terminal result. Dropping the stream without calling `finish` or `cancel` performs
    /// the same cancellation/observation cleanup but discards the result.
    pub fn cancel(mut self) -> Result<LocalRunResult, RuntimeError> {
        self.runtime.abandon_stream(self.handle)?;
        let result = self.runtime.finish_run(self.handle)?;
        self.closed = true;
        Ok(result)
    }

    /// Yield the next live event, preserving provisional-before-final ordering.
    pub async fn next_event(&mut self) -> Result<Option<StreamingRunEvent<R>>, RuntimeError> {
        let mut remaining_passes = self.runtime.config.max_schedule_passes;
        let mut target_effect_deadline = None;
        loop {
            match self.receiver.try_recv() {
                Ok(event) => {
                    if matches!(event, RunEvent::Terminal(_)) {
                        self.terminal_seen = true;
                    }
                    if let RunEvent::ProviderFinal { operation_id, .. } = event {
                        let entity = self.runtime.validate_handle(self.handle)?;
                        let raw = self
                            .runtime
                            .world
                            .get::<RawFinalRecord>(entity)
                            .filter(|raw| raw.operation_id == operation_id)
                            .ok_or(RuntimeError::RunFailed {
                                code: "missing_provider_final".to_string(),
                                diagnostic: "accepted provider final side channel is unavailable"
                                    .to_string(),
                            })?;
                        let actual = raw.raw.type_name();
                        let response =
                            raw.raw
                                .downcast::<R>()
                                .ok_or(RuntimeError::RawFinalTypeMismatch {
                                    expected: std::any::type_name::<R>(),
                                    actual,
                                })?;
                        return Ok(Some(StreamingRunEvent::ProviderFinal {
                            operation_id,
                            response,
                        }));
                    }
                    return Ok(Some(StreamingRunEvent::Runtime(Box::new(event))));
                }
                Err(broadcast::error::TryRecvError::Lagged(skipped)) => {
                    return Err(RuntimeError::EventStreamLagged { skipped });
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    return Err(RuntimeError::IngressClosed);
                }
                Err(broadcast::error::TryRecvError::Empty) if self.terminal_seen => {
                    return Ok(None);
                }
                Err(broadcast::error::TryRecvError::Empty) => {}
            }

            let mut capacity_rx = self.runtime.effect_capacity_tx.subscribe();
            match self.runtime.step_with_ingress_limit(self.handle, 1).await? {
                RunStepStatus::Terminal => {}
                RunStepStatus::Progressed => {
                    if remaining_passes == 0 {
                        crate::schedule::fail_effect_wait(
                            &mut self.runtime.world,
                            self.handle.run_id,
                            RuntimeError::Livelock.to_string(),
                        );
                        return Err(RuntimeError::Livelock);
                    }
                    remaining_passes -= 1;
                }
                RunStepStatus::EffectProgressed => {
                    remaining_passes = self.runtime.config.max_schedule_passes;
                    target_effect_deadline = None;
                }
                RunStepStatus::Quiescent => {
                    if self
                        .runtime
                        .wait_quiescent(self.handle, &mut capacity_rx, &mut target_effect_deadline)
                        .await?
                    {
                        remaining_passes = self.runtime.config.max_schedule_passes;
                    }
                }
            }
        }
    }

    /// Materialize the canonical result after the terminal event was observed.
    pub fn finish(mut self) -> Result<LocalRunResult, RuntimeError> {
        if !self.terminal_seen {
            return Err(RuntimeError::Livelock);
        }
        let result = self.runtime.finish_run(self.handle)?;
        self.closed = true;
        LocalRuntime::ensure_completed(result)
    }
}

impl<R> Drop for LocalStreamingRun<'_, R> {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.runtime.abandon_stream(self.handle);
        }
    }
}

struct LocalRunGuard<'runtime> {
    runtime: &'runtime mut LocalRuntime,
    handle: RunHandle,
    closed: bool,
}

impl Drop for LocalRunGuard<'_> {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.runtime.abandon_stream(self.handle);
        }
    }
}

/// Authoritative native ECS runtime.
pub struct LocalRuntime {
    pub(crate) world: World,
    pub(crate) schedule: Schedule,
    pub(crate) config: RuntimeConfig,
    pub(crate) models: HashMap<ModelId, Arc<dyn ModelBinding>>,
    pub(crate) tools: HashMap<CapabilityId, Arc<dyn ToolBinding>>,
    pub(crate) memories: HashMap<MemoryId, Arc<dyn MemoryBinding>>,
    pub(crate) model_tenants: HashMap<ModelId, TenantId>,
    pub(crate) memory_tenants: HashMap<MemoryId, TenantId>,
    pub(crate) model_binding_identities: HashMap<ModelId, BindingIdentity>,
    pub(crate) tool_binding_identities: HashMap<CapabilityId, BindingIdentity>,
    pub(crate) memory_binding_identities: HashMap<MemoryId, BindingIdentity>,
    ingress_tx: IngressSender,
    ingress_rx: mpsc::Receiver<EffectIngress>,
    effects: JoinSet<()>,
    aborts: BTreeMap<OperationId, (RunId, AbortHandle)>,
    global_limit: Arc<Semaphore>,
    model_limit: Arc<Semaphore>,
    tool_limit: Arc<Semaphore>,
    effect_capacity_tx: watch::Sender<u64>,
}

impl LocalRuntime {
    /// Construct a native ECS runtime with bounded defaults.
    pub fn new() -> Result<Self, RuntimeError> {
        Self::with_config(RuntimeConfig::default())
    }

    /// Construct a runtime with explicit bounds and retention policy.
    pub fn with_config(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        config.validate()?;
        let runtime_id = RuntimeId::new();
        let (sender, ingress_rx) = mpsc::channel(config.ingress_capacity);
        let (effect_capacity_tx, _) = watch::channel(0);
        let (ingress_progress_tx, _) = watch::channel(0);
        let ingress_tx = IngressSender {
            sender,
            run_notifies: Arc::new(Mutex::new(HashMap::new())),
            ingress_progress_tx,
        };
        let mut world = World::new();
        world.insert_resource(RuntimeIdentity(runtime_id));
        world.insert_resource(RuntimeTick::default());
        world.insert_resource(TopologyIndex::default());
        world.insert_resource(PendingIngress::default());
        world.insert_resource(PendingEffects::default());
        world.insert_resource(RejectionLog::default());
        world.insert_resource(CapabilityReferences::default());
        world.insert_resource(CapabilitiesToDrop::default());
        world.insert_resource(crate::schedule::RuntimeConfigResource(config.clone()));
        let schedule = crate::schedule::build_schedule();
        Ok(Self {
            world,
            schedule,
            models: HashMap::new(),
            tools: HashMap::new(),
            memories: HashMap::new(),
            model_tenants: HashMap::new(),
            memory_tenants: HashMap::new(),
            model_binding_identities: HashMap::new(),
            tool_binding_identities: HashMap::new(),
            memory_binding_identities: HashMap::new(),
            ingress_tx,
            ingress_rx,
            effects: JoinSet::new(),
            aborts: BTreeMap::new(),
            global_limit: Arc::new(Semaphore::new(config.max_effects)),
            model_limit: Arc::new(Semaphore::new(config.max_model_calls)),
            tool_limit: Arc::new(Semaphore::new(config.max_tool_calls)),
            effect_capacity_tx,
            config,
        })
    }

    /// Return this world's stable runtime identity.
    #[must_use]
    pub fn id(&self) -> RuntimeId {
        self.world.resource::<RuntimeIdentity>().0
    }

    /// Borrow the authoritative ECS world for read-only inspection.
    #[must_use]
    pub const fn world(&self) -> &World {
        &self.world
    }

    /// Register a concrete model under a generated stable binding identity.
    pub fn register_model<M>(&mut self, tenant_id: TenantId, model: M) -> ModelId
    where
        M: CompletionModel + Send + Sync + 'static,
        M::Response: Any + Send + Sync,
        M::StreamingResponse: Any + Send + Sync,
    {
        let id = ModelId::new();
        self.bind_model(id, tenant_id, None, model);
        id
    }

    /// Register a tenant-owned model with a stable snapshot rebinding identity.
    pub fn register_persistable_model<M>(
        &mut self,
        tenant_id: TenantId,
        identity: BindingIdentity,
        model: M,
    ) -> ModelId
    where
        M: CompletionModel + Send + Sync + 'static,
        M::Response: Any + Send + Sync,
        M::StreamingResponse: Any + Send + Sync,
    {
        let id = ModelId::new();
        self.bind_model(id, tenant_id, Some(identity), model);
        id
    }

    pub(crate) fn bind_model<M>(
        &mut self,
        id: ModelId,
        tenant_id: TenantId,
        identity: Option<BindingIdentity>,
        model: M,
    ) where
        M: CompletionModel + Send + Sync + 'static,
        M::Response: Any + Send + Sync,
        M::StreamingResponse: Any + Send + Sync,
    {
        self.models.insert(id, Arc::new(TypedModelBinding(model)));
        self.model_tenants.insert(id, tenant_id);
        if let Some(identity) = identity {
            self.model_binding_identities.insert(id, identity);
        }
    }

    /// Register a concrete conversation memory under a generated stable identity.
    pub fn register_memory<M>(&mut self, tenant_id: TenantId, memory: M) -> MemoryId
    where
        M: ConversationMemory + Send + Sync + 'static,
    {
        let id = MemoryId::new();
        self.bind_memory(id, tenant_id, None, memory);
        id
    }

    /// Register tenant-owned memory with a stable snapshot rebinding identity.
    pub fn register_persistable_memory<M>(
        &mut self,
        tenant_id: TenantId,
        identity: BindingIdentity,
        memory: M,
    ) -> MemoryId
    where
        M: ConversationMemory + Send + Sync + 'static,
    {
        let id = MemoryId::new();
        self.bind_memory(id, tenant_id, Some(identity), memory);
        id
    }

    pub(crate) fn bind_memory<M>(
        &mut self,
        id: MemoryId,
        tenant_id: TenantId,
        identity: Option<BindingIdentity>,
        memory: M,
    ) where
        M: ConversationMemory + Send + Sync + 'static,
    {
        self.memories
            .insert(id, Arc::new(TypedMemoryBinding(Arc::new(memory))));
        self.memory_tenants.insert(id, tenant_id);
        if let Some(identity) = identity {
            self.memory_binding_identities.insert(id, identity);
        }
    }

    /// Spawn a built ECS agent and register its concrete model implementation.
    pub fn spawn_agent<M>(
        &mut self,
        definition: BevyAgentDefinition<M>,
    ) -> Result<AgentId, RuntimeError>
    where
        M: CompletionModel + Send + Sync + 'static,
        M::Response: Any + Send + Sync,
        M::StreamingResponse: Any + Send + Sync,
    {
        self.validate_agent_memory(&definition.spec)?;
        self.bind_model(
            definition.spec.model_id,
            definition.spec.tenant_id,
            definition.binding_identity,
            definition.model,
        );
        self.spawn_agent_spec(definition.spec)
    }

    fn validate_agent_memory(&self, spec: &AgentSpec) -> Result<(), RuntimeError> {
        match (spec.memory_id, spec.conversation_id.is_some()) {
            (Some(_), false) => {
                return Err(RuntimeError::InvalidAgentSpec {
                    reason: "memory requires a conversation identifier",
                });
            }
            (None, true) => {
                return Err(RuntimeError::InvalidAgentSpec {
                    reason: "a conversation identifier requires memory",
                });
            }
            (Some(_), true) | (None, false) => {}
        }
        if let Some(memory_id) = spec.memory_id
            && !self.memories.contains_key(&memory_id)
        {
            return Err(RuntimeError::UnknownMemory(memory_id));
        }
        if let Some(memory_id) = spec.memory_id
            && let Some(owner) = self.memory_tenants.get(&memory_id)
            && *owner != spec.tenant_id
        {
            return Err(RuntimeError::TenantMismatch {
                expected: *owner,
                actual: spec.tenant_id,
            });
        }
        Ok(())
    }

    /// Spawn an agent from an already rebound model identity.
    pub fn spawn_agent_spec(&mut self, spec: AgentSpec) -> Result<AgentId, RuntimeError> {
        self.validate_agent_memory(&spec)?;
        if !self.models.contains_key(&spec.model_id) {
            return Err(RuntimeError::UnknownModel(spec.model_id));
        }
        if let Some(owner) = self.model_tenants.get(&spec.model_id)
            && *owner != spec.tenant_id
        {
            return Err(RuntimeError::TenantMismatch {
                expected: *owner,
                actual: spec.tenant_id,
            });
        }
        let id = AgentId::new();
        let composes_native_output_with_tools = self
            .models
            .get(&spec.model_id)
            .is_some_and(|model| model.composes_native_output_with_tools());
        let entity = self
            .world
            .spawn(AgentNode {
                id,
                spec,
                composes_native_output_with_tools,
            })
            .id();
        self.world
            .resource_mut::<TopologyIndex>()
            .agents
            .insert(id, entity);
        Ok(id)
    }

    /// Install and grant one portable tool to an agent.
    pub fn install_tool<T>(&mut self, agent_id: AgentId, tool: T) -> Result<ToolGrant, RuntimeError>
    where
        T: PortableTool + Send + Sync + 'static,
        T::Args: Send + Sync + 'static,
        T::Output: Send + 'static,
    {
        let binding: Arc<dyn ToolBinding> = Arc::new(TypedToolBinding(Arc::new(tool)));
        self.install_tool_binding(agent_id, 1, CapabilityKind::Tool, None, binding)
    }

    /// Install a portable tool with a stable implementation/configuration identity.
    pub fn install_persistable_tool<T>(
        &mut self,
        agent_id: AgentId,
        identity: BindingIdentity,
        tool: T,
    ) -> Result<ToolGrant, RuntimeError>
    where
        T: PortableTool + Send + Sync + 'static,
        T::Args: Send + Sync + 'static,
        T::Output: Send + 'static,
    {
        let binding: Arc<dyn ToolBinding> = Arc::new(TypedToolBinding(Arc::new(tool)));
        self.install_tool_binding(agent_id, 1, CapabilityKind::Tool, Some(identity), binding)
    }

    /// Install and grant a portable vector index as an executable retrieval capability.
    ///
    /// Vector indexes use `rig-core`'s portable tool implementation, while the
    /// ECS topology records their distinct store capability kind. Retrieval
    /// calls therefore use the same immutable grant snapshot, owned effect,
    /// correlation validation, and deterministic commit path as other tools.
    pub fn install_vector_store<I>(
        &mut self,
        agent_id: AgentId,
        index: I,
    ) -> Result<ToolGrant, RuntimeError>
    where
        I: rig_core::vector_store::VectorStoreIndex + PortableTool + Send + Sync + 'static,
        I::Args: Send + Sync + 'static,
        I::Output: Send + 'static,
    {
        let binding: Arc<dyn ToolBinding> = Arc::new(TypedToolBinding(Arc::new(index)));
        self.install_tool_binding(agent_id, 1, CapabilityKind::Store, None, binding)
    }

    /// Install a portable vector index with a stable snapshot rebinding identity.
    pub fn install_persistable_vector_store<I>(
        &mut self,
        agent_id: AgentId,
        identity: BindingIdentity,
        index: I,
    ) -> Result<ToolGrant, RuntimeError>
    where
        I: rig_core::vector_store::VectorStoreIndex + PortableTool + Send + Sync + 'static,
        I::Args: Send + Sync + 'static,
        I::Output: Send + 'static,
    {
        let binding: Arc<dyn ToolBinding> = Arc::new(TypedToolBinding(Arc::new(index)));
        self.install_tool_binding(agent_id, 1, CapabilityKind::Store, Some(identity), binding)
    }

    /// Install a runtime-authored context-free dynamic tool.
    pub fn install_dynamic_tool(
        &mut self,
        agent_id: AgentId,
        tool: PortableDynamicTool,
    ) -> Result<ToolGrant, RuntimeError> {
        let binding: Arc<dyn ToolBinding> = Arc::new(DynamicToolBinding(tool));
        self.install_tool_binding(agent_id, 1, CapabilityKind::Tool, None, binding)
    }

    /// Install a dynamic tool with a stable snapshot rebinding identity.
    pub fn install_persistable_dynamic_tool(
        &mut self,
        agent_id: AgentId,
        identity: BindingIdentity,
        tool: PortableDynamicTool,
    ) -> Result<ToolGrant, RuntimeError> {
        let binding: Arc<dyn ToolBinding> = Arc::new(DynamicToolBinding(tool));
        self.install_tool_binding(agent_id, 1, CapabilityKind::Tool, Some(identity), binding)
    }

    fn install_tool_binding(
        &mut self,
        agent_id: AgentId,
        revision: u64,
        kind: CapabilityKind,
        binding_identity: Option<BindingIdentity>,
        binding: Arc<dyn ToolBinding>,
    ) -> Result<ToolGrant, RuntimeError> {
        let agent_entity = self
            .world
            .resource::<TopologyIndex>()
            .agents
            .get(&agent_id)
            .copied()
            .ok_or(RuntimeError::UnknownAgent(agent_id))?;
        let tenant_id = self
            .world
            .get::<AgentNode>(agent_entity)
            .ok_or(RuntimeError::UnknownAgent(agent_id))?
            .spec
            .tenant_id;
        let definition = binding.definition();
        self.ensure_unique_active_tool_name(agent_id, &definition.name, None)?;
        let capability_id = CapabilityId::new();
        let grant_id = GrantId::new();
        let capability_entity = self
            .world
            .spawn(CapabilityNode {
                id: capability_id,
                tenant_id,
                kind,
                definition: Some(definition),
                revision,
                retired: false,
            })
            .id();
        let grant_entity = self
            .world
            .spawn(GrantNode {
                id: grant_id,
                agent_id,
                capability_id,
                tenant_id,
                revoked: false,
            })
            .id();
        let mut index = self.world.resource_mut::<TopologyIndex>();
        index.capabilities.insert(capability_id, capability_entity);
        index.grants.insert(grant_id, grant_entity);
        self.tools.insert(capability_id, binding);
        if let Some(identity) = binding_identity {
            self.tool_binding_identities.insert(capability_id, identity);
        }
        Ok(ToolGrant {
            capability_id,
            grant_id,
            revision,
        })
    }

    fn ensure_unique_active_tool_name(
        &self,
        agent_id: AgentId,
        name: &str,
        excluded_grant: Option<GrantId>,
    ) -> Result<(), RuntimeError> {
        let index = self.world.resource::<TopologyIndex>();
        let collision = index.grants.values().any(|grant_entity| {
            let Some(grant) = self.world.get::<GrantNode>(*grant_entity) else {
                return false;
            };
            if grant.revoked || grant.agent_id != agent_id || Some(grant.id) == excluded_grant {
                return false;
            }
            index
                .capabilities
                .get(&grant.capability_id)
                .and_then(|entity| self.world.get::<CapabilityNode>(*entity))
                .is_some_and(|capability| {
                    !capability.retired
                        && capability
                            .definition
                            .as_ref()
                            .is_some_and(|definition| definition.name == name)
                })
        });
        if collision {
            return Err(RuntimeError::DuplicateToolName {
                agent_id,
                name: name.to_string(),
            });
        }
        Ok(())
    }

    /// Replace a tool with a new immutable capability version and grant.
    pub fn replace_tool<T>(&mut self, grant_id: GrantId, tool: T) -> Result<ToolGrant, RuntimeError>
    where
        T: PortableTool + Send + Sync + 'static,
        T::Args: Send + Sync + 'static,
        T::Output: Send + 'static,
    {
        self.replace_tool_with_identity(grant_id, CapabilityKind::Tool, None, tool)
    }

    /// Replace a persistable tool or store capability with a new exact binding identity.
    pub fn replace_persistable_tool<T>(
        &mut self,
        grant_id: GrantId,
        identity: BindingIdentity,
        tool: T,
    ) -> Result<ToolGrant, RuntimeError>
    where
        T: PortableTool + Send + Sync + 'static,
        T::Args: Send + Sync + 'static,
        T::Output: Send + 'static,
    {
        self.replace_tool_with_identity(grant_id, CapabilityKind::Tool, Some(identity), tool)
    }

    /// Replace a vector-store capability with a new immutable store version.
    pub fn replace_vector_store<I>(
        &mut self,
        grant_id: GrantId,
        index: I,
    ) -> Result<ToolGrant, RuntimeError>
    where
        I: rig_core::vector_store::VectorStoreIndex + PortableTool + Send + Sync + 'static,
        I::Args: Send + Sync + 'static,
        I::Output: Send + 'static,
    {
        self.replace_tool_with_identity(grant_id, CapabilityKind::Store, None, index)
    }

    /// Replace a persistable vector store with a new exact binding identity.
    pub fn replace_persistable_vector_store<I>(
        &mut self,
        grant_id: GrantId,
        identity: BindingIdentity,
        index: I,
    ) -> Result<ToolGrant, RuntimeError>
    where
        I: rig_core::vector_store::VectorStoreIndex + PortableTool + Send + Sync + 'static,
        I::Args: Send + Sync + 'static,
        I::Output: Send + 'static,
    {
        self.replace_tool_with_identity(grant_id, CapabilityKind::Store, Some(identity), index)
    }

    fn replace_tool_with_identity<T>(
        &mut self,
        grant_id: GrantId,
        expected_kind: CapabilityKind,
        binding_identity: Option<BindingIdentity>,
        tool: T,
    ) -> Result<ToolGrant, RuntimeError>
    where
        T: PortableTool + Send + Sync + 'static,
        T::Args: Send + Sync + 'static,
        T::Output: Send + 'static,
    {
        let grant_entity = self
            .world
            .resource::<TopologyIndex>()
            .grants
            .get(&grant_id)
            .copied()
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        let old_grant = self
            .world
            .get::<GrantNode>(grant_entity)
            .cloned()
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        if old_grant.revoked {
            return Err(RuntimeError::RevokedGrant(grant_id));
        }
        let capability_entity = self
            .world
            .resource::<TopologyIndex>()
            .capabilities
            .get(&old_grant.capability_id)
            .copied()
            .ok_or(RuntimeError::UnknownCapability(old_grant.capability_id))?;
        let (old_revision, old_kind, old_retired) = self
            .world
            .get::<CapabilityNode>(capability_entity)
            .map(|capability| (capability.revision, capability.kind, capability.retired))
            .ok_or(RuntimeError::UnknownCapability(old_grant.capability_id))?;
        if old_retired {
            return Err(RuntimeError::RetiredCapability(old_grant.capability_id));
        }
        if old_kind != expected_kind {
            return Err(RuntimeError::CapabilityKindMismatch {
                capability_id: old_grant.capability_id,
                expected: expected_kind,
                actual: old_kind,
            });
        }
        if binding_identity.is_none()
            && self
                .tool_binding_identities
                .contains_key(&old_grant.capability_id)
        {
            return Err(RuntimeError::PersistenceIdentityRequired(
                old_grant.capability_id,
            ));
        }
        let new_revision =
            old_revision
                .checked_add(1)
                .ok_or(RuntimeError::CapabilityRevisionOverflow(
                    old_grant.capability_id,
                ))?;
        let binding: Arc<dyn ToolBinding> = Arc::new(TypedToolBinding(Arc::new(tool)));
        let replacement_name = binding.definition().name;
        self.ensure_unique_active_tool_name(old_grant.agent_id, &replacement_name, Some(grant_id))?;
        if let Some(mut capability) = self.world.get_mut::<CapabilityNode>(capability_entity) {
            capability.retired = true;
        }
        if let Some(mut grant) = self.world.get_mut::<GrantNode>(grant_entity) {
            grant.revoked = true;
        }
        self.install_tool_binding(
            old_grant.agent_id,
            new_revision,
            old_kind,
            binding_identity,
            binding,
        )
    }

    /// Retire the capability version behind a grant and revoke it for new turns.
    pub fn retire_tool(&mut self, grant_id: GrantId) -> Result<(), RuntimeError> {
        let grant_entity = self
            .world
            .resource::<TopologyIndex>()
            .grants
            .get(&grant_id)
            .copied()
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        let capability_id = self
            .world
            .get::<GrantNode>(grant_entity)
            .ok_or(RuntimeError::UnknownGrant(grant_id))?
            .capability_id;
        if let Some(mut grant) = self.world.get_mut::<GrantNode>(grant_entity) {
            grant.revoked = true;
        }
        let capability_entity = self
            .world
            .resource::<TopologyIndex>()
            .capabilities
            .get(&capability_id)
            .copied()
            .ok_or(RuntimeError::UnknownCapability(capability_id))?;
        if let Some(mut capability) = self.world.get_mut::<CapabilityNode>(capability_entity) {
            capability.retired = true;
        }
        Ok(())
    }

    /// Revoke one grant without retiring its capability implementation.
    pub fn revoke_grant(&mut self, grant_id: GrantId) -> Result<(), RuntimeError> {
        let entity = self
            .world
            .resource::<TopologyIndex>()
            .grants
            .get(&grant_id)
            .copied()
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        let mut grant = self
            .world
            .get_mut::<GrantNode>(entity)
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        grant.revoked = true;
        Ok(())
    }

    /// Return the exact capability and grant represented by a stable grant ID.
    pub fn tool_grant(&self, grant_id: GrantId) -> Result<ToolGrant, RuntimeError> {
        let index = self.world.resource::<TopologyIndex>();
        let grant_entity = index
            .grants
            .get(&grant_id)
            .copied()
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        let grant = self
            .world
            .get::<GrantNode>(grant_entity)
            .ok_or(RuntimeError::UnknownGrant(grant_id))?;
        let capability_entity = index
            .capabilities
            .get(&grant.capability_id)
            .copied()
            .ok_or(RuntimeError::UnknownCapability(grant.capability_id))?;
        let capability = self
            .world
            .get::<CapabilityNode>(capability_entity)
            .ok_or(RuntimeError::UnknownCapability(grant.capability_id))?;
        Ok(ToolGrant {
            capability_id: capability.id,
            grant_id,
            revision: capability.revision,
        })
    }

    /// Return the executable category recorded for a stable capability identity.
    pub fn capability_kind(
        &self,
        capability_id: CapabilityId,
    ) -> Result<CapabilityKind, RuntimeError> {
        let entity = self
            .world
            .resource::<TopologyIndex>()
            .capabilities
            .get(&capability_id)
            .copied()
            .ok_or(RuntimeError::UnknownCapability(capability_id))?;
        self.world
            .get::<CapabilityNode>(entity)
            .map(|capability| capability.kind)
            .ok_or(RuntimeError::UnknownCapability(capability_id))
    }

    /// Begin one run without waiting for provider or tool I/O.
    pub fn start_run(
        &mut self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
    ) -> Result<RunHandle, RuntimeError> {
        let agent_entity = self
            .world
            .resource::<TopologyIndex>()
            .agents
            .get(&agent_id)
            .copied()
            .ok_or(RuntimeError::UnknownAgent(agent_id))?;
        let agent = self
            .world
            .get::<AgentNode>(agent_entity)
            .cloned()
            .ok_or(RuntimeError::UnknownAgent(agent_id))?;
        let prompt = prompt.into();
        if matches!(prompt, Message::Assistant { .. }) {
            return Err(RuntimeError::InvalidPrompt {
                reason: "an assistant-role prompt cannot precede a model reply canonically",
            });
        }
        let validation = crate::persistence::TranscriptValidator::over(std::slice::from_ref(
            &prompt,
        ))
        .map_err(|_| RuntimeError::InvalidPrompt {
            reason: "prompt content violates canonical transcript invariants",
        })?;
        let run_id = RunId::new();
        let generation = Generation::default();
        let tick = self.world.resource::<RuntimeTick>().0;
        let telemetry_agent_name = match (
            agent.spec.record_telemetry_content,
            agent.spec.name.as_deref(),
        ) {
            (true, Some(name)) => name,
            (false, Some(_)) => "<redacted>",
            (_, None) => "",
        };
        let run_span = tracing::info_span!(
            target: "rig::bevy",
            "rig.bevy.run",
            run.id = %run_id,
            run.generation = generation.0,
            run.streaming = ?agent.spec.streaming,
            run.tenant = "<redacted>",
            rig.agent.name = %telemetry_agent_name,
        );
        let mut entity = self.world.spawn((
            RunNode {
                id: run_id,
                agent_id,
                model_id: agent.spec.model_id,
                tenant_id: agent.spec.tenant_id,
                generation,
                phase: if agent.spec.memory_id.is_some() {
                    RunPhase::LoadingMemory
                } else {
                    RunPhase::ReadyModel
                },
                streaming: agent.spec.streaming,
                created_tick: tick,
            },
            CanonicalTranscript {
                messages: vec![prompt],
                new_messages_start: 0,
            },
            crate::persistence::TranscriptValidation(validation),
            RunAccounting::default(),
            ActiveOperations::default(),
            AcceptedDeltas::default(),
            RecoveryFeedback::default(),
            ResponseRetryState::default(),
            InvalidToolRetryState::default(),
            RunEvents::new(self.config.event_capacity),
            RunProgress::default(),
            RunTelemetrySpan(run_span),
        ));
        if let (Some(memory_id), Some(conversation_id)) =
            (agent.spec.memory_id, agent.spec.conversation_id.clone())
        {
            entity.insert(MemoryProgress {
                memory_id,
                conversation_id,
                loaded: false,
                appended: false,
            });
        }
        if let Some((schema, policy)) = agent.spec.structured_output.clone() {
            entity.insert(StructuredOutputState {
                schema,
                policy,
                resolved_mode: crate::OutputMode::Auto,
                output_tool_name: None,
                retries: 0,
                value: None,
            });
        }
        let run_entity = entity.id();
        self.world
            .resource_mut::<TopologyIndex>()
            .runs
            .insert(run_id, run_entity);
        Ok(RunHandle {
            runtime_id: self.id(),
            run_id,
            generation,
            tenant_id: agent.spec.tenant_id,
        })
    }

    pub(crate) fn validate_handle(&self, handle: RunHandle) -> Result<Entity, RuntimeError> {
        if handle.runtime_id != self.id() {
            return Err(RuntimeError::ForeignRuntime {
                expected: self.id(),
                actual: handle.runtime_id,
            });
        }
        let entity = self
            .world
            .resource::<TopologyIndex>()
            .runs
            .get(&handle.run_id)
            .copied()
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        let run = self
            .world
            .get::<RunNode>(entity)
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        if run.tenant_id != handle.tenant_id {
            return Err(RuntimeError::TenantMismatch {
                expected: run.tenant_id,
                actual: handle.tenant_id,
            });
        }
        if run.generation != handle.generation {
            return Err(RuntimeError::StaleHandle {
                run_id: handle.run_id,
            });
        }
        Ok(entity)
    }

    /// Reacquire the valid handle for a retained run owned by `tenant_id`.
    ///
    /// This is the supported way to address a run after [`LocalRuntime::restore`],
    /// which advances every restored run's generation and thereby staleness-rejects
    /// handles minted by the snapshotted runtime.
    ///
    /// Note that a [`RunHandle`] is therefore not an unforgeable capability:
    /// any code that learns `(run_id, tenant_id)` — run identities appear in
    /// events and telemetry — can mint a live handle and, for example, cancel
    /// the run. Isolation between untrusting callers inside one tenant must be
    /// enforced by the embedder, not by handle possession.
    pub fn run_handle(
        &self,
        run_id: RunId,
        tenant_id: TenantId,
    ) -> Result<RunHandle, RuntimeError> {
        let entity = self
            .world
            .resource::<TopologyIndex>()
            .runs
            .get(&run_id)
            .copied()
            .ok_or(RuntimeError::UnknownRun(run_id))?;
        let run = self
            .world
            .get::<RunNode>(entity)
            .ok_or(RuntimeError::UnknownRun(run_id))?;
        if run.tenant_id != tenant_id {
            return Err(RuntimeError::TenantMismatch {
                expected: run.tenant_id,
                actual: tenant_id,
            });
        }
        Ok(RunHandle {
            runtime_id: self.id(),
            run_id,
            generation: run.generation,
            tenant_id,
        })
    }

    /// Request cancellation. The cancellation system wins before pending ingress.
    pub fn cancel(&mut self, handle: RunHandle) -> Result<(), RuntimeError> {
        let entity = self.validate_handle(handle)?;
        if !self.world.entity(entity).contains::<TerminalState>() {
            self.world.entity_mut(entity).insert(CancellationRequest);
        }
        Ok(())
    }

    fn abandon_stream(&mut self, handle: RunHandle) -> Result<(), RuntimeError> {
        let entity = self.validate_handle(handle)?;
        if !self.world.entity(entity).contains::<TerminalState>() {
            self.world.entity_mut(entity).insert(CancellationRequest);
            self.stage_available_ingress(None, self.config.ingress_capacity);
            self.schedule.run(&mut self.world);
            self.spawn_pending_effects();
            self.prune_effect_tasks();
        }
        let entity = self.validate_handle(handle)?;
        if !self.world.entity(entity).contains::<TerminalState>() {
            return Err(RuntimeError::Livelock);
        }
        let tick = self.world.resource::<RuntimeTick>().0;
        self.world.entity_mut(entity).insert(TerminalObservation {
            observed_tick: tick,
        });
        Ok(())
    }

    /// Subscribe to bounded lifecycle events for an active or retained run.
    pub fn subscribe(
        &mut self,
        handle: RunHandle,
    ) -> Result<broadcast::Receiver<RunEvent>, RuntimeError> {
        let entity = self.validate_handle(handle)?;
        let events = self
            .world
            .get::<RunEvents>(entity)
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        Ok(events.subscribe())
    }

    /// Return all rejected effect records retained by the runtime.
    #[must_use]
    pub fn effect_rejections(&self) -> &[crate::EffectRejection] {
        &self.world.resource::<RejectionLog>().0
    }

    /// Inject ingress for testing or an externally hosted executor.
    pub fn ingest(&mut self, ingress: EffectIngress) -> Result<(), RuntimeError> {
        self.ingress_tx.try_send(ingress)
    }

    /// Return active effect headers for diagnostics and hosted execution.
    pub fn active_effect_headers(
        &self,
        handle: RunHandle,
    ) -> Result<Vec<EffectHeader>, RuntimeError> {
        let entity = self.validate_handle(handle)?;
        let operations = self
            .world
            .get::<ActiveOperations>(entity)
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        Ok(operations
            .0
            .values()
            .filter(|operation| !operation.completed)
            .map(|operation| operation.header)
            .collect())
    }

    fn spawn_pending_effects(&mut self) {
        let intents = std::mem::take(&mut self.world.resource_mut::<PendingEffects>().0);
        let mut deferred = Vec::new();
        for intent in intents {
            let operation_id = intent.header().operation_id;
            let run_id = intent.header().run_id;
            let run_is_active = self
                .world
                .resource::<TopologyIndex>()
                .runs
                .get(&run_id)
                .copied()
                .is_some_and(|entity| !self.world.entity(entity).contains::<TerminalState>());
            if !run_is_active {
                continue;
            }
            let parent = self
                .world
                .resource::<TopologyIndex>()
                .runs
                .get(&run_id)
                .and_then(|entity| self.world.get::<RunTelemetrySpan>(*entity))
                .map(|telemetry| telemetry.0.clone())
                .unwrap_or_else(tracing::Span::none);
            let effect_span = match &intent {
                EffectIntent::Model(_) => tracing::info_span!(
                    target: "rig::completion_parent",
                    parent: &parent,
                    "chat",
                    rig.completion_parent = true,
                    gen_ai.operation.name = tracing::field::Empty,
                    gen_ai.provider.name = tracing::field::Empty,
                    gen_ai.request.model = tracing::field::Empty,
                    gen_ai.system_instructions = tracing::field::Empty,
                    gen_ai.response.id = tracing::field::Empty,
                    gen_ai.response.model = tracing::field::Empty,
                    gen_ai.usage.input_tokens = tracing::field::Empty,
                    gen_ai.usage.output_tokens = tracing::field::Empty,
                    gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                    gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                    gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                    gen_ai.usage.reasoning_tokens = tracing::field::Empty,
                    gen_ai.input.messages = tracing::field::Empty,
                    gen_ai.output.messages = tracing::field::Empty,
                    rig.run.id = %run_id,
                    rig.operation.id = %operation_id,
                ),
                EffectIntent::Tool(intent) => tracing::info_span!(
                    target: "rig::bevy",
                    parent: &parent,
                    "rig.bevy.tool_effect",
                    rig.run.id = %run_id,
                    rig.operation.id = %operation_id,
                    tool.name = %intent.name,
                ),
                EffectIntent::Memory(intent) => {
                    let kind = match intent {
                        crate::effects::MemoryEffectIntent::Load { .. } => "load",
                        crate::effects::MemoryEffectIntent::Append { .. } => "append",
                    };
                    tracing::info_span!(
                        target: "rig::bevy",
                        parent: &parent,
                        "rig.bevy.memory_effect",
                        rig.run.id = %run_id,
                        rig.operation.id = %operation_id,
                        memory.operation = kind,
                    )
                }
            };
            let ingress = self.ingress_tx.clone();
            let timeout = self.config.effect_timeout;
            let kind_limit = match &intent {
                EffectIntent::Model(_) => Some(&self.model_limit),
                EffectIntent::Tool(_) => Some(&self.tool_limit),
                EffectIntent::Memory(_) => None,
            };
            let Some(permits) = try_acquire_permits(&self.global_limit, kind_limit) else {
                deferred.push(intent);
                continue;
            };
            let task: NativeFuture<()> = match intent {
                EffectIntent::Model(intent) => {
                    let binding = self
                        .model_tenants
                        .get(&intent.model_id)
                        .filter(|owner| **owner == intent.header.tenant_id)
                        .and_then(|_| self.models.get(&intent.model_id))
                        .cloned();
                    Box::pin(async move {
                        let _permits = permits;
                        let result = match binding {
                            Some(binding) => {
                                let future = if intent.streaming {
                                    binding.stream(intent.request, intent.header, ingress.clone())
                                } else {
                                    binding.complete(intent.request)
                                };
                                match tokio::time::timeout(timeout, future).await {
                                    Ok(result) => result,
                                    Err(_) => Err(ModelEffectError::TimedOut),
                                }
                            }
                            None => Err(ModelEffectError::MissingBinding {
                                model_id: intent.model_id,
                            }),
                        };
                        let _ = ingress
                            .send(EffectIngress::Completion(EffectCompletion::Model {
                                header: intent.header,
                                result,
                            }))
                            .await;
                    })
                }
                EffectIntent::Tool(intent) => {
                    let binding = intent
                        .header
                        .capability_id
                        .and_then(|id| self.tools.get(&id).cloned());
                    Box::pin(async move {
                        let _permits = permits;
                        let result = match binding {
                            Some(binding) if binding.definition().name == intent.name => {
                                match tokio::time::timeout(
                                    timeout,
                                    binding.execute(intent.arguments),
                                )
                                .await
                                {
                                    Ok(Ok(output)) => {
                                        crate::effects::ToolEffectOutput::Success(output)
                                    }
                                    Ok(Err(error)) => {
                                        crate::effects::ToolEffectOutput::Failure(error)
                                    }
                                    Err(_) => crate::effects::ToolEffectOutput::Failure(
                                        ToolExecutionError::timeout("tool effect timed out"),
                                    ),
                                }
                            }
                            Some(_) => crate::effects::ToolEffectOutput::Failure(
                                ToolExecutionError::permission_denied(
                                    "advertised tool identity did not match its exact capability binding",
                                ),
                            ),
                            None => crate::effects::ToolEffectOutput::Failure(
                                ToolExecutionError::not_found(
                                    "exact tool capability implementation is unavailable",
                                ),
                            ),
                        };
                        let _ = ingress
                            .send(EffectIngress::Completion(EffectCompletion::Tool {
                                header: intent.header,
                                tool_call_id: intent.tool_call_id,
                                order: intent.order,
                                result,
                            }))
                            .await;
                    })
                }
                EffectIntent::Memory(intent) => {
                    let (header, memory_id) = match &intent {
                        crate::effects::MemoryEffectIntent::Load {
                            header, memory_id, ..
                        }
                        | crate::effects::MemoryEffectIntent::Append {
                            header, memory_id, ..
                        } => (*header, *memory_id),
                    };
                    let binding = self
                        .memory_tenants
                        .get(&memory_id)
                        .filter(|owner| **owner == header.tenant_id)
                        .and_then(|_| self.memories.get(&memory_id))
                        .cloned();
                    Box::pin(async move {
                        let _permits = permits;
                        let result = match (binding, intent) {
                            (
                                Some(binding),
                                crate::effects::MemoryEffectIntent::Load {
                                    conversation_id, ..
                                },
                            ) => match tokio::time::timeout(timeout, binding.load(conversation_id))
                                .await
                            {
                                Ok(result) => {
                                    result.map(crate::effects::MemoryEffectOutput::Loaded)
                                }
                                Err(_) => Err(MemoryEffectError::TimedOut),
                            },
                            (
                                Some(binding),
                                crate::effects::MemoryEffectIntent::Append {
                                    conversation_id,
                                    messages,
                                    ..
                                },
                            ) => match tokio::time::timeout(
                                timeout,
                                binding.append(conversation_id, messages),
                            )
                            .await
                            {
                                Ok(result) => {
                                    result.map(|()| crate::effects::MemoryEffectOutput::Appended)
                                }
                                Err(_) => Err(MemoryEffectError::TimedOut),
                            },
                            (None, _) => Err(MemoryEffectError::MissingBinding { memory_id }),
                        };
                        let _ = ingress
                            .send(EffectIngress::Completion(EffectCompletion::Memory {
                                header,
                                result,
                            }))
                            .await;
                    })
                }
            };
            let capacity_guard = EffectCapacityGuard(self.effect_capacity_tx.clone());
            let task = async move {
                let _capacity_guard = capacity_guard;
                task.await;
            };
            let abort = self.effects.spawn(task.instrument(effect_span));
            self.aborts.insert(operation_id, (run_id, abort));
        }
        self.world.resource_mut::<PendingEffects>().0 = deferred;
    }

    fn prune_effect_tasks(&mut self) {
        while self.effects.try_join_next().is_some() {}
        let terminal_runs = self
            .world
            .query::<(&RunNode, Option<&TerminalState>)>()
            .iter(&self.world)
            .filter_map(|(run, terminal)| terminal.map(|_| run.id))
            .collect::<Vec<_>>();
        for (run_id, abort) in self.aborts.values() {
            if terminal_runs.contains(run_id) {
                abort.abort();
            }
        }
        self.aborts.retain(|operation_id, (run_id, _)| {
            if terminal_runs.contains(run_id) {
                return false;
            }
            self.world
                .resource::<TopologyIndex>()
                .runs
                .get(run_id)
                .and_then(|entity| self.world.get::<ActiveOperations>(*entity))
                .and_then(|operations| operations.0.get(operation_id))
                .is_some_and(|operation| !operation.completed)
        });
        for capability_id in std::mem::take(&mut self.world.resource_mut::<CapabilitiesToDrop>().0)
        {
            self.tools.remove(&capability_id);
            self.tool_binding_identities.remove(&capability_id);
        }
    }

    fn stage_available_ingress(&mut self, target_run: Option<RunId>, limit: usize) -> bool {
        let (already_staged, mut target_progress) = {
            let pending = self.world.resource::<PendingIngress>();
            (
                pending.0.len(),
                pending
                    .0
                    .iter()
                    .any(|ingress| target_run.is_some_and(|target| target == ingress.run_id())),
            )
        };
        for _ in 0..limit.saturating_sub(already_staged) {
            let Ok(ingress) = self.ingress_rx.try_recv() else {
                break;
            };
            let run_id = ingress.run_id();
            target_progress |= target_run.is_some_and(|target| run_id == target);
            self.world.resource_mut::<PendingIngress>().0.push(ingress);
        }
        target_progress
    }

    /// Run one bounded deterministic schedule pass and dispatch newly owned effects.
    pub async fn step(&mut self, handle: RunHandle) -> Result<RunStepStatus, RuntimeError> {
        self.step_with_ingress_limit(handle, self.config.ingress_capacity)
            .await
    }

    async fn step_with_ingress_limit(
        &mut self,
        handle: RunHandle,
        ingress_limit: usize,
    ) -> Result<RunStepStatus, RuntimeError> {
        let entity = self.validate_handle(handle)?;
        let target_ingress_staged =
            self.stage_available_ingress(Some(handle.run_id), ingress_limit);
        let before = self
            .world
            .get::<RunProgress>(entity)
            .map_or(0, |progress| progress.0);
        self.schedule.run(&mut self.world);
        self.spawn_pending_effects();
        self.prune_effect_tasks();
        if self
            .world
            .get_entity(entity)
            .is_ok_and(|entity_ref| entity_ref.contains::<TerminalState>())
        {
            // Returning `Terminal` to the handle holder is an observation, and
            // each one renews the retention lease: a caller actively polling a
            // terminal run never loses its result, while a handle that stops
            // being stepped ages out from its last observation.
            let tick = self.world.resource::<RuntimeTick>().0;
            self.world.entity_mut(entity).insert(TerminalObservation {
                observed_tick: tick,
            });
            return Ok(RunStepStatus::Terminal);
        }
        if !self
            .world
            .resource::<TopologyIndex>()
            .runs
            .contains_key(&handle.run_id)
        {
            return Err(RuntimeError::UnknownRun(handle.run_id));
        }
        let after = self
            .world
            .get::<RunProgress>(entity)
            .map_or(before, |progress| progress.0);
        if after == before {
            Ok(RunStepStatus::Quiescent)
        } else if target_ingress_staged {
            Ok(RunStepStatus::EffectProgressed)
        } else {
            Ok(RunStepStatus::Progressed)
        }
    }

    /// Wait for one bounded effect message and stage it for the next schedule pass.
    pub async fn wait_for_effect(&mut self) -> Result<(), RuntimeError> {
        let deadline = tokio::time::Instant::now() + self.config.effect_timeout;
        self.wait_for_effect_for(None, deadline).await.map(|_| ())
    }

    async fn wait_for_effect_for(
        &mut self,
        target_run: Option<RunId>,
        deadline: tokio::time::Instant,
    ) -> Result<bool, RuntimeError> {
        let ingress = tokio::time::timeout_at(deadline, self.ingress_rx.recv())
            .await
            .map_err(|_| RuntimeError::EffectWaitTimedOut)?
            .ok_or(RuntimeError::IngressClosed)?;
        let run_id = ingress.run_id();
        self.world.resource_mut::<PendingIngress>().0.push(ingress);
        Ok(target_run.is_none_or(|target| target == run_id))
    }

    fn waits_for_effect_capacity(&self, handle: RunHandle) -> Result<bool, RuntimeError> {
        let entity = self.validate_handle(handle)?;
        Ok(self.world.entity(entity).contains::<EffectQueueWait>())
    }

    async fn wait_for_effect_capacity(
        &self,
        receiver: &mut watch::Receiver<u64>,
    ) -> Result<(), RuntimeError> {
        tokio::time::timeout(self.config.effect_timeout, receiver.changed())
            .await
            .map_err(|_| RuntimeError::EffectWaitTimedOut)?
            .map_err(|_| RuntimeError::IngressClosed)
    }

    // The one quiescent-wait contract shared by the blocking and streaming
    // local drivers. Only real progress (`EffectProgressed`, observed by the
    // caller) clears the effect deadline: staged-but-rejected ingress cannot
    // extend an effect wait. Capacity release also wakes the pass so a
    // semaphore-deferred intent spawns even when its owner was aborted without
    // ingress. Returns whether the caller should reset its pass budget. The
    // hosted driver intentionally has its own wait (short world locks, notify
    // plus two watch channels, timeout-to-finish) and must not be folded in.
    async fn wait_quiescent(
        &mut self,
        handle: RunHandle,
        capacity_rx: &mut watch::Receiver<u64>,
        target_effect_deadline: &mut Option<tokio::time::Instant>,
    ) -> Result<bool, RuntimeError> {
        let has_active = !self.active_effect_headers(handle)?.is_empty();
        if has_active {
            let deadline = *target_effect_deadline
                .get_or_insert_with(|| tokio::time::Instant::now() + self.config.effect_timeout);
            tokio::select! {
                result = self.wait_for_effect_for(Some(handle.run_id), deadline) => match result {
                    Err(error) => {
                        crate::schedule::fail_effect_wait(
                            &mut self.world,
                            handle.run_id,
                            error.to_string(),
                        );
                        Ok(false)
                    }
                    Ok(target_staged) => Ok(target_staged),
                },
                changed = capacity_rx.changed() => match changed {
                    Ok(()) => Ok(true),
                    Err(_) => {
                        crate::schedule::fail_effect_wait(
                            &mut self.world,
                            handle.run_id,
                            RuntimeError::IngressClosed.to_string(),
                        );
                        Ok(false)
                    }
                },
            }
        } else if self.waits_for_effect_capacity(handle)? {
            *target_effect_deadline = None;
            if let Err(error) = self.wait_for_effect_capacity(capacity_rx).await {
                crate::schedule::fail_effect_wait(
                    &mut self.world,
                    handle.run_id,
                    error.to_string(),
                );
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            crate::schedule::fail_effect_wait(
                &mut self.world,
                handle.run_id,
                "run became quiescent before reaching terminal state".to_string(),
            );
            Ok(false)
        }
    }

    /// Drive one run to terminal state without collapsing its terminal reason into an error.
    pub async fn drive_to_terminal(
        &mut self,
        handle: RunHandle,
    ) -> Result<LocalRunResult, RuntimeError> {
        let mut remaining_passes = self.config.max_schedule_passes;
        let mut target_effect_deadline = None;
        loop {
            let mut capacity_rx = self.effect_capacity_tx.subscribe();
            match self.step(handle).await? {
                RunStepStatus::Terminal => return self.finish_run(handle),
                RunStepStatus::Progressed => {
                    if remaining_passes == 0 {
                        break;
                    }
                    remaining_passes -= 1;
                }
                RunStepStatus::EffectProgressed => {
                    remaining_passes = self.config.max_schedule_passes;
                    target_effect_deadline = None;
                }
                RunStepStatus::Quiescent => {
                    if self
                        .wait_quiescent(handle, &mut capacity_rx, &mut target_effect_deadline)
                        .await?
                    {
                        remaining_passes = self.config.max_schedule_passes;
                    }
                }
            }
        }
        crate::schedule::fail_effect_wait(
            &mut self.world,
            handle.run_id,
            RuntimeError::Livelock.to_string(),
        );
        self.finish_run(handle)
    }

    /// Start and drive a run using the agent's configured model surface.
    pub async fn run(
        &mut self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
    ) -> Result<LocalRunResult, RuntimeError> {
        self.run_with_mode(agent_id, prompt, None).await
    }

    /// Start and drive a run through the provider blocking surface.
    pub async fn run_blocking(
        &mut self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
    ) -> Result<LocalRunResult, RuntimeError> {
        self.run_with_mode(agent_id, prompt, Some(StreamingMode::Blocking))
            .await
    }

    async fn run_with_mode(
        &mut self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
        mode: Option<StreamingMode>,
    ) -> Result<LocalRunResult, RuntimeError> {
        let handle = self.start_run(agent_id, prompt)?;
        if let Some(mode) = mode {
            let entity = self.validate_handle(handle)?;
            if let Some(mut run) = self.world.get_mut::<RunNode>(entity) {
                run.streaming = mode;
            }
        }
        let mut guard = LocalRunGuard {
            runtime: self,
            handle,
            closed: false,
        };
        let result = guard.runtime.drive_to_terminal(handle).await?;
        guard.closed = true;
        Self::ensure_completed(result)
    }

    /// Start and drive a provider-streaming run with concrete typed final events.
    pub async fn run_streaming<R>(
        &mut self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
    ) -> Result<StreamingRunResult<R>, RuntimeError>
    where
        R: Any + Send + Sync + 'static,
    {
        let mut stream = self.start_streaming::<R>(agent_id, prompt)?;
        let mut events = Vec::new();
        while let Some(event) = stream.next_event().await? {
            if events.len() == stream.runtime.config.event_capacity {
                return Err(RuntimeError::EventCollectionLimit {
                    capacity: stream.runtime.config.event_capacity,
                });
            }
            events.push(event);
        }
        let result = stream.finish()?;
        Ok(StreamingRunResult { result, events })
    }

    /// Start a live provider-streaming run with a concrete provider-final type.
    pub fn start_streaming<R>(
        &mut self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
    ) -> Result<LocalStreamingRun<'_, R>, RuntimeError>
    where
        R: Any + Send + Sync + 'static,
    {
        let handle = self.start_run(agent_id, prompt)?;
        let entity = self.validate_handle(handle)?;
        if let Some(mut run) = self.world.get_mut::<RunNode>(entity) {
            run.streaming = StreamingMode::Streaming;
        }
        let receiver = self.subscribe(handle)?;
        Ok(LocalStreamingRun {
            runtime: self,
            handle,
            receiver,
            terminal_seen: false,
            closed: false,
            response: PhantomData,
        })
    }

    fn ensure_completed(result: LocalRunResult) -> Result<LocalRunResult, RuntimeError> {
        match &result.terminal_reason {
            TerminalReason::Completed => Ok(result),
            TerminalReason::Cancelled => Err(RuntimeError::Cancelled),
            TerminalReason::Stopped => Err(RuntimeError::Stopped),
            TerminalReason::ModelCallBudgetExhausted => Err(RuntimeError::ModelCallBudgetExhausted),
            TerminalReason::Failed { code } => {
                let diagnostic = result
                    .failure_diagnostic
                    .clone()
                    .unwrap_or_else(|| code.clone());
                match code.as_str() {
                    "model_effect" => match result.failure_cause {
                        Some(TerminalCause::Model(error)) => Err(RuntimeError::ModelEffect(error)),
                        _ => Err(RuntimeError::RunFailed {
                            code: code.clone(),
                            diagnostic,
                        }),
                    },
                    "memory_effect" => match result.failure_cause {
                        Some(TerminalCause::Memory(error)) => {
                            Err(RuntimeError::MemoryEffect(error))
                        }
                        _ => Err(RuntimeError::RunFailed {
                            code: code.clone(),
                            diagnostic,
                        }),
                    },
                    "structured_output" => Err(RuntimeError::StructuredOutput(diagnostic)),
                    _ => Err(RuntimeError::RunFailed {
                        code: code.clone(),
                        diagnostic,
                    }),
                }
            }
        }
    }

    /// Materialize and observe a retained terminal run without triggering cleanup.
    pub fn finish_run(&mut self, handle: RunHandle) -> Result<LocalRunResult, RuntimeError> {
        let entity = self.validate_handle(handle)?;
        let terminal = self
            .world
            .get::<TerminalState>(entity)
            .cloned()
            .ok_or(RuntimeError::Livelock)?;
        let tick = self.world.resource::<RuntimeTick>().0;
        self.world.entity_mut(entity).insert(TerminalObservation {
            observed_tick: tick,
        });
        let transcript = self
            .world
            .get::<CanonicalTranscript>(entity)
            .cloned()
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        let accounting = self
            .world
            .get::<RunAccounting>(entity)
            .cloned()
            .ok_or(RuntimeError::UnknownRun(handle.run_id))?;
        let events = self
            .world
            .get::<RunEvents>(entity)
            .map(|events| events.events.iter().cloned().collect())
            .unwrap_or_default();
        let raw = self.world.get::<RawFinalRecord>(entity);
        let raw_final_operation = raw.map(|raw| raw.operation_id);
        let raw_final = raw.map(|raw| raw.raw.clone());
        let structured_output = self
            .world
            .get::<StructuredOutputState>(entity)
            .and_then(|state| state.value.clone());
        let failure_diagnostic = self
            .world
            .get::<TerminalDiagnostic>(entity)
            .map(|diagnostic| diagnostic.message.clone());
        let failure_cause = self.world.get::<TerminalCause>(entity).cloned();
        let text = transcript.messages.iter().rev().find_map(|message| {
            let Message::Assistant { content, .. } = message else {
                return None;
            };
            let text = content
                .iter()
                .filter_map(|item| match item {
                    rig_core::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            (!text.is_empty()).then_some(text)
        });
        Ok(LocalRunResult {
            run_id: handle.run_id,
            text,
            transcript: transcript.messages,
            usage: accounting.usage,
            model_calls: accounting.model_calls,
            terminal_reason: terminal.reason,
            structured_output,
            events,
            raw_final_operation,
            failure_diagnostic,
            raw_final,
            failure_cause,
        })
    }
}

impl Drop for LocalRuntime {
    fn drop(&mut self) {
        self.effects.abort_all();
    }
}

/// Serialized hosted driver that invokes the same schedule and effect adapters.
#[derive(Clone)]
pub struct HostedRuntime {
    inner: Arc<tokio::sync::Mutex<LocalRuntime>>,
    run_notifies: Arc<Mutex<HashMap<RunId, Arc<Notify>>>>,
    effect_capacity_tx: watch::Sender<u64>,
    ingress_progress_tx: watch::Sender<u64>,
    driving_runs: Arc<Mutex<BTreeSet<RunId>>>,
}

struct HostedDriveGuard {
    driving_runs: Arc<Mutex<BTreeSet<RunId>>>,
    run_id: RunId,
}

impl Drop for HostedDriveGuard {
    fn drop(&mut self) {
        lock_unpoisoned(&self.driving_runs).remove(&self.run_id);
    }
}

impl HostedRuntime {
    /// Wrap a local runtime without changing schedule semantics.
    #[must_use]
    pub fn new(runtime: LocalRuntime) -> Self {
        let run_notifies = Arc::clone(&runtime.ingress_tx.run_notifies);
        let effect_capacity_tx = runtime.effect_capacity_tx.clone();
        let ingress_progress_tx = runtime.ingress_tx.ingress_progress_tx.clone();
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(runtime)),
            run_notifies,
            effect_capacity_tx,
            ingress_progress_tx,
            driving_runs: Arc::new(Mutex::new(BTreeSet::new())),
        }
    }

    fn run_notifies(&self) -> MutexGuard<'_, HashMap<RunId, Arc<Notify>>> {
        lock_unpoisoned(&self.run_notifies)
    }

    fn run_notify(&self, run_id: RunId) -> Arc<Notify> {
        Arc::clone(
            self.run_notifies()
                .entry(run_id)
                .or_insert_with(|| Arc::new(Notify::new())),
        )
    }

    /// Start a run while holding the world lock only for the synchronous mutation.
    pub async fn start_run(
        &self,
        agent_id: AgentId,
        prompt: impl Into<Message>,
    ) -> Result<RunHandle, RuntimeError> {
        let handle = self.inner.lock().await.start_run(agent_id, prompt)?;
        self.run_notify(handle.run_id);
        Ok(handle)
    }

    /// Request cancellation without waiting for a convenience `run` lock.
    pub async fn cancel(&self, handle: RunHandle) -> Result<(), RuntimeError> {
        self.inner.lock().await.cancel(handle)?;
        self.run_notify(handle.run_id).notify_one();
        Ok(())
    }

    /// Reacquire the valid handle for a retained run owned by `tenant_id`.
    pub async fn run_handle(
        &self,
        run_id: RunId,
        tenant_id: TenantId,
    ) -> Result<RunHandle, RuntimeError> {
        self.inner.lock().await.run_handle(run_id, tenant_id)
    }

    /// Run one short schedule pass without holding the lock across effect I/O.
    pub async fn step(&self, handle: RunHandle) -> Result<RunStepStatus, RuntimeError> {
        let mut runtime = self.inner.lock().await;
        let result = runtime.step(handle).await;
        // Cleaned-up runs must release their notify entries, or abandoned hosted
        // handles would grow this map without bound.
        let retained_runs = &runtime.world.resource::<TopologyIndex>().runs;
        self.run_notifies()
            .retain(|run_id, _| retained_runs.contains_key(run_id));
        result
    }

    /// Observe a retained terminal result.
    pub async fn finish_run(&self, handle: RunHandle) -> Result<LocalRunResult, RuntimeError> {
        let result = self.inner.lock().await.finish_run(handle);
        if result.is_ok() {
            self.run_notifies().remove(&handle.run_id);
        }
        result
    }

    /// Drive a hosted run while yielding between short world locks.
    ///
    /// Dropping this future releases its single-driver lease but intentionally leaves the run
    /// and its owned effects active. A hosted caller may resume the same handle or cancel it
    /// explicitly; dropping a driver is not an implicit cancellation boundary.
    pub async fn drive_to_terminal(
        &self,
        handle: RunHandle,
    ) -> Result<LocalRunResult, RuntimeError> {
        {
            let inserted = lock_unpoisoned(&self.driving_runs).insert(handle.run_id);
            if !inserted {
                return Err(RuntimeError::RunAlreadyDriven(handle.run_id));
            }
        }
        let _drive_guard = HostedDriveGuard {
            driving_runs: Arc::clone(&self.driving_runs),
            run_id: handle.run_id,
        };
        let (max_schedule_passes, effect_timeout) = {
            let runtime = self.inner.lock().await;
            (
                runtime.config.max_schedule_passes,
                runtime.config.effect_timeout,
            )
        };
        let run_notify = self.run_notify(handle.run_id);
        let mut remaining_passes = max_schedule_passes;
        let mut target_effect_deadline = None;
        loop {
            let mut capacity_rx = self.effect_capacity_tx.subscribe();
            let mut ingress_progress_rx = self.ingress_progress_tx.subscribe();
            match self.step(handle).await? {
                RunStepStatus::Terminal => return self.finish_run(handle).await,
                RunStepStatus::Progressed => {
                    if remaining_passes == 0 {
                        break;
                    }
                    remaining_passes -= 1;
                    tokio::task::yield_now().await;
                }
                RunStepStatus::EffectProgressed => {
                    remaining_passes = max_schedule_passes;
                    target_effect_deadline = None;
                    tokio::task::yield_now().await;
                }
                RunStepStatus::Quiescent => {
                    let notified = run_notify.notified();
                    let (has_active, waits_for_capacity) = {
                        let runtime = self.inner.lock().await;
                        (
                            !runtime.active_effect_headers(handle)?.is_empty(),
                            runtime.waits_for_effect_capacity(handle)?,
                        )
                    };
                    if !has_active && !waits_for_capacity {
                        let mut runtime = self.inner.lock().await;
                        crate::schedule::fail_effect_wait(
                            &mut runtime.world,
                            handle.run_id,
                            "run became quiescent before reaching terminal state".to_string(),
                        );
                        return runtime.finish_run(handle);
                    }
                    let wait = async {
                        tokio::select! {
                            () = notified => Ok(()),
                            changed = ingress_progress_rx.changed() => changed,
                            changed = capacity_rx.changed() => changed,
                        }
                    };
                    let wait_result = if has_active {
                        let deadline = *target_effect_deadline
                            .get_or_insert_with(|| tokio::time::Instant::now() + effect_timeout);
                        tokio::time::timeout_at(deadline, wait).await
                    } else {
                        target_effect_deadline = None;
                        tokio::time::timeout(effect_timeout, wait).await
                    };
                    if !matches!(wait_result, Ok(Ok(()))) {
                        let mut runtime = self.inner.lock().await;
                        crate::schedule::fail_effect_wait(
                            &mut runtime.world,
                            handle.run_id,
                            "hosted effect wait timed out".to_string(),
                        );
                        return runtime.finish_run(handle);
                    }
                    remaining_passes = max_schedule_passes;
                }
            }
        }
        let mut runtime = self.inner.lock().await;
        crate::schedule::fail_effect_wait(
            &mut runtime.world,
            handle.run_id,
            RuntimeError::Livelock.to_string(),
        );
        runtime.finish_run(handle)
    }

    /// Return a redacted provider-final diagnostic for hosted/erased consumers.
    pub async fn provider_diagnostic(
        &self,
        handle: RunHandle,
    ) -> Result<Option<HostedProviderDiagnostic>, RuntimeError> {
        let runtime = self.inner.lock().await;
        let entity = runtime.validate_handle(handle)?;
        Ok(runtime
            .world
            .get::<crate::components::RawFinalRecord>(entity)
            .map(|record| HostedProviderDiagnostic {
                operation_id: record.operation_id,
                provider_type: record.raw.type_name(),
            }))
    }
}

#[cfg(test)]
mod portable_adapter_tests {
    use rig_core::tool::{PortableTool, ToolErrorKind};
    use rig_runtime_conformance::{
        PortableEmbeddingFixture, portable_dynamic_fixture, portable_fixture_output,
    };

    use super::{DynamicToolBinding, ToolBinding, TypedToolBinding};

    #[tokio::test]
    async fn portable_embedding_binding_preserves_classification_and_rich_error() {
        let binding =
            TypedToolBinding(std::sync::Arc::new(PortableEmbeddingFixture::new("shared")));

        assert_eq!(
            binding.definition().name,
            <PortableEmbeddingFixture as PortableTool>::NAME
        );
        let result = binding
            .execute(serde_json::json!({"value": "ignored", "fail": true}))
            .await;
        assert!(result.is_err(), "portable fixture unexpectedly succeeded");
        let Err(error) = result else {
            return;
        };
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert_eq!(error.code(), Some("portable_fixture"));
        assert_eq!(
            error.model_output(),
            &portable_fixture_output("portable failure")
        );
    }

    #[tokio::test]
    async fn portable_dynamic_binding_preserves_classification_and_rich_error() {
        let binding = DynamicToolBinding(portable_dynamic_fixture());
        let result = binding
            .execute(serde_json::json!({"value": "ignored", "fail": true}))
            .await;
        assert!(result.is_err(), "portable fixture unexpectedly succeeded");
        let Err(error) = result else {
            return;
        };
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert_eq!(error.code(), Some("portable_dynamic_fixture"));
        assert_eq!(
            error.model_output(),
            &portable_fixture_output("portable dynamic failure")
        );
    }
}

// This module panics inside a provider double to model an effect task that dies
// without sending ingress; expectations are confined to test setup.
#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod driver_wait_tests {
    use std::time::Duration;

    use rig_core::{
        completion::{CompletionError, CompletionRequest, CompletionResponse},
        streaming::StreamingCompletionResponse,
        test_utils::{MockCompletionModel, MockResponse},
    };

    use super::{CompletionModel, HostedRuntime, LocalRuntime};
    use crate::{
        AgentSpec, CorrelationId, EffectCompletion, EffectHeader, EffectIngress, ModelEffectError,
        OperationId, RuntimeConfig, TenantId, TerminalReason,
    };

    /// A model whose spawned effect task dies without ever sending ingress.
    #[derive(Clone)]
    struct SilentModel;

    impl CompletionModel for SilentModel {
        type Response = MockResponse;
        type StreamingResponse = MockResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            panic!("silent model never completes")
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            panic!("silent model never streams")
        }
    }

    #[tokio::test]
    async fn rejected_target_ingress_cannot_extend_the_effect_deadline() {
        let mut runtime = LocalRuntime::with_config(RuntimeConfig {
            effect_timeout: Duration::from_millis(200),
            ..RuntimeConfig::default()
        })
        .expect("runtime");
        let tenant_id = TenantId::new();
        let model_id = runtime.register_model(tenant_id, SilentModel);
        let agent = runtime
            .spawn_agent_spec(AgentSpec::new(model_id, tenant_id))
            .expect("agent");
        let handle = runtime.start_run(agent, "hang").expect("run");
        let ingress = runtime.ingress_tx.clone();
        let runtime_id = runtime.id();
        let flooder = tokio::spawn(async move {
            loop {
                let header = EffectHeader {
                    runtime_id,
                    run_id: handle.run_id,
                    operation_id: OperationId::new(),
                    generation: handle.generation,
                    correlation_id: CorrelationId::new(),
                    tenant_id: handle.tenant_id,
                    capability_id: None,
                    grant_id: None,
                    capability_revision: None,
                };
                let garbage = EffectIngress::Completion(EffectCompletion::Model {
                    header,
                    result: Err(ModelEffectError::TimedOut),
                });
                if ingress.send(garbage).await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });

        let result =
            tokio::time::timeout(Duration::from_secs(5), runtime.drive_to_terminal(handle))
                .await
                .expect("rejected target ingress must not extend the effect deadline")
                .expect("driver result");
        flooder.abort();

        assert!(matches!(
            result.terminal_reason,
            TerminalReason::Failed { ref code } if code == "effect_wait"
        ));
    }

    #[tokio::test]
    async fn hosted_notify_entries_are_pruned_with_cleaned_runs() {
        let mut local = LocalRuntime::with_config(RuntimeConfig {
            terminal_retention_ticks: 64,
            unobserved_terminal_retention_ticks: 2,
            ..RuntimeConfig::default()
        })
        .expect("runtime");
        let tenant_id = TenantId::new();
        let abandoned_model =
            local.register_model(tenant_id, MockCompletionModel::text("abandoned"));
        let keeper_model = local.register_model(tenant_id, MockCompletionModel::text("keeper"));
        let abandoned_agent = local
            .spawn_agent_spec(AgentSpec::new(abandoned_model, tenant_id))
            .expect("agent");
        let keeper_agent = local
            .spawn_agent_spec(AgentSpec::new(keeper_model, tenant_id))
            .expect("agent");
        let hosted = HostedRuntime::new(local);
        let abandoned = hosted
            .start_run(abandoned_agent, "abandoned")
            .await
            .expect("run");
        hosted.cancel(abandoned).await.expect("cancel");
        assert!(hosted.run_notifies().contains_key(&abandoned.run_id));
        let keeper = hosted.start_run(keeper_agent, "keeper").await.expect("run");

        for _ in 0..64 {
            if !hosted.run_notifies().contains_key(&abandoned.run_id) {
                break;
            }
            let _ = hosted.step(keeper).await;
            tokio::time::sleep(Duration::from_millis(2)).await;
        }

        assert!(!hosted.run_notifies().contains_key(&abandoned.run_id));
    }
}
