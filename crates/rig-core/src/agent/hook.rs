//! Event-specific hooks for observing and steering an agent run.
//!
//! Each lifecycle method returns only the action type that event can honor, so
//! invalid event/action combinations are rejected by the Rust type system.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Serialize;

use crate::{
    OneOrMany,
    completion::{CompletionModel, Document, Usage},
    json_utils,
    message::{AssistantContent, Message, ToolChoice},
    tool::{ToolContext, ToolExecution},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(String);
impl RunId {
    pub(crate) fn generate() -> Self {
        Self(crate::id::generate())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Default)]
pub struct Scratchpad {
    inner: Arc<std::sync::Mutex<ToolContext>>,
}
impl Scratchpad {
    fn lock(&self) -> std::sync::MutexGuard<'_, ToolContext> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn insert<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(
        &self,
        value: T,
    ) -> Option<T> {
        self.lock().insert(value)
    }
    pub fn get<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(&self) -> Option<T> {
        self.lock().get::<T>().cloned()
    }
    pub fn contains<T: WasmCompatSend + WasmCompatSync + 'static>(&self) -> bool {
        self.lock().get::<T>().is_some()
    }
    pub fn remove<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(&self) -> Option<T> {
        self.lock().remove::<T>()
    }
    pub fn update<T, R>(&self, f: impl FnOnce(&mut T) -> R) -> R
    where
        T: Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    {
        let mut guard = self.lock();
        let mut value = guard.remove::<T>().unwrap_or_default();
        let result = f(&mut value);
        guard.insert(value);
        result
    }
}
impl std::fmt::Debug for Scratchpad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratchpad").finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct HookContext {
    run_id: RunId,
    turn: AtomicUsize,
    is_streaming: bool,
    agent_name: Option<String>,
    scratchpad: Scratchpad,
}
impl HookContext {
    pub(crate) fn new(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            run_id: RunId::generate(),
            turn: AtomicUsize::new(0),
            is_streaming,
            agent_name,
            scratchpad: Scratchpad::default(),
        }
    }
    pub(crate) fn set_turn(&self, turn: usize) {
        self.turn.store(turn, Ordering::Relaxed);
    }
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }
    pub fn turn(&self) -> usize {
        self.turn.load(Ordering::Relaxed)
    }
    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }
    pub fn scratchpad(&self) -> &Scratchpad {
        &self.scratchpad
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InvalidToolCallContext {
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub internal_call_id: Option<String>,
    pub args: Option<String>,
    pub available_tools: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub tool_choice: Option<ToolChoice>,
    pub chat_history: Vec<Message>,
    pub is_streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidToolCallHookAction {
    Fail,
    Retry { feedback: String },
    Repair { tool_name: String },
    Skip { reason: String },
}
impl InvalidToolCallHookAction {
    pub fn fail() -> Self {
        Self::Fail
    }
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RequestPatch {
    pub preamble: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tool_choice: Option<ToolChoice>,
    pub active_tools: Option<Vec<String>>,
    pub additional_params: Option<serde_json::Value>,
    pub extra_context: Vec<Document>,
    pub history: Option<Vec<Message>>,
}
fn merge_last_wins<T>(earlier: Option<T>, later: Option<T>, field: &str) -> Option<T> {
    match (earlier, later) {
        (Some(_), Some(value)) => {
            tracing::warn!(patch_field = field, "later hook value wins");
            Some(value)
        }
        (earlier, later) => later.or(earlier),
    }
}
impl RequestPatch {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn preamble(mut self, value: impl Into<String>) -> Self {
        self.preamble = Some(value.into());
        self
    }
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature = Some(value);
        self
    }
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = Some(value);
        self
    }
    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.tool_choice = Some(value);
        self
    }
    pub fn active_tools<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.active_tools = Some(values.into_iter().map(Into::into).collect());
        self
    }
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.additional_params = Some(value);
        self
    }
    pub fn extra_context<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Document>,
    {
        self.extra_context.extend(values);
        self
    }
    pub fn context(mut self, value: Document) -> Self {
        self.extra_context.push(value);
        self
    }
    pub fn history<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.history = Some(values.into_iter().collect());
        self
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.preamble.is_none()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.tool_choice.is_none()
            && self.active_tools.is_none()
            && self.additional_params.is_none()
            && self.extra_context.is_empty()
            && self.history.is_none()
    }
    pub(crate) fn merge(mut self, later: Self) -> Self {
        self.extra_context.extend(later.extra_context);
        self.additional_params = match (self.additional_params.take(), later.additional_params) {
            (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
                Some(json_utils::merge(base, patch))
            }
            (base, patch) => patch.or(base),
        };
        self.preamble = merge_last_wins(self.preamble, later.preamble, "preamble");
        self.temperature = merge_last_wins(self.temperature, later.temperature, "temperature");
        self.max_tokens = merge_last_wins(self.max_tokens, later.max_tokens, "max_tokens");
        self.tool_choice = merge_last_wins(self.tool_choice, later.tool_choice, "tool_choice");
        self.history = merge_last_wins(self.history, later.history, "history");
        self.active_tools = match (self.active_tools.take(), later.active_tools) {
            (Some(a), Some(b)) => {
                let b: std::collections::BTreeSet<_> = b.iter().collect();
                Some(a.into_iter().filter(|v| b.contains(v)).collect())
            }
            (a, b) => a.or(b),
        };
        self
    }
}

/// Inputs visible before one model completion request is built.
#[derive(Clone, Copy)]
pub struct CompletionCall<'a> {
    pub prompt: &'a Message,
    pub history: &'a [Message],
    pub turn: usize,
}
/// Non-streaming provider response accepted for the current turn.
pub struct CompletionResponse<'a, M: CompletionModel> {
    pub prompt: &'a Message,
    pub response: &'a crate::completion::CompletionResponse<M::Response>,
}
/// Normalized accepted model turn emitted on streaming and non-streaming runs.
#[derive(Clone, Copy)]
pub struct ModelTurnFinished<'a> {
    pub turn: usize,
    pub content: &'a OneOrMany<AssistantContent>,
    pub usage: Usage,
}
/// Tool invocation visible immediately before framework dispatch.
#[derive(Clone, Copy)]
pub struct ToolCall<'a> {
    pub tool_name: &'a str,
    pub tool_call_id: Option<&'a str>,
    pub internal_call_id: &'a str,
    pub args: &'a str,
}
/// Raw tool execution visible before its presentation is sent to the model.
///
/// Rewriting [`result`](Self::result) changes presentation only; the canonical
/// [`execution`](Self::execution) remains available to later hooks unchanged.
#[derive(Clone, Copy)]
pub struct ToolResult<'a> {
    pub tool_name: &'a str,
    pub tool_call_id: Option<&'a str>,
    pub internal_call_id: &'a str,
    pub args: &'a str,
    pub result: &'a str,
    pub execution: &'a ToolExecution,
}
/// Incremental text emitted by a streaming model response.
#[derive(Clone, Copy)]
pub struct TextDelta<'a> {
    pub delta: &'a str,
    pub aggregated: &'a str,
}
/// Incremental tool-call data emitted by a streaming model response.
#[derive(Clone, Copy)]
pub struct ToolCallDelta<'a> {
    pub tool_call_id: &'a str,
    pub internal_call_id: &'a str,
    pub tool_name: Option<&'a str>,
    pub delta: &'a str,
}
/// Provider-specific streaming response accepted at the end of a text turn.
pub struct StreamResponseFinish<'a, M: CompletionModel> {
    pub prompt: &'a Message,
    pub response: &'a M::StreamingResponse,
}

impl<M: CompletionModel> Copy for CompletionResponse<'_, M> {}
impl<M: CompletionModel> Clone for CompletionResponse<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M: CompletionModel> Copy for StreamResponseFinish<'_, M> {}
impl<M: CompletionModel> Clone for StreamResponseFinish<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Lifecycle event kinds used by [`AgentHook::observes`] to avoid unnecessary
/// high-frequency streaming callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HookEvent {
    CompletionCall,
    CompletionResponse,
    ModelTurnFinished,
    InvalidToolCall,
    ToolCall,
    ToolResult,
    TextDelta,
    ToolCallDelta,
    StreamResponseFinish,
}

/// Actions available to observe-only lifecycle methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserveAction {
    Continue,
    Stop { reason: String },
}
impl ObserveAction {
    pub fn cont() -> Self {
        Self::Continue
    }
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}
/// Actions available before a completion request is sent.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionCallAction {
    Continue,
    Patch(RequestPatch),
    Stop { reason: String },
}
impl CompletionCallAction {
    pub fn cont() -> Self {
        Self::Continue
    }
    pub fn patch(patch: RequestPatch) -> Self {
        Self::Patch(patch)
    }
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}
/// Actions available before a tool body executes.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallAction {
    Run,
    Rewrite(serde_json::Value),
    Skip { reason: String },
    Stop { reason: String },
}
impl ToolCallAction {
    pub fn run() -> Self {
        Self::Run
    }
    pub fn rewrite(args: impl Into<serde_json::Value>) -> Self {
        Self::Rewrite(args.into())
    }
    pub fn try_rewrite<T: Serialize>(args: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::Rewrite(serde_json::to_value(args)?))
    }
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}
/// Actions available after execution and before model-visible presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolResultAction {
    Keep,
    Rewrite(String),
    Stop { reason: String },
}
impl ToolResultAction {
    pub fn keep() -> Self {
        Self::Keep
    }
    pub fn rewrite(result: impl Into<String>) -> Self {
        Self::Rewrite(result.into())
    }
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}
/// Recovery actions available for a model-emitted invalid tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidToolCallAction {
    /// Decline to resolve this event and allow the next registered hook to act.
    Continue,
    Fail,
    Retry {
        feedback: String,
    },
    Repair {
        tool_name: String,
    },
    Skip {
        reason: String,
    },
    Stop {
        reason: String,
    },
}
impl InvalidToolCallAction {
    pub fn cont() -> Self {
        Self::Continue
    }
    pub fn fail() -> Self {
        Self::Fail
    }
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}

/// Per-run lifecycle observer and steerer with event-specific methods.
///
/// Each method returns only actions valid for that lifecycle event. Default
/// implementations continue without changing the run.
pub trait AgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    fn on_completion_call(
        &self,
        _: &HookContext,
        _: CompletionCall<'_>,
    ) -> impl Future<Output = CompletionCallAction> + WasmCompatSend {
        async { CompletionCallAction::Continue }
    }
    fn on_completion_response(
        &self,
        _: &HookContext,
        _: CompletionResponse<'_, M>,
    ) -> impl Future<Output = ObserveAction> + WasmCompatSend {
        async { ObserveAction::Continue }
    }
    fn on_model_turn_finished(
        &self,
        _: &HookContext,
        _: ModelTurnFinished<'_>,
    ) -> impl Future<Output = ObserveAction> + WasmCompatSend {
        async { ObserveAction::Continue }
    }
    fn on_invalid_tool_call(
        &self,
        _: &HookContext,
        _: &InvalidToolCallContext,
    ) -> impl Future<Output = InvalidToolCallAction> + WasmCompatSend {
        async { InvalidToolCallAction::Continue }
    }
    fn on_tool_call(
        &self,
        _: &HookContext,
        _: ToolCall<'_>,
    ) -> impl Future<Output = ToolCallAction> + WasmCompatSend {
        async { ToolCallAction::Run }
    }
    #[doc(hidden)]
    fn resolve_tool_call<'a>(
        &'a self,
        context: &'a HookContext,
        event: ToolCall<'a>,
    ) -> impl Future<Output = (ToolCallAction, Option<serde_json::Value>)> + WasmCompatSend + 'a
    where
        M: 'a,
    {
        async move { (self.on_tool_call(context, event).await, None) }
    }
    fn on_tool_result(
        &self,
        _: &HookContext,
        _: ToolResult<'_>,
    ) -> impl Future<Output = ToolResultAction> + WasmCompatSend {
        async { ToolResultAction::Keep }
    }
    fn on_text_delta(
        &self,
        _: &HookContext,
        _: TextDelta<'_>,
    ) -> impl Future<Output = ObserveAction> + WasmCompatSend {
        async { ObserveAction::Continue }
    }
    fn on_tool_call_delta(
        &self,
        _: &HookContext,
        _: ToolCallDelta<'_>,
    ) -> impl Future<Output = ObserveAction> + WasmCompatSend {
        async { ObserveAction::Continue }
    }
    fn on_stream_response_finish(
        &self,
        _: &HookContext,
        _: StreamResponseFinish<'_, M>,
    ) -> impl Future<Output = ObserveAction> + WasmCompatSend {
        async { ObserveAction::Continue }
    }
    fn observes(&self, _: HookEvent) -> bool {
        true
    }
}
impl<M: CompletionModel> AgentHook<M> for () {
    fn observes(&self, _: HookEvent) -> bool {
        false
    }
}

trait DynAgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    fn completion_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionCall<'a>,
    ) -> WasmBoxedFuture<'a, CompletionCallAction>
    where
        M: 'a;
    fn completion_response<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionResponse<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn model_turn<'a>(
        &'a self,
        c: &'a HookContext,
        e: ModelTurnFinished<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn invalid<'a>(
        &'a self,
        c: &'a HookContext,
        e: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, InvalidToolCallAction>
    where
        M: 'a;
    fn resolve_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCall<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>
    where
        M: 'a;
    fn tool_result<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolResult<'a>,
    ) -> WasmBoxedFuture<'a, ToolResultAction>
    where
        M: 'a;
    fn text_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: TextDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn tool_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCallDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn stream_finish<'a>(
        &'a self,
        c: &'a HookContext,
        e: StreamResponseFinish<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn observes(&self, e: HookEvent) -> bool;
}
impl<M: CompletionModel, H: AgentHook<M>> DynAgentHook<M> for H {
    fn completion_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionCall<'a>,
    ) -> WasmBoxedFuture<'a, CompletionCallAction>
    where
        M: 'a,
    {
        Box::pin(self.on_completion_call(c, e))
    }
    fn completion_response<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionResponse<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_completion_response(c, e))
    }
    fn model_turn<'a>(
        &'a self,
        c: &'a HookContext,
        e: ModelTurnFinished<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_model_turn_finished(c, e))
    }
    fn invalid<'a>(
        &'a self,
        c: &'a HookContext,
        e: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, InvalidToolCallAction>
    where
        M: 'a,
    {
        Box::pin(self.on_invalid_tool_call(c, e))
    }
    fn resolve_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCall<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>
    where
        M: 'a,
    {
        Box::pin(AgentHook::resolve_tool_call(self, c, e))
    }
    fn tool_result<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolResult<'a>,
    ) -> WasmBoxedFuture<'a, ToolResultAction>
    where
        M: 'a,
    {
        Box::pin(self.on_tool_result(c, e))
    }
    fn text_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: TextDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_text_delta(c, e))
    }
    fn tool_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCallDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_tool_call_delta(c, e))
    }
    fn stream_finish<'a>(
        &'a self,
        c: &'a HookContext,
        e: StreamResponseFinish<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_stream_response_finish(c, e))
    }
    fn observes(&self, e: HookEvent) -> bool {
        AgentHook::observes(self, e)
    }
}

/// Ordered heterogeneous collection of [`AgentHook`] implementations.
///
/// Completion patches merge, tool rewrites chain, and terminal actions
/// short-circuit in registration order.
pub struct HookStack<M: CompletionModel> {
    hooks: Vec<Arc<dyn DynAgentHook<M>>>,
}
impl<M: CompletionModel> Clone for HookStack<M> {
    fn clone(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
        }
    }
}
impl<M: CompletionModel> Default for HookStack<M> {
    fn default() -> Self {
        Self { hooks: Vec::new() }
    }
}
impl<M: CompletionModel> HookStack<M> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with<H: AgentHook<M> + 'static>(hook: H) -> Self {
        let mut s = Self::new();
        s.push(hook);
        s
    }
    pub fn push<H: AgentHook<M> + 'static>(&mut self, hook: H) {
        self.hooks.push(Arc::new(hook))
    }
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}
impl<M: CompletionModel> std::fmt::Debug for HookStack<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookStack")
            .field("len", &self.len())
            .finish()
    }
}

impl<M: CompletionModel> AgentHook<M> for HookStack<M> {
    async fn on_completion_call(
        &self,
        c: &HookContext,
        e: CompletionCall<'_>,
    ) -> CompletionCallAction {
        let mut merged: Option<RequestPatch> = None;
        for h in &self.hooks {
            match h.completion_call(c, e).await {
                CompletionCallAction::Continue => {}
                CompletionCallAction::Patch(p) => {
                    merged = Some(match merged {
                        Some(a) => a.merge(p),
                        None => p,
                    })
                }
                stop => return stop,
            }
        }
        match merged {
            Some(p) if !p.is_empty() => CompletionCallAction::Patch(p),
            _ => CompletionCallAction::Continue,
        }
    }
    async fn on_completion_response(
        &self,
        c: &HookContext,
        e: CompletionResponse<'_, M>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.completion_response(c, e).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_model_turn_finished(
        &self,
        c: &HookContext,
        e: ModelTurnFinished<'_>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.model_turn(c, e).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_invalid_tool_call(
        &self,
        c: &HookContext,
        e: &InvalidToolCallContext,
    ) -> InvalidToolCallAction {
        for h in &self.hooks {
            let a = h.invalid(c, e).await;
            if !matches!(a, InvalidToolCallAction::Continue) {
                return a;
            }
        }
        InvalidToolCallAction::Continue
    }
    async fn on_tool_call(&self, c: &HookContext, e: ToolCall<'_>) -> ToolCallAction {
        AgentHook::resolve_tool_call(self, c, e).await.0
    }
    async fn resolve_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCall<'a>,
    ) -> (ToolCallAction, Option<serde_json::Value>)
    where
        M: 'a,
    {
        let mut value: Option<serde_json::Value> = None;
        for hook in &self.hooks {
            let rendered = value.as_ref().map(json_utils::value_to_json_string);
            let current = ToolCall {
                args: rendered.as_deref().unwrap_or(e.args),
                ..e
            };
            let (action, nested_rewrite) = hook.resolve_tool_call(c, current).await;
            if let Some(rewrite) = nested_rewrite {
                value = Some(rewrite);
            }
            match action {
                ToolCallAction::Run => {}
                ToolCallAction::Rewrite(rewrite) => value = Some(rewrite),
                terminal => return (terminal, value),
            }
        }
        match value {
            Some(rewrite) => (ToolCallAction::Rewrite(rewrite), None),
            None => (ToolCallAction::Run, None),
        }
    }
    async fn on_tool_result(&self, c: &HookContext, e: ToolResult<'_>) -> ToolResultAction {
        let mut value = None;
        for h in &self.hooks {
            let current = ToolResult {
                result: value.as_deref().unwrap_or(e.result),
                ..e
            };
            match h.tool_result(c, current).await {
                ToolResultAction::Keep => {}
                ToolResultAction::Rewrite(v) => value = Some(v),
                other => return other,
            }
        }
        value
            .map(ToolResultAction::Rewrite)
            .unwrap_or(ToolResultAction::Keep)
    }
    async fn on_text_delta(&self, c: &HookContext, e: TextDelta<'_>) -> ObserveAction {
        for h in &self.hooks {
            let a = h.text_delta(c, e).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_tool_call_delta(&self, c: &HookContext, e: ToolCallDelta<'_>) -> ObserveAction {
        for h in &self.hooks {
            let a = h.tool_delta(c, e).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_stream_response_finish(
        &self,
        c: &HookContext,
        e: StreamResponseFinish<'_, M>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.stream_finish(c, e).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    fn observes(&self, e: HookEvent) -> bool {
        self.hooks.iter().any(|h| h.observes(e))
    }
}
#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, OnceLock};

    use super::{
        AgentHook, CompletionCall, CompletionCallAction, HookContext, HookEvent, HookStack,
        InvalidToolCallAction, InvalidToolCallContext, ObserveAction, RequestPatch, Scratchpad,
        TextDelta, ToolCall, ToolCallAction,
    };
    use crate::test_utils::MockCompletionModel;
    use serde_json::{Value, json};

    type M = MockCompletionModel;

    fn ctx() -> HookContext {
        HookContext::new(false, Some("test-agent".to_string()))
    }

    fn tool_call_event(args: &'static str) -> ToolCall<'static> {
        ToolCall {
            tool_name: "add",
            tool_call_id: Some("tc1"),
            internal_call_id: "ic1",
            args,
        }
    }

    fn completion_call_event() -> CompletionCall<'static> {
        static PROMPT: OnceLock<crate::message::Message> = OnceLock::new();
        CompletionCall {
            prompt: PROMPT.get_or_init(|| crate::message::Message::user("hi")),
            history: &[],
            turn: 1,
        }
    }

    struct ToolRecorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }

    impl AgentHook<M> for ToolRecorder {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                ToolCallAction::stop("stop")
            } else {
                ToolCallAction::run()
            }
        }
    }

    struct TextRecorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }

    impl AgentHook<M> for TextRecorder {
        async fn on_text_delta(&self, _: &HookContext, _: TextDelta<'_>) -> ObserveAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                ObserveAction::stop("stop")
            } else {
                ObserveAction::cont()
            }
        }
    }

    struct ObservesOnly(HookEvent);
    impl AgentHook<M> for ObservesOnly {
        fn observes(&self, kind: HookEvent) -> bool {
            kind == self.0
        }
    }

    struct Patcher {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        patch: RequestPatch,
    }
    impl AgentHook<M> for Patcher {
        async fn on_completion_call(
            &self,
            _: &HookContext,
            _: CompletionCall<'_>,
        ) -> CompletionCallAction {
            self.log.lock().expect("log").push(self.label);
            CompletionCallAction::patch(self.patch.clone())
        }
    }

    struct CompletionStopper {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
    }
    impl AgentHook<M> for CompletionStopper {
        async fn on_completion_call(
            &self,
            _: &HookContext,
            _: CompletionCall<'_>,
        ) -> CompletionCallAction {
            self.log.lock().expect("log").push(self.label);
            CompletionCallAction::stop("stop")
        }
    }

    #[tokio::test]
    async fn runs_hooks_in_registration_order_and_consults_all_on_continue() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(ToolRecorder {
            label: 1,
            log: log.clone(),
            stop: false,
        });
        stack.push(ToolRecorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });
        assert!(matches!(
            stack.on_tool_call(&ctx(), tool_call_event("{}")).await,
            ToolCallAction::Run
        ));
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn first_stop_short_circuits_on_chained_tool_call() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(ToolRecorder {
            label: 1,
            log: log.clone(),
            stop: true,
        });
        stack.push(ToolRecorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });
        assert!(matches!(
            stack.on_tool_call(&ctx(), tool_call_event("{}")).await,
            ToolCallAction::Stop { .. }
        ));
        assert_eq!(*log.lock().expect("log"), vec![1]);
    }

    #[tokio::test]
    async fn first_stop_short_circuits_on_observe_only_events() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(TextRecorder {
            label: 1,
            log: log.clone(),
            stop: true,
        });
        stack.push(TextRecorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });
        assert!(matches!(
            stack
                .on_text_delta(
                    &ctx(),
                    TextDelta {
                        delta: "hi",
                        aggregated: "hi",
                    },
                )
                .await,
            ObserveAction::Stop { .. }
        ));
        assert_eq!(*log.lock().expect("log"), vec![1]);
    }

    #[tokio::test]
    async fn completion_call_patches_accumulate_and_consult_every_hook() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.1),
        });
        stack.push(Patcher {
            label: 2,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(256),
        });
        let action = stack
            .on_completion_call(&ctx(), completion_call_event())
            .await;
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
        match action {
            CompletionCallAction::Patch(patch) => {
                assert_eq!(patch.temperature, Some(0.1));
                assert_eq!(patch.max_tokens, Some(256));
            }
            other => panic!("expected merged patch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn completion_call_stop_short_circuits_and_discards_patch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.1),
        });
        stack.push(CompletionStopper {
            label: 2,
            log: log.clone(),
        });
        stack.push(Patcher {
            label: 3,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(256),
        });
        assert!(matches!(
            stack
                .on_completion_call(&ctx(), completion_call_event())
                .await,
            CompletionCallAction::Stop { .. }
        ));
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn nested_stack_composes_patches_without_inner_short_circuit() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut inner = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.2),
        });
        inner.push(Patcher {
            label: 2,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(128),
        });
        let mut outer = HookStack::<M>::with(inner);
        outer.push(Patcher {
            label: 3,
            log: log.clone(),
            patch: RequestPatch::new().preamble("outer"),
        });
        match outer
            .on_completion_call(&ctx(), completion_call_event())
            .await
        {
            CompletionCallAction::Patch(patch) => {
                assert_eq!(patch.temperature, Some(0.2));
                assert_eq!(patch.max_tokens, Some(128));
                assert_eq!(patch.preamble.as_deref(), Some("outer"));
            }
            other => panic!("expected merged patch, got {other:?}"),
        }
        assert_eq!(*log.lock().expect("log"), vec![1, 2, 3]);
    }

    #[test]
    fn stack_observes_is_the_or_of_its_members() {
        let mut stack = HookStack::<M>::with(ObservesOnly(HookEvent::ToolCall));
        stack.push(ObservesOnly(HookEvent::ToolResult));
        assert!(stack.observes(HookEvent::ToolCall));
        assert!(stack.observes(HookEvent::ToolResult));
        assert!(!stack.observes(HookEvent::TextDelta));
    }

    #[tokio::test]
    async fn empty_stack_uses_event_defaults_and_observes_nothing() {
        let stack = HookStack::<M>::new();
        assert!(stack.is_empty());
        assert!(!stack.observes(HookEvent::ToolCall));
        assert!(matches!(
            stack.on_tool_call(&ctx(), tool_call_event("{}")).await,
            ToolCallAction::Run
        ));
        assert!(matches!(
            stack
                .on_invalid_tool_call(
                    &ctx(),
                    &InvalidToolCallContext {
                        tool_name: "missing".into(),
                        tool_call_id: None,
                        internal_call_id: None,
                        args: None,
                        available_tools: vec![],
                        allowed_tools: vec![],
                        tool_choice: None,
                        chat_history: vec![],
                        is_streaming: false,
                    },
                )
                .await,
            InvalidToolCallAction::Continue
        ));
    }

    #[test]
    fn unit_hook_observes_no_event_kind() {
        let all = [
            HookEvent::CompletionCall,
            HookEvent::CompletionResponse,
            HookEvent::ModelTurnFinished,
            HookEvent::InvalidToolCall,
            HookEvent::ToolCall,
            HookEvent::ToolResult,
            HookEvent::TextDelta,
            HookEvent::ToolCallDelta,
            HookEvent::StreamResponseFinish,
        ];
        let stack = HookStack::<M>::with(());
        for kind in all {
            assert!(!<() as AgentHook<M>>::observes(&(), kind));
            assert!(!stack.observes(kind));
        }
    }

    #[test]
    fn request_patch_merge_rules_are_preserved() {
        let doc = |id: &str| crate::completion::Document {
            id: id.to_string(),
            text: String::new(),
            additional_props: Default::default(),
        };
        let merged = RequestPatch::new()
            .temperature(0.1)
            .active_tools(["search", "add", "sub"])
            .additional_params(json!({"x": 1, "y": 2}))
            .context(doc("a"))
            .merge(
                RequestPatch::new()
                    .temperature(0.9)
                    .active_tools(["add", "sub", "mul"])
                    .additional_params(json!({"y": 3, "z": 4}))
                    .context(doc("b")),
            );
        assert_eq!(merged.temperature, Some(0.9));
        assert_eq!(merged.active_tools, Some(vec!["add".into(), "sub".into()]));
        assert_eq!(
            merged.additional_params,
            Some(json!({"x": 1, "y": 3, "z": 4}))
        );
        assert_eq!(
            merged
                .extra_context
                .iter()
                .map(|doc| doc.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(
            RequestPatch::new()
                .active_tools(["search"])
                .merge(RequestPatch::new().active_tools(["add"]))
                .active_tools,
            Some(vec![])
        );
    }

    #[test]
    fn scratchpad_is_typed_mutable_and_shared_across_clones() {
        #[derive(Clone, Default, Debug, PartialEq)]
        struct Count(u32);
        let pad = Scratchpad::default();
        let clone = pad.clone();
        pad.update(|count: &mut Count| count.0 += 2);
        assert_eq!(clone.get::<Count>(), Some(Count(2)));
        assert_eq!(pad.remove::<Count>(), Some(Count(2)));
    }

    #[test]
    fn hook_context_reports_identity_and_turn() {
        let context = HookContext::new(true, Some("agent".to_string()));
        assert!(context.is_streaming());
        assert_eq!(context.agent_name(), Some("agent"));
        context.set_turn(3);
        assert_eq!(context.turn(), 3);
        assert!(!context.run_id().as_str().is_empty());
    }

    struct RewriteHook(Value);
    impl AgentHook<M> for RewriteHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::rewrite(self.0.clone())
        }
    }

    struct SkipHook;
    impl AgentHook<M> for SkipHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::skip("denied")
        }
    }

    struct StopHook;
    impl AgentHook<M> for StopHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::stop("stop")
        }
    }

    #[derive(Clone, Default)]
    struct ArgsSpy(Arc<Mutex<Vec<String>>>);
    impl AgentHook<M> for ArgsSpy {
        async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            self.0.lock().expect("spy").push(event.args.to_string());
            ToolCallAction::run()
        }
    }

    async fn resolve(stack: &HookStack<M>) -> (ToolCallAction, Option<Value>) {
        stack.resolve_tool_call(&ctx(), tool_call_event("{}")).await
    }

    #[tokio::test]
    async fn nested_rewrite_then_skip_preserves_rewrite() {
        let mut inner = HookStack::<M>::new();
        inner.push(RewriteHook(json!({"x": 41})));
        inner.push(SkipHook);
        let outer = HookStack::<M>::with(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Skip { .. }));
        assert_eq!(salvaged, Some(json!({"x": 41})));
    }

    #[tokio::test]
    async fn nested_rewrite_then_stop_preserves_rewrite() {
        let mut inner = HookStack::<M>::new();
        inner.push(RewriteHook(json!({"x": 7})));
        inner.push(StopHook);
        let outer = HookStack::<M>::with(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Stop { .. }));
        assert_eq!(salvaged, Some(json!({"x": 7})));
    }

    #[tokio::test]
    async fn outer_rewrite_is_threaded_into_nested_stack_before_skip() {
        let spy = ArgsSpy::default();
        let mut inner = HookStack::<M>::new();
        inner.push(spy.clone());
        inner.push(SkipHook);
        let mut outer = HookStack::<M>::new();
        outer.push(RewriteHook(json!({"x": 1, "y": 2})));
        outer.push(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Skip { .. }));
        assert_eq!(salvaged, Some(json!({"x": 1, "y": 2})));
        assert_eq!(
            spy.0.lock().expect("spy").as_slice(),
            [serde_json::to_string(&json!({"x": 1, "y": 2})).unwrap()]
        );
    }

    #[tokio::test]
    async fn deeply_nested_rewrite_then_skip_preserves_rewrite() {
        let mut deepest = HookStack::<M>::new();
        deepest.push(RewriteHook(json!({"deep": true})));
        deepest.push(SkipHook);
        let middle = HookStack::<M>::with(deepest);
        let outer = HookStack::<M>::with(middle);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Skip { .. }));
        assert_eq!(salvaged, Some(json!({"deep": true})));
    }

    #[tokio::test]
    async fn nested_proceeding_rewrite_surfaces_as_rewrite_action() {
        let inner = HookStack::<M>::with(RewriteHook(json!({"x": 5})));
        let outer = HookStack::<M>::with(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert_eq!(action, ToolCallAction::Rewrite(json!({"x": 5})));
        assert_eq!(salvaged, None);
    }

    struct InvalidRecorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        action: InvalidToolCallAction,
    }
    impl AgentHook<M> for InvalidRecorder {
        async fn on_invalid_tool_call(
            &self,
            _: &HookContext,
            _: &InvalidToolCallContext,
        ) -> InvalidToolCallAction {
            self.log.lock().expect("log").push(self.label);
            self.action.clone()
        }
    }

    #[tokio::test]
    async fn invalid_hooks_can_decline_and_first_decision_wins() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(InvalidRecorder {
            label: 1,
            log: log.clone(),
            action: InvalidToolCallAction::Continue,
        });
        stack.push(InvalidRecorder {
            label: 2,
            log: log.clone(),
            action: InvalidToolCallAction::Fail,
        });
        stack.push(InvalidRecorder {
            label: 3,
            log: log.clone(),
            action: InvalidToolCallAction::retry("unused"),
        });
        let action = stack
            .on_invalid_tool_call(
                &ctx(),
                &InvalidToolCallContext {
                    tool_name: "missing".into(),
                    tool_call_id: None,
                    internal_call_id: None,
                    args: None,
                    available_tools: vec![],
                    allowed_tools: vec![],
                    tool_choice: None,
                    chat_history: vec![],
                    is_streaming: false,
                },
            )
            .await;
        assert_eq!(action, InvalidToolCallAction::Fail);
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }
}
