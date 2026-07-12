//! Runtime control and context for interactive agent runs.
//!
//! A [`crate::runtime::RunControlHandle`] is cloneable host-side state for one run. The shared
//! agent driver observes it only at explicit boundaries and inserts the paired
//! [`crate::runtime::RunContext`] into every tool call's extensions.

use crate::{agent::RunId, completion::Message};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Cooperative cancellation shared by a run and all descendant work.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Creates an uncancelled token.
    pub fn new() -> Self {
        Self::default()
    }
    /// Requests cancellation. Repeated requests are harmless.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Returns whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Provider-neutral terminal reason for a completion or agent run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TerminalReason {
    /// A final answer completed normally.
    #[default]
    Completed,
    /// Cancellation was requested.
    Cancelled,
    /// The run deadline elapsed.
    DeadlineExceeded,
    /// The configured model-turn budget was exhausted.
    MaxTurns,
    /// The provider's output-token limit was reached.
    MaxTokens,
    /// A hook or host policy terminated execution.
    PolicyTerminated,
    /// A provider safety policy filtered the response.
    ContentFiltered,
    /// The model stopped to invoke tools.
    ToolCalls,
    /// Execution failed.
    Failed,
    /// Provider-specific reason not understood by Rig.
    Other(String),
}

/// Observable state of an interactive run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunStatus {
    /// The run can make progress.
    Running,
    /// Cancellation has been requested while work settles.
    Cancelling,
    /// A checkpoint was requested and will be honored at the next safe boundary.
    Checkpointing,
    /// The run is paused at a safe [`AgentRun`](crate::agent::AgentRun) boundary.
    Checkpointed,
    /// The run has settled.
    Terminal(TerminalReason),
}

#[derive(Debug)]
struct ControlState {
    status: Mutex<RunStatus>,
    steering: Mutex<VecDeque<Message>>,
    follow_ups: Mutex<VecDeque<Message>>,
    checkpoint_state: Mutex<Option<String>>,
    cancellation: CancellationToken,
}

/// Error returned when a run-control command cannot be accepted.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RunControlError {
    /// The run already settled.
    #[error("run is already terminal")]
    Terminal,
    /// Resume was requested before the driver reached a checkpoint.
    #[error("run has not reached a checkpoint")]
    NotCheckpointed,
    /// Follow-ups were requested before terminal settlement.
    #[error("run has not settled")]
    NotTerminal,
}

/// Cloneable control handle for exactly one agent run.
#[derive(Clone, Debug)]
pub struct RunControlHandle {
    run_id: RunId,
    state: Arc<ControlState>,
}

pub(crate) struct RunLease(RunControlHandle);

impl Drop for RunLease {
    fn drop(&mut self) {
        self.0.finish(TerminalReason::Cancelled);
    }
}

impl RunControlHandle {
    /// Creates independent settlement ownership for a child while sharing the
    /// supplied context's cancellation and deadline.
    pub(crate) fn for_child_context(context: &RunContext) -> Self {
        Self {
            run_id: context.run_id.clone(),
            state: Arc::new(ControlState {
                status: Mutex::new(RunStatus::Running),
                steering: Mutex::new(VecDeque::new()),
                follow_ups: Mutex::new(VecDeque::new()),
                checkpoint_state: Mutex::new(None),
                cancellation: context.cancellation.clone(),
            }),
        }
    }

    /// Creates a handle and the context inherited by work in the run.
    pub fn new(conversation_id: Option<String>, deadline: Option<Instant>) -> (Self, RunContext) {
        let run_id = RunId::generate();
        let cancellation = CancellationToken::new();
        let state = Arc::new(ControlState {
            status: Mutex::new(RunStatus::Running),
            steering: Mutex::new(VecDeque::new()),
            follow_ups: Mutex::new(VecDeque::new()),
            checkpoint_state: Mutex::new(None),
            cancellation: cancellation.clone(),
        });
        let handle = Self {
            run_id: run_id.clone(),
            state,
        };
        let context = RunContext {
            run_id,
            conversation_id,
            cancellation,
            deadline,
            ancestry: Arc::from([]),
            current_call_id: None,
        };
        (handle, context)
    }

    /// Returns the stable run identifier.
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// Returns a consistent status snapshot.
    pub fn status(&self) -> RunStatus {
        self.state
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Requests cancellation and returns false when the run was already terminal.
    pub fn cancel(&self) -> bool {
        let mut status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*status, RunStatus::Terminal(_)) {
            return false;
        }
        self.state.cancellation.cancel();
        *status = RunStatus::Cancelling;
        true
    }

    /// Queues a user message for the next pre-model safe boundary.
    pub fn steer(&self, message: impl Into<Message>) -> Result<(), RunControlError> {
        self.enqueue(message.into(), false)
    }

    /// Queues a user message for consumption only after this run settles.
    pub fn follow_up(&self, message: impl Into<Message>) -> Result<(), RunControlError> {
        self.enqueue(message.into(), true)
    }

    fn enqueue(&self, message: Message, follow_up: bool) -> Result<(), RunControlError> {
        let status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*status, RunStatus::Terminal(_)) {
            return Err(RunControlError::Terminal);
        }
        let queue = if follow_up {
            &self.state.follow_ups
        } else {
            &self.state.steering
        };
        queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(message);
        drop(status);
        Ok(())
    }

    /// Requests a pause at the next safe state-machine boundary.
    pub fn checkpoint(&self) -> Result<(), RunControlError> {
        let mut status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*status, RunStatus::Terminal(_)) {
            return Err(RunControlError::Terminal);
        }
        if matches!(*status, RunStatus::Running) {
            *status = RunStatus::Checkpointing;
        }
        Ok(())
    }

    /// Resumes execution after the driver reports [`RunStatus::Checkpointed`].
    pub fn resume(&self) -> Result<(), RunControlError> {
        let mut status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if *status != RunStatus::Checkpointed {
            return Err(RunControlError::NotCheckpointed);
        }
        *status = RunStatus::Running;
        Ok(())
    }

    /// Drains follow-ups in FIFO order after terminal settlement.
    pub fn drain_follow_ups(&self) -> Result<Vec<Message>, RunControlError> {
        let status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(*status, RunStatus::Terminal(_)) {
            return Err(RunControlError::NotTerminal);
        }
        let messages = self
            .state
            .follow_ups
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
            .collect();
        drop(status);
        Ok(messages)
    }

    pub(crate) fn drain_steering(&self) -> Vec<Message> {
        self.state
            .steering
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
            .collect()
    }

    pub(crate) fn defer_as_follow_ups(&self, messages: Vec<Message>) {
        self.state
            .follow_ups
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(messages);
    }

    /// Returns the serialized [`AgentRun`](crate::agent::AgentRun) captured at
    /// the current or most recent safe checkpoint.
    ///
    /// It contains conversation data and follows `AgentRun`'s same-version
    /// resume contract, so hosts must store it as sensitive data.
    pub fn checkpoint_state(&self) -> Option<String> {
        self.state
            .checkpoint_state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub(crate) fn checkpoint_requested(&self) -> bool {
        self.status() == RunStatus::Checkpointing
    }

    pub(crate) fn reach_checkpoint(&self, serialized_run: String) {
        let mut status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if *status == RunStatus::Checkpointing {
            *self
                .state
                .checkpoint_state
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(serialized_run);
            *status = RunStatus::Checkpointed;
        }
    }

    pub(crate) fn checkpoint_blocked(&self) -> bool {
        self.status() == RunStatus::Checkpointed
    }

    pub(crate) fn lease(&self) -> RunLease {
        RunLease(self.clone())
    }

    pub(crate) fn finish(&self, mut reason: TerminalReason) -> bool {
        let mut status = self.state.status.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*status, RunStatus::Terminal(_)) {
            return false;
        }
        if reason == TerminalReason::Completed && self.state.cancellation.is_cancelled() {
            reason = TerminalReason::Cancelled;
        }
        *status = RunStatus::Terminal(reason);
        true
    }
}

/// Immutable context automatically inherited by tools and nested agents.
#[derive(Clone, Debug)]
pub struct RunContext {
    run_id: RunId,
    conversation_id: Option<String>,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
    ancestry: Arc<[String]>,
    current_call_id: Option<String>,
}

impl RunContext {
    /// Returns the root run identifier.
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }
    /// Returns the conversation identifier, when configured.
    pub fn conversation_id(&self) -> Option<&str> {
        self.conversation_id.as_deref()
    }
    /// Returns the inherited cancellation token.
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }
    /// Returns the absolute deadline.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    /// Returns the remaining duration, saturating at zero.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
    /// Returns whether cancellation or deadline requires work to stop.
    pub fn should_stop(&self) -> bool {
        self.cancellation.is_cancelled() || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
    /// Returns ancestor names, oldest first.
    pub fn ancestry(&self) -> &[String] {
        &self.ancestry
    }
    /// Returns the current framework-generated call identifier.
    pub fn current_call_id(&self) -> Option<&str> {
        self.current_call_id.as_deref()
    }

    pub(crate) fn with_conversation(mut self, id: Option<String>) -> Self {
        self.conversation_id = id;
        self
    }

    pub(crate) fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    pub(crate) fn fresh_child(&self, name: String, call_id: String) -> Self {
        let mut ancestry = self.ancestry.to_vec();
        ancestry.push(format!("{}:{name}", self.run_id));
        Self {
            run_id: RunId::generate(),
            conversation_id: self.conversation_id.clone(),
            cancellation: self.cancellation.clone(),
            deadline: self.deadline,
            ancestry: ancestry.into(),
            current_call_id: Some(call_id),
        }
    }

    pub(crate) fn child(&self, name: String, call_id: String) -> Self {
        let mut ancestry = self.ancestry.to_vec();
        ancestry.push(name);
        Self {
            run_id: self.run_id.clone(),
            conversation_id: self.conversation_id.clone(),
            cancellation: self.cancellation.clone(),
            deadline: self.deadline,
            ancestry: ancestry.into(),
            current_call_id: Some(call_id),
        }
    }

    pub(crate) async fn stopped(&self) -> TerminalReason {
        loop {
            if self.cancellation.is_cancelled() {
                return TerminalReason::Cancelled;
            }
            if self.deadline.is_some_and(|d| Instant::now() >= d) {
                return TerminalReason::DeadlineExceeded;
            }
            let delay = self.remaining().map_or(Duration::from_millis(5), |left| {
                left.min(Duration::from_millis(5))
            });
            futures_timer::Delay::new(delay).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::{AgentBuilder, AgentHook, Flow, HookContext, StepEvent},
        completion::{
            CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Prompt,
            PromptError, ProviderToolDefinition,
        },
        streaming::{StreamingCompletionResponse, StreamingPrompt},
        test_utils::{MockAddTool, MockCompletionModel, MockResponse, MockStreamEvent, MockTurn},
        tool::{
            Tool, ToolReturn,
            server::{
                NestedToolPolicy, ScopedToolExecutor, ToolCatalogKind, ToolCatalogMetadata,
                ToolServer,
            },
        },
    };
    use futures::StreamExt;
    use std::{
        convert::Infallible,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    #[derive(Clone)]
    struct PendingModel {
        started: Arc<tokio::sync::Notify>,
    }

    impl CompletionModel for PendingModel {
        type Response = MockResponse;
        type StreamingResponse = MockResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self {
                started: Arc::new(tokio::sync::Notify::new()),
            }
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            self.started.notify_one();
            futures::future::pending().await
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            self.started.notify_one();
            futures::future::pending().await
        }
    }

    #[derive(Clone)]
    struct RegenerateArgsHook;

    impl AgentHook<MockCompletionModel> for RegenerateArgsHook {
        async fn on_event(
            &self,
            _context: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            if matches!(event, StepEvent::InvalidToolCall(_)) {
                Flow::regenerate_args("regenerate the call with valid arguments")
            } else {
                Flow::cont()
            }
        }
    }

    #[derive(Clone)]
    struct NestedHook(Arc<AtomicUsize>);

    impl AgentHook<MockCompletionModel> for NestedHook {
        async fn on_event(
            &self,
            _context: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::ToolCall {
                    tool_name: "add",
                    parent_internal_call_id: Some(_),
                    ..
                } => {
                    self.0.fetch_add(1, Ordering::SeqCst);
                    Flow::rewrite_args(serde_json::json!({"x": 40, "y": 2}))
                }
                StepEvent::ToolResult {
                    tool_name: "add",
                    parent_internal_call_id: Some(_),
                    ..
                } => {
                    self.0.fetch_add(1, Ordering::SeqCst);
                    Flow::rewrite_result("nested-rewritten")
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct NestedCaller(Arc<Mutex<Option<String>>>);

    impl Tool for NestedCaller {
        const NAME: &'static str = "nested_caller";
        type Error = std::io::Error;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "call add through scoped executor".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn output_schema(&self) -> Option<serde_json::Value> {
            Some(serde_json::json!({"type": "string"}))
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Err(std::io::Error::other("scoped executor missing"))
        }
        async fn call_with_extensions(
            &self,
            _args: Self::Args,
            extensions: &crate::tool::ToolCallExtensions,
        ) -> Result<Self::Output, Self::Error> {
            let executor = extensions
                .get::<ScopedToolExecutor>()
                .ok_or_else(|| std::io::Error::other("scoped executor missing"))?;
            let nested = executor.call("add", r#"{"x":2,"y":3}"#).await;
            assert!(nested.parent_internal_call_id.is_some());
            let output = nested.result.model_output().to_string();
            *self.0.lock().unwrap_or_else(|error| error.into_inner()) = Some(output.clone());
            Ok(output)
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ErrorTag(&'static str);

    #[derive(Clone)]
    struct SchedulingTool {
        active: Arc<AtomicUsize>,
        maximum: Arc<AtomicUsize>,
    }

    impl Tool for SchedulingTool {
        const NAME: &'static str = "scheduled";
        type Error = Infallible;
        type Args = serde_json::Value;
        type Output = String;
        fn description(&self) -> String {
            "scheduling probe".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.maximum.fetch_max(active, Ordering::SeqCst);
            futures_timer::Delay::new(Duration::from_millis(10)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok("ok".into())
        }
    }

    #[derive(Clone)]
    struct FinalTool;

    impl Tool for FinalTool {
        const NAME: &'static str = "final_tool";
        type Error = Infallible;
        type Args = serde_json::Value;
        type Output = String;
        fn description(&self) -> String {
            "final result".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("tool-final".into())
        }
    }

    #[derive(Clone)]
    struct ClassifiedErrorTool;

    impl Tool for ClassifiedErrorTool {
        const NAME: &'static str = "classified_error";
        type Error = std::io::Error;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "fails with metadata".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn classify_error(&self, error: &Self::Error) -> crate::tool::ToolFailure {
            crate::tool::ToolFailure::provider(error.to_string())
        }
        fn error_extensions(&self, _error: &Self::Error) -> crate::tool::ToolResultExtensions {
            let mut extensions = crate::tool::ToolResultExtensions::new();
            extensions.insert(ErrorTag("classified"));
            extensions
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Err(std::io::Error::other("boom"))
        }
    }

    #[derive(Clone)]
    struct RichTool;

    impl Tool for RichTool {
        const NAME: &'static str = "rich";
        type Error = Infallible;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "rich output".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("fallback".into())
        }
        async fn call_structured(
            &self,
            _args: Self::Args,
            _extensions: &crate::tool::ToolCallExtensions,
        ) -> Result<ToolReturn<Self::Output>, Self::Error> {
            Ok(ToolReturn::success("fallback".into()).with_content(
                crate::OneOrMany::many([
                    crate::message::ToolResultContent::text("first"),
                    crate::message::ToolResultContent::text("second"),
                ])
                .unwrap(),
            ))
        }
    }

    #[derive(Clone)]
    struct RetryTool(Arc<AtomicUsize>);

    impl Tool for RetryTool {
        const NAME: &'static str = "retry";
        type Error = std::io::Error;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "fails once".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn classify_error(&self, error: &Self::Error) -> crate::tool::ToolFailure {
            crate::tool::ToolFailure::network(error.to_string())
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(std::io::Error::other("transient"))
            } else {
                Ok("recovered".into())
            }
        }
    }

    #[derive(Clone)]
    struct PendingTool {
        started: Arc<tokio::sync::Notify>,
    }

    impl Tool for PendingTool {
        const NAME: &'static str = "pending";
        type Error = Infallible;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "wait forever".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.started.notify_one();
            futures::future::pending().await
        }
    }

    #[tokio::test]
    async fn steering_is_fifo_and_identical_on_unary_and_streaming() {
        let unary_model = MockCompletionModel::new([MockTurn::text("done")]);
        let unary_request = AgentBuilder::new(unary_model.clone())
            .build()
            .prompt("original")
            .extended_details();
        let unary_control = unary_request.control_handle();
        unary_control.steer("first steer").unwrap();
        unary_control.steer("second steer").unwrap();
        let unary = unary_request.await.unwrap();
        assert_eq!(unary.terminal_reason(), &TerminalReason::Completed);
        let unary_text: Vec<_> = unary_model.requests()[0]
            .chat_history
            .iter()
            .filter_map(Message::rag_text)
            .collect();
        assert!(unary_text.ends_with(&["first steer".to_string(), "second steer".to_string(),]));

        let stream_model =
            MockCompletionModel::from_stream_turns([[MockStreamEvent::text("done")]]);
        let stream_request = AgentBuilder::new(stream_model.clone())
            .build()
            .stream_prompt("original");
        let stream_control = stream_request.control_handle();
        stream_control.steer("first steer").unwrap();
        stream_control.steer("second steer").unwrap();
        let mut stream = stream_request.await;
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
        let stream_text: Vec<_> = stream_model.requests()[0]
            .chat_history
            .iter()
            .filter_map(Message::rag_text)
            .collect();
        assert!(stream_text.ends_with(&["first steer".to_string(), "second steer".to_string(),]));
        assert_eq!(
            stream_control.status(),
            RunStatus::Terminal(TerminalReason::Completed)
        );
    }

    #[tokio::test]
    async fn regenerated_arguments_share_repair_and_turn_budgets_on_both_surfaces() {
        let unary = AgentBuilder::new(MockCompletionModel::new([
            MockTurn::tool_call("bad", "missing", serde_json::json!({"x": 1})),
            MockTurn::tool_call("good", "add", serde_json::json!({"x": 2, "y": 3})),
            MockTurn::text("done"),
        ]))
        .tool(MockAddTool)
        .build()
        .prompt("start")
        .add_hook(RegenerateArgsHook)
        .max_tool_repairs(1)
        .max_turns(3)
        .await
        .unwrap();
        assert_eq!(unary, "done");

        let streaming = AgentBuilder::new(MockCompletionModel::from_stream_turns([
            [MockStreamEvent::tool_call(
                "bad",
                "missing",
                serde_json::json!({"x": 1}),
            )],
            [MockStreamEvent::tool_call(
                "good",
                "add",
                serde_json::json!({"x": 2, "y": 3}),
            )],
            [MockStreamEvent::text("done")],
        ]))
        .tool(MockAddTool)
        .build()
        .stream_prompt("start")
        .add_hook(RegenerateArgsHook)
        .max_tool_repairs(1)
        .max_turns(3);
        let mut stream = streaming.await;
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
    }

    #[tokio::test]
    async fn scoped_executor_inherits_context_and_dispatches_nested_tool() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("outer", "nested_caller", serde_json::json!({})),
            MockTurn::text("done"),
        ]);
        let observed = Arc::new(Mutex::new(None));
        let hook_events = Arc::new(AtomicUsize::new(0));
        let agent = AgentBuilder::new(model.clone())
            .nested_tool_policy(NestedToolPolicy {
                allowlist: Some(["add".to_string()].into_iter().collect()),
                ..Default::default()
            })
            .tool(MockAddTool)
            .tool(NestedCaller(observed.clone()))
            .build();
        agent
            .prompt("start")
            .add_hook(NestedHook(hook_events.clone()))
            .max_turns(2)
            .await
            .unwrap();
        assert_eq!(
            model.requests()[0]
                .tools
                .iter()
                .find(|definition| definition.name == "nested_caller")
                .unwrap()
                .output_schema,
            Some(serde_json::json!({"type": "string"}))
        );
        assert_eq!(
            observed
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_deref(),
            Some("nested-rewritten")
        );
        assert_eq!(hook_events.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn serial_and_parallel_safe_metadata_control_batch_scheduling() {
        async fn maximum_for(scheduling: crate::tool::server::ToolScheduling) -> usize {
            let active = Arc::new(AtomicUsize::new(0));
            let maximum = Arc::new(AtomicUsize::new(0));
            let server = ToolServer::new().run();
            server
                .add_tool_with_metadata(
                    SchedulingTool {
                        active,
                        maximum: maximum.clone(),
                    },
                    ToolCatalogMetadata {
                        scheduling,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let turn = MockTurn::from_contents([
                crate::message::AssistantContent::ToolCall(crate::message::ToolCall::new(
                    "one".to_string(),
                    crate::message::ToolFunction::new(
                        "scheduled".to_string(),
                        serde_json::json!({}),
                    ),
                )),
                crate::message::AssistantContent::ToolCall(crate::message::ToolCall::new(
                    "two".to_string(),
                    crate::message::ToolFunction::new(
                        "scheduled".to_string(),
                        serde_json::json!({}),
                    ),
                )),
            ])
            .unwrap();
            AgentBuilder::new(MockCompletionModel::new([turn, MockTurn::text("done")]))
                .tool_server_handle(server)
                .build()
                .prompt("start")
                .tool_concurrency(2)
                .max_turns(2)
                .await
                .unwrap();
            maximum.load(Ordering::SeqCst)
        }
        assert_eq!(
            maximum_for(crate::tool::server::ToolScheduling::Serial).await,
            1
        );
        assert_eq!(
            maximum_for(crate::tool::server::ToolScheduling::ParallelSafe).await,
            2
        );
    }

    #[tokio::test]
    async fn final_result_metadata_finishes_without_another_model_call() {
        let server = ToolServer::new().run();
        server
            .add_tool_with_metadata(
                FinalTool,
                ToolCatalogMetadata {
                    final_result: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "final",
            "final_tool",
            serde_json::json!({}),
        )]);
        let output = AgentBuilder::new(model.clone())
            .tool_server_handle(server)
            .build()
            .prompt("start")
            .max_turns(2)
            .await
            .unwrap();
        assert_eq!(output, "tool-final");
        assert_eq!(model.request_count(), 1);
    }

    #[tokio::test]
    async fn ordinary_tool_errors_attach_classification_extensions() {
        let server = ToolServer::new().tool(ClassifiedErrorTool).run();
        let result = server
            .call_tool_structured(
                "classified_error",
                "{}",
                &crate::tool::ToolCallExtensions::new(),
            )
            .await;
        assert!(
            result
                .outcome()
                .is_error_kind(crate::tool::ToolFailureKind::Provider)
        );
        assert_eq!(
            result.extensions().get::<ErrorTag>(),
            Some(&ErrorTag("classified"))
        );
    }

    #[tokio::test]
    async fn explicit_rich_tool_parts_reach_model_history_without_envelope_parsing() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("rich-call", "rich", serde_json::json!({})),
            MockTurn::text("done"),
        ]);
        AgentBuilder::new(model.clone())
            .tool(RichTool)
            .build()
            .prompt("start")
            .max_turns(2)
            .await
            .unwrap();
        let has_two_parts = model.requests()[1].chat_history.iter().any(|message| {
            if let Message::User { content } = message {
                content.iter().any(|item| match item {
                    crate::message::UserContent::ToolResult(result) => result.content.len() == 2,
                    _ => false,
                })
            } else {
                false
            }
        });
        assert!(has_two_parts);
    }

    #[tokio::test]
    async fn catalog_retry_budget_only_retries_classified_transient_failures() {
        let server = ToolServer::new().run();
        let calls = Arc::new(AtomicUsize::new(0));
        server
            .add_tool_with_metadata(
                RetryTool(calls.clone()),
                ToolCatalogMetadata {
                    max_retries: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let result = server
            .call_tool_structured("retry", "{}", &crate::tool::ToolCallExtensions::new())
            .await;
        assert!(result.outcome().is_success());
        assert_eq!(result.model_output(), "recovered");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn catalog_preserves_order_and_host_metadata_never_enters_schema() {
        let server = ToolServer::new().run();
        let mut metadata = ToolCatalogMetadata {
            source: Some("application".into()),
            ..Default::default()
        };
        metadata
            .host
            .insert("secret".into(), serde_json::json!(true));
        server
            .add_tool_with_metadata(MockAddTool, metadata)
            .await
            .unwrap();
        server
            .add_provider_tool(
                ProviderToolDefinition::new("web_search"),
                ToolCatalogMetadata::default(),
            )
            .await;
        let catalog = server.catalog().await;
        assert_eq!(catalog[0].kind, ToolCatalogKind::Native);
        assert_eq!(catalog[1].kind, ToolCatalogKind::ProviderHosted);
        let provider_schema =
            serde_json::to_value(catalog[0].definition.as_ref().unwrap()).unwrap();
        assert!(!provider_schema.to_string().contains("secret"));
        server.remove_tool("add").await.unwrap();
        assert_eq!(server.catalog().await.len(), 1);
    }

    #[tokio::test]
    async fn agent_provider_tools_are_advertised_but_not_executable() {
        let model = MockCompletionModel::new([MockTurn::text("done")]);
        let hosted = ProviderToolDefinition::new("web_search")
            .with_config("search_context_size", serde_json::json!("low"));
        let agent = AgentBuilder::new(model.clone())
            .provider_tool(hosted)
            .build();
        assert!(
            agent
                .tool_catalog()
                .await
                .iter()
                .any(|entry| entry.kind == ToolCatalogKind::ProviderHosted)
        );
        agent.prompt("search").await.unwrap();
        let request = &model.requests()[0];
        assert!(request.tools.is_empty());
        assert_eq!(
            request.additional_params.as_ref().unwrap()["tools"][0]["type"],
            "web_search"
        );
        let result = agent
            .tool_server_handle
            .call_tool_structured("web_search", "{}", &crate::tool::ToolCallExtensions::new())
            .await;
        assert!(
            result
                .outcome()
                .is_error_kind(crate::tool::ToolFailureKind::NotFound)
        );
    }

    #[tokio::test]
    async fn concurrent_steering_is_lossless_and_run_scoped() {
        let model = MockCompletionModel::new([MockTurn::text("done")]);
        let request = AgentBuilder::new(model.clone()).build().prompt("original");
        let control = request.control_handle();
        let sends = (0..16).map(|index| {
            let control = control.clone();
            tokio::spawn(async move { control.steer(format!("steer-{index}")).unwrap() })
        });
        for send in sends {
            send.await.unwrap();
        }
        request.await.unwrap();
        let texts: std::collections::HashSet<_> = model.requests()[0]
            .chat_history
            .iter()
            .filter_map(Message::rag_text)
            .collect();
        for index in 0..16 {
            assert!(texts.contains(&format!("steer-{index}")));
        }
    }

    #[tokio::test]
    async fn checkpoint_pauses_before_model_and_resume_continues() {
        let model = MockCompletionModel::new([MockTurn::text("done")]);
        let request = AgentBuilder::new(model.clone())
            .build()
            .prompt("hello")
            .extended_details();
        let control = request.control_handle();
        control.checkpoint().unwrap();
        let task = tokio::spawn(async move { request.await });
        for _ in 0..100 {
            if control.status() == RunStatus::Checkpointed {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(control.status(), RunStatus::Checkpointed);
        assert_eq!(model.request_count(), 0);
        let checkpoint = control.checkpoint_state().expect("safe state is captured");
        let restored: crate::agent::AgentRun = serde_json::from_str(&checkpoint).unwrap();
        assert_eq!(restored.turn(), 0);
        control.resume().unwrap();
        assert_eq!(task.await.unwrap().unwrap().output(), "done");
    }

    #[tokio::test]
    async fn deadline_cancels_pending_model_on_unary_and_streaming() {
        for streaming in [false, true] {
            if streaming {
                let model =
                    MockCompletionModel::from_stream_turns([[MockStreamEvent::text("never")]]);
                let request = AgentBuilder::new(model)
                    .build()
                    .stream_prompt("hello")
                    .deadline(Duration::ZERO);
                let control = request.control_handle();
                let mut stream = request.await;
                let err = stream.next().await.unwrap().unwrap_err();
                assert!(matches!(err, crate::agent::StreamingError::Prompt(_)));
                assert_eq!(
                    control.status(),
                    RunStatus::Terminal(TerminalReason::DeadlineExceeded)
                );
            } else {
                let request = AgentBuilder::new(MockCompletionModel::text("never"))
                    .run_deadline(Duration::ZERO)
                    .build()
                    .prompt("hello");
                let control = request.control_handle();
                let err = request.await.unwrap_err();
                assert!(matches!(err, PromptError::PromptCancelled { .. }));
                assert_eq!(
                    control.status(),
                    RunStatus::Terminal(TerminalReason::DeadlineExceeded)
                );
            }
        }
    }

    #[tokio::test]
    async fn cancellation_drops_pending_model_future_on_both_surfaces() {
        let started = Arc::new(tokio::sync::Notify::new());
        let request = AgentBuilder::new(PendingModel {
            started: started.clone(),
        })
        .build()
        .prompt("start");
        let control = request.control_handle();
        let task = tokio::spawn(async move { request.await });
        started.notified().await;
        control.cancel();
        assert!(matches!(
            task.await.unwrap(),
            Err(PromptError::PromptCancelled { .. })
        ));
        assert_eq!(
            control.status(),
            RunStatus::Terminal(TerminalReason::Cancelled)
        );

        let started = Arc::new(tokio::sync::Notify::new());
        let request = AgentBuilder::new(PendingModel {
            started: started.clone(),
        })
        .build()
        .stream_prompt("start");
        let control = request.control_handle();
        let task = tokio::spawn(async move {
            let mut stream = request.await;
            stream.next().await
        });
        started.notified().await;
        control.cancel();
        assert!(task.await.unwrap().unwrap().is_err());
        assert_eq!(
            control.status(),
            RunStatus::Terminal(TerminalReason::Cancelled)
        );
    }

    #[tokio::test]
    async fn cancellation_drops_pending_tool_future_on_both_surfaces() {
        let started = Arc::new(tokio::sync::Notify::new());
        let unary = AgentBuilder::new(MockCompletionModel::new([
            MockTurn::tool_call("call", "pending", serde_json::json!({})),
            MockTurn::text("unreachable"),
        ]))
        .tool(PendingTool {
            started: started.clone(),
        })
        .build()
        .prompt("start")
        .max_turns(2);
        let unary_control = unary.control_handle();
        let unary_task = tokio::spawn(async move { unary.await });
        started.notified().await;
        assert!(unary_control.cancel());
        assert!(matches!(
            unary_task.await.unwrap(),
            Err(PromptError::PromptCancelled { .. })
        ));
        assert_eq!(
            unary_control.status(),
            RunStatus::Terminal(TerminalReason::Cancelled)
        );

        let started = Arc::new(tokio::sync::Notify::new());
        let streaming = AgentBuilder::new(MockCompletionModel::from_stream_turns([
            [MockStreamEvent::tool_call(
                "call",
                "pending",
                serde_json::json!({}),
            )],
            [MockStreamEvent::text("unreachable")],
        ]))
        .tool(PendingTool {
            started: started.clone(),
        })
        .build()
        .stream_prompt("start")
        .max_turns(2);
        let stream_control = streaming.control_handle();
        let stream_task = tokio::spawn(async move {
            let mut stream = streaming.await;
            while let Some(item) = stream.next().await {
                item?;
            }
            Ok::<_, crate::agent::StreamingError>(())
        });
        started.notified().await;
        assert!(stream_control.cancel());
        assert!(stream_task.await.unwrap().is_err());
        assert_eq!(
            stream_control.status(),
            RunStatus::Terminal(TerminalReason::Cancelled)
        );
    }

    #[test]
    fn child_settlement_is_independent_while_cancellation_is_shared() {
        let (outer, context) = RunControlHandle::new(None, None);
        let child = RunControlHandle::for_child_context(&context);
        assert!(child.finish(TerminalReason::Completed));
        assert_eq!(outer.status(), RunStatus::Running);
        outer.cancel();
        assert!(context.cancellation().is_cancelled());
        assert_eq!(
            child.status(),
            RunStatus::Terminal(TerminalReason::Completed)
        );
    }

    #[test]
    fn follow_ups_are_hidden_until_terminal_settlement() {
        let (control, _) = RunControlHandle::new(None, None);
        control.follow_up("next").unwrap();
        assert_eq!(
            control.drain_follow_ups(),
            Err(RunControlError::NotTerminal)
        );
        control.finish(TerminalReason::Completed);
        assert_eq!(control.drain_follow_ups().unwrap().len(), 1);
    }
}
