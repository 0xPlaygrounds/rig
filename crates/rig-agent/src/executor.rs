//! Automatic tool execution over the session drivers.
//!
//! [`ToolExecutor`] holds an ordered set of executable
//! [`PortableDynamicTool`] records and answers a
//! [`SessionEvent::ToolCallsReady`](crate::session::SessionEvent::ToolCallsReady)
//! / [`AgentStreamItem::ToolCallsReady`](crate::stream::AgentStreamItem::ToolCallsReady)
//! batch the way the classic runner's tool loop did:
//!
//! - bounded concurrency: at most `min(tool_concurrency, batch size)` calls
//!   in flight (default 1, i.e. sequential in call order);
//! - preresolved results (invalid tool-call recovery) pass through verbatim
//!   without executing anything;
//! - an unknown tool name produces the registry's `not_found` result shape,
//!   delivered to the model like any other tool failure;
//! - the batch commits atomically in call order: results are collected and
//!   returned only once every call settles, ordered by call index regardless
//!   of completion order;
//! - when multiple calls fail, the **lowest call-index** failure is the one
//!   reported on [`ToolBatchOutput::first_error`] — failures stay
//!   model-visible as failed tool results (classic semantics: tool errors are
//!   delivered to the model, never a driver-level error), and the report lets
//!   the host observe the deterministic first failure;
//! - each executed call runs inside a per-tool `execute_tool` tracing span
//!   carrying the classic `gen_ai.tool.*` fields.
//!
//! Drive a whole run with [`AgentSession::run_with_tools`](crate::session::AgentSession::run_with_tools),
//! or a stream with [`AgentStream::next_item_with_tools`](crate::stream::AgentStream::next_item_with_tools).

use futures::{StreamExt, stream};
use indexmap::IndexMap;
use tracing_futures::Instrument;

use crate::agent::prepare::ToolCatalog;
use crate::agent::run::PendingToolCall;
use crate::agent::telemetry::new_execute_tool_span;
use crate::tool::router_support::{not_found, shape_result};
use crate::tool::{PortableDynamicTool, ToolExecutionError, ToolResult};
use rig_core::message::{ToolCall, UserContent};

/// The deterministic first failure of a batch: the failed call's index in the
/// batch, its tool name, and the execution error (also delivered to the model
/// as a failed tool result).
#[derive(Debug, Clone)]
pub struct ToolBatchError {
    /// Zero-based index of the failed call within the batch.
    pub index: usize,
    /// The tool name the model called.
    pub tool_name: String,
    /// The execution failure.
    pub error: ToolExecutionError,
}

/// The settled outcome of one tool-call batch.
///
/// `results` always carries one shaped tool-result content per pending call,
/// in call order — the exact answer for
/// [`AgentSession::provide_tool_results`](crate::session::AgentSession::provide_tool_results).
/// Execution failures are embedded in those results (model-visible, classic
/// semantics); `first_error` additionally reports the lowest call-index
/// failure for host observation.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolBatchOutput {
    /// One tool-result content per call, ordered by call index.
    pub results: Vec<UserContent>,
    /// The lowest call-index failure, when any call failed.
    pub first_error: Option<ToolBatchError>,
    /// The **structured** execution result for each call whose body ran,
    /// keyed by the provider tool-call id: the classification
    /// (`success`/`failed`/`skipped`/`denied`, error kind, HTTP status) that
    /// the model-visible `results` content flattens away. A driver surfacing a
    /// post-execution decision point hands this to its host so a policy can
    /// inspect the outcome exactly as the classic tool-result hook could.
    pub raw_results: Vec<(String, ToolResult)>,
    /// The `execute_tool` span opened for each call whose body ran, keyed by
    /// the provider tool-call id. A driver that lets a hook rewrite the
    /// model-visible presentation records the rewritten value onto the
    /// matching span, exactly as the classic driver did.
    pub call_spans: Vec<(String, tracing::Span)>,
    /// Id of the last `execute_tool` span this batch opened, for drivers that
    /// keep a linear `follows_from` span chain (the blocking driver does; see
    /// [`ToolExecutor::execute_batch_following`]). `None` when no tool body
    /// ran or tracing is disabled.
    pub last_span_id: Option<u64>,
}

/// An ordered registry of executable tools that answers `ToolCallsReady`
/// batches for the session drivers. See the [module docs](self).
///
/// Typed tools are erased into concrete [`PortableDynamicTool`] records at
/// registration time; the executor itself remains concrete.
///
/// ```
/// # use rig_agent::executor::ToolExecutor;
/// # use rig_core::tool::PortableDynamicTool;
/// # fn dynamic(name: &str) -> PortableDynamicTool {
/// #     PortableDynamicTool::new(name.to_string(), "d", serde_json::json!({}), |args| async move {
/// #         Ok(rig_core::tool::ToolOutput::json(args))
/// #     })
/// # }
/// let executor = ToolExecutor::new()
///     .register(dynamic("first"))
///     .register_all([dynamic("second"), dynamic("third")])
///     .tool_concurrency(4);
///
/// assert_eq!(executor.len(), 3);
/// assert!(executor.get("second").is_some());
/// ```
#[derive(Debug, Clone)]
pub struct ToolExecutor {
    tools: IndexMap<String, PortableDynamicTool>,
    concurrency: usize,
    /// Whether tool arguments and results are recorded on `execute_tool`
    /// spans (the classic `record_content_telemetry` setting; off by
    /// default, since arguments and results are sensitive).
    record_telemetry_content: bool,
    /// Whether recording the result is left to the driver. A driver that
    /// surfaces a post-execution decision point records the **post-hook**
    /// presentation itself, so recording the raw result here would leak what
    /// a redaction hook removed (the classic driver recorded only once, after
    /// the hook).
    defer_result_telemetry: bool,
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolExecutor {
    /// Create an empty executor (sequential execution by default).
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            concurrency: 1,
            record_telemetry_content: false,
            defer_result_telemetry: false,
        }
    }

    /// Build an executor from an ordered set of tools.
    pub fn from_tools(tools: impl IntoIterator<Item = PortableDynamicTool>) -> Self {
        Self::new().register_all(tools)
    }

    /// Register a typed tool, erasing it to a concrete record immediately.
    ///
    /// Re-registering a name replaces it in place, preserving order.
    pub fn tool<T>(self, tool: T) -> Self
    where
        T: rig_core::tool::PortableTool + 'static,
    {
        self.register(PortableDynamicTool::from_portable(tool))
    }

    /// Register a tool, replacing any earlier registration of the same name
    /// in place (registration order is preserved).
    pub fn register(mut self, tool: PortableDynamicTool) -> Self {
        self.tools.insert(tool.name().to_owned(), tool);
        self
    }

    /// Register tool records in iteration order.
    ///
    /// Each record follows [`Self::register`] semantics, including replacing a
    /// duplicate name in place without changing its registration position.
    pub fn register_all(self, tools: impl IntoIterator<Item = PortableDynamicTool>) -> Self {
        tools.into_iter().fold(self, Self::register)
    }

    /// Bound how many tools of one batch run concurrently. Values are clamped
    /// to at least 1; the effective bound per batch is
    /// `min(tool_concurrency, batch size)`. Mirrors the classic
    /// `tool_concurrency` builder setting (default 1, sequential).
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Opt in or out of recording tool arguments and results on
    /// `execute_tool` spans (`gen_ai.tool.call.arguments` /
    /// `gen_ai.tool.call.result`). Off by default; structural fields (name,
    /// call id, outcome, error type) are always recorded. Mirrors the classic
    /// `record_content_telemetry` setting.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.record_telemetry_content = enabled;
        self
    }

    /// Leave `gen_ai.tool.call.result` to the caller. Set this when the
    /// driver surfaces a post-execution decision point and records the
    /// post-hook presentation itself, so a redaction rewrite is not preceded
    /// by the raw value on the same span (classic parity).
    pub fn defer_result_telemetry(mut self, deferred: bool) -> Self {
        self.defer_result_telemetry = deferred;
        self
    }

    /// A copy of this executor keeping only the named tools (per-turn
    /// `active_tools` narrowing), preserving registration order and the
    /// concurrency bound. Records are `Arc`-backed, so this is cheap.
    pub fn narrowed(&self, keep: &std::collections::BTreeSet<String>) -> Self {
        let mut narrowed = self.clone();
        narrowed.tools.retain(|name, _| keep.contains(name));
        narrowed
    }

    /// The registered tool with this name, if any.
    pub fn get(&self, name: &str) -> Option<&PortableDynamicTool> {
        self.tools.get(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The advertised [`ToolCatalog`] for this executor's tools: every
    /// registered tool's definition, all executable. Pair it with a session
    /// via [`AgentSession::with_tools`](crate::session::AgentSession::with_tools)
    /// (or the stream equivalent) so the driver advertises exactly what this
    /// executor can run.
    pub fn catalog(&self) -> ToolCatalog {
        ToolCatalog::new(
            self.tools
                .values()
                .map(PortableDynamicTool::definition)
                .collect(),
        )
    }

    /// Execute one batch of pending calls to a settled, call-ordered
    /// [`ToolBatchOutput`]. Infallible by design: like the classic loop,
    /// every failure is shaped into a model-visible failed tool result
    /// rather than surfaced as a driver-level error. See the
    /// [module docs](self) for the batch semantics.
    // A single lifetime for both borrows: an `async fn` whose future captures
    // two independent elided lifetimes is invariant over each of them, which
    // makes the future fail higher-ranked `Send` bounds in callers such as the
    // Discord integration's `async_trait` handlers.
    pub async fn execute_batch<'a>(&'a self, calls: &'a [PendingToolCall]) -> ToolBatchOutput {
        self.execute_batch_following(calls, None).await
    }

    /// [`ToolExecutor::execute_batch`], chaining this batch's `execute_tool`
    /// spans onto `follows_from` (and onto each other, in call order) so a
    /// driver can keep the classic blocking surface's linear causal trace
    /// `chat -> execute_tool -> chat`. The id of the last span opened comes
    /// back as [`ToolBatchOutput::last_span_id`] so the caller can continue
    /// the chain.
    pub async fn execute_batch_following<'a>(
        &'a self,
        calls: &'a [PendingToolCall],
        follows_from: Option<u64>,
    ) -> ToolBatchOutput {
        let concurrency = self.concurrency.min(calls.len()).max(1);

        // One slot per call, prefilled with preresolved passthroughs so
        // executed results land back in call order regardless of completion
        // order (mirrors `dispatch_pending` / classic `drive_tool_calls`).
        let mut slots: Vec<Option<UserContent>> = calls
            .iter()
            .map(|call| call.preresolved_result.clone())
            .collect();
        let mut first_error: Option<ToolBatchError> = None;

        // Own the per-call work up front: each future then borrows only
        // `&self`, so the batch future carries a single lifetime and stays
        // `Send` for every caller lifetime (a future capturing two
        // independent borrows is invariant over each, which breaks
        // higher-ranked `Send` bounds such as the Discord integration's
        // `async_trait` handlers).
        //
        // Spans are opened here, in call order, so the `follows_from` chain is
        // deterministic at any concurrency (mirroring the classic driver,
        // which also chained at prepare time rather than completion time).
        let mut chain = follows_from;
        let mut last_span_id = None;
        let jobs: Vec<(usize, ToolCall, tracing::Span)> = calls
            .iter()
            .enumerate()
            .filter(|(_, call)| call.preresolved_result.is_none())
            .map(|(index, call)| {
                let span = new_execute_tool_span();
                let span = match chain {
                    Some(id) => span
                        .follows_from(tracing::span::Id::from_u64(id))
                        .to_owned(),
                    None => span,
                };
                if let Some(id) = span.id() {
                    chain = Some(id.into_u64());
                    last_span_id = chain;
                }
                (index, call.tool_call.clone(), span)
            })
            .collect();
        let call_spans: Vec<(String, tracing::Span)> = jobs
            .iter()
            .map(|(_, tool_call, span)| (tool_call.id.clone(), span.clone()))
            .collect();
        let mut executed = stream::iter(jobs)
            .map(|(index, tool_call, span)| async move {
                let (content, result) = self.execute_one(&tool_call, span).await;
                (index, tool_call, content, result)
            })
            .buffer_unordered(concurrency);
        let mut raw_results: Vec<(String, ToolResult)> = Vec::new();
        while let Some((index, tool_call, content, result)) = executed.next().await {
            if let Some(slot) = slots.get_mut(index) {
                *slot = Some(content);
            }
            let error = result.error().cloned();
            raw_results.push((tool_call.id.clone(), result));
            // Deterministic error selection: the lowest call-index failure
            // wins regardless of completion order.
            if let Some(error) = error
                && first_error.as_ref().is_none_or(|first| index < first.index)
            {
                first_error = Some(ToolBatchError {
                    index,
                    tool_name: tool_call.function.name.clone(),
                    error,
                });
            }
        }

        // Every slot is filled by construction (preresolved upfront, executed
        // by index above); the fallback keeps this lint-clean and fail-safe.
        let results = calls
            .iter()
            .zip(slots)
            .map(|(pending, slot)| {
                slot.unwrap_or_else(|| {
                    shape_result(
                        &pending.tool_call,
                        &ToolResult::failed(ToolExecutionError::other(
                            "tool result unavailable".to_string(),
                        )),
                    )
                })
            })
            .collect();
        // Structured results in call order, matching `results`.
        raw_results.sort_by_key(|(id, _)| {
            calls
                .iter()
                .position(|call| call.tool_call.id == *id)
                .unwrap_or(usize::MAX)
        });
        ToolBatchOutput {
            results,
            first_error,
            raw_results,
            call_spans,
            last_span_id,
        }
    }

    /// Execute a single call inside its `execute_tool` span, returning the
    /// shaped model-visible content plus the execution error, if any.
    async fn execute_one<'a>(
        &'a self,
        tool_call: &'a ToolCall,
        span: tracing::Span,
    ) -> (UserContent, ToolResult) {
        let tool_name = &tool_call.function.name;
        let record_content = self.record_telemetry_content;

        async {
            let tool_span = tracing::Span::current();
            tool_span.record("gen_ai.tool.name", tool_name.as_str());
            tool_span.record("gen_ai.tool.call.id", &tool_call.id);
            if record_content {
                tool_span.record(
                    "gen_ai.tool.call.arguments",
                    crate::json_utils::serialize_json_value(&tool_call.function.arguments),
                );
            }

            let result = match self.tools.get(tool_name) {
                Some(tool) => match tool.execute(tool_call.function.arguments.clone()).await {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                // Unknown tool: the registry's not-found shape, delivered to
                // the model like any other failure.
                None => not_found(tool_name),
            };

            tool_span.record("gen_ai.tool.call.outcome", result.status_name());
            if let Some(error) = result.error() {
                tool_span.record("gen_ai.tool.error.type", error.kind().as_str());
            }
            let content = shape_result(tool_call, &result);
            if record_content && !self.defer_result_telemetry {
                // The model-visible presentation, matching what the classic
                // driver recorded. A driver that lets a hook rewrite the
                // presentation records it post-hook instead (see
                // `defer_result_telemetry`).
                tool_span.record("gen_ai.tool.call.result", result.output().render());
            }
            (content, result)
        }
        .instrument(span)
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::agent::AgentConfig;
    use crate::provider::{MockScript, ProviderConfig, Runtime};
    use crate::session::AgentSession;
    use crate::tool::{ToolErrorKind, ToolOutput};
    use rig_core::OneOrMany;
    use rig_core::completion::{CompletionResponse, FinishReason};
    use rig_core::message::{AssistantContent, ToolFunction};

    fn call(id: &str, name: &str, args: serde_json::Value) -> PendingToolCall {
        PendingToolCall::new(ToolCall::new(
            id.to_owned(),
            ToolFunction {
                name: name.to_owned(),
                arguments: args,
            },
        ))
    }

    fn echo_tool(name: &str, reply: &'static str, delay_ms: u64) -> PortableDynamicTool {
        PortableDynamicTool::new(
            name,
            "echo",
            serde_json::json!({"type": "object"}),
            move |_args| async move {
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Ok(ToolOutput::json(serde_json::json!({"reply": reply})))
            },
        )
    }

    /// A typed tool, to prove `.tool()` erases it to the same record shape
    /// `PortableDynamicTool::from_portable` produces.
    struct TypedEcho;

    impl rig_core::tool::PortableTool for TypedEcho {
        const NAME: &'static str = "typed_echo";
        type Args = serde_json::Value;
        type Output = serde_json::Value;
        type Error = crate::tool::ToolExecutionError;

        fn description(&self) -> String {
            "typed echo".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args)
        }
    }

    #[test]
    fn direct_configuration_preserves_registration_and_settings() {
        let executor = ToolExecutor::new()
            .register(echo_tool("first", "a", 0))
            .register_all([echo_tool("second", "b", 0), echo_tool("third", "c", 0)])
            .tool_concurrency(4)
            .record_content_telemetry(true)
            .defer_result_telemetry(true);

        let names: Vec<String> = executor
            .catalog()
            .definitions
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(names, ["first", "second", "third"]);
        assert_eq!(executor.concurrency, 4);
        assert!(executor.record_telemetry_content);
        assert!(executor.defer_result_telemetry);
    }

    #[test]
    fn typed_tool_registration_matches_manual_erasure() {
        let built = ToolExecutor::new().tool(TypedEcho);
        let explicit = ToolExecutor::new().register(PortableDynamicTool::from_portable(TypedEcho));

        assert_eq!(built.len(), explicit.len());
        assert!(built.get("typed_echo").is_some());
        assert_eq!(
            built.get("typed_echo").expect("registered").name(),
            explicit.get("typed_echo").expect("registered").name()
        );
    }

    #[test]
    fn re_registering_a_name_replaces_in_place_preserving_order() {
        let executor = ToolExecutor::new()
            .register(echo_tool("first", "a", 0))
            .register(echo_tool("second", "b", 0))
            .register(echo_tool("first", "replaced", 0));

        assert_eq!(executor.len(), 2);
        let names: Vec<String> = executor
            .catalog()
            .definitions
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn default_matches_new() {
        let default = ToolExecutor::default();
        let new = ToolExecutor::new();

        assert_eq!(default.len(), new.len());
        assert!(default.is_empty());
        assert!(new.is_empty());
        assert_eq!(default.concurrency, new.concurrency);
        assert_eq!(default.concurrency, 1);
        assert_eq!(
            default.record_telemetry_content,
            new.record_telemetry_content
        );
        assert_eq!(default.defer_result_telemetry, new.defer_result_telemetry);
    }

    fn failing_tool(name: &str, message: &'static str, delay_ms: u64) -> PortableDynamicTool {
        PortableDynamicTool::new(
            name,
            "fail",
            serde_json::json!({"type": "object"}),
            move |_args| async move {
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Err(ToolExecutionError::other(message.to_string()))
            },
        )
    }

    fn result_id(content: &UserContent) -> &str {
        match content {
            UserContent::ToolResult(result) => &result.id,
            other => panic!("expected a tool result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn concurrent_batch_preserves_call_order() {
        // The first call finishes last (it sleeps); with concurrency 2 the
        // second completes first, but results still land in call order.
        let executor = ToolExecutor::new()
            .register(echo_tool("slow", "slow-reply", 30))
            .register(echo_tool("fast", "fast-reply", 0))
            .tool_concurrency(2);

        let calls = vec![
            call("call_0", "slow", serde_json::json!({})),
            call("call_1", "fast", serde_json::json!({})),
        ];
        let batch = executor.execute_batch(&calls).await;

        assert!(batch.first_error.is_none());
        assert_eq!(batch.results.len(), 2);
        let ids: Vec<&str> = batch.results.iter().map(result_id).collect();
        assert_eq!(ids, vec!["call_0", "call_1"]);
    }

    #[tokio::test]
    async fn lowest_call_index_error_is_reported() {
        // Both calls fail; the lower-index one finishes LAST, but its error
        // is still the one reported.
        let executor = ToolExecutor::new()
            .register(failing_tool("boom_slow", "first failure", 30))
            .register(failing_tool("boom_fast", "second failure", 0))
            .tool_concurrency(2);

        let calls = vec![
            call("call_0", "boom_slow", serde_json::json!({})),
            call("call_1", "boom_fast", serde_json::json!({})),
        ];
        let batch = executor.execute_batch(&calls).await;

        // Both failures stay model-visible as results, in call order.
        assert_eq!(batch.results.len(), 2);
        let first_error = batch.first_error.expect("both calls failed");
        assert_eq!(first_error.index, 0);
        assert_eq!(first_error.tool_name, "boom_slow");
        assert_eq!(first_error.error.message(), "first failure");
    }

    #[tokio::test]
    async fn preresolved_results_pass_through_without_executing() {
        let executions = Arc::new(AtomicUsize::new(0));
        let counter = executions.clone();
        let counted = PortableDynamicTool::new(
            "counted",
            "counts executions",
            serde_json::json!({"type": "object"}),
            move |_args| {
                counter.fetch_add(1, Ordering::SeqCst);
                async move { Ok(ToolOutput::json(serde_json::json!(null))) }
            },
        );
        let executor = ToolExecutor::new().register(counted);

        let preresolved = UserContent::tool_result(
            "call_0",
            OneOrMany::one(rig_core::message::ToolResultContent::text("preresolved")),
        );
        let calls = vec![
            call("call_0", "counted", serde_json::json!({}))
                .with_preresolved_result(preresolved.clone()),
        ];
        let batch = executor.execute_batch(&calls).await;

        assert!(batch.first_error.is_none());
        assert_eq!(batch.results.len(), 1);
        // Verbatim passthrough, nothing executed.
        assert_eq!(
            serde_json::to_value(&batch.results).expect("serializable"),
            serde_json::to_value(vec![preresolved]).expect("serializable"),
        );
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unknown_tool_produces_the_registry_not_found_shape() {
        let executor = ToolExecutor::new();
        let calls = vec![call("call_0", "nope", serde_json::json!({}))];
        let batch = executor.execute_batch(&calls).await;

        assert_eq!(batch.results.len(), 1);
        assert_eq!(
            result_id(batch.results.first().expect("one result")),
            "call_0"
        );
        // Identical shaping to the registry miss path.
        let expected = shape_result(
            &calls.first().expect("one call").tool_call,
            &not_found("nope"),
        );
        assert_eq!(
            serde_json::to_value(&batch.results).expect("serializable"),
            serde_json::to_value(vec![expected]).expect("serializable"),
        );
        let first_error = batch.first_error.expect("miss is reported");
        assert_eq!(first_error.error.kind(), ToolErrorKind::NotFound);
    }

    #[tokio::test]
    async fn run_with_tools_round_trips_a_tool_turn() {
        let mut usage = crate::completion::Usage::new();
        usage.total_tokens = 3;
        let tool_turn = CompletionResponse::new(
            OneOrMany::one(AssistantContent::tool_call(
                "call_1",
                "add",
                serde_json::json!({"a": 1, "b": 2}),
            )),
            usage,
            "mock",
        )
        .with_finish_reason(FinishReason::ToolCalls);
        let final_turn = CompletionResponse::new(
            OneOrMany::one(AssistantContent::text("the sum is 3")),
            usage,
            "mock",
        )
        .with_message_id("msg_1")
        .with_finish_reason(FinishReason::Stop);
        let script = MockScript::from_responses(vec![tool_turn, final_turn]);

        let adder = PortableDynamicTool::new(
            "add",
            "Adds two numbers",
            serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]
            }),
            |args| async move {
                let a = args
                    .get("a")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0);
                let b = args
                    .get("b")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0);
                Ok(ToolOutput::json(serde_json::json!(a + b)))
            },
        );
        let executor = ToolExecutor::new().register(adder);

        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        let mut session = AgentSession::new(
            config,
            ProviderConfig::Mock(script),
            Arc::new(Runtime::new()),
            "what is 1 + 2?",
        )
        .with_tools(executor.catalog());

        let done = session
            .run_with_tools(&executor)
            .await
            .expect("run should succeed");
        assert_eq!(done.output, "the sum is 3");
        assert_eq!(session.run_state().completion_calls().len(), 2);
    }
}
