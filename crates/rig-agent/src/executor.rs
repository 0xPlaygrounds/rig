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
use crate::agent::runner::new_execute_tool_span;
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
pub struct ToolBatchOutput {
    /// One tool-result content per call, ordered by call index.
    pub results: Vec<UserContent>,
    /// The lowest call-index failure, when any call failed.
    pub first_error: Option<ToolBatchError>,
}

/// An ordered registry of executable tools that answers `ToolCallsReady`
/// batches for the session drivers. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct ToolExecutor {
    tools: IndexMap<String, PortableDynamicTool>,
    concurrency: usize,
}

impl ToolExecutor {
    /// Create an empty executor (sequential execution by default).
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            concurrency: 1,
        }
    }

    /// Build an executor from an ordered set of tools.
    pub fn from_tools(tools: impl IntoIterator<Item = PortableDynamicTool>) -> Self {
        tools.into_iter().fold(Self::new(), Self::register)
    }

    /// Register a tool, replacing any earlier registration of the same name
    /// in place (registration order is preserved).
    pub fn register(mut self, tool: PortableDynamicTool) -> Self {
        self.tools.insert(tool.name().to_owned(), tool);
        self
    }

    /// Bound how many tools of one batch run concurrently. Values are clamped
    /// to at least 1; the effective bound per batch is
    /// `min(tool_concurrency, batch size)`. Mirrors the classic
    /// `tool_concurrency` builder setting (default 1, sequential).
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
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
    pub async fn execute_batch(&self, calls: &[PendingToolCall]) -> ToolBatchOutput {
        let concurrency = self.concurrency.min(calls.len()).max(1);

        // One slot per call, prefilled with preresolved passthroughs so
        // executed results land back in call order regardless of completion
        // order (mirrors `dispatch_pending` / classic `drive_tool_calls`).
        let mut slots: Vec<Option<UserContent>> = calls
            .iter()
            .map(|call| call.preresolved_result.clone())
            .collect();
        let mut first_error: Option<ToolBatchError> = None;

        let mut executed = stream::iter(
            calls
                .iter()
                .enumerate()
                .filter(|(_, call)| call.preresolved_result.is_none()),
        )
        .map(|(index, pending)| async move {
            let (content, error) = self.execute_one(&pending.tool_call).await;
            (index, pending, content, error)
        })
        .buffer_unordered(concurrency);
        while let Some((index, pending, content, error)) = executed.next().await {
            if let Some(slot) = slots.get_mut(index) {
                *slot = Some(content);
            }
            // Deterministic error selection: the lowest call-index failure
            // wins regardless of completion order.
            if let Some(error) = error
                && first_error.as_ref().is_none_or(|first| index < first.index)
            {
                first_error = Some(ToolBatchError {
                    index,
                    tool_name: pending.tool_call.function.name.clone(),
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
        ToolBatchOutput {
            results,
            first_error,
        }
    }

    /// Execute a single call inside its `execute_tool` span, returning the
    /// shaped model-visible content plus the execution error, if any.
    async fn execute_one(&self, tool_call: &ToolCall) -> (UserContent, Option<ToolExecutionError>) {
        let tool_name = &tool_call.function.name;
        let span = new_execute_tool_span();

        async {
            let tool_span = tracing::Span::current();
            tool_span.record("gen_ai.tool.name", tool_name.as_str());
            tool_span.record("gen_ai.tool.call.id", &tool_call.id);

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
            (content, result.error().cloned())
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
            move |_args| {
                Box::pin(async move {
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    Ok(ToolOutput::json(serde_json::json!({"reply": reply})))
                })
            },
        )
    }

    fn failing_tool(name: &str, message: &'static str, delay_ms: u64) -> PortableDynamicTool {
        PortableDynamicTool::new(
            name,
            "fail",
            serde_json::json!({"type": "object"}),
            move |_args| {
                Box::pin(async move {
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    Err(ToolExecutionError::other(message.to_string()))
                })
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
                Box::pin(async move { Ok(ToolOutput::json(serde_json::json!(null))) })
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
            |args| {
                Box::pin(async move {
                    let a = args
                        .get("a")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0);
                    let b = args
                        .get("b")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0);
                    Ok(ToolOutput::json(serde_json::json!(a + b)))
                })
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
