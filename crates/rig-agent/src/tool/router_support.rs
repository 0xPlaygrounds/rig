//! Runtime support for the `#[derive(ToolRouter)]` macro.
//!
//! These functions are the shared, monomorphic building blocks the derive
//! expands calls to. They are extracted from (or call directly into) the
//! classic runner pipeline so a derived router behaves exactly like the
//! classic loop for argument parsing, error normalization, and tool-result
//! shaping — without duplicating that logic.
//!
//! This module is public so generated code can reference it from downstream
//! crates; it is not intended to be a hand-written user API, but every
//! function is safe to call directly.

use std::future::Future;

use futures::{StreamExt, stream};

use rig_core::message::{ToolCall, UserContent};

use super::{IntoToolOutput, Tool, ToolContext, ToolExecutionError, ToolResult, parse_tool_args};
use crate::agent::prompt_request::{tool_result_message, tool_result_output};
use crate::agent::run::PendingToolCall;

/// Execute one typed tool against the model-emitted JSON arguments, mirroring
/// the classic erased-dispatch semantics monomorphically:
///
/// - arguments are serialized to their compact JSON string and parsed with the
///   classic parser (including the `null` → `{}` fallback for tools whose
///   argument struct has no required fields);
/// - the tool runs against a fresh [`ToolContext`];
/// - the output is normalized through [`IntoToolOutput`];
/// - typed errors are normalized through [`Tool::map_error`].
///
/// Every failure becomes a [`ToolResult::failed`], never a panic or a
/// separate error channel — errors stay model-visible, as in the classic loop.
pub async fn execute_typed<T: Tool>(tool: &T, arguments: &serde_json::Value) -> ToolResult {
    let serialized = rig_core::json_utils::serialize_json_value(arguments);
    let args = match parse_tool_args::<T::Args>(&serialized) {
        Ok(args) => args,
        Err(error) => return ToolResult::failed(error),
    };
    let mut context = ToolContext::new();
    match tool.call(&mut context, args).await {
        Ok(output) => match output.into_tool_output() {
            Ok(output) => ToolResult::success(output),
            Err(error) => ToolResult::failed(error),
        },
        Err(error) => ToolResult::failed(tool.map_error(error)),
    }
}

/// The result a derived router returns for an unknown tool name — the same
/// shape the classic registry produces when dispatch misses.
pub fn not_found(name: &str) -> ToolResult {
    ToolResult::failed(
        ToolExecutionError::not_found(format!("no tool named `{name}` is registered"))
            .with_model_feedback(format!("tool `{name}` not found")),
    )
}

/// Shape a structured [`ToolResult`] into the tool-result message content the
/// classic loop delivers to the model: the result's presentation output
/// (success output or the error's model-visible output) wrapped as a
/// `UserContent::tool_result` correlated by the call's ids.
pub fn shape_result(call: &ToolCall, result: &ToolResult) -> UserContent {
    tool_result_output(
        call.id.clone(),
        call.call_id.clone(),
        result.output().clone(),
    )
}

/// Execute a batch of pending tool calls and return their shaped results in
/// call order.
///
/// Preresolved calls (invalid tool-call recovery) return their
/// `preresolved_result` verbatim without executing anything, exactly as the
/// classic drivers do. The remaining calls run through `dispatch` with at most
/// `concurrency.max(1)` in flight, and every result — success or failure — is
/// shaped into model-visible tool-result content via [`shape_result`]. This is
/// infallible by design: like the classic loop, tool errors are delivered to
/// the model as results, never surfaced as a driver-level failure.
pub async fn dispatch_pending<'calls, F, Fut>(
    calls: &'calls [PendingToolCall],
    concurrency: usize,
    dispatch: F,
) -> Vec<UserContent>
where
    F: Fn(&'calls ToolCall) -> Fut,
    Fut: Future<Output = ToolResult>,
{
    let concurrency = concurrency.max(1);
    // One slot per call, prefilled with preresolved passthroughs so executed
    // results land back in call order regardless of completion order.
    let mut slots: Vec<Option<UserContent>> = calls
        .iter()
        .map(|call| call.preresolved_result.clone())
        .collect();

    let dispatch = &dispatch;
    let mut executed = stream::iter(
        calls
            .iter()
            .enumerate()
            .filter(|(_, call)| call.preresolved_result.is_none()),
    )
    .map(|(index, pending)| {
        let future = dispatch(&pending.tool_call);
        async move { (index, shape_result(&pending.tool_call, &future.await)) }
    })
    .buffer_unordered(concurrency);
    while let Some((index, content)) = executed.next().await {
        if let Some(slot) = slots.get_mut(index) {
            *slot = Some(content);
        }
    }

    // Every slot is filled by construction (preresolved upfront, executed by
    // index above). The fallback is unreachable but keeps this lint-clean and
    // fail-safe: a synthetic textual result rather than a panic or a dropped
    // (misaligned) tool result.
    calls
        .iter()
        .zip(slots)
        .map(|(pending, slot)| {
            slot.unwrap_or_else(|| {
                tool_result_message(
                    pending.tool_call.id.clone(),
                    pending.tool_call.call_id.clone(),
                    "tool result unavailable".to_string(),
                )
            })
        })
        .collect()
}
