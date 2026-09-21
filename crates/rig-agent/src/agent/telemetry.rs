//! Shared run, completion-turn, and tool-execution tracing spans.

use tracing::info_span;

/// Build a completion-parent span with the supplied literal name and operation.
/// Literal names are required because tracing stores them in static metadata.
macro_rules! build_chat_span {
    ($runner:expr, $effective_preamble:expr, $name:literal, $operation:literal) => {{
        let system_instructions = $crate::core::telemetry::system_instructions_json(
            $effective_preamble,
            $runner.config.record_telemetry_content,
        );
        // Reuse the core parent marker so completion instrumentation recognizes this span.
        $crate::core::telemetry::completion_parent_span!(
            target: "rig::agent_chat",
            name: $name,
            operation: $operation,
            system_instructions: system_instructions.as_deref(),
            gen_ai.agent.name = $runner.agent_name_or_default(),
        )
    }};
}
pub(crate) use build_chat_span;

/// Adopt an enabled ambient span or create a root `invoke_agent` span.
/// Returns the span and whether it was created. Pass the span captured when the
/// run terminal was called, not its poller's span; record run-level usage only
/// on newly created spans to avoid modifying caller-owned accounting.
pub(crate) fn acquire_agent_span(
    ambient: tracing::Span,
    agent_name: &str,
    preamble: Option<&str>,
    record_content: bool,
) -> (tracing::Span, bool) {
    if ambient.is_disabled() {
        let system_instructions =
            rig_core::telemetry::system_instructions_json(preamble, record_content);
        let span = info_span!(
            parent: None,
            "invoke_agent",
            gen_ai.operation.name = "invoke_agent",
            gen_ai.agent.name = agent_name,
            gen_ai.system_instructions = system_instructions.as_deref(),
            gen_ai.prompt = tracing::field::Empty,
            gen_ai.completion = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = tracing::field::Empty,
        );
        (span, true)
    } else {
        (ambient, false)
    }
}

/// Create an `execute_tool` span parented to the current span, with empty
/// tool identity, arguments, result, outcome, and error fields.
pub(crate) fn new_execute_tool_span() -> tracing::Span {
    info_span!(
        "execute_tool",
        gen_ai.operation.name = "execute_tool",
        gen_ai.tool.type = "function",
        gen_ai.tool.name = tracing::field::Empty,
        gen_ai.tool.call.id = tracing::field::Empty,
        gen_ai.tool.call.arguments = tracing::field::Empty,
        gen_ai.tool.call.result = tracing::field::Empty,
        gen_ai.tool.call.outcome = tracing::field::Empty,
        gen_ai.tool.error.type = tracing::field::Empty
    )
}
