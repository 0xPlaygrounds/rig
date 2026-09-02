//! Span shapes shared by both drivers: the run-level `invoke_agent` span, the
//! per-turn `chat` span, and the per-tool `execute_tool` span. Defined once so
//! the blocking and streaming surfaces cannot drift in what they record.

use tracing::info_span;

/// Build the per-turn `chat` span shared by both turn sources.
///
/// The span *name* must be a string literal — `tracing` bakes it into static
/// metadata — so this is a macro parameterized by the name rather than a
/// function (the two surfaces keep distinct names, `chat` vs `chat_streaming`,
/// which dashboards split on). The matching operation value is passed with the
/// name; every other field is identical across the two surfaces, so it lives
/// here once instead of being copy-pasted into each `TurnSource::open_chat_span`.
macro_rules! build_chat_span {
    ($runner:expr, $effective_preamble:expr, $name:literal, $operation:literal) => {{
        let system_instructions = $crate::core::telemetry::system_instructions_json(
            $effective_preamble,
            $runner.config.record_telemetry_content,
        );
        // The core macro is the single source of the completion-parent
        // contract (marker + required fields); only the agent-specific field
        // is declared here.
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

/// Build (or adopt) the top-level `invoke_agent` span for a run, shared by the
/// blocking and streaming drivers so the run-level span shape is defined once.
///
/// Returns the span plus whether it was newly created. When the caller is
/// already inside a span we adopt it and report `false`, so the driver can avoid
/// recording run-level usage onto a span it does not own (see the
/// `created_agent_span` guard in both drivers' `Done` handling).
pub(crate) fn acquire_agent_span(
    agent_name: &str,
    preamble: Option<&str>,
    record_content: bool,
) -> (tracing::Span, bool) {
    if tracing::Span::current().is_disabled() {
        let system_instructions =
            rig_core::telemetry::system_instructions_json(preamble, record_content);
        let span = info_span!(
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
        (tracing::Span::current(), false)
    }
}

/// Build the per-tool `execute_tool` span carrying the `gen_ai.tool.*` fields
/// that [`run_single_tool`] records on the current span. Parented to the
/// contextual current span; the blocking driver additionally chains it via
/// `follows_from`, while the streaming driver uses it as-is. Shared by both
/// drivers so the span shape stays defined in one place.
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
