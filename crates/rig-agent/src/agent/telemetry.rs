//! GenAI span shapes shared by the session drivers.
//!
//! The run-level `invoke_agent` span, the per-model-call `chat` /
//! `chat_streaming` spans, the per-tool `execute_tool` span, and the usage
//! recorder all live here so [`AgentSession`](crate::session::AgentSession),
//! [`AgentStream`](crate::stream::AgentStream), and
//! [`ToolExecutor`](crate::executor::ToolExecutor) emit one span shape between
//! them — the same shape dashboards were built against.

use tracing::info_span;

use super::UNKNOWN_AGENT_NAME;

/// Build the per-turn `chat` span shared by both drivers.
///
/// The span *name* must be a string literal — `tracing` bakes it into static
/// metadata — so this is a macro parameterized by the name rather than a
/// function (the two surfaces keep distinct names, `chat` vs `chat_streaming`,
/// which dashboards split on). Every other field is identical across the two
/// surfaces, so it lives here once.
macro_rules! build_chat_span {
    ($params:expr, $request:expr, $name:literal, $operation:literal) => {{
        // Derived from the prepared request, not a scalar preamble: the scalar
        // path missed history system messages, output-mode preamble
        // augmentation, and per-turn overrides.
        let system_instructions = $crate::core::telemetry::system_instructions_json($request);
        // The core macro is the single source of the completion-parent
        // contract (marker + required fields); only the agent-specific field
        // is declared here.
        $crate::core::telemetry::completion_parent_span!(
            target: "rig::agent_chat",
            name: $name,
            operation: $operation,
            system_instructions: system_instructions.as_deref(),
            gen_ai.agent.name = $params.agent_name_or_default(),
        )
    }};
}

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
            rig_core::telemetry::configured_system_instructions_json(preamble, record_content);
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

/// The name/content-recording pair the drivers feed into [`build_chat_span!`].
pub(crate) struct SessionSpanParams<'a> {
    pub(crate) agent_name: Option<&'a str>,
}

impl SessionSpanParams<'_> {
    pub(crate) fn agent_name_or_default(&self) -> &str {
        self.agent_name.unwrap_or(UNKNOWN_AGENT_NAME)
    }
}

/// Build the per-model-call `chat` span for [`AgentSession`]'s unary calls.
///
/// [`AgentSession`]: crate::session::AgentSession
pub(crate) fn new_session_chat_span(
    params: &SessionSpanParams<'_>,
    request: &rig_core::completion::CompletionRequest,
) -> tracing::Span {
    build_chat_span!(params, request, "chat", "chat")
}

/// Build the per-model-call `chat_streaming` span for [`AgentStream`]'s
/// streamed calls.
///
/// [`AgentStream`]: crate::stream::AgentStream
pub(crate) fn new_session_chat_streaming_span(
    params: &SessionSpanParams<'_>,
    request: &rig_core::completion::CompletionRequest,
) -> tracing::Span {
    build_chat_span!(params, request, "chat_streaming", "chat")
}

/// Build the per-tool `execute_tool` span carrying the `gen_ai.tool.*` fields
/// the executor records.
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

/// Record aggregated GenAI token usage onto `span`.
pub(crate) fn record_usage_on_span(span: &tracing::Span, usage: crate::completion::Usage) {
    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
    span.record(
        "gen_ai.usage.cache_read.input_tokens",
        usage.cached_input_tokens,
    );
    span.record(
        "gen_ai.usage.cache_creation.input_tokens",
        usage.cache_creation_input_tokens,
    );
    span.record(
        "gen_ai.usage.tool_use_prompt_tokens",
        usage.tool_use_prompt_tokens,
    );
    span.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);
}
