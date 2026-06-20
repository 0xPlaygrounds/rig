//! How an agent enforces its `output_schema` (see issue #1928).
//!
//! Mirrors pydantic-ai's output modes. When an agent has both tools and an
//! `output_schema`, applying the provider's native structured-output constraint
//! (`format`/`response_format`) on every turn suppresses tool calls — the model
//! is forced to emit schema JSON instead of using its tools. `OutputMode` lets
//! the structured output be produced as a *tool call* the model makes after
//! using its tools, which composes correctly.

use serde::{Deserialize, Serialize};

/// Controls how an agent's `output_schema` is enforced.
///
/// # Strictness
///
/// [`Native`](OutputMode::Native) is the only mode whose output is *constrained*
/// by the provider — the response is guaranteed to match the schema (on
/// providers that support it). [`Tool`](OutputMode::Tool) and
/// [`Prompted`](OutputMode::Prompted) are **best-effort**: the schema is offered
/// to the model (as a tool or in the prompt) but the model is asked, not forced,
/// to honor it, so the agent re-prompts a bounded number of times and otherwise
/// validate the returned JSON before relying on it. The default
/// [`Auto`](OutputMode::Auto) is provider-aware: for a tool + schema agent it
/// routes to `Tool` only on providers whose native constraint would suppress
/// tool calls, and keeps guaranteed `Native` structured output on providers that
/// compose the two (e.g. OpenAI, Anthropic).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OutputMode {
    /// Resolve at request time: [`OutputMode::Tool`] when the agent has an
    /// `output_schema` **and** at least one function tool (and the tool choice
    /// permits the output-tool call), otherwise [`OutputMode::Native`]. This is
    /// the default and only changes behavior for the tool + schema case (which is
    /// broken under `Native` on providers whose native constraint suppresses
    /// tool calls).
    #[default]
    Auto,
    /// Register the schema as a synthetic "output tool" the model calls to
    /// finalize. No native structured-output constraint is sent, so the model can
    /// freely call its other tools first. Best-effort: the model is instructed to
    /// call the output tool but is not forced to, so validate the result.
    /// (pydantic-ai `ToolOutput`.)
    Tool,
    /// Use the provider's native structured output (`format`/`response_format`).
    /// Constrains every turn, so the output is guaranteed to match the schema,
    /// but may suppress tool calls on some providers (e.g. Ollama). (pydantic-ai
    /// `NativeOutput`.)
    Native,
    /// Inject the schema into the system prompt and return the model's final text
    /// verbatim. The caller parses it — the text is *not* guaranteed to be clean
    /// JSON and may include prose or markdown fences, so extract/validate before
    /// deserializing. Useful for weak/local models that lack reliable tool calling
    /// or native structured output. (pydantic-ai `PromptedOutput`.)
    Prompted,
}
