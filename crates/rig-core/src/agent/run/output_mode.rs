//! How an agent enforces its `output_schema` (see issue #1928).
//!
//! Mirrors pydantic-ai's output modes. When an agent has both tools and an
//! `output_schema`, applying the provider's native structured-output constraint
//! (`format`/`response_format`) on every turn suppresses tool calls — the model
//! is forced to emit schema JSON instead of using its tools. `OutputMode` lets
//! the structured output be produced as a *tool call* the model makes after
//! using its tools, which composes correctly.

/// Controls how an agent's `output_schema` is enforced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputMode {
    /// Resolve at request time: [`OutputMode::Tool`] when the agent has an
    /// `output_schema` **and** at least one function tool, otherwise
    /// [`OutputMode::Native`]. This is the default and only changes behavior for
    /// the tool + schema case (which is broken under `Native`).
    #[default]
    Auto,
    /// Register the schema as a synthetic "output tool" the model calls to
    /// finalize. No native structured-output constraint is sent, so the model
    /// can freely call its other tools first. (pydantic-ai `ToolOutput`.)
    Tool,
    /// Use the provider's native structured output (`format`/`response_format`).
    /// Constrains every turn; may suppress tool calls on some providers.
    /// (pydantic-ai `NativeOutput`.)
    Native,
    /// Inject the schema into the system prompt and parse the model's final text
    /// as JSON. Works with weak/local models that lack reliable tool calling or
    /// native structured output. (pydantic-ai `PromptedOutput`.)
    Prompted,
}
