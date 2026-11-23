//! Structured-output enforcement modes for agent requests.
//!
//! ```
//! use rig_agent::run::output::OutputMode;
//! let mode = OutputMode::default();
//! assert_eq!(mode, OutputMode::Auto);
//! ```

use serde::{Deserialize, Serialize};

/// Select how requests convey an agent's output schema.
///
/// Whatever the mode, and whether or not the provider enforces the schema, a
/// run ends only on an answer that is JSON fitting the schema, as
/// [`rig_core::tool::validate::validate`] checks it: it asks the model again
/// while its output-retry budget lasts, then fails.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputMode {
    /// Resolve to tool output when a schema and function tools are present,
    /// tool choice permits it, and native output cannot compose with tools.
    /// Otherwise use native output.
    #[default]
    Auto,
    /// Offer the schema as a synthetic final-answer tool instead of a native
    /// constraint. A text answer that satisfies the schema is also accepted.
    Tool,
    /// Send a native structured-output constraint on every turn.
    /// Enforcement and compatibility with tool calls depend on provider
    /// capabilities; a provider that drops the constraint is checked by the run.
    Native,
    /// Put the schema in the system prompt and return final text verbatim.
    /// Prose or a markdown fence around the JSON may be present.
    Prompted,
}
