//! Structured-output enforcement modes for agent requests.
//!
//! ```
//! use rig_agent::run::output::OutputMode;
//! let mode = OutputMode::default();
//! assert_eq!(mode, OutputMode::Auto);
//! ```

use serde::{Deserialize, Serialize};

/// Select how requests convey an agent's output schema.
/// Native enforcement depends on provider support. Tool and prompted modes are
/// best-effort; callers must validate returned values before relying on the schema.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputMode {
    /// Resolve to tool output when a schema and function tools are present,
    /// tool choice permits it, and native output cannot compose with tools.
    /// Otherwise use native output.
    #[default]
    Auto,
    /// Offer the schema as a synthetic final-answer tool instead of a native
    /// constraint. Calling it is best-effort; callers must validate the result.
    Tool,
    /// Send a native structured-output constraint on every turn.
    /// Enforcement and compatibility with tool calls depend on provider capabilities.
    Native,
    /// Put the schema in the system prompt and return final text verbatim.
    /// Callers must extract and validate JSON; prose and markdown may be present.
    Prompted,
}
