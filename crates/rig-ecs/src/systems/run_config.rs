//! Named inputs for starting a run without positional policy flags.

use crate::agent::MessageParts;

/// Optional inputs to [`super::RunCommands::spawn_run`].
///
/// The prompt follows `history`. An absent `max_turns` inherits the agent's
/// setting; `streamed` selects streamed model delivery, not a separate driver.
/// Both command and world entry points copy the history into the run graph;
/// the borrowed slice need not outlive the call.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunConfig<'a> {
    /// Prior messages, in conversation order, before the new prompt.
    pub history: &'a [MessageParts],
    /// Request streamed model delivery.
    pub streamed: bool,
    /// Override the agent's model-call budget for this run.
    pub max_turns: Option<usize>,
}
