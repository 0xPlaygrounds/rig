//! rig-agents: helper agents for the model, each a child session of the rig
//! agent server started, steered, waited on, answered and stopped through
//! the worker's channel.

use rig_harness::prelude::*;

/// The helper-agent tools.
#[derive(Default)]
pub struct AgentsPlugin;

impl Plugin for AgentsPlugin {
    fn build(&self, _app: &mut App) {}
}
