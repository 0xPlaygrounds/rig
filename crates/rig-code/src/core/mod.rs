//! The agent core: agents, the agent loop, effects, models, registries and
//! saving. It knows nothing about any view, so it can be extracted as its
//! own crate.

pub(crate) mod app;

mod agent;
mod dispatch;
mod models;
mod registry;
mod session;
mod turn;

use bevy::app::OnAppExitSystems;
use bevy::prelude::*;

pub use agent::{
    Agent, AgentId, AgentStatus, Conversation, EffortChoice, ModelChoice, SystemPrompt, ToolAccess,
    Work, WorkOf,
};
pub use app::DataDir;
pub use dispatch::Effects;
pub use models::{
    AgentDefaults, ModelEndpoint, available_models, effort_label, effort_options, model_reference,
};
pub use registry::{
    AgentAppExt, Choice, Interrupt, Notice, NoticeLevel, OfferChoices, RunCommand, SlashCommand,
    Submit, ToolEntry, TurnFinished,
};
pub use session::Session;
pub use turn::{ModelCall, StreamingText, ToolCallDone, ToolCallRun};

/// The stages of the agent loop, in order, in `Update`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentSet {
    /// Input becomes prompts and commands.
    Route,
    /// Model calls start.
    Start,
    /// Calls in flight are polled.
    Poll,
    /// Finished tool calls feed the next model call.
    Finish,
}

/// Registers the agent core. Added by [`crate::run`] before the plugin
/// list.
pub(crate) struct CorePlugin;

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        let defaults = app
            .world()
            .get_resource::<DataDir>()
            .map(|data| AgentDefaults::load(&data.0))
            .unwrap_or_default();
        app.insert_resource(defaults)
            .init_resource::<Effects>()
            .add_message::<Submit>()
            .add_message::<Interrupt>()
            .add_message::<Notice>()
            .add_message::<OfferChoices>()
            .add_message::<TurnFinished>()
            .add_observer(models::connect_model)
            .configure_sets(
                Update,
                (
                    AgentSet::Route,
                    AgentSet::Start,
                    AgentSet::Poll,
                    AgentSet::Finish,
                )
                    .chain(),
            )
            .add_systems(Startup, session::start_session)
            .add_systems(
                Update,
                (
                    registry::route_input.in_set(AgentSet::Route),
                    turn::start_model_calls.in_set(AgentSet::Start),
                    (turn::poll_model_calls, turn::poll_tool_calls)
                        .chain()
                        .in_set(AgentSet::Poll),
                    turn::collect_tool_results.in_set(AgentSet::Finish),
                    turn::report_turns.after(AgentSet::Finish),
                ),
            )
            .add_systems(
                Last,
                (
                    dispatch::flush_effects,
                    models::save_defaults.run_if(resource_changed::<AgentDefaults>),
                    // Autosave after every turn, and save on every exit.
                    session::save_session
                        .in_set(OnAppExitSystems)
                        .run_if(on_message::<TurnFinished>.or_eager(on_message::<AppExit>)),
                ),
            );
    }
}
