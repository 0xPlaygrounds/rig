//! What the app's commands ask of a view, whatever draws it: show an agent
//! ([`Focus`]), let the user pick one of several command lines
//! ([`PickRequest`]), and send what the user typed ([`send_input`]), with
//! the images it names as `@path` read.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

use crate::attach;
use rig_ecs::agent::Notice;
use rig_ecs::commands::RunCommand;
use rig_ecs::inbox::{Deliver, DeliveryMode};

/// Registers [`PickRequest`].
pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PickRequest>();
    }
}

/// Ask the views to show the agent and send what is typed to it, such as
/// a subagent picked with `/agents`.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Focus {
    /// The agent.
    pub entity: Entity,
}

/// One choice of a [`PickRequest`]: what it shows, and the command line,
/// without its `/`, that choosing it runs for the agent, such as
/// `model openai/gpt-5`.
#[derive(Clone, Debug)]
pub struct PickItem {
    /// What it shows.
    pub label: String,
    /// The command line it runs.
    pub command: String,
}

/// Asks a view to let the user pick one of `items` for the agent; the
/// view runs the chosen item's command with [`RunCommand`].
#[derive(Message, Clone, Debug)]
pub struct PickRequest {
    /// The agent.
    pub agent: Entity,
    /// What is picked.
    pub title: String,
    /// The choices.
    pub items: Vec<PickItem>,
    /// The position of the choice selected at first.
    pub selected: usize,
}

/// Sends what the user typed to `agent`: a slash command when it starts
/// with `/`, which runs now, otherwise the user's message with the images
/// it names as `@path`, delivered as `mode` says.
pub fn send_input(commands: &mut Commands, agent: Entity, text: String, mode: DeliveryMode) {
    if let Some(line) = text.trim_start().strip_prefix('/') {
        commands.trigger(RunCommand {
            entity: agent,
            line: line.to_owned(),
        });
        return;
    }
    let (attachments, notes) = attach::attachments(&text);
    for note in notes {
        commands.write_message(Notice::info(agent, note));
    }
    commands.trigger(Deliver::user(agent, text, mode, attachments));
}
