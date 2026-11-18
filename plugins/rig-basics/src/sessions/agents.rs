//! Forks of this session's agents: `/fork` makes one from the shown agent
//! (the kernel's [`ForkAgent`]) and shows it, and, with the `tui` feature,
//! rewind: select an earlier message of yours with Alt+Up and press Enter,
//! and the shown agent forks just before it, the fork is shown and the
//! message is back in the input. [`AgentsPlugin`] is off by default; it adds
//! the [`AgentPickerPlugin`]'s `/agents`, which shows the other agents.

use crate::agent_picker::{AgentPickerPlugin, show};
use crate::session_picker::{SessionPicker, SessionPickerPlugin, SessionSource};
use rig_ecs::agent::{AgentId, Notice};
use rig_ecs::fork::{ForkAgent, ForkResult};
use rig_ecs::inbox::Recalled;
use rig_ecs::prelude::MessageId;
use rig_ecs::requests::RequestOf;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::pick::Picker;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

#[cfg(feature = "tui")]
mod rewind;

/// `/fork`, rewind with the `tui` feature, and the `/agents` it adds.
#[derive(Default)]
pub struct AgentsPlugin;

impl Plugin for AgentsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionPickerPlugin>() {
            app.add_plugins(SessionPickerPlugin);
        }
        if !app.is_plugin_added::<AgentPickerPlugin>() {
            app.add_plugins(AgentPickerPlugin);
        }
        app.add_command(
            "fork",
            "Fork the shown agent into a new one and show it: /fork at the end, \
             /fork at to pick a message, /fork <message> through it",
            fork,
        );
        #[cfg(feature = "tui")]
        rewind::add(app);
    }
}

/// On a fork this plugin asked for: the text to put back in the input once
/// the fork is shown.
#[derive(Component)]
struct Recall(Option<String>);

/// Forks `agent` through `through`, then shows the fork with `recall` in
/// its input, or says why there is none.
fn ask_fork(
    commands: &mut Commands,
    (agent, through): (Entity, Option<MessageId>),
    recall: Option<String>,
) {
    let fork = ForkAgent::new(agent, through);
    commands.spawn((fork, Recall(recall))).observe(forked);
}

/// `/fork`: forks the agent the command is for at the end, through the
/// message named (`m4`), or, with `at`, through the message picked from
/// its entries.
fn fork(
    In(args): In<CommandArgs>,
    ids: Query<&AgentId>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let through = match args.args.trim() {
        "" => None,
        "at" => {
            let agent = ids.get(args.agent).ok().map(|id| id.0.clone());
            commands.spawn((
                Picker::new("Fork through which message?").for_agent(args.agent),
                SessionPicker {
                    source: SessionSource::Entries {
                        session: None,
                        agent,
                    },
                    command: "fork {entry}".to_owned(),
                    ..SessionPicker::default()
                },
            ));
            return;
        }
        named => match named.parse::<MessageId>() {
            Ok(id) => Some(id),
            Err(_) => {
                let why = format!("`{named}` is no message, such as `m4`; /fork at picks one.");
                notices.write(Notice::error(args.agent, why));
                return;
            }
        },
    };
    ask_fork(&mut commands, (args.agent, through), None);
}

/// Shows a fork this plugin asked for, with what it puts back in the
/// input, or says why it was not made.
fn forked(
    made: On<Add<ForkResult>>,
    requests: Query<(&ForkResult, &RequestOf, &Recall)>,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((ForkResult(result), &RequestOf(agent), Recall(recall))) = requests.get(made.entity)
    else {
        return;
    };
    let fork = match result {
        Ok(fork) => *fork,
        Err(why) => {
            notices.write(Notice::error(
                agent,
                format!("Could not fork the agent: {why}."),
            ));
            return;
        }
    };
    show(fork, &mut commands);
    let said = "Forked into a new agent; /agents shows the other.";
    notices.write(Notice::info(fork, said));
    if let Some(text) = recall {
        recalled.write(Recalled {
            agent: fork,
            text: text.clone(),
            why: None,
        });
    }
}

#[cfg(test)]
mod tests;
