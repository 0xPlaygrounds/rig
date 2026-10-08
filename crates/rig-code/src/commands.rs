//! Slash commands as entities, the routing of submitted text, and the
//! built-in commands.
//!
//! A command is an entity holding a [`SlashCommand`]: its name and a
//! registered one-shot system taking [`CommandArgs`]. Plugins add commands
//! with [`RigCodeAppExt::add_command`](crate::RigCodeAppExt::add_command);
//! [`CommandsPlugin`] registers `/model`, `/effort`, `/help` and `/quit` the
//! same way.

use bevy_app::{App, Plugin};
use bevy_ecs::{prelude::*, system::SystemId};
use rig_core::message::Message;

use crate::{
    RigCodeAppExt as _,
    agent::{
        Agent, AgentCalls, AgentStatus, Choose, Conversation, Effort, ModelChoice, NeedsReply,
        Notice, Quit, Submit, turn_running,
    },
    model::{self, Credentials},
};

/// A command typed as `/name args`.
#[derive(Component, Debug)]
pub struct SlashCommand {
    /// The name, without the slash.
    pub name: String,
    /// One line for `/help`.
    pub description: String,
    /// The system run with the command's arguments.
    pub run: SystemId<In<CommandArgs>>,
}

/// What a command runs with.
#[derive(Debug, Clone)]
pub struct CommandArgs {
    /// The agent the command was typed for.
    pub agent: Entity,
    /// The text after the command name, trimmed.
    pub args: String,
}

/// Turn each [`Submit`] into a command run or a prompt. A prompt is taken
/// only while the agent is idle.
pub(crate) fn route_submit(
    mut submits: MessageReader<Submit>,
    slash: Query<&SlashCommand>,
    mut agents: Query<
        (
            &mut Conversation,
            &ModelChoice,
            &AgentStatus,
            Has<NeedsReply>,
            Has<AgentCalls>,
        ),
        With<Agent>,
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for Submit { agent, text } in submits.read() {
        let (agent, text) = (*agent, text.trim());
        if text.is_empty() {
            continue;
        }
        if let Some(line) = text.strip_prefix('/') {
            let (name, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            match slash.iter().find(|command| command.name == name) {
                Some(command) => commands.run_system_with(
                    command.run,
                    CommandArgs {
                        agent,
                        args: args.trim().to_owned(),
                    },
                ),
                None => {
                    notices.write(Notice {
                        agent,
                        text: format!("Unknown command /{name}. Type /help for the list."),
                    });
                }
            }
            continue;
        }
        let Ok((mut conversation, choice, status, needs_reply, calls)) = agents.get_mut(agent)
        else {
            continue;
        };
        let refused = if turn_running(*status, needs_reply, calls) {
            Some("A turn is running. Press Esc to stop it first.")
        } else if choice.0.is_none() {
            Some("Pick a model with /model first.")
        } else {
            None
        };
        if let Some(text) = refused {
            notices.write(Notice {
                agent,
                text: text.to_owned(),
            });
            continue;
        }
        conversation.0.push(Message::user(text));
        commands.entity(agent).insert(NeedsReply);
    }
}

/// Registers `/model`, `/effort`, `/help` and `/quit`.
pub struct CommandsPlugin;

impl Plugin for CommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "model",
            "Pick the model, or /model vendor/model",
            model_command,
        )
        .add_command(
            "effort",
            "Pick the reasoning effort for the model",
            effort_command,
        )
        .add_command("help", "List the commands", help_command)
        .add_command("quit", "Leave rig-code", quit_command);
    }
}

fn model_command(
    In(CommandArgs { agent, args }): In<CommandArgs>,
    mut agents: Query<(&mut ModelChoice, &mut Effort)>,
    mut credentials: ResMut<Credentials>,
    mut choose: MessageWriter<Choose>,
    mut notices: MessageWriter<Notice>,
) {
    let mut notice = |text: String| {
        notices.write(Notice { agent, text });
    };
    if args.is_empty() {
        let options = model::model_choices(&mut credentials);
        if options.is_empty() {
            notice(
                "No catalog model has a credential. Set a provider key such as \
                 OPENAI_API_KEY or DEEPSEEK_API_KEY and restart."
                    .to_owned(),
            );
        } else {
            choose.write(Choose {
                agent,
                title: "Model".to_owned(),
                options,
            });
        }
        return;
    }
    let Some(spec) = model::resolve(&args) else {
        notice(format!("{args} is not in the model catalog."));
        return;
    };
    if !credentials.usable(spec) {
        let key = spec.provider.api_key_env().unwrap_or("its credential");
        notice(format!("{args} needs {key} to be set."));
        return;
    }
    let Ok((mut choice, mut effort)) = agents.get_mut(agent) else {
        return;
    };
    let reference = model::reference(spec);
    choice.0 = Some(reference.clone());
    if let Some(reasoning) = effort.0
        && model::refusal(spec, reasoning).is_some()
    {
        effort.0 = None;
    }
    notice(format!(
        "Model: {} ({reference}), effort {}.",
        spec.display_name,
        model::describe(effort.0)
    ));
}

fn effort_command(
    In(CommandArgs { agent, args }): In<CommandArgs>,
    mut agents: Query<(&ModelChoice, &mut Effort)>,
    mut choose: MessageWriter<Choose>,
    mut notices: MessageWriter<Notice>,
) {
    let mut notice = |text: String| {
        notices.write(Notice { agent, text });
    };
    let Ok((choice, mut effort)) = agents.get_mut(agent) else {
        return;
    };
    let Some(spec) = choice.0.as_deref().and_then(model::resolve) else {
        notice("Pick a model first with /model.".to_owned());
        return;
    };
    if args.is_empty() {
        let options = model::effort_choices(spec);
        if options.is_empty() {
            notice(format!("{} takes no reasoning setting.", spec.display_name));
        } else {
            choose.write(Choose {
                agent,
                title: format!("Effort for {}", spec.display_name),
                options,
            });
        }
        return;
    }
    let Some(reasoning) = model::parse_effort(&args) else {
        notice(format!(
            "{args} is not an effort. Use off, a level such as high, or a token budget."
        ));
        return;
    };
    match model::refusal(spec, reasoning) {
        Some(reason) => notice(reason),
        None => {
            effort.0 = Some(reasoning);
            notice(format!("Effort: {}.", model::describe(effort.0)));
        }
    }
}

fn help_command(
    In(CommandArgs { agent, .. }): In<CommandArgs>,
    slash: Query<&SlashCommand>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines: Vec<String> = slash
        .iter()
        .map(|command| format!("/{:<10} {}", command.name, command.description))
        .collect();
    lines.sort();
    lines.push("Esc stops a running turn. Ctrl+C clears the input, then quits.".to_owned());
    notices.write(Notice {
        agent,
        text: lines.join("\n"),
    });
}

fn quit_command(_: In<CommandArgs>, mut quit: MessageWriter<Quit>) {
    quit.write(Quit);
}
