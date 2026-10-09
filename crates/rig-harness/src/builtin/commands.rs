//! The built-in slash commands: `/model`, `/effort`, `/usage`, `/retry`,
//! `/compact`, `/agents`, `/help` and `/quit`, with `/login` and `/logout`
//! from the [`LoginPlugin`] they add.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::LoginPlugin;
use crate::view::{Focus, PickItem, PickRequest};
use rig_ecs::agent::{
    ActiveTurn, Agent, AgentId, Compact, Connection, Effort, ModelChoice, Notice, Retry, SetEffort,
    SetModel, Spawned, SpawnedBy,
};
use rig_ecs::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use rig_ecs::models::{self, ModelConnector};
use rig_ecs::usage::{Spending, TurnSpending};

/// Registers the built-in commands with [`AppCommandsExt::add_command`].
#[derive(Default)]
pub struct BuiltinCommandsPlugin;

impl Plugin for BuiltinCommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "model",
            "Pick the model, or set it with /model vendor/model",
            model,
        )
        .add_command(
            "effort",
            "Pick the reasoning setting, or set it with /effort <level>",
            effort,
        )
        .add_command(
            "usage",
            "Show the tokens, cost and context the session used",
            usage,
        )
        .add_command(
            "retry",
            "Send the conversation again after a failed model call",
            retry,
        )
        .add_command(
            "compact",
            "Summarize the older conversation to free context; /compact <focus> says what to keep",
            compact,
        )
        .add_command(
            "agents",
            "List the agents and subagents and show one; /agents <number or title> shows it",
            agents,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Quit; the session stays for /resume", quit)
        .add_plugins(LoginPlugin);
    }
}

fn model(
    In(args): In<CommandArgs>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        let items: Vec<PickItem> = connector
            .available()
            .into_iter()
            .map(|spec| {
                let reference = spec.reference();
                let note = match connector.plan(spec) {
                    Some(plan) => format!("  ({plan} plan)"),
                    None if spec.provider.requires_credential() => String::new(),
                    None => "  (no key needed)".to_owned(),
                };
                PickItem {
                    label: format!("{reference}  {}{note}", spec.display_name),
                    command: format!("model {reference}"),
                }
            })
            .collect();
        if items.is_empty() {
            notices.write(Notice::error(
                args.agent,
                "No provider with tool-calling models can be reached: set a key such as \
                 OPENAI_API_KEY, or sign in with /login chatgpt.",
            ));
            return;
        }
        picks.write(PickRequest {
            agent: args.agent,
            title: "Model".to_owned(),
            items,
            selected: 0,
        });
    } else {
        commands.trigger(SetModel {
            entity: args.agent,
            model: args.args,
        });
    }
}

fn effort(
    In(args): In<CommandArgs>,
    agents: Query<&Connection>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Connection { spec, .. }) = agents.get(args.agent) else {
        notices.write(Notice::info(
            args.agent,
            "Pick a model with /model first.".to_owned(),
        ));
        return;
    };
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            title: format!("Reasoning for {}", spec.display_name),
            items: spec
                .reasoning
                .choices()
                .into_iter()
                .map(|choice| PickItem {
                    label: choice.label(),
                    command: format!("effort {}", choice.name),
                })
                .collect(),
            selected: 0,
        });
        return;
    }
    match models::effort_named(spec, &args.args) {
        Ok(effort) => {
            commands.trigger(SetEffort {
                entity: args.agent,
                effort: Effort(effort),
            });
        }
        Err(why) => {
            notices.write(Notice::error(args.agent, format!("{why}.")));
        }
    }
}

fn usage(
    In(args): In<CommandArgs>,
    agents: Query<(&Spending, Option<&Connection>, Option<&ActiveTurn>)>,
    turns: Query<&TurnSpending>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((spent, connection, turn)) = agents.get(args.agent) else {
        return;
    };
    if spent.calls == 0 {
        notices.write(Notice::info(args.agent, "No model call yet."));
        return;
    }
    let mut lines = vec![format!("Session: {}.", spent.summary())];
    if let Some(TurnSpending(turn)) = turn.and_then(|turn| turns.get(turn.turn()).ok())
        && turn.calls > 0
    {
        lines.push(format!("This turn: {}.", turn.summary()));
    }
    let spec = connection.map(|connection| &*connection.spec);
    match spent.context_use(spec) {
        Some(context) => lines.push(format!(
            "Context: {} tokens{}.",
            context.label(),
            if context.window.is_none() {
                ", the model's window is not in the catalog"
            } else {
                ""
            }
        )),
        None => lines.push("Context: not reported by the provider.".to_owned()),
    }
    if spent.unpriced > 0 {
        lines.push(format!(
            "{} of {} calls had no price: their provider did not report one and the catalog \
             lists none, or the model is local or billed by subscription.",
            spent.unpriced, spent.calls
        ));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

fn retry(In(args): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Retry { entity: args.agent });
}

fn compact(In(args): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Compact {
        entity: args.agent,
        focus: args.args,
    });
}

fn help(
    In(args): In<CommandArgs>,
    commands: Query<&SlashCommand>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines: Vec<String> = commands
        .iter()
        .map(|command| format!("/{:<10} {}", command.name, command.help))
        .collect();
    lines.sort();
    lines.push(
        "Esc stops a running turn. Ctrl+C clears the input. In the terminal view, \
         Shift+Enter or Ctrl+J adds a line, Up and Down browse earlier prompts, Tab completes \
         /commands and @paths. While a turn runs, \
         Enter steers it and Tab queues a follow-up. @path attaches an image file, and \
         Ctrl+V pastes the clipboard's image. Esc stops only the shown agent."
            .to_owned(),
    );
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

fn quit(In(_): In<CommandArgs>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}

/// `/agents`: opens the agent picker, or shows the agent whose number or
/// title is given.
fn agents(
    In(args): In<CommandArgs>,
    agent_tree: RosterQuery,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let entries = roster(&agent_tree);
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            title: "Show an agent".to_owned(),
            selected: entries
                .iter()
                .position(|entry| entry.agent == args.agent)
                .unwrap_or(0),
            items: entries
                .iter()
                .enumerate()
                .map(|(index, entry)| PickItem {
                    label: format!("{}. {}", index + 1, entry.label),
                    command: format!("agents {}", index + 1),
                })
                .collect(),
        });
        return;
    }
    let wanted = args.args.to_lowercase();
    let chosen = args
        .args
        .parse::<usize>()
        .ok()
        .and_then(|number| number.checked_sub(1))
        .and_then(|index| entries.get(index))
        .or_else(|| {
            entries
                .iter()
                .find(|entry| entry.label.to_lowercase().contains(&wanted))
        });
    match chosen {
        Some(entry) => commands.trigger(Focus {
            entity: entry.agent,
        }),
        None => {
            let listed: Vec<String> = entries
                .iter()
                .enumerate()
                .map(|(index, entry)| format!("{}. {}", index + 1, entry.label))
                .collect();
            notices.write(Notice::error(
                args.agent,
                format!(
                    "No agent matches `{}`. The agents:\n{}",
                    args.args,
                    listed.join("\n")
                ),
            ));
        }
    }
}

/// One agent in [`roster`]: a line describing it.
#[derive(Clone, Debug)]
struct RosterEntry {
    agent: Entity,
    /// Its title, model, state and cost, indented by depth.
    label: String,
}

/// What [`roster`] reads of each agent.
type RosterQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AgentId,
        Option<&'static Name>,
        Option<&'static SpawnedBy>,
        Option<&'static Spawned>,
        Option<&'static ModelChoice>,
        Has<ActiveTurn>,
        &'static Spending,
    ),
    With<Agent>,
>;

/// Every agent as a tree: the agents nothing spawned, by id, each followed
/// by the agents it spawned in the order it spawned them. A spawned agent
/// is titled by its [`Name`].
fn roster(agents: &RosterQuery) -> Vec<RosterEntry> {
    let mut roots: Vec<(Entity, &AgentId)> = agents
        .iter()
        .filter(|(_, _, _, of, ..)| of.is_none_or(|of| !agents.contains(of.0)))
        .map(|(entity, id, ..)| (entity, id))
        .collect();
    roots.sort_by(|a, b| a.1.0.cmp(&b.1.0));
    let several = roots.len() > 1;
    let total = agents.iter().count();
    let mut stack: Vec<(Entity, usize)> = roots.iter().rev().map(|(root, _)| (*root, 0)).collect();
    let mut entries = Vec::new();
    while let Some((agent, depth)) = stack.pop() {
        // A relationship loop cannot happen, but a bound costs nothing.
        if entries.len() >= total {
            break;
        }
        let Ok((_, id, name, of, spawned, model, busy, spent)) = agents.get(agent) else {
            continue;
        };
        let title = match (name, of) {
            (Some(name), Some(_)) => name.as_str().to_owned(),
            _ if several => format!("agent {}", id.short()),
            _ => "main agent".to_owned(),
        };
        let mut label = format!(
            "{}{title} · {} · {}",
            "  ".repeat(depth),
            model.map_or("no model", |model| model.0.as_str()),
            if busy { "working" } else { "idle" }
        );
        if spent.calls > 0 {
            label.push_str(&format!(" · {}", spent.cost_or_tokens()));
        }
        entries.push(RosterEntry { agent, label });
        for child in spawned.into_iter().flat_map(|spawned| spawned.iter().rev()) {
            stack.push((child, depth + 1));
        }
    }
    entries
}
