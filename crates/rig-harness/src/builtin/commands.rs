//! The built-in slash commands: `/model`, `/effort`, `/login`, `/logout`,
//! `/usage`, `/retry`, `/compact`, `/agents`, `/rewind`, `/fork`,
//! `/approvals`, `/help` and `/quit`.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::core::agent::{
    ActiveTurn, Compact, Connection, Conversation, Effort, Focus, Notice, PickKind, PickRequest,
    Retry, SetEffort, SetModel,
};
use crate::core::approval::{ApprovalMode, Permission, Policy, Rule};
use crate::core::commands::{AppCommandsExt, CommandArgs, SlashCommand};
use crate::core::login::{SignIn, SignOut};
use crate::core::models;
use crate::core::rewind::{self, Fork, History, Point, Rewind, UndoRewind};
use crate::core::subagents::{self, RosterQuery};
use crate::core::usage::{Spending, TurnSpending};

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
            "login",
            "Sign in with your ChatGPT plan: /login chatgpt opens the browser (--device shows a \
             code to enter instead); /login again or Esc cancels",
            login,
        )
        .add_command(
            "logout",
            "Forget a sign-in: /logout chatgpt",
            logout,
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
        .add_command(
            "rewind",
            "Go back to a checkpoint, conversation and files; /rewind <n> [chat] keeps the files \
             with chat, /rewind undo undoes it",
            rewind,
        )
        .add_command(
            "fork",
            "Clone this agent at a checkpoint into a new one; /fork now clones it as it is",
            fork,
        )
        .add_command(
            "approvals",
            "Show which tool calls ask first; /approvals auto|ask|read-only sets the mode, \
             /approvals allow|ask|deny <tool> [subject] adds a rule, /approvals forget <n> drops one",
            approvals,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Save and quit", quit);
    }
}

fn model(In(args): In<CommandArgs>, mut commands: Commands, mut picks: MessageWriter<PickRequest>) {
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            kind: PickKind::Model,
        });
    } else {
        commands.trigger(SetModel {
            entity: args.agent,
            model: args.args,
        });
    }
}

fn login(In(args): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(SignIn {
        entity: args.agent,
        provider: args.args,
    });
}

fn logout(In(args): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(SignOut {
        entity: args.agent,
        provider: args.args,
    });
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
            kind: PickKind::Effort,
        });
        return;
    }
    let options = models::effort_options(spec);
    match options.iter().find(|option| option.0 == args.args) {
        Some(option) => {
            commands.trigger(SetEffort {
                entity: args.agent,
                effort: Effort(option.1),
            });
        }
        None => {
            let names: Vec<&str> = options.iter().map(|option| option.0).collect();
            notices.write(Notice::error(
                args.agent,
                format!("{} takes: {}.", spec.display_name, names.join(", ")),
            ));
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
    let spec = connection.map(|connection| connection.spec);
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

/// Shows or changes the agent's approval [`Policy`].
fn approvals(
    In(args): In<CommandArgs>,
    mut policies: Query<&mut Policy>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(mut policy) = policies.get_mut(agent) else {
        return;
    };
    let mut words = args.args.split_whitespace();
    let first = words.next();
    let permission = match first {
        Some("allow") => Some(Permission::Allow),
        Some("ask") if words.clone().next().is_some() => Some(Permission::Ask),
        Some("deny") => Some(Permission::Deny),
        _ => None,
    };
    if let Some(permission) = permission {
        let Some(tool) = words.next() else {
            notices.write(Notice::error(
                agent,
                "Name the tool: /approvals allow shell git status*",
            ));
            return;
        };
        let subject = words.collect::<Vec<_>>().join(" ");
        let rule = Rule {
            tool: tool.to_owned(),
            subject: (!subject.is_empty()).then_some(subject),
            permission,
        };
        notices.write(Notice::info(agent, format!("Added: {}.", rule.label())));
        policy.rules.push(rule);
        return;
    }
    match first {
        None => {}
        Some("forget") => {
            let index = words
                .next()
                .and_then(|number| number.parse::<usize>().ok())
                .and_then(|number| number.checked_sub(1))
                .filter(|index| *index < policy.rules.len());
            match index {
                Some(index) => {
                    let rule = policy.rules.remove(index);
                    notices.write(Notice::info(agent, format!("Dropped: {}.", rule.label())));
                }
                None => {
                    notices.write(Notice::error(
                        agent,
                        "Name a rule by its number, as /approvals lists them.",
                    ));
                }
            }
            return;
        }
        Some(name) => match ApprovalMode::parse(name) {
            Some(mode) => {
                policy.mode = mode;
                notices.write(Notice::info(agent, format!("Approvals: {}.", mode.name())));
                return;
            }
            None => {
                notices.write(Notice::error(
                    agent,
                    format!(
                        "No mode `{name}`: auto runs every call, ask asks before any call that \
                         changes something, read-only refuses those."
                    ),
                ));
                return;
            }
        },
    }
    let mut lines = vec![format!(
        "Approvals: {} (auto, ask or read-only; calls that only read always run).",
        policy.mode.name()
    )];
    if policy.rules.is_empty() {
        lines.push("No rules. /approvals allow <tool> [subject] adds one.".to_owned());
    } else {
        lines.push("Rules, the last that matches a call decides:".to_owned());
        lines.extend(
            policy
                .rules
                .iter()
                .enumerate()
                .map(|(index, rule)| format!("{}. {}", index + 1, rule.label())),
        );
    }
    notices.write(Notice::info(agent, lines.join("\n")));
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

fn agents(
    In(args): In<CommandArgs>,
    roster: RosterQuery,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        picks.write(PickRequest {
            agent: args.agent,
            kind: PickKind::Agent,
        });
        return;
    }
    let entries = subagents::roster(&roster);
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

/// The agent's checkpoints, newest first, or a notice that it has none.
fn checkpoints(
    agent: Entity,
    agents: &Query<(&Conversation, &History)>,
    notices: &mut MessageWriter<Notice>,
) -> Option<Vec<Point>> {
    let points = agents
        .get(agent)
        .map(|(conversation, history)| rewind::points(conversation, history))
        .unwrap_or_default();
    if points.is_empty() {
        notices.write(Notice::info(
            agent,
            "No checkpoint yet: each model call makes one.",
        ));
        return None;
    }
    Some(points)
}

/// The checkpoint numbered `number` (1 is the newest), or a notice.
fn numbered<'a>(
    agent: Entity,
    points: &'a [Point],
    number: usize,
    command: &str,
    notices: &mut MessageWriter<Notice>,
) -> Option<&'a Point> {
    let point = number.checked_sub(1).and_then(|index| points.get(index));
    if point.is_none() {
        notices.write(Notice::error(
            agent,
            format!(
                "No checkpoint {number}: there are {}. /{command} lists them.",
                points.len()
            ),
        ));
    }
    point
}

fn rewind(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, &History)>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let words: Vec<&str> = args.args.split_whitespace().collect();
    if words == ["undo"] {
        commands.trigger(UndoRewind { entity: agent });
        return;
    }
    let mut files = true;
    let mut number = None;
    for word in &words {
        match (*word, word.parse::<usize>()) {
            ("chat", _) => files = false,
            (_, Ok(parsed)) if number.is_none() => number = Some(parsed),
            _ => {
                notices.write(Notice::error(
                    agent,
                    "Use /rewind, /rewind <number> [chat] or /rewind undo.",
                ));
                return;
            }
        }
    }
    let Some(points) = checkpoints(agent, &agents, &mut notices) else {
        return;
    };
    match number {
        None => {
            picks.write(PickRequest {
                agent,
                kind: PickKind::Rewind { files },
            });
        }
        Some(number) => {
            if let Some(point) = numbered(agent, &points, number, "rewind", &mut notices) {
                commands.trigger(Rewind {
                    entity: agent,
                    to: point.effect,
                    files,
                });
            }
        }
    }
}

fn fork(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, &History)>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    match args.args.as_str() {
        "" => {
            picks.write(PickRequest {
                agent,
                kind: PickKind::Fork,
            });
        }
        "now" => {
            commands.trigger(Fork {
                entity: agent,
                at: None,
            });
        }
        text => {
            let Ok(number) = text.parse::<usize>() else {
                notices.write(Notice::error(
                    agent,
                    "Use /fork, /fork <number> or /fork now.",
                ));
                return;
            };
            let Some(points) = checkpoints(agent, &agents, &mut notices) else {
                return;
            };
            if let Some(point) = numbered(agent, &points, number, "fork", &mut notices) {
                commands.trigger(Fork {
                    entity: agent,
                    at: Some(point.effect),
                });
            }
        }
    }
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
         Ctrl+V pastes the clipboard's image. /agents shows a subagent's work, and what is \
         typed then goes to it. /rewind goes back to an earlier \
         checkpoint, files included, and /fork tries another way in a new agent."
            .to_owned(),
    );
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

fn quit(In(_): In<CommandArgs>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
