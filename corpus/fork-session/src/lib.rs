//! Plugin corpus: 43. Fork and branch sessions (spec `43-fork-session.md`).
//!
//! `/fork [name]` copies the running session into a new one with the
//! harness's [`ForkSession`] and switches to it. `/branch` picks one of the
//! shown agent's own messages and forks the session as of just before it
//! (the kernel cuts every log there, so saved plugin state comes back as it
//! was then); the new session starts with that message in the input line.
//! The harness records each fork's parent and fork point ([`ForkedFrom`]),
//! and `/tree` shows the family. Files on disk are never branched, except
//! that `/branch --files` (with the `checkpoints` feature and spec 31's
//! plugin) puts them back as that plugin's checkpoint before the message
//! kept them.

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_sessions::{ResumeLabel, SessionTitle, SessionsPlugin, list};
use rig_tools::attach::attached_file;
use rig_tools::fs::{read_json, write_json};
use rig_tools::shorten;
use rig_tui::{Focus, TuiInput};
use serde::{Deserialize, Serialize};
use std::fs;

/// The file a fork made by this plugin starts with: the name and input
/// its first start takes, then removes.
pub const SEED: &str = "fork-seed.json";

/// Most characters of a message in the `/branch` picker.
const LABEL_CHARS: usize = 70;

/// How deep `/tree` follows a family, against a loop in edited files.
const DEPTH: usize = 64;

/// What the user is told after a fork.
const FILES: &str = "Files on disk are not branched: the working tree stays as it is.";

/// `/fork`, `/branch` and `/tree`.
#[derive(Default)]
pub struct ForkSessionPlugin;

impl Plugin for ForkSessionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionsPlugin>() {
            app.add_plugins(SessionsPlugin);
        }
        app.init_resource::<Pending>()
            .add_message::<SessionForked>()
            .add_command(
                "fork",
                "Copy this session into a new one and switch to it: /fork [name]",
                fork,
            )
            .add_command(
                "branch",
                "Pick a message of yours to start a new session before it [--files]",
                branch,
            )
            .add_command("tree", "Show this session's forks and branches", tree)
            .add_systems(Startup, arrive)
            .add_systems(Update, forked);
    }
}

/// What a fork's first start takes, in its [`SEED`] file.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(default)]
pub struct Seed {
    /// The name `/fork` gave it.
    pub name: Option<String>,
    /// What goes in an agent's input line: its id and the text.
    pub input: Option<(String, String)>,
}

/// The fork asked for, until the harness wrote it.
#[derive(Resource, Default)]
struct Pending(Option<Asked>);

struct Asked {
    agent: Entity,
    before: Option<(AgentId, u64)>,
    seed: Seed,
    /// The command that puts the files back, for `/branch --files`.
    files: Option<String>,
}

/// The messages the user typed in `conversation`, by index, as text.
fn typed(conversation: &Conversation) -> Vec<(usize, String)> {
    let mut typed = Vec::new();
    for (at, message) in conversation.messages().iter().enumerate() {
        let Message::User { content } = message else {
            continue;
        };
        let texts: Vec<&str> = content
            .iter()
            .enumerate()
            .filter_map(|(item, content)| match content {
                UserContent::Text(text)
                    if !text.text.trim().is_empty()
                        && conversation.origin(at, item).is_none()
                        && attached_file(&text.text).is_none() =>
                {
                    Some(text.text.as_str())
                }
                _ => None,
            })
            .collect();
        if !texts.is_empty() {
            typed.push((at, texts.join("\n\n")));
        }
    }
    typed
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Why nothing can be forked now, if so.
fn forkable(
    agent: Entity,
    turns: &Query<(), With<TurnOf>>,
    paths: Option<&SessionPaths>,
) -> Result<(), Notice> {
    if !turns.is_empty() {
        return Err(Notice::turn_running(agent));
    }
    match paths {
        Some(paths) if paths.is_saved() => Ok(()),
        _ => Err(Notice::error(
            agent,
            "Nothing of this session is saved yet.",
        )),
    }
}

/// `/fork [name]`.
fn fork(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    mut pending: ResMut<Pending>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if let Err(refused) = forkable(args.agent, &turns, paths.as_deref()) {
        notices.write(refused);
        return;
    }
    let name = args.args.trim();
    pending.0 = Some(Asked {
        agent: args.agent,
        before: None,
        seed: Seed {
            name: (!name.is_empty()).then(|| one_line(name)),
            input: None,
        },
        files: None,
    });
    commands.trigger(ForkSession::default());
}

/// `/branch [--files]`: a picker of the agent's typed messages.
fn branch(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    conversations: Query<&Conversation>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if let Err(refused) = forkable(agent, &turns, paths.as_deref()) {
        notices.write(refused);
        return;
    }
    let files = match args.args.trim() {
        "" => false,
        "--files" if cfg!(feature = "checkpoints") => true,
        "--files" => {
            let why = "/branch --files needs this plugin built with its `checkpoints` feature.";
            notices.write(Notice::error(agent, why));
            return;
        }
        other => {
            notices.write(Notice::error(agent, format!("Unknown option `{other}`.")));
            return;
        }
    };
    let typed = conversations.get(agent).map(typed).unwrap_or_default();
    if typed.is_empty() {
        notices.write(Notice::info(agent, "No message of yours to branch from."));
        return;
    }
    let picker = typed.iter().fold(
        Picker::new("Branch a new session before this message").for_agent(agent),
        |picker, (_, text)| picker.item(shorten(&one_line(text), LABEL_CHARS), ""),
    );
    let selected = picker.items.len().saturating_sub(1);
    commands.spawn(Picker { selected, ..picker }).observe(
        move |picked: On<Picked>,
              agents: Query<(&AgentId, &Conversation)>,
              snapshots: Snapshots<'_, '_>,
              mut pending: ResMut<Pending>,
              mut commands: Commands,
              mut notices: MessageWriter<Notice>| {
            let Some(Chosen::Item(index)) = picked.chosen else {
                return;
            };
            let (Some((at, text)), Ok((id, conversation))) = (typed.get(index), agents.get(agent))
            else {
                return;
            };
            let Some(seq) = conversation.seq(*at) else {
                notices.write(Notice::error(agent, "That message is not saved yet."));
                return;
            };
            let files = match files.then(|| restore_files(&snapshots, agent, seq)) {
                Some(Err(why)) => {
                    notices.write(Notice::error(agent, why));
                    return;
                }
                Some(Ok(line)) => Some(line),
                None => None,
            };
            let before = Some((id.clone(), seq));
            pending.0 = Some(Asked {
                agent,
                before: before.clone(),
                seed: Seed {
                    name: None,
                    input: Some((id.0.clone(), text.clone())),
                },
                files,
            });
            commands.trigger(ForkSession { before });
        },
    );
}

/// The checkpoints spec 31's plugin keeps with each agent.
#[cfg(feature = "checkpoints")]
type Snapshots<'w, 's> = Query<'w, 's, &'static corpus_checkpoint_rewind::Checkpoints>;
#[cfg(not(feature = "checkpoints"))]
type Snapshots<'w, 's> = ();

/// The command of spec 31's plugin that puts the files back as they were
/// before the message logged as `seq`.
#[cfg(feature = "checkpoints")]
fn restore_files(snapshots: &Snapshots<'_, '_>, agent: Entity, seq: u64) -> Result<String, String> {
    snapshots
        .get(agent)
        .ok()
        .and_then(|kept| kept.list.iter().find(|kept| kept.seq == Some(seq)))
        .map(|checkpoint| format!("rewind --files-only {}", checkpoint.turn))
        .ok_or_else(|| "No checkpoint of the files before that message.".to_owned())
}

#[cfg(not(feature = "checkpoints"))]
fn restore_files(_: &Snapshots<'_, '_>, _: Entity, _: u64) -> Result<String, String> {
    Err("/branch --files needs spec 31's checkpoints.".to_owned())
}

/// Seeds each fork this plugin asked for, then switches to it.
fn forked(
    mut forks: MessageReader<SessionForked>,
    mut pending: ResMut<Pending>,
    home: Res<RigHome>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for fork in forks.read() {
        let Some(asked) = pending.0.take_if(|asked| asked.before == fork.before) else {
            continue;
        };
        let file = home.0.session(&fork.to).path().join(SEED);
        if asked.seed != Seed::default()
            && let Err(failure) = write_json(&file, &asked.seed)
        {
            let why = format!("Could not seed session {}: {failure}", fork.to);
            notices.write(Notice::error(asked.agent, why));
            continue;
        }
        if let Some(line) = asked.files {
            commands.trigger(RunCommand::new(asked.agent, line));
        }
        let made = format!("Made session {}. {FILES}", fork.to);
        notices.write(Notice::info(asked.agent, made));
        commands.trigger(Exit(ExitReason::Switch(Some(fork.to.clone()))));
    }
}

/// A fork's start: says where it came from in `/resume`, and the first
/// time takes its seed: the name `/fork` gave it, and the message it was
/// branched before in the input line, showing its agent.
fn arrive(
    paths: Option<Res<SessionPaths>>,
    agents: PrimaryQuery,
    mut title: ResMut<SessionTitle>,
    input: Option<ResMut<TuiInput>>,
    mut commands: Commands,
) {
    let Some(paths) = paths else {
        return;
    };
    let Some(parent) = ForkedFrom::of(&paths) else {
        return;
    };
    let how = if parent.before.is_some() {
        "branch"
    } else {
        "fork"
    };
    commands.spawn(ResumeLabel::new(
        "family",
        format!("{how} of {}", parent.session),
    ));
    let file = paths.path().join(SEED);
    let Some(seed) = read_json::<Seed>(&file) else {
        return;
    };
    // Once: a later start of the session leaves its input alone.
    fs::remove_file(&file).ok();
    if seed.name.is_some() {
        title.name = seed.name;
    }
    if let Some((agent, text)) = seed.input {
        let agent = agents.iter().find(|(_, id, _)| id.0 == agent);
        if let Some(entity) = agent.map(|(agent, ..)| agent).or_else(|| primary(&agents)) {
            commands.trigger(Focus { entity });
        }
        if let Some(mut input) = input {
            input.set(text);
        }
    }
}

/// A session of the family: its id, its line, and how it was forked from
/// which session.
struct Node {
    id: String,
    label: String,
    parent: Option<ForkedFrom>,
}

/// `/tree`: the sessions forked and branched from this one's root, as a
/// picker that resumes the one chosen.
fn tree(
    In(args): In<CommandArgs>,
    paths: Option<Res<SessionPaths>>,
    home: Res<RigHome>,
    title: Res<SessionTitle>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(paths) = paths else {
        return;
    };
    let Some(current) = paths.id().map(|id| id.to_string()) else {
        return;
    };
    let this = title.name.clone().unwrap_or_else(|| title.typed.clone());
    let mut nodes: Vec<Node> = list(&home.0, paths.path())
        .into_iter()
        .map(|entry| Node {
            parent: ForkedFrom::of(&home.0.session(&entry.id)),
            id: entry.id.to_string(),
            label: entry.label(),
        })
        .collect();
    nodes.push(Node {
        id: current.clone(),
        label: format!("● this session  {this}"),
        parent: ForkedFrom::of(&paths),
    });
    let parent_of = |id: &str| -> Option<String> {
        let node = nodes.iter().find(|node| node.id == id)?;
        let parent = node.parent.as_ref()?.session.to_string();
        nodes.iter().any(|node| node.id == parent).then_some(parent)
    };
    let mut root = current.clone();
    for _ in 0..DEPTH {
        match parent_of(&root) {
            Some(parent) => root = parent,
            None => break,
        }
    }
    let mut picker = Picker::new("This session's family").for_agent(args.agent);
    let mut stack = vec![(root, 0usize)];
    while let Some((id, depth)) = stack.pop() {
        let Some(node) = nodes.iter().find(|node| node.id == id) else {
            continue;
        };
        if depth > DEPTH || picker.items.len() > 500 {
            break;
        }
        let how = match &node.parent {
            Some(parent) if parent.before.is_some() => "  · branch",
            Some(_) => "  · fork",
            None => "",
        };
        let command = if id == current {
            picker.selected = picker.items.len();
            String::new()
        } else {
            format!("resume {id}")
        };
        let line = format!("{}{}{how}", "  ".repeat(depth), node.label);
        picker = picker.item(line, command);
        // Session ids start with their creation time: oldest first once
        // popped.
        let mut children: Vec<&str> = nodes
            .iter()
            .filter(|child| parent_of(&child.id).as_deref() == Some(id.as_str()))
            .map(|child| child.id.as_str())
            .collect();
        children.sort_by(|a, b| b.cmp(a));
        stack.extend(
            children
                .into_iter()
                .map(|child| (child.to_owned(), depth + 1)),
        );
    }
    if picker.items.len() < 2 {
        let none = "This session has no forks or branches; /fork or /branch makes one.";
        notices.write(Notice::info(args.agent, none));
        return;
    }
    commands.spawn(picker);
}

#[cfg(test)]
mod tests;
