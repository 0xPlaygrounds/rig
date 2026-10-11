//! Plugin corpus: 43. Fork and branch sessions (spec `43-fork-session.md`).
//!
//! `/fork [name]` copies the running session (every agent's log, so its
//! saved plugin state too, the saved resources, the images and the
//! listing cache) into a new session and switches to it. `/branch` picks
//! one of the shown agent's own messages and makes a new session whose
//! logs end just before it, so the conversation and every saved component
//! are as they were then; the message waits in the new session's input.
//! Each new session's [`FAMILY`] file names its parent and the fork point,
//! and `/tree` shows the family. Files on disk are never branched.

use std::fs;
use std::io;
use std::path::Path;

use rig_cassette::journal::{JournalStore, JsonlDirStore};
use rig_core::completion::Message;
use rig_ecs::WriteJournal;
use rig_ecs::journal::{SESSION, SessionStore, now_ms};
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_sessions::{Meta, SessionsPlugin, SwitchSession, list};
use rig_tools::attach::attached_file;
use rig_tools::fs::{read_json, write_json};
use rig_tools::shorten;
use rig_tui::{Focus, PickItem, PickRequest};
use serde::{Deserialize, Serialize};

/// The file of a session directory that says where the session came from.
pub const FAMILY: &str = "family.json";

/// Most characters of a message in the `/branch` picker.
const LABEL_CHARS: usize = 70;

/// How deep `/tree` follows parents, against a loop in edited files.
const DEPTH: usize = 64;

/// What the files are told after a fork.
const FILES: &str = "Files on disk are not branched: the working tree stays as it is.";

/// `/fork`, `/branch` and `/tree`.
#[derive(Default)]
pub struct ForkSessionPlugin;

impl Plugin for ForkSessionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionsPlugin>() {
            app.add_plugins(SessionsPlugin);
        }
        app.init_resource::<Requested>()
            .add_message::<PickRequest>()
            .add_command(
                "fork",
                "Copy this session into a new one and switch to it: /fork [name]",
                fork,
            )
            .add_command(
                "branch",
                "Pick a message of yours to branch a new session from, before it",
                branch,
            )
            .add_command("tree", "Show this session's forks and branches", tree)
            .add_observer(recall_input)
            .add_systems(
                Last,
                carry_out
                    .after(WriteJournal)
                    .run_if(|requested: Res<Requested>| requested.0.is_some()),
            );
    }
}

/// Where a session came from, in its [`FAMILY`] file.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Family {
    /// The session it was forked or branched from.
    pub parent: Option<String>,
    /// Where: `fork`, or the start of the message it was branched before.
    pub point: String,
    /// When, in milliseconds since the Unix epoch.
    pub created: u64,
    /// What goes in an agent's input at the session's next start, once.
    pub input: Option<Input>,
}

/// Text for an agent's input line.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Input {
    /// The agent's id.
    pub agent: String,
    /// The text.
    pub text: String,
}

/// A fork the commands asked for, made once the frame's records are on
/// disk.
#[derive(Resource, Default)]
struct Requested(Option<Request>);

enum Request {
    Fork { agent: Entity, name: Option<String> },
    Branch { agent: Entity, message: usize },
}

/// Where the logs of a branch end.
struct Cut {
    /// The log of the agent branched, cut before this record.
    agent: String,
    seq: u64,
    /// Every other log, cut before this time.
    t: u64,
}

/// The fields of a log line a cut reads.
#[derive(Deserialize)]
struct Line {
    seq: u64,
    t: u64,
    #[serde(rename = "type")]
    kind: String,
    message: Option<Message>,
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

/// Why nothing can be forked now, if so.
fn refusal(turns: &Query<(), With<TurnOf>>, paths: Option<&SessionPaths>) -> Option<String> {
    if !turns.is_empty() {
        return Some("A turn is running. Press Esc to stop it first.".to_owned());
    }
    if !paths.is_some_and(|paths| paths.is_saved()) {
        return Some("Nothing of this session is saved yet.".to_owned());
    }
    None
}

/// `/fork [name]`.
fn fork(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    mut requested: ResMut<Requested>,
    mut notices: MessageWriter<Notice>,
) {
    if let Some(why) = refusal(&turns, paths.as_deref()) {
        notices.write(Notice::error(args.agent, why));
        return;
    }
    let name = args.args.trim();
    requested.0 = Some(Request::Fork {
        agent: args.agent,
        name: (!name.is_empty()).then(|| name.to_owned()),
    });
}

/// `/branch` picks a message; `/branch <index>` branches before it.
fn branch(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    conversations: Query<&Conversation>,
    mut requested: ResMut<Requested>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if let Some(why) = refusal(&turns, paths.as_deref()) {
        notices.write(Notice::error(args.agent, why));
        return;
    }
    let typed = conversations.get(args.agent).map(typed).unwrap_or_default();
    let arg = args.args.trim();
    if arg == "--files" {
        notices.write(Notice::error(
            args.agent,
            "/branch --files needs a checkpoint plugin, and none is loaded.",
        ));
        return;
    }
    if arg.is_empty() {
        if typed.is_empty() {
            notices.write(Notice::info(
                args.agent,
                "No message of yours to branch from.",
            ));
            return;
        }
        let items: Vec<PickItem> = typed
            .iter()
            .rev()
            .map(|(at, text)| PickItem {
                label: shorten(&one_line(text), LABEL_CHARS),
                command: format!("branch {at}"),
            })
            .collect();
        picks.write(PickRequest {
            agent: args.agent,
            title: "Branch a new session before this message".to_owned(),
            items,
            selected: 0,
        });
        return;
    }
    match arg.parse::<usize>() {
        Ok(message) if typed.iter().any(|(at, _)| *at == message) => {
            requested.0 = Some(Request::Branch {
                agent: args.agent,
                message,
            });
        }
        _ => {
            notices.write(Notice::error(
                args.agent,
                format!("`{arg}` is none of your messages; /branch alone lists them."),
            ));
        }
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Makes the fork asked for, once this frame's records are on disk, and
/// switches to it.
fn carry_out(
    mut requested: ResMut<Requested>,
    store: Option<Res<SessionStore>>,
    paths: Option<Res<SessionPaths>>,
    agents: Query<(&AgentId, &Conversation)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(request) = requested.0.take() else {
        return;
    };
    let (Some(SessionStore(store)), Some(paths)) = (store.as_deref(), paths.as_deref()) else {
        return;
    };
    let (agent, made) = match request {
        Request::Fork { agent, name } => {
            let family = Family {
                parent: Some(session_id(paths)),
                point: "fork".to_owned(),
                created: now_ms(),
                input: None,
            };
            (agent, copy(&**store, paths, None, family, name))
        }
        Request::Branch { agent, message } => {
            let made = agents
                .get(agent)
                .map_err(|_| io::Error::other("the agent is gone"))
                .and_then(|(id, conversation)| {
                    let text = typed(conversation)
                        .into_iter()
                        .find_map(|(at, text)| (at == message).then_some(text))
                        .unwrap_or_default();
                    let cut = cut_before(&**store, id, conversation, message)?;
                    let family = Family {
                        parent: Some(session_id(paths)),
                        point: format!("before “{}”", shorten(&one_line(&text), 40)),
                        created: now_ms(),
                        input: Some(Input {
                            agent: id.0.clone(),
                            text,
                        }),
                    };
                    copy(&**store, paths, Some(cut), family, None)
                });
            (agent, made)
        }
    };
    match made {
        Ok(id) => {
            notices.write(Notice::info(agent, format!("Made session {id}. {FILES}")));
            commands.trigger(SwitchSession {
                session: Some(id.to_string()),
            });
        }
        Err(failure) => {
            notices.write(Notice::error(agent, format!("Could not fork: {failure}")));
        }
    }
}

/// The running session's id: its directory's name.
fn session_id(paths: &SessionDir) -> String {
    paths
        .path()
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The complete lines of the log of `agent` in `store`, with what a cut
/// reads of each.
fn lines(store: &dyn JournalStore, agent: &str) -> io::Result<Vec<(Vec<u8>, Line)>> {
    let bytes = store.read(agent)?;
    Ok(bytes
        .split_inclusive(|byte| *byte == b'\n')
        .filter(|line| line.ends_with(b"\n"))
        .filter_map(|line| Some((line.to_vec(), serde_json::from_slice(line).ok()?)))
        .collect())
}

/// Where the logs end for a branch before message `message` of the agent
/// `id`: the record that began that message. The log is folded as the
/// conversation was (a user message after a user message goes into it, a
/// retract takes the last out), and aligned with the conversation at the
/// end, since a restore after a summary starts the conversation later.
fn cut_before(
    store: &dyn JournalStore,
    id: &AgentId,
    conversation: &Conversation,
    message: usize,
) -> io::Result<Cut> {
    let mut starts: Vec<(u64, u64, bool)> = Vec::new();
    for (_, line) in lines(store, &id.0)? {
        match line.kind.as_str() {
            "message" => {
                let user = matches!(line.message, Some(Message::User { .. }));
                if !(user && starts.last().is_some_and(|(.., last)| *last)) {
                    starts.push((line.seq, line.t, user));
                }
            }
            "retract" => {
                starts.pop();
            }
            _ => {}
        }
    }
    let at = (message + starts.len()).checked_sub(conversation.messages().len());
    let (seq, t, _) = at
        .and_then(|at| starts.get(at))
        .copied()
        .ok_or_else(|| io::Error::other("that message is not in the session's log"))?;
    Ok(Cut {
        agent: id.0.clone(),
        seq,
        t,
    })
}

/// A new session id under `home`'s sessions, unused.
fn new_id(sessions: &Path) -> io::Result<SessionId> {
    let base = SessionId::generate();
    for n in 0..1000 {
        let id = if n == 0 {
            base.to_string()
        } else {
            format!("{base}-{n}")
        };
        if !sessions.join(&id).exists() {
            return id.parse().map_err(io::Error::other);
        }
    }
    Err(io::Error::other("no free session id"))
}

/// Copies the session in `store` and `from` into a new session beside it,
/// its logs cut as `cut` says, with `family` and the listing cache, named
/// `name` if given; the new session's id.
fn copy(
    store: &dyn JournalStore,
    from: &SessionDir,
    cut: Option<Cut>,
    family: Family,
    name: Option<String>,
) -> io::Result<SessionId> {
    let sessions = from
        .path()
        .parent()
        .ok_or_else(|| io::Error::other("the session has no sessions directory"))?;
    let id = new_id(sessions)?;
    let to = Home::from_env().session(&id);
    if to.path().parent() != Some(sessions) {
        return Err(io::Error::other("the session is not under RIG_HOME"));
    }
    fs::create_dir_all(to.path())?;
    let target = JsonlDirStore::new(to.path());
    for agent in store.agents()? {
        let kept: Vec<u8> = lines(store, &agent)?
            .into_iter()
            .filter(|(_, line)| match &cut {
                None => true,
                Some(cut) if cut.agent == agent => line.seq < cut.seq,
                Some(cut) => line.t < cut.t,
            })
            .flat_map(|(bytes, _)| bytes)
            .collect();
        // A log of only its header is an agent spawned after the cut.
        let records = kept.iter().filter(|byte| **byte == b'\n').count();
        if records > 1 || (agent == SESSION && records > 0) {
            target.append(&agent, &kept)?;
        }
    }
    copy_dir(&from.path().join("blobs"), &to.path().join("blobs"))?;
    copy_dir(&from.images(), &to.images())?;
    if from.directory().exists() {
        fs::copy(from.directory(), to.directory())?;
    }
    let mut meta: Meta = read_json(&from.meta()).unwrap_or_default();
    if name.is_some() {
        meta.title.name = name;
    }
    meta.updated = now_ms();
    write_json(&to.meta(), &meta)?;
    write_json(&to.path().join(FAMILY), &family)?;
    Ok(id)
}

/// Copies the files of `from`, if it exists, into `to`.
fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    let Ok(entries) = fs::read_dir(from) else {
        return Ok(());
    };
    fs::create_dir_all(to)?;
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            fs::copy(entry.path(), to.join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// Puts the message a branch was made before in its agent's input, once,
/// and shows that agent.
fn recall_input(
    restored: On<Restored>,
    ids: Query<&AgentId>,
    paths: Option<Res<SessionPaths>>,
    mut recalled: MessageWriter<Recalled>,
    mut commands: Commands,
) {
    let Some(paths) = paths else {
        return;
    };
    let file = paths.path().join(FAMILY);
    let Some(mut family) = read_json::<Family>(&file) else {
        return;
    };
    let Ok(id) = ids.get(restored.entity) else {
        return;
    };
    if family
        .input
        .as_ref()
        .is_none_or(|input| input.agent != id.0)
    {
        return;
    }
    if let Some(input) = family.input.take() {
        commands.trigger(Focus {
            entity: restored.entity,
        });
        recalled.write(Recalled {
            agent: restored.entity,
            text: input.text,
            why: None,
        });
        write_json(&file, &family).ok();
    }
}

/// `/tree`: the sessions forked and branched from this one's root, as a
/// picker that resumes the one chosen.
fn tree(
    In(args): In<CommandArgs>,
    paths: Option<Res<SessionPaths>>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(paths) = paths else {
        return;
    };
    let home = Home::from_env();
    let current = session_id(&paths);
    let family_of = |dir: &SessionDir| read_json::<Family>(&dir.path().join(FAMILY));
    // Every other saved session, with where it came from.
    let others: Vec<_> = list(&home, paths.path())
        .into_iter()
        .map(|entry| {
            let family = family_of(&home.session(&entry.id)).unwrap_or_default();
            (entry, family)
        })
        .collect();
    let parent_of = |id: &str| -> Option<String> {
        if id == current {
            return family_of(&paths).and_then(|family| family.parent);
        }
        others
            .iter()
            .find(|(entry, _)| entry.id.as_str() == id)
            .and_then(|(_, family)| family.parent.clone())
    };
    let mut root = current.clone();
    for _ in 0..DEPTH {
        match parent_of(&root) {
            Some(parent)
                if parent == current || others.iter().any(|(e, _)| e.id.as_str() == parent) =>
            {
                root = parent;
            }
            _ => break,
        }
    }
    let mut items = Vec::new();
    let mut stack = vec![(root, 0usize)];
    while let Some((id, depth)) = stack.pop() {
        if items.len() > 500 || depth > DEPTH {
            break;
        }
        let indent = "  ".repeat(depth);
        let (label, point) = if id == current {
            let point = family_of(&paths)
                .map(|family| family.point)
                .unwrap_or_default();
            ("● this session".to_owned(), point)
        } else {
            match others.iter().find(|(entry, _)| entry.id.as_str() == id) {
                Some((entry, family)) => (entry.label(), family.point.clone()),
                None => continue,
            }
        };
        let point = if point.is_empty() {
            String::new()
        } else {
            format!("  · {point}")
        };
        items.push(PickItem {
            label: format!("{indent}{label}{point}"),
            command: if id == current {
                "name".to_owned()
            } else {
                format!("resume {id}")
            },
        });
        let mut children: Vec<(u64, String)> = others
            .iter()
            .filter(|(_, family)| family.parent.as_deref() == Some(id.as_str()))
            .map(|(entry, family)| (family.created, entry.id.to_string()))
            .collect();
        if family_of(&paths)
            .and_then(|family| family.parent)
            .as_deref()
            == Some(id.as_str())
        {
            let created = family_of(&paths).map_or(0, |family| family.created);
            children.push((created, current.clone()));
        }
        // Oldest first once popped.
        children.sort_by(|a, b| b.cmp(a));
        stack.extend(children.into_iter().map(|(_, child)| (child, depth + 1)));
    }
    if items.len() < 2 {
        notices.write(Notice::info(
            args.agent,
            "This session has no forks or branches; /fork or /branch makes one.",
        ));
        return;
    }
    picks.write(PickRequest {
        agent: args.agent,
        title: "Sessions forked from the same one".to_owned(),
        items,
        selected: 0,
    });
}

#[cfg(test)]
mod tests;
