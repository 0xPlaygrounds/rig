//! Plugin corpus: 31. Checkpoints and rewind (spec `31-checkpoint-rewind.md`).
//!
//! Before each turn of an agent the user started, the working tree, with
//! its untracked files that are not ignored, is written as a commit under a
//! private ref, `refs/rig/checkpoints/<session>/<turn>`, through a
//! temporary index: the user's branch, index and stash stay as they are.
//! `/rewind` picks one of the user's messages and puts the files back as
//! they were before it, takes the conversation back to just before it,
//! and puts the message back in the input line.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_tui::{PickItem, PickRequest};
use serde_json::Value;

/// Where the checkpoints are kept, below the session's name.
const REFS: &str = "refs/rig/checkpoints";

/// The repository the checkpoints are of, and the session that names
/// their refs. Inserted by the plugin unless the app has one.
#[derive(Resource, Clone, Debug)]
pub struct CheckpointRepo {
    /// A directory inside the repository.
    pub root: PathBuf,
    /// The session, as in the refs' names.
    pub session: String,
}

/// The files before one of the user's messages.
#[derive(Reflect, Clone, Debug, Default)]
pub struct Checkpoint {
    /// The turn's number, as in its ref's name.
    pub turn: u64,
    /// The ref holding the files: an earlier turn's when nothing changed.
    pub reference: String,
    /// The files, as a git tree.
    pub tree: String,
    /// The position of the user's message in the conversation.
    pub message: usize,
    /// The message's text.
    pub text: String,
    /// The files the user changed since the turn before ended.
    pub user_edits: Vec<String>,
}

/// An agent's checkpoints, oldest first, kept with the session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved)]
pub struct Checkpoints {
    /// The checkpoints.
    pub list: Vec<Checkpoint>,
    /// The number of the next turn's ref.
    pub next_turn: u64,
    /// The files as the last turn left them, as a git tree.
    pub settled: Option<String>,
}

#[derive(Default)]
pub struct CheckpointRewindPlugin;

impl Plugin for CheckpointRewindPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<CheckpointRepo>() {
            let paths = app.world().get_resource::<SessionPaths>();
            let session = paths
                .and_then(|paths| paths.path().file_name())
                .map_or_else(|| "local".to_owned(), |name| name.to_string_lossy().into());
            let root = std::env::current_dir().unwrap_or_default();
            app.insert_resource(CheckpointRepo { root, session });
        }
        app.register_required_components::<Agent, Checkpoints>()
            .add_message::<PickRequest>()
            .add_systems(Startup, prune)
            .add_observer(checkpoint)
            .add_observer(settle)
            .add_command(
                "rewind",
                "Rewind files and conversation to before a message [--files-only|--chat-only]",
                rewind,
            );
    }
}

/// Snapshots the files as a turn of an agent the user started begins,
/// unless its message has a checkpoint already (a retried turn).
fn checkpoint(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    mut agents: Query<(&Conversation, &mut Checkpoints), Without<SpawnedBy>>,
    repo: Res<CheckpointRepo>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    let Ok((conversation, mut checkpoints)) = agents.get_mut(agent) else {
        return;
    };
    let messages = conversation.messages();
    let Some((message, text)) = messages
        .len()
        .checked_sub(1)
        .and_then(|last| Some((last, user_text(messages.get(last)?)?)))
    else {
        return;
    };
    if checkpoints
        .list
        .last()
        .is_some_and(|last| last.message == message)
    {
        return;
    }
    // Synchronous: the turn's tool calls must not run before it is taken.
    let Ok(tree) = working_tree(&repo.root) else {
        return;
    };
    let user_edits = checkpoints
        .settled
        .as_deref()
        .map(|settled| changed(&repo.root, settled, &tree))
        .unwrap_or_default();
    let number = checkpoints.next_turn.max(1);
    let reference = match checkpoints.list.last() {
        Some(last) if last.tree == tree => Ok(last.reference.clone()),
        _ => store(&repo, number, &tree),
    };
    match reference {
        Ok(reference) => {
            checkpoints.next_turn = number + 1;
            checkpoints.list.push(Checkpoint {
                turn: number,
                reference,
                tree,
                message,
                text,
                user_edits,
            });
        }
        Err(failure) => {
            notices.write(Notice::error(agent, format!("No checkpoint: {failure}")));
        }
    }
}

/// Notes the files as each turn left them, so edits made between turns
/// are told apart.
fn settle(
    ended: On<TurnEnded>,
    mut agents: Query<&mut Checkpoints, Without<SpawnedBy>>,
    repo: Res<CheckpointRepo>,
) {
    if let Ok(mut checkpoints) = agents.get_mut(ended.entity)
        && let Ok(tree) = working_tree(&repo.root)
    {
        checkpoints.settled = Some(tree);
    }
}

/// What `/rewind` was asked.
#[derive(Default)]
struct Asked {
    files: bool,
    chat: bool,
    confirmed: bool,
    turn: Option<u64>,
    flags: String,
}

fn parse(args: &str) -> Result<Asked, String> {
    let mut asked = Asked {
        files: true,
        chat: true,
        ..Asked::default()
    };
    for word in args.split_whitespace() {
        match word {
            "--files-only" => asked.chat = false,
            "--chat-only" => asked.files = false,
            "--confirm" => asked.confirmed = true,
            number => {
                let turn = number
                    .parse()
                    .map_err(|_| format!("Not a turn: `{number}`."))?;
                asked.turn = Some(turn);
            }
        }
    }
    if !asked.files && !asked.chat {
        return Err("--files-only and --chat-only exclude each other.".to_owned());
    }
    let flags = [
        (!asked.chat, "--files-only "),
        (!asked.files, "--chat-only "),
    ];
    asked.flags = flags
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, flag)| *flag)
        .collect();
    Ok(asked)
}

/// `/rewind [--files-only|--chat-only] [turn]`: without a turn, a picker of
/// the user's messages with checkpoints.
fn rewind(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, &mut Checkpoints, Option<&Condensed>)>,
    busy: Query<(), With<ActiveTurn>>,
    repo: Res<CheckpointRepo>,
    mut notices: MessageWriter<Notice>,
    mut picks: MessageWriter<PickRequest>,
    mut recalled: MessageWriter<Recalled>,
    mut commands: Commands,
) {
    let agent = args.agent;
    if busy.contains(agent) {
        notices.write(Notice::error(agent, "A turn is running; stop it first."));
        return;
    }
    let asked = match parse(&args.args) {
        Ok(asked) => asked,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let Ok((mut conversation, mut checkpoints, condensed)) = agents.get_mut(agent) else {
        return;
    };
    let Some(turn) = asked.turn else {
        if checkpoints.list.is_empty() {
            notices.write(Notice::error(agent, "No checkpoints yet."));
            return;
        }
        let items = checkpoints.list.iter().map(|checkpoint| PickItem {
            label: format!("{}. {}", checkpoint.turn, first_line(&checkpoint.text)),
            command: format!("rewind {}{}", asked.flags, checkpoint.turn),
        });
        let items: Vec<PickItem> = items.collect();
        let selected = items.len().saturating_sub(1);
        picks.write(PickRequest {
            agent,
            title: "Rewind to before".to_owned(),
            items,
            selected,
        });
        return;
    };
    let Some(at) = checkpoints.list.iter().position(|kept| kept.turn == turn) else {
        notices.write(Notice::error(agent, format!("No checkpoint {turn}.")));
        return;
    };
    let Some(target) = checkpoints.list.get(at).cloned() else {
        return;
    };
    if asked.files {
        let Ok(current) = working_tree(&repo.root) else {
            notices.write(Notice::error(
                agent,
                "The files are not in a git repository.",
            ));
            return;
        };
        let mut edits: BTreeSet<String> = checkpoints
            .list
            .iter()
            .skip(at + 1)
            .flat_map(|later| later.user_edits.iter().cloned())
            .collect();
        if let Some(settled) = &checkpoints.settled {
            edits.extend(changed(&repo.root, settled, &current));
        }
        if !edits.is_empty() && !asked.confirmed {
            let edits: Vec<String> = edits.into_iter().collect();
            let why = format!("This also undoes your edits to {}.", edits.join(", "));
            picks.write(PickRequest {
                agent,
                title: why.clone(),
                items: vec![PickItem {
                    label: format!("Rewind to before turn {turn} anyway"),
                    command: format!("rewind {}--confirm {turn}", asked.flags),
                }],
                selected: 0,
            });
            notices.write(Notice::info(agent, format!("{why} Confirm to rewind.")));
            return;
        }
        if let Err(failure) = restore(&repo.root, &current, &target.tree) {
            notices.write(Notice::error(agent, format!("Could not rewind: {failure}")));
            return;
        }
        checkpoints.settled = Some(target.tree.clone());
    }
    if asked.chat {
        let Some(message) = locate(conversation.messages(), &target) else {
            notices.write(Notice::error(
                agent,
                "The message is no longer in the conversation.",
            ));
            return;
        };
        let Some(cut) = truncated(&conversation, message) else {
            notices.write(Notice::error(agent, "Could not cut the conversation."));
            return;
        };
        *conversation = cut;
        if condensed.is_some_and(|condensed| condensed.upto > message) {
            commands.entity(agent).remove::<Condensed>();
        }
        let dropped = checkpoints.list.split_off(at);
        let kept: BTreeSet<&str> = checkpoints
            .list
            .iter()
            .map(|kept| kept.reference.as_str())
            .collect();
        let refs: BTreeSet<&str> = dropped.iter().map(|gone| gone.reference.as_str()).collect();
        for reference in refs.difference(&kept) {
            git(&repo.root, None, &["update-ref", "-d", reference]).ok();
        }
        recalled.write(Recalled {
            agent,
            text: target.text.clone(),
            why: None,
        });
    }
    let what = match (asked.files, asked.chat) {
        (true, false) => "the files",
        (false, true) => "the conversation",
        _ => "the files and the conversation",
    };
    notices.write(Notice::info(
        agent,
        format!("Rewound {what} to before turn {turn}."),
    ));
}

/// Deletes the checkpoints of sessions that no longer exist.
fn prune(repo: Res<CheckpointRepo>, paths: Option<Res<SessionPaths>>) {
    let Some(sessions) = paths.as_ref().and_then(|paths| paths.path().parent()) else {
        return;
    };
    let Ok(listed) = git(
        &repo.root,
        None,
        &["for-each-ref", "--format=%(refname)", REFS],
    ) else {
        return;
    };
    for reference in listed.lines() {
        let session = reference
            .strip_prefix(REFS)
            .and_then(|rest| rest.trim_start_matches('/').split('/').next());
        if let Some(session) = session
            && session != repo.session
            && !sessions.join(session).is_dir()
        {
            git(&repo.root, None, &["update-ref", "-d", reference]).ok();
        }
    }
}

/// The text of a user's message, without tool results.
fn user_text(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };
    let texts: Vec<&str> = content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    (!texts.is_empty()).then(|| texts.join("\n"))
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

/// Where the checkpoint's message is now: where it was, else the last user
/// message with its text (a restored session starts after its summary).
fn locate(messages: &[Message], checkpoint: &Checkpoint) -> Option<usize> {
    let matches = |message: &Message| {
        user_text(message).is_some_and(|text| text.starts_with(&checkpoint.text))
    };
    if messages.get(checkpoint.message).is_some_and(matches) {
        return Some(checkpoint.message);
    }
    messages.iter().rposition(matches)
}

/// `conversation` with its first `len` messages only. The kernel has no
/// public way to take messages out, so this rebuilds it through its serde
/// form; the session log still holds the messages taken out.
fn truncated(conversation: &Conversation, len: usize) -> Option<Conversation> {
    let mut value = serde_json::to_value(conversation).ok()?;
    let object = value.as_object_mut()?;
    if let Some(Value::Array(messages)) = object.get_mut("messages") {
        messages.truncate(len);
    }
    if let Some(Value::Array(origins)) = object.get_mut("origins") {
        let len = u64::try_from(len).unwrap_or(u64::MAX);
        origins.retain(|origin| {
            origin
                .get(0)
                .and_then(Value::as_u64)
                .is_some_and(|at| at < len)
        });
    }
    object.insert("halted".to_owned(), Value::Null);
    serde_json::from_value(value).ok()
}

/// Runs git in `root`, with `index` as its index when given; its output.
fn git(root: &Path, index: Option<&Path>, args: &[&str]) -> Result<String, String> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_AUTHOR_NAME", "rig")
        .env("GIT_AUTHOR_EMAIL", "rig@localhost")
        .env("GIT_COMMITTER_NAME", "rig")
        .env("GIT_COMMITTER_EMAIL", "rig@localhost");
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    let output = command.output().map_err(|failure| failure.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .to_owned())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
    }
}

/// Runs `work` with a fresh temporary index in the repository's git
/// directory, removed afterwards.
fn with_index<T>(root: &Path, work: impl FnOnce(&Path) -> Result<T, String>) -> Result<T, String> {
    let name = format!("rig-checkpoint-{}.index", std::process::id());
    let index = root.join(git(root, None, &["rev-parse", "--git-path", &name])?);
    fs::remove_file(&index).ok();
    let done = work(&index);
    fs::remove_file(&index).ok();
    done
}

/// The working tree as a git tree: the tracked files and the untracked
/// ones that are not ignored.
fn working_tree(root: &Path) -> Result<String, String> {
    let head = git(root, None, &["rev-parse", "--verify", "-q", "HEAD"]).is_ok();
    with_index(root, |index| {
        if head {
            git(root, Some(index), &["read-tree", "HEAD"])?;
        }
        git(root, Some(index), &["add", "-A"])?;
        git(root, Some(index), &["write-tree"])
    })
}

/// The paths that differ between the trees `from` and `to`, or that only
/// one has with `filter`.
fn diff(root: &Path, from: &str, to: &str, filter: Option<&str>) -> Vec<String> {
    let filter = filter.map(|filter| format!("--diff-filter={filter}"));
    let mut args = vec!["diff-tree", "-r", "--name-only", "--no-renames"];
    args.extend(filter.as_deref());
    args.extend([from, to]);
    git(root, None, &args)
        .map(|listed| listed.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn changed(root: &Path, from: &str, to: &str) -> Vec<String> {
    diff(root, from, to, None)
}

/// Keeps `tree` as a commit under the session's ref for turn `number`.
fn store(repo: &CheckpointRepo, number: u64, tree: &str) -> Result<String, String> {
    let message = format!("rig checkpoint: session {}, turn {number}", repo.session);
    let commit = git(&repo.root, None, &["commit-tree", tree, "-m", &message])?;
    let reference = format!("{REFS}/{}/{number}", repo.session);
    git(&repo.root, None, &["update-ref", &reference, &commit])?;
    Ok(reference)
}

/// Makes the working tree, now `current`, hold `target`'s files: removes
/// the files `target` lacks and writes those that differ.
fn restore(root: &Path, current: &str, target: &str) -> Result<(), String> {
    let top = PathBuf::from(git(root, None, &["rev-parse", "--show-toplevel"])?);
    for added in diff(root, target, current, Some("A")) {
        fs::remove_file(top.join(added)).map_err(|failure| failure.to_string())?;
    }
    let differing = diff(root, target, current, Some("DMT"));
    if differing.is_empty() {
        return Ok(());
    }
    with_index(&top, |index| {
        git(&top, Some(index), &["read-tree", target])?;
        let mut args = vec!["checkout-index", "-f", "--"];
        args.extend(differing.iter().map(String::as_str));
        git(&top, Some(index), &args).map(drop)
    })
}

#[cfg(test)]
mod tests;
