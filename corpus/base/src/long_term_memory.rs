//! Long-term memory: facts the agent keeps across sessions, as markdown
//! lines a person can edit, in `RIG_HOME/memory/user.md` (the user's) and
//! `<project>/.rig/memory.md` (the project's).
//!
//! A memory is a `- fact` line; `- [pinned] fact` is recalled on every
//! turn. Its id is its scope's letter and a hash of its normalized text, so
//! ids survive hand edits and the same fact is never stored twice. The
//! files are read at startup and again whenever they change on disk
//! (`FileWatch`). The `remember` and `forget` tools change them; at the
//! first model request of each turn, the memories that share the most
//! words with the user's message, and every pinned one, are added to that
//! message in each request of the turn, after a cache point of the
//! request, and never to the saved conversation. `/memory` lists them,
//! `/memory edit` opens files in `$VISUAL` or `$EDITOR` in the terminal view, and
//! `/memory off` and `/memory on` stop and restart the recall for the
//! session.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CachePoint;
use rig_harness::watch::WatchPlugin;
use rig_tui::{FRONT, RunInTerminal, TerminalExit};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// How many memories a turn recalls, the pinned ones included.
const MOST: usize = 10;

/// The marker of a memory recalled on every turn.
const PINNED: &str = "[pinned]";

/// Words too common to tell memories apart.
const COMMON: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "you", "how", "what", "are", "use", "can", "does",
    "from", "not", "have", "has", "was", "his", "her", "its", "our", "your", "into", "about",
];

/// The memory files, the tools, the recall and `/memory`.
#[derive(Default)]
pub struct LongTermMemoryPlugin;

impl Plugin for LongTermMemoryPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<WatchPlugin>() {
            app.add_plugins(WatchPlugin);
        }
        app.init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .init_resource::<MemoryOff>()
            .add_open_tool(
                "remember",
                "Remember a fact across sessions; returns its id (an existing one for a fact \
                 already remembered). Use it when the user asks you to remember something \
                 (\"remember that …\"), or states a lasting preference or project fact, with it \
                 as one short fact: scope `project` for this repository, `user` for the user \
                 everywhere.",
                ToolOptions::default(),
                remember,
            )
            .add_open_tool(
                "forget",
                "Forget a remembered fact by its id. Memories recalled for a message come in a \
                 <memory> block after it; forget one when it is wrong or the user asks.",
                ToolOptions::default(),
                forget,
            )
            .add_command(
                "memory",
                "List what the agent remembers; `edit` to edit, `off` or `on` for the session",
                memory,
            )
            .add_systems(Startup, open)
            .add_systems(Update, (reload, edited).before(AgentLoop))
            .configure_sets(AgentPass, MemorySystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, recall.in_set(MemorySystems));
    }

    fn finish(&self, app: &mut App) {
        let shown = app.runs_front(FRONT);
        app.insert_resource(TerminalView(shown));
    }
}

/// The plugin's Prepare system, which adds the recalled memories to each
/// turn's requests: after the default prompt went in, before compaction
/// measures what is sent.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemorySystems;

/// Whether the terminal view runs, which `/memory edit` needs.
#[derive(Resource)]
struct TerminalView(bool);

/// Whether memories stay out of requests, for the session (`/memory off`).
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
pub struct MemoryOff(pub bool);

/// The memory files and what they hold.
#[derive(Resource, Debug)]
pub struct Memories {
    /// The user's file, for every project.
    pub user: PathBuf,
    /// The project's file.
    pub project: PathBuf,
    /// Every memory, the project's first.
    known: Vec<Memory>,
}

impl Memories {
    /// The files of the rig home `home` and the project `root`, read.
    pub fn new(home: &Home, root: &Path) -> Self {
        let mut memories = Self {
            user: home.root().join("memory").join("user.md"),
            project: root.join(".rig").join("memory.md"),
            known: Vec::new(),
        };
        memories.read();
        memories
    }

    fn path(&self, scope: Scope) -> &Path {
        match scope {
            Scope::User => &self.user,
            Scope::Project => &self.project,
        }
    }

    /// Reads the files again; a missing one holds nothing.
    fn read(&mut self) {
        self.known = [Scope::Project, Scope::User]
            .into_iter()
            .flat_map(|scope| {
                let text = fs::read_to_string(self.path(scope)).unwrap_or_default();
                text.lines()
                    .filter_map(|line| Memory::parse(scope, line))
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    /// Stores `fact` in `scope`, or finds it stored already; its id, and
    /// whether it was new.
    fn remember(&mut self, fact: &str, scope: Scope) -> std::io::Result<(String, bool)> {
        let fact = fact.split_whitespace().collect::<Vec<_>>().join(" ");
        let key = normalize(&fact);
        if key.is_empty() {
            return Err(std::io::Error::other("the fact is empty"));
        }
        // What is on disk now, in case it changed since the watch looked.
        self.read();
        if let Some(known) = self.known.iter().find(|memory| memory.key == key) {
            return Ok((known.id.clone(), false));
        }
        let path = self.path(scope);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let fresh = !path.exists();
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        if fresh {
            writeln!(
                file,
                "# Memories\n\nOne fact per `- ` line; `- {PINNED} fact` is recalled on every turn.\n"
            )?;
        }
        writeln!(file, "- {fact}")?;
        self.read();
        Ok((id(scope, &key), true))
    }

    /// Takes the memory `id` out of its file; the fact it was.
    fn forget(&mut self, id: &str) -> std::io::Result<Option<String>> {
        let Some(scope) = Scope::of(id) else {
            return Ok(None);
        };
        let path = self.path(scope);
        let text = fs::read_to_string(path).unwrap_or_default();
        let mut forgotten = None;
        let kept: Vec<&str> = text
            .lines()
            .filter(|line| match Memory::parse(scope, line) {
                Some(memory) if memory.id == id => {
                    forgotten = Some(memory.fact);
                    false
                }
                _ => true,
            })
            .collect();
        if forgotten.is_some() {
            fs::write(path, kept.join("\n") + "\n")?;
        }
        self.read();
        Ok(forgotten)
    }

    /// The memories to recall for `message`: every pinned one, then those
    /// sharing the most words with it, up to [`MOST`] in all.
    fn relevant(&self, message: &str) -> Vec<&Memory> {
        let asked = words(message);
        let (pinned, others): (Vec<&Memory>, Vec<&Memory>) =
            self.known.iter().partition(|memory| memory.pinned);
        let mut scored: Vec<(usize, &Memory)> = others
            .into_iter()
            .map(|memory| (words(&memory.fact).intersection(&asked).count(), memory))
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        let room = MOST.saturating_sub(pinned.len());
        pinned
            .into_iter()
            .chain(scored.into_iter().take(room).map(|(_, memory)| memory))
            .collect()
    }

    /// The list `/memory` shows.
    fn list(&self) -> Vec<String> {
        self.known.iter().map(Memory::line).collect()
    }
}

/// Whose a memory is.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Scope {
    /// This project's.
    Project,
    /// The user's, in every project.
    User,
}

impl Scope {
    fn letter(self) -> char {
        match self {
            Self::Project => 'p',
            Self::User => 'u',
        }
    }

    /// The scope of a memory's `id`.
    fn of(id: &str) -> Option<Self> {
        match id.chars().next()? {
            'p' => Some(Self::Project),
            'u' => Some(Self::User),
            _ => None,
        }
    }
}

/// One line of a memory file.
#[derive(Clone, Debug)]
struct Memory {
    id: String,
    fact: String,
    /// The fact normalized, which ids and duplicates go by.
    key: String,
    pinned: bool,
}

impl Memory {
    /// The memory a `- fact` line of `scope`'s file holds.
    fn parse(scope: Scope, line: &str) -> Option<Self> {
        let item = line.trim();
        let item = item
            .strip_prefix("- ")
            .or_else(|| item.strip_prefix("* "))?;
        let (pinned, fact) = match item.strip_prefix(PINNED) {
            Some(fact) => (true, fact.trim()),
            None => (false, item.trim()),
        };
        let key = normalize(fact);
        (!key.is_empty()).then(|| Self {
            id: id(scope, &key),
            fact: fact.to_owned(),
            key,
            pinned,
        })
    }

    /// How it shows in a list.
    fn line(&self) -> String {
        let pin = if self.pinned { " (pinned)" } else { "" };
        format!("- [{}]{pin} {}", self.id, self.fact)
    }
}

/// `fact` in lower case with single spaces and no closing punctuation.
fn normalize(fact: &str) -> String {
    let words: Vec<String> = fact.split_whitespace().map(str::to_lowercase).collect();
    words
        .join(" ")
        .trim_end_matches(['.', '!', ';', ','])
        .to_owned()
}

/// The id of the memory `key` in `scope`: its letter and 32-bit FNV-1a
/// hash, the same in every build.
fn id(scope: Scope, key: &str) -> String {
    let hash = key.bytes().fold(0x811c_9dc5_u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    format!("{}{hash:08x}", scope.letter())
}

/// The words of `text` worth matching, plural `s` dropped.
fn words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|word| word.len() > 2 && !COMMON.contains(&word.as_str()))
        .map(|word| match word.strip_suffix('s') {
            Some(stem) if stem.len() > 2 => stem.to_owned(),
            _ => word,
        })
        .collect()
}

/// The block a request carries for `memories`, if any.
fn block(memories: &[&Memory]) -> Option<String> {
    if memories.is_empty() {
        return None;
    }
    let lines: Vec<String> = memories.iter().map(|memory| memory.line()).collect();
    Some(format!(
        "<memory>\nWhat you remember from earlier sessions (ids for `forget`):\n{}\n</memory>",
        lines.join("\n")
    ))
}

/// On a memory file's watch.
#[derive(Component)]
struct MemoryFile;

/// Reads the memory files of this project and `RIG_HOME`, and watches them.
fn open(root: Res<ProjectRoot>, home: Res<RigHome>, mut commands: Commands) {
    let memories = Memories::new(&home.0, &root.0);
    commands.spawn((MemoryFile, FileWatch::new(&memories.user)));
    commands.spawn((MemoryFile, FileWatch::new(&memories.project)));
    commands.insert_resource(memories);
}

/// Reads the files again when one changed on disk; a watch's first
/// report, when it is armed, covers a change since `open` read them.
fn reload(
    mut changes: MessageReader<FileChanges>,
    files: Query<(), With<MemoryFile>>,
    memories: Option<ResMut<Memories>>,
) {
    let changed = changes.read().any(|change| files.contains(change.watch));
    if let Some(mut memories) = memories.filter(|_| changed) {
        memories.read();
    }
}

/// What a turn recalled at its first request, on the turn: each of its
/// requests carries the same, so the prompt cache holds.
#[derive(Component)]
struct Recollection(Option<String>);

/// The turns' model calls, as Prepare sees each once.
type NewTurnCalls<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut ModelRequest,
        &'static CallOf,
        &'static ForAgent,
    ),
    (Added<Admitted>, ModelCalls, With<TurnCall>),
>;

/// Recalls the memories for the user's message the turn answers at its
/// first request, and adds them to that message in each of its requests,
/// only in what is sent; a cache point of the request ends before that
/// message, so the block stays after the cached prefix.
fn recall(
    mut calls: NewTurnCalls,
    turns: Query<(&Turn, Option<&Recollection>)>,
    agents: Query<&Conversation>,
    memories: Res<Memories>,
    off: Res<MemoryOff>,
    mut commands: Commands,
) {
    for (mut call, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok((state, recollection)) = turns.get(turn) else {
            continue;
        };
        let asked = agents
            .get(agent)
            .ok()
            .and_then(|conversation| conversation.request(state.start).map(|entry| entry.message));
        let Some(asked @ message::Message::User { content }) = asked else {
            continue;
        };
        if text_of(content).is_empty() {
            continue;
        }
        let recalled = match recollection {
            Some(Recollection(recalled)) => recalled.clone(),
            None => {
                let recalled = (!off.0)
                    .then(|| block(&memories.relevant(&text_of(content))))
                    .flatten();
                commands
                    .entity(turn)
                    .try_insert(Recollection(recalled.clone()));
                recalled
            }
        };
        let Some(history) = &mut call.history else {
            continue;
        };
        let Some((at, recalled)) = history
            .messages()
            .iter()
            .rposition(|message| **message == *asked)
            .zip(recalled)
        else {
            continue;
        };
        if let Some(mut entry) = history.entry(at)
            && let message::Message::User { content } = std::sync::Arc::make_mut(&mut entry.message)
        {
            content.push(UserContent::text(recalled));
            history.splice(at..=at, [entry], Author::plugin("long-term-memory"));
            // What comes before the message is the stable prefix to cache.
            let turns = history
                .messages()
                .iter()
                .take(at)
                .filter(|message| !matches!(&***message, message::Message::System { .. }))
                .count();
            let point = CachePoint::Messages(turns);
            let request = call.request_mut();
            if !request.cache_points.contains(&point) {
                request.cache_points.push(point);
            }
        }
    }
}

/// The text of a message's content.
fn text_of(content: &[UserContent]) -> String {
    let texts: Vec<&str> = content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text()),
            _ => None,
        })
        .collect();
    texts.join("\n")
}

/// `/memory`, `/memory edit`, `/memory off`, `/memory on`.
fn memory(
    In(args): In<CommandArgs>,
    memories: Res<Memories>,
    view: Res<TerminalView>,
    processes: Res<Processes>,
    mut off: ResMut<MemoryOff>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let text = match args.args.as_str() {
        "" | "list" => {
            let lines = memories.list();
            let state = if off.0 {
                " (not recalled: /memory on)"
            } else {
                ""
            };
            if lines.is_empty() {
                format!("Nothing remembered yet{state}.")
            } else {
                format!("Remembered{state}:\n{}", lines.join("\n"))
            }
        }
        "off" => {
            off.0 = true;
            "Memories are not recalled for the rest of the session.".to_owned()
        }
        "on" => {
            off.0 = false;
            "Memories are recalled again.".to_owned()
        }
        "edit" if view.0 => {
            let line = corpus_shared::editor::line();
            let (program, args) = corpus_shared::editor::words(line.as_deref());
            let mut command = processes.launcher().command(program);
            command.args(args);
            for path in [&memories.project, &memories.user] {
                if let Some(dir) = path.parent() {
                    fs::create_dir_all(dir).ok();
                }
                command.arg(path);
            }
            commands.spawn((RunInTerminal(command), Editing));
            return;
        }
        "edit" => format!(
            "Edit the memories in {} (project) and {} (user); they are read again when they \
             change.",
            memories.project.display(),
            memories.user.display()
        ),
        other => {
            notices.write(Notice::error(
                agent,
                format!("/memory takes `edit`, `off` or `on`, not `{other}`"),
            ));
            return;
        }
    };
    notices.write(Notice::info(agent, text));
}

/// On the `$EDITOR` run of `/memory edit`.
#[derive(Component)]
struct Editing;

/// Reads the files again once `$EDITOR` closed, at once rather than at
/// the watch's next look.
fn edited(
    done: Query<(Entity, &TerminalExit), (Added<TerminalExit>, With<Editing>)>,
    memories: Option<ResMut<Memories>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(mut memories) = memories else {
        return;
    };
    for (run, exit) in &done {
        if let Err(why) = &exit.0 {
            notices.write(Notice::error(None, format!("$EDITOR did not start: {why}")));
        }
        memories.read();
        commands.entity(run).despawn();
    }
}

/// What `remember` takes.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RememberArgs {
    /// The fact, one short sentence, such as "This repo runs tests with cargo nextest".
    fact: String,
    /// `project` for this repository, `user` for the user everywhere.
    scope: Scope,
}

fn remember(
    called: On<ToolCalled<RememberArgs>>,
    mut memories: ResMut<Memories>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let result = match memories.remember(&called.args.fact, called.args.scope) {
        Ok((id, true)) => ToolResult::success(format!("Remembered as {id}.").into()),
        Ok((id, false)) => ToolResult::success(format!("Already remembered as {id}.").into()),
        Err(error) => ToolResult::failed(ToolExecutionError::from_error(error)),
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("long-term-memory"), result);
    }
}

/// What `forget` takes.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ForgetArgs {
    /// The memory's id, such as `p1a2b3c4d`.
    id: String,
}

fn forget(
    called: On<ToolCalled<ForgetArgs>>,
    mut memories: ResMut<Memories>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let id = &called.args.id;
    let result = match memories.forget(id) {
        Ok(Some(fact)) => ToolResult::success(format!("Forgot: {fact}").into()),
        Ok(None) => ToolResult::failed(ToolExecutionError::not_found(format!(
            "No memory has the id {id}."
        ))),
        Err(error) => ToolResult::failed(ToolExecutionError::from_error(error)),
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("long-term-memory"), result);
    }
}
