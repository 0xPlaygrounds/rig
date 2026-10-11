//! Long-term memory: facts the agent keeps across sessions, as markdown
//! lines a person can edit, in `RIG_HOME/memory/user.md` (the user's) and
//! `<project>/.rig/memory.md` (the project's).
//!
//! A memory is a `- fact` line; `- [pinned] fact` is recalled on every
//! turn. Its id is its scope's letter and a hash of its normalized text, so
//! ids survive hand edits and the same fact is never stored twice. The
//! `remember` and `forget` tools change the files; before the first model
//! request of each turn, the memories that share the most words with the
//! user's message, and every pinned one, go with that message, after the
//! cached prefix, and never into the saved conversation. The files are read
//! afresh each time, so a change on disk counts at once. `/memory` lists
//! them, `/memory off` and `/memory on` stop and restart the recall for
//! the session.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::serde_json;
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// How many memories a turn recalls, besides the pinned ones.
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
        let files = app
            .world_mut()
            .get_resource_or_insert_with(MemoryFiles::from_env)
            .clone();
        app.init_resource::<MemoryOff>()
            .add_tool_with(
                Remember(files.clone()),
                ToolOptions {
                    rules: &[
                        "When the user asks you to remember something (\"remember that …\"), or \
                         states a lasting preference or project fact, call `remember` with it as \
                         one short fact: scope `project` for this repository, `user` for the \
                         user everywhere.",
                        "Memories recalled for a message come in a <memory> block after it; \
                         `forget` one by its id when it is wrong or the user asks.",
                    ],
                    footprint: Footprint::default(),
                },
            )
            .add_tool(Forget(files))
            .add_command(
                "memory",
                "List what the agent remembers; `edit`, `off` or `on` for the session",
                memory,
            )
            .add_observer(recall);
    }
}

/// Where the memories are kept.
#[derive(Resource, Clone, Debug)]
pub struct MemoryFiles {
    /// The user's, for every project.
    pub user: PathBuf,
    /// The project's.
    pub project: PathBuf,
}

impl MemoryFiles {
    /// Those of `RIG_HOME` and of the working directory.
    pub fn from_env() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::new(Home::from_env().root(), &cwd)
    }

    /// Those of the rig home `home` and the project directory `project`.
    pub fn new(home: &Path, project: &Path) -> Self {
        Self {
            user: home.join("memory").join("user.md"),
            project: project.join(".rig").join("memory.md"),
        }
    }

    fn path(&self, scope: Scope) -> &Path {
        match scope {
            Scope::User => &self.user,
            Scope::Project => &self.project,
        }
    }

    /// Every memory, the project's first; a missing file has none.
    fn read(&self) -> Vec<Memory> {
        [Scope::Project, Scope::User]
            .into_iter()
            .flat_map(|scope| {
                let text = fs::read_to_string(self.path(scope)).unwrap_or_default();
                text.lines()
                    .filter_map(|line| Memory::parse(scope, line))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Stores `fact` in `scope`, or finds it stored already; its id, and
    /// whether it was new.
    fn remember(&self, fact: &str, scope: Scope) -> std::io::Result<(String, bool)> {
        let fact = fact.split_whitespace().collect::<Vec<_>>().join(" ");
        let key = normalize(&fact);
        if key.is_empty() {
            return Err(std::io::Error::other("the fact is empty"));
        }
        if let Some(known) = self.read().into_iter().find(|memory| memory.key == key) {
            return Ok((known.id, false));
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
        Ok((id(scope, &key), true))
    }

    /// Takes the memory `id` out of its file; the fact it was.
    fn forget(&self, id: &str) -> std::io::Result<Option<String>> {
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
        Ok(forgotten)
    }
}

/// Whether memories stay out of requests, for the session (`/memory off`).
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
pub struct MemoryOff(pub bool);

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

/// The memories to recall for `message`: every pinned one, then those
/// sharing the most words with it, up to [`MOST`] in all.
fn relevant(memories: Vec<Memory>, message: &str) -> Vec<Memory> {
    let asked = words(message);
    let (pinned, others): (Vec<Memory>, Vec<Memory>) =
        memories.into_iter().partition(|memory| memory.pinned);
    let mut scored: Vec<(usize, Memory)> = others
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

/// The block a request carries for `memories`, if any.
fn block(memories: &[Memory]) -> Option<String> {
    if memories.is_empty() {
        return None;
    }
    let lines: Vec<String> = memories.iter().map(Memory::line).collect();
    Some(format!(
        "<memory>\nWhat you remember from earlier sessions (ids for `forget`):\n{}\n</memory>",
        lines.join("\n")
    ))
}

/// What a turn recalled at its first request, which each of its requests
/// carries, so they agree and the prompt cache holds.
#[derive(Component)]
struct Recollection(Option<String>);

/// The content of the user's message the turn answers: the last user
/// message with text and no tool results.
fn asked(messages: &mut [Message]) -> Option<&mut Vec<UserContent>> {
    messages.iter_mut().rev().find_map(|message| match message {
        Message::User { content }
            if !content
                .iter()
                .any(|item| matches!(item, UserContent::ToolResult(_))) =>
        {
            Some(content)
        }
        _ => None,
    })
}

/// Recalls the memories for the user's message at the turn's first
/// request, and adds them after that message in each of its requests:
/// after the cached prefix, and only in what is sent.
fn recall(
    mut prepare: On<PrepareRequest>,
    turns: Query<&Recollection>,
    files: Res<MemoryFiles>,
    off: Res<MemoryOff>,
    mut commands: Commands,
) {
    let turn = prepare.entity;
    let recalled = match turns.get(turn) {
        Ok(Recollection(recalled)) => recalled.clone(),
        Err(_) => {
            let message = asked(&mut prepare.messages).map(|content| text_of(content));
            let recalled = message
                .filter(|_| !off.0)
                .and_then(|message| block(&relevant(files.read(), &message)));
            commands
                .entity(turn)
                .try_insert(Recollection(recalled.clone()));
            recalled
        }
    };
    if let Some(recalled) = recalled
        && let Some(content) = asked(&mut prepare.messages)
    {
        content.push(UserContent::text(recalled));
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
    files: Res<MemoryFiles>,
    mut off: ResMut<MemoryOff>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let text = match args.args.as_str() {
        "" | "list" => {
            let lines: Vec<String> = files.read().iter().map(Memory::line).collect();
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
        // The terminal view cannot hand the terminal to `$EDITOR`: say
        // where the files are instead.
        "edit" => format!(
            "Edit the memories in {} (project) and {} (user); each turn reads them afresh.",
            files.project.display(),
            files.user.display()
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

/// The `remember` tool.
struct Remember(MemoryFiles);

/// What `remember` takes.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RememberArgs {
    /// The fact, one short sentence, such as "This repo runs tests with cargo nextest".
    fact: String,
    /// `project` for this repository, `user` for the user everywhere.
    scope: Scope,
}

impl PortableTool for Remember {
    const NAME: &'static str = "remember";
    type Args = RememberArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Remember a fact across sessions; returns its id (an existing one for a fact already \
         remembered)."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<RememberArgs>()
    }

    async fn call(&self, args: RememberArgs) -> Result<String, ToolExecutionError> {
        let files = self.0.clone();
        blocking(move || {
            let (id, new) = files
                .remember(&args.fact, args.scope)
                .map_err(ToolExecutionError::from_error)?;
            Ok(if new {
                format!("Remembered as {id}.")
            } else {
                format!("Already remembered as {id}.")
            })
        })
        .await
    }
}

/// The `forget` tool.
struct Forget(MemoryFiles);

/// What `forget` takes.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ForgetArgs {
    /// The memory's id, such as `p1a2b3c4d`.
    id: String,
}

impl PortableTool for Forget {
    const NAME: &'static str = "forget";
    type Args = ForgetArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Forget a remembered fact by its id.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ForgetArgs>()
    }

    async fn call(&self, args: ForgetArgs) -> Result<String, ToolExecutionError> {
        let files = self.0.clone();
        blocking(move || {
            match files
                .forget(&args.id)
                .map_err(ToolExecutionError::from_error)?
            {
                Some(fact) => Ok(format!("Forgot: {fact}")),
                None => Err(ToolExecutionError::not_found(format!(
                    "No memory has the id {}.",
                    args.id
                ))),
            }
        })
        .await
    }
}
