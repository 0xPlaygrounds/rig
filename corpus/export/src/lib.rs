//! Plugin corpus: 41. Session export (spec `41-export.md`).
//!
//! `/export [md|html|json] [path] [--thinking] [--session <id>]` writes a
//! session as Markdown (default, to `./rig-<session-name>.md`), a
//! self-contained HTML page or JSON. It reads the *saved* session, not the
//! running agents: the session's agent logs are folded back by the
//! kernel's own restore, in a second, headless app over a store that
//! drops every write, so `--session <id>` exports a session that is not
//! running the same way as the current one.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_cassette::journal::{JournalStore, JsonlDirStore};
use rig_core::completion::UsageTotals;
use rig_ecs::journal::SessionStore;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_sessions::{Meta, SessionTitle};
use rig_telemetry::Spending;

mod render;

pub use render::{Format, JSON_FORMAT};

/// Adds `/export`.
#[derive(Default)]
pub struct ExportPlugin;

impl Plugin for ExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "export",
            "Export the session: /export [md|html|json] [path] [--thinking] [--session <id>]",
            export,
        );
    }
}

/// One agent of a saved session, as its log restores it.
#[derive(Clone)]
pub struct SavedAgent {
    /// Its id.
    pub id: AgentId,
    /// The agent that spawned it, if any.
    pub parent: Option<AgentId>,
    /// Its conversation, from the newest compaction on.
    pub conversation: Conversation,
    /// The summary that replaced its older messages, if it was compacted.
    pub condensed: Option<String>,
    /// Its catalog model.
    pub model: Option<String>,
    /// What its model calls used and cost, when the usage plugin saved it.
    pub usage: Option<UsageTotals>,
}

/// A saved session: its id, its name and its agents.
#[derive(Clone, Default)]
pub struct SavedSession {
    /// Its id, when known.
    pub id: Option<String>,
    /// What it is listed as.
    pub name: String,
    /// When it started, in seconds since the Unix epoch, when known.
    pub started: Option<u64>,
    /// Its agents, in no particular order.
    pub agents: Vec<SavedAgent>,
}

/// What `/export` was asked for.
#[derive(Debug, PartialEq, Eq)]
struct Request {
    format: Format,
    path: Option<PathBuf>,
    thinking: bool,
    session: Option<String>,
}

impl Request {
    fn parse(args: &str) -> Result<Self, String> {
        let mut request = Self {
            format: Format::Markdown,
            path: None,
            thinking: false,
            session: None,
        };
        let mut words = args.split_whitespace();
        while let Some(word) = words.next() {
            match word {
                "--thinking" => request.thinking = true,
                "--session" => {
                    let id = words.next().ok_or("--session needs a session id")?;
                    request.session = Some(id.to_owned());
                }
                flag if flag.starts_with("--") => return Err(format!("unknown option `{flag}`")),
                word => match (Format::parse(word), &request.path) {
                    (Some(format), None) => request.format = format,
                    (_, None) => request.path = Some(PathBuf::from(word)),
                    (_, Some(_)) => return Err(format!("unexpected `{word}`")),
                },
            }
        }
        Ok(request)
    }
}

/// `/export`.
fn export(
    In(args): In<CommandArgs>,
    store: Option<Res<SessionStore>>,
    paths: Option<Res<SessionPaths>>,
    title: Option<Res<SessionTitle>>,
    mut notices: MessageWriter<Notice>,
) {
    let said = Request::parse(&args.args).and_then(|request| {
        let session = match &request.session {
            Some(id) => other_session(id)?,
            None => {
                let store = store.as_ref().ok_or("this session is not saved")?;
                let id = paths.as_ref().and_then(|paths| session_id(paths.path()));
                let name = title.as_deref().map(listed).unwrap_or_default();
                load(store.0.clone(), id, name)
            }
        };
        let path = request.path.clone().unwrap_or_else(|| {
            let name = slug(&session.name);
            PathBuf::from(format!("rig-{name}.{}", request.format.extension()))
        });
        let files = render::render(&session, request.format, request.thinking, &path);
        for (file, bytes) in &files {
            write(file, bytes)
                .map_err(|failure| format!("writing {}: {failure}", file.display()))?;
        }
        Ok(format!("Exported the session to {}.", path.display()))
    });
    match said {
        Ok(text) => notices.write(Notice::info(args.agent, text)),
        Err(why) => notices.write(Notice::error(args.agent, format!("/export: {why}"))),
    };
}

/// The saved session `id` under `RIG_HOME`, whether it runs or not.
fn other_session(id: &str) -> Result<SavedSession, String> {
    let id: SessionId = id.parse()?;
    let dir = Home::from_env().session(&id);
    if !dir.is_saved() {
        return Err(format!("no saved session {id}"));
    }
    let meta: Option<Meta> = rig_tools::fs::read_json(&dir.meta());
    let name = meta.map(|meta| listed(&meta.title)).unwrap_or_default();
    let store = Arc::new(JsonlDirStore::new(dir.path()));
    Ok(load(store, Some(id.to_string()), name))
}

/// A session directory's id: its last path component.
fn session_id(dir: &Path) -> Option<String> {
    Some(dir.file_name()?.to_str()?.to_owned())
}

/// The name `/name` gave, else the start of the first message typed.
fn listed(title: &SessionTitle) -> String {
    title.name.clone().unwrap_or_else(|| title.typed.clone())
}

/// `name` as a file name part: letters, digits and dashes.
fn slug(name: &str) -> String {
    let dashed: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let words: Vec<&str> = dashed.split('-').filter(|word| !word.is_empty()).collect();
    let slug: String = words.join("-").chars().take(60).collect();
    if slug.is_empty() {
        "session".to_owned()
    } else {
        slug
    }
}

fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, bytes)
}

/// A session store that reads and never writes: restoring a session to
/// read it must not cut a torn line, answer open calls or log anything.
struct ReadOnly(Arc<dyn JournalStore>);

impl JournalStore for ReadOnly {
    fn agents(&self) -> io::Result<Vec<String>> {
        self.0.agents()
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        self.0.read(agent)
    }

    fn truncate(&self, _: &str, _: u64) -> io::Result<()> {
        Ok(())
    }

    fn append(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn put_blob(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.0.blob(name)
    }
}

/// The session saved in `store`, folded back by the kernel's restore in a
/// headless app of its own whose agents stay idle.
pub fn load(store: Arc<dyn JournalStore>, id: Option<String>, name: String) -> SavedSession {
    let mut app = App::new();
    app.insert_resource(SessionStore(Arc::new(ReadOnly(store))))
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_observer(|mut restored: On<Restored>| restored.resume = false);
    app.finish();
    app.cleanup();
    app.update();
    let world = app.world_mut();
    let mut agents = world.query_filtered::<(
        &AgentId,
        &Conversation,
        Option<&SpawnedBy>,
        Option<&Condensed>,
        Option<&ModelChoice>,
        Option<&Spending>,
    ), With<Agent>>();
    let mut ids = world.query::<&AgentId>();
    let world = &*world;
    let agents = agents
        .iter(world)
        .filter(|(_, conversation, ..)| !conversation.messages().is_empty())
        .map(
            |(id, conversation, parent, condensed, model, usage)| SavedAgent {
                id: id.clone(),
                parent: parent.and_then(|parent| ids.get(world, parent.0).ok().cloned()),
                conversation: conversation.clone(),
                condensed: condensed.map(|condensed| condensed.summary.clone()),
                model: model.map(|model| model.0.clone()),
                usage: usage.map(|usage| usage.0),
            },
        )
        .collect();
    // A session id starts with the Unix time it was made at.
    let started = id
        .as_deref()
        .and_then(|id| id.split('-').next()?.parse().ok());
    let name = if name.is_empty() {
        id.as_ref()
            .map_or_else(|| "session".to_owned(), |id| format!("session {id}"))
    } else {
        name
    };
    SavedSession {
        id,
        name,
        started,
        agents,
    }
}

#[cfg(test)]
mod tests;
