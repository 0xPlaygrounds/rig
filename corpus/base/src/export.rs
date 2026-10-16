//! Plugin corpus: 41. Session export (spec `41-export.md`).
//!
//! `/export [md|html|json] [path] [--thinking] [--session <id>]` writes a
//! session as Markdown (default, to `./rig-<session-name>.md`), a
//! self-contained HTML page or JSON; `rig --export=<session>` (with
//! `--export-format`, `--export-to` and `--export-thinking`) does the same
//! from the command line and exits. Either reads the *saved* session with
//! rig-memory's `Session::read_lenient`, which neither restores nor writes
//! it, so a session that is not running exports the same way as this one,
//! and a log that cannot be read is named and left out; each model call's
//! usage and cost comes from the session's effect log.

use std::any::TypeId;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use bevy_ecs::system::SystemParam;
use bevy_reflect::TypeRegistry;
use bevy_reflect::serde::TypedReflectDeserializer;
use rig_cassette::effect_log::jsonl::{self, SESSION_EFFECT_LOG};
use rig_core::completion::{Usage, UsageTotals};
use rig_core::effect::{Answered, Disposition, EffectId, EffectKind, EffectRecord};
use rig_ecs::session_log::SessionStore;
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_sessions::SessionTitle;
use serde::de::DeserializeSeed;

mod render;

pub use render::{Format, JSON_FORMAT};

/// Adds `/export` and the `--export` front.
#[derive(Default)]
pub struct ExportPlugin;

impl Plugin for ExportPlugin {
    fn build(&self, app: &mut App) {
        let flag = Flag::new("export", "Export a saved session, then exit").value("SESSION");
        app.add_command(
            "export",
            "Export the session: /export [md|html|json] [path] [--thinking] [--session <id>]",
            export,
        )
        // Its run exports another session and keeps none of its own.
        .add_front(Front {
            flag: Some(flag),
            ..Front::new("export").sessionless()
        })
        .add_flag(Flag::new("export-format", "--export's format: md, html or json").value("FORMAT"))
        .add_flag(Flag::new("export-to", "The file --export writes").value("PATH"))
        .add_flag(Flag::new(
            "export-thinking",
            "--export's Markdown shows reasoning",
        ));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front("export") {
            app.add_systems(Startup, export_and_exit);
        }
    }
}

/// A saved session as an export shows it.
pub struct Export {
    /// Its id, when it has a directory.
    pub id: Option<String>,
    /// What it is listed as.
    pub name: String,
    /// When its first log was started, in milliseconds since the Unix epoch.
    pub started_ms: Option<u64>,
    /// Its agents, in the order their logs were started.
    pub agents: Vec<ExportedAgent>,
    /// The logs that could not be read, each with why: the export holds
    /// the rest.
    pub left_out: Vec<String>,
}

/// One agent of an [`Export`].
pub struct ExportedAgent {
    /// What its log holds.
    pub saved: SavedAgent,
    /// Its catalog model, as saved.
    pub model: Option<String>,
    /// Its model calls, in the order they were made.
    pub calls: Vec<ModelCall>,
}

/// One model call of an agent, from the session's effect log.
pub struct ModelCall {
    /// The effect's id in the log.
    pub effect: EffectId,
    /// What it was made for, such as `turn` or `compaction`.
    pub purpose: Option<String>,
    /// The agent's turn it was made in, the first being 0.
    pub turn: Option<u64>,
    /// The model that answered, as `provider/model`.
    pub model: String,
    /// The tokens it used and what it cost.
    pub usage: Usage,
    /// The provider's response id, which the assistant message's origin
    /// names too.
    pub response_id: Option<String>,
    /// How it ended: ran, superseded, cancelled, …
    pub disposition: Disposition,
}

impl ExportedAgent {
    /// What its calls used, summed.
    pub fn totals(&self) -> UsageTotals {
        let mut totals = UsageTotals::default();
        for call in &self.calls {
            totals.record(&call.usage);
        }
        totals
    }
}

/// What `/export` or `--export` was asked for.
#[derive(Debug, PartialEq, Eq)]
struct Request {
    format: Format,
    path: Option<PathBuf>,
    thinking: bool,
    session: Option<SessionId>,
}

impl Request {
    /// `/export`'s arguments.
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
                    request.session = Some(id.parse()?);
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

    /// The `--export` options.
    fn of(invoked: &Invoked) -> Result<Self, String> {
        let format = match invoked.flag_value("export-format") {
            Some(word) => Format::parse(word).ok_or(format!("unknown format `{word}`"))?,
            None => Format::Markdown,
        };
        let session = invoked
            .flag_value("export")
            .ok_or("--export needs a session id")?;
        Ok(Self {
            format,
            path: invoked.flag_value("export-to").map(PathBuf::from),
            thinking: invoked.flag("export-thinking"),
            session: Some(session.parse()?),
        })
    }
}

/// Where an export reads from: this session, or another under `RIG_HOME`.
#[derive(SystemParam)]
struct Sessions<'w> {
    store: Option<Res<'w, SessionStore>>,
    stores: Option<Res<'w, StoreOpener>>,
    paths: Option<Res<'w, SessionPaths>>,
    title: Option<Res<'w, SessionTitle>>,
    home: Option<Res<'w, RigHome>>,
    root: Option<Res<'w, ProjectRoot>>,
    registry: Res<'w, AppTypeRegistry>,
}

impl Sessions<'_> {
    /// Writes the export `request` asks for; the file it wrote.
    fn export(&self, request: &Request) -> Result<PathBuf, String> {
        let registry = self.registry.read();
        let session = match &request.session {
            Some(id) => {
                let home = self.home.as_ref().map(|home| home.0.clone());
                let dir = home.unwrap_or_else(Home::from_env).session(id);
                if !dir.is_saved() {
                    return Err(format!("no saved session {id}"));
                }
                let stores = self.stores.as_deref().cloned().unwrap_or_default();
                let store = stores
                    .open(&dir)
                    .map_err(|failure| format!("opening session {id}: {failure}"))?;
                let name = SessionTitle::read(&dir, &*store).map(|title| title.shown().to_owned());
                load(&*store, Some(&dir), name.unwrap_or_default(), &registry)?
            }
            None => {
                let store = self.store.as_ref().ok_or("this session is not saved")?;
                let name = self.title.as_deref().map(SessionTitle::shown);
                let name = name.unwrap_or_default().to_owned();
                let dir = self.paths.as_deref().map(|paths| &paths.0);
                load(store.0.as_ref(), dir, name, &registry)?
            }
        };
        let path = request.path.clone().unwrap_or_else(|| {
            let name = slug(&session.name);
            PathBuf::from(format!("rig-{name}.{}", request.format.extension()))
        });
        // Paths the user names are the project's.
        let path = match &self.root {
            Some(root) => root.0.join(path),
            None => path,
        };
        for (file, bytes) in render::render(&session, request.format, request.thinking, &path) {
            write(&file, &bytes)
                .map_err(|failure| format!("writing {}: {failure}", file.display()))?;
        }
        Ok(path)
    }
}

/// `/export`.
fn export(In(args): In<CommandArgs>, sessions: Sessions, mut notices: MessageWriter<Notice>) {
    let notice = match Request::parse(&args.args).and_then(|request| sessions.export(&request)) {
        Ok(path) => Notice::info(
            args.agent,
            format!("Exported the session to {}.", path.display()),
        ),
        Err(why) => Notice::error(args.agent, format!("/export: {why}")),
    };
    notices.write(notice);
}

/// `rig --export=<session>`: exports, says where to, and exits.
fn export_and_exit(invoked: Option<Res<Invoked>>, sessions: Sessions, mut commands: Commands) {
    let invoked = invoked.as_deref().cloned().unwrap_or_default();
    let reason = match Request::of(&invoked).and_then(|request| sessions.export(&request)) {
        Ok(path) => {
            println!("{}", path.display());
            ExitReason::Quit
        }
        Err(why) => {
            eprintln!("rig --export: {why}");
            ExitReason::Failed(1)
        }
    };
    commands.trigger(Exit(reason));
}

/// The session in `store`, with the model calls its effect log holds.
fn load(
    store: &dyn Store,
    dir: Option<&SessionDir>,
    name: String,
    registry: &TypeRegistry,
) -> Result<Export, String> {
    let (saved, unreadable) = Session::read_lenient(store)
        .map_err(|failure| format!("reading the session: {failure}"))?;
    let left_out = unreadable
        .into_iter()
        .map(|log| format!("{} ({})", log.log, log.error))
        .collect();
    let records = store
        .read_file(SESSION_EFFECT_LOG)
        .ok()
        .and_then(|log| jsonl::read_from(log.as_slice()).ok())
        .map(|log| log.records)
        .unwrap_or_default();
    let id = dir.and_then(SessionDir::id).map(|id| id.to_string());
    let agents: Vec<ExportedAgent> = saved
        .agents
        .into_iter()
        .filter(|agent| !agent.conversation.messages().is_empty())
        .map(|saved| ExportedAgent {
            model: saved_value::<ModelChoice>(&saved.saved, registry).map(|choice| choice.0),
            calls: records
                .iter()
                .filter(|record| record.scope.as_deref() == Some(saved.id.as_str()))
                .filter_map(model_call)
                .collect(),
            saved,
        })
        .collect();
    let name = match (name.is_empty(), &id) {
        (false, _) => name,
        (true, Some(id)) => format!("session {id}"),
        (true, None) => "session".to_owned(),
    };
    Ok(Export {
        started_ms: agents.iter().map(|agent| agent.saved.started_ms).min(),
        id,
        name,
        agents,
        left_out,
    })
}

/// A model call's record as a [`ModelCall`]; `None` for other effects and
/// calls that got no reply.
fn model_call(record: &EffectRecord) -> Option<ModelCall> {
    if !matches!(record.kind, EffectKind::Completion { .. }) {
        return None;
    }
    let outcome = record.reviewed.as_ref().unwrap_or(&record.outcome);
    let Ok(Answered::Completion(response)) = outcome else {
        return None;
    };
    Some(ModelCall {
        effect: record.id,
        purpose: record.purpose.as_ref().map(|purpose| purpose.to_string()),
        turn: record.turn,
        model: format!("{}/{}", response.origin.provider, response.origin.model),
        usage: response.usage,
        response_id: response.origin.response_id.clone(),
        disposition: record.disposition,
    })
}

/// The saved value of `T` among an agent's saved components.
fn saved_value<T: Reflect + TypePath + FromReflect>(
    saved: &BTreeMap<String, serde_json::Value>,
    registry: &TypeRegistry,
) -> Option<T> {
    let value = saved.get(T::type_path())?;
    let registration = registry.get(TypeId::of::<T>())?;
    let reflected = TypedReflectDeserializer::new(registration, registry)
        .deserialize(value)
        .ok()?;
    T::from_reflect(&*reflected)
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

#[cfg(test)]
mod tests;
