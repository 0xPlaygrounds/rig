//! Plugin corpus (held-out): H26: Raw provider stream log (spec
//! `H26-raw-stream-log.md`).
//!
//! `/rawlog on|off` (or `--rawlog`) records every model call, an agent's,
//! a subagent's or a plugin's own, to `<session>/raw/<turn>-<call>-<model>.jsonl`:
//! the request as the model's handler got it, then each stream item as it
//! came, with the milliseconds since the request started (unknown provider
//! items verbatim, the response with its provider request id), then the
//! error, if any. A retry is a new record in the same file.
//!
//! - Prepare: each admitted model call gets an adapter, innermost, that
//!   wraps the model's handler in a tap. The tap passes every item on
//!   unchanged and hands a copy to a writer thread.
//! - The writer thread redacts secret-looking strings, writes the lines,
//!   keeps the folder under 50 MB (oldest files first) and reports back as
//!   a stream on an entity: one notice when it first trims, and one warning
//!   when a write fails, which turns the log off.
//! - Whether the log is on is a saved resource.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use async_channel::{Receiver, Sender};
use bevy_tasks::{IoTaskPool, TaskPool};
use futures::StreamExt as _;
use regex::Regex;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::{EffectKind, HandlerDescriptor, Outcome as Answered, family};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_harness::rig_core::streaming::{Item, Relayed};
use rig_harness::rig_ecs::effects::Handler;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// The command, the flag and the adapter's name.
const NAME: &str = "rawlog";

/// The most the session's raw folder holds.
const CAP: u64 = 50 * 1024 * 1024;

/// What a secret is written as.
const REDACTED: &str = "[redacted]";

/// Records every model call's request and raw stream.
#[derive(Default)]
pub struct RawStreamLogPlugin;

/// The plugin's stage system, for others to order against: it puts its
/// adapter first on each admitted model call.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawLogSystems;

/// Whether model calls are recorded, and how many were, saved with the
/// session.
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource, Saved)]
pub struct RawLog {
    /// Whether new model calls are recorded.
    pub on: bool,
    /// The calls recorded in this session, numbering their files.
    calls: u64,
}

impl Plugin for RawStreamLogPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(Flag::new(
            NAME,
            "Record each model call's request and raw stream in the session's raw/ folder",
        ));
        let flagged = flagged(app.world());
        app.insert_resource(RawLog {
            on: flagged,
            calls: 0,
        })
        .register_type::<RawLog>()
        .configure_sets(AgentPass, RawLogSystems.in_set(Stage::Prepare))
        .add_systems(Startup, start_writer)
        .add_systems(Update, keep_flag.run_if(resource_added::<SessionRestored>))
        .add_systems(AgentPass, tap_model_calls.in_set(RawLogSystems))
        .add_observer(writer_reported)
        .add_command(
            NAME,
            "Record each model call's raw stream: /rawlog on, off or last",
            rawlog,
        );
    }
}

/// Whether the agent was started with `--rawlog`.
fn flagged(world: &World) -> bool {
    world
        .get_resource::<Invoked>()
        .is_some_and(|invoked| invoked.flag(NAME))
}

/// `--rawlog` wins over the state a restored session saved.
fn keep_flag(world: &mut World) {
    if flagged(world)
        && let Some(mut log) = world.get_resource_mut::<RawLog>()
    {
        log.on = true;
    }
}

/// The writer thread's input, and the newest file it wrote.
#[derive(Resource, Clone)]
struct Writer {
    lines: Sender<Line>,
    newest: Arc<Mutex<Option<PathBuf>>>,
    dir: PathBuf,
}

/// What the writer thread is told.
enum Line {
    /// Write `entry` of the record `attempt` into `file`.
    Entry {
        file: String,
        attempt: u32,
        entry: Entry,
    },
    /// Writing may go on after a failure.
    Resume,
}

/// One line of a record.
enum Entry {
    /// The request, as the model's handler got it.
    Request(Box<CompletionRequest>),
    /// A stream item, `ms` after the request started.
    Item {
        ms: u128,
        item: Result<Relayed, ErrorReport>,
    },
    /// A whole answer, `ms` after the request started, when the handler
    /// did not stream (a failure before the stream, say).
    Answer {
        ms: u128,
        outcome: Result<Answered, ErrorReport>,
    },
}

/// What the writer thread reports.
enum WriterEvent {
    /// It removed the oldest files to stay under the cap, the first time.
    Trimmed,
    /// A write failed; it writes nothing more until resumed.
    Failed(String),
}

/// On the entity the writer's reports stream to.
#[derive(Component)]
struct WriterReports;

/// Starts the writer thread on the session's raw folder, when the session
/// has a directory.
fn start_writer(paths: Option<Res<SessionPaths>>, wake: Option<Res<Wake>>, mut commands: Commands) {
    let Some(paths) = paths else {
        return;
    };
    let dir = paths.path().join("raw");
    let (lines, received) = async_channel::unbounded();
    let (reports, events) = async_channel::unbounded();
    let newest = Arc::new(Mutex::new(None));
    let files = Files::scan(dir.clone(), newest.clone(), reports);
    let spawned = std::thread::Builder::new()
        .name("rawlog".to_owned())
        .spawn(move || files.run(&received));
    if spawned.is_err() {
        commands.write_message(Notice::error(None, "The raw log's writer could not start."));
        return;
    }
    let wake = wake.map(|wake| wake.clone()).unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((WriterReports, Running::stream(pool, &wake, events)));
    commands.insert_resource(Writer { lines, newest, dir });
}

/// Says what the writer reported; a failed write turns the log off.
fn writer_reported(
    reported: On<Insert<Done<WriterEvent>>>,
    reports: Query<&Done<WriterEvent>, With<WriterReports>>,
    mut log: ResMut<RawLog>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Done(event)) = reports.get(reported.entity) else {
        return;
    };
    match event {
        WriterEvent::Trimmed => notices.write(Notice::info(
            None,
            "The raw log passed 50 MB: its oldest files go first.",
        )),
        WriterEvent::Failed(why) => {
            log.on = false;
            notices.write(Notice::error(
                None,
                format!("The raw log could not write ({why}); it is off."),
            ))
        }
    };
}

/// Where a model call's attempts are recorded: the file, named on the
/// first attempt's model, and which attempt this is.
#[derive(Component, Clone)]
struct RawFile {
    stem: String,
    name: Arc<OnceLock<String>>,
    attempt: u32,
}

/// Puts the tap first on each model call admitted while the log is on; a
/// call that supersedes a recorded one is its next attempt, in its file.
fn tap_model_calls(
    mut calls: Query<
        (Entity, &mut Adapters, Option<&CallOf>, Option<&Supersedes>),
        Added<Admitted>,
    >,
    recorded: Query<&RawFile>,
    turns: Query<&Turn>,
    writer: Option<Res<Writer>>,
    mut log: ResMut<RawLog>,
    mut commands: Commands,
) {
    let Some(writer) = writer.filter(|_| log.on) else {
        return;
    };
    for (call, mut adapters, of, supersedes) in &mut calls {
        let earlier = supersedes.and_then(|&Supersedes(earlier)| recorded.get(earlier).ok());
        let raw = match earlier {
            Some(earlier) => RawFile {
                attempt: earlier.attempt.saturating_add(1),
                ..earlier.clone()
            },
            None => {
                log.calls += 1;
                let turn = of.and_then(|&CallOf(turn)| turns.get(turn).ok());
                RawFile {
                    stem: format!("{}-{}", turn.map_or(0, |turn| turn.index), log.calls),
                    name: Arc::default(),
                    attempt: 1,
                }
            }
        };
        adapters.0.insert(0, tap(raw.clone(), writer.lines.clone()));
        commands.entity(call).insert(raw);
    }
}

/// The adapter that wraps whatever model a call finally goes to in a tap.
fn tap(raw: RawFile, lines: Sender<Line>) -> Adapter {
    Adapter::new(NAME, move |mut connection: Connection| {
        let model = file_safe(&connection.spec.reference());
        let file = raw
            .name
            .get_or_init(|| format!("{}-{model}.jsonl", raw.stem));
        let tap = RawTap {
            inner: connection.handler.0.clone(),
            sink: Sink {
                file: file.clone(),
                attempt: raw.attempt,
                lines: lines.clone(),
            },
        };
        connection.handler = Handler(ErasedHandler::new(tap));
        connection
    })
}

/// `text` as a part of a file name.
fn file_safe(text: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_');
    text.chars()
        .map(|c| if safe(c) { c } else { '_' })
        .collect()
}

/// Where one attempt's lines go.
#[derive(Clone)]
struct Sink {
    file: String,
    attempt: u32,
    lines: Sender<Line>,
}

impl Sink {
    /// Hands `entry` to the writer thread, without waiting.
    fn send(&self, entry: Entry) {
        let line = Line::Entry {
            file: self.file.clone(),
            attempt: self.attempt,
            entry,
        };
        self.lines.try_send(line).ok();
    }
}

/// A model's handler, recorded: every request and stream item passes
/// through unchanged, and a copy goes to the writer thread.
struct RawTap {
    inner: ErasedHandler,
    sink: Sink,
}

impl Serve for RawTap {
    type Family = family::Dynamic;

    /// The model's own: the tap changes nothing a replay would check.
    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let started = Instant::now();
        if let EffectKind::Completion { request, .. } = &kind {
            self.sink.send(Entry::Request(Box::new(request.clone())));
        }
        let sink = self.sink.clone();
        let ms = move || started.elapsed().as_millis();
        match self.inner.handle(kind, dispatch).await {
            Reply::Outcome(outcome) => {
                let ms = ms();
                sink.send(Entry::Answer {
                    ms,
                    outcome: outcome.clone(),
                });
                Reply::Outcome(outcome)
            }
            Reply::Stream(events) => Reply::Stream(
                events
                    .inspect(move |item| {
                        let ms = ms();
                        sink.send(Entry::Item {
                            ms,
                            item: item.clone(),
                        });
                    })
                    .boxed(),
            ),
        }
    }
}

/// `/rawlog on|off|last`.
fn rawlog(
    In(args): In<CommandArgs>,
    writer: Option<Res<Writer>>,
    mut log: ResMut<RawLog>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Some(writer) = writer else {
        notices.write(Notice::error(agent, "/rawlog needs a session directory."));
        return;
    };
    let text = match args.args.as_str() {
        "on" => {
            writer.lines.try_send(Line::Resume).ok();
            log.on = true;
            format!("Recording model calls in {}.", writer.dir.display())
        }
        "off" => {
            log.on = false;
            "Model calls are no longer recorded.".to_owned()
        }
        "last" => {
            let newest = writer.newest.lock().unwrap_or_else(PoisonError::into_inner);
            match newest.as_deref() {
                Some(newest) => newest.display().to_string(),
                None => "Nothing is recorded yet.".to_owned(),
            }
        }
        "" => format!("The raw log is {}.", if log.on { "on" } else { "off" }),
        _ => {
            notices.write(Notice::error(agent, "Usage: /rawlog on, off or last."));
            return;
        }
    };
    notices.write(Notice::info(agent, text));
}

/// The writer thread's state: the raw folder's files, oldest first, with
/// their sizes.
struct Files {
    dir: PathBuf,
    order: VecDeque<(PathBuf, u64)>,
    total: u64,
    open: HashMap<PathBuf, File>,
    trimmed: bool,
    failed: bool,
    newest: Arc<Mutex<Option<PathBuf>>>,
    reports: Sender<WriterEvent>,
}

impl Files {
    /// The files an earlier run left in `dir`, oldest first.
    fn scan(
        dir: PathBuf,
        newest: Arc<Mutex<Option<PathBuf>>>,
        reports: Sender<WriterEvent>,
    ) -> Self {
        let mut found: Vec<(std::time::SystemTime, PathBuf, u64)> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let meta = entry.metadata().ok()?;
                Some((meta.modified().ok()?, entry.path(), meta.len()))
            })
            .collect();
        found.sort();
        let order: VecDeque<_> = found
            .into_iter()
            .map(|(_, path, size)| (path, size))
            .collect();
        if let Some((last, _)) = order.back() {
            *newest.lock().unwrap_or_else(PoisonError::into_inner) = Some(last.clone());
        }
        Self {
            total: order.iter().map(|(_, size)| size).sum(),
            dir,
            order,
            open: HashMap::new(),
            trimmed: false,
            failed: false,
            newest,
            reports,
        }
    }

    /// Writes each line it is sent until the app is gone.
    fn run(mut self, lines: &Receiver<Line>) {
        while let Ok(line) = lines.recv_blocking() {
            match line {
                Line::Resume => self.failed = false,
                Line::Entry { .. } if self.failed => {}
                Line::Entry {
                    file,
                    attempt,
                    entry,
                } => {
                    let text = render(attempt, entry);
                    if let Err(why) = self.write(&self.dir.join(file), &text) {
                        self.failed = true;
                        self.open.clear();
                        let failed = WriterEvent::Failed(why.to_string());
                        self.reports.send_blocking(failed).ok();
                    }
                }
            }
        }
    }

    /// Appends `text` as a line of `path`, then trims the folder.
    fn write(&mut self, path: &Path, text: &str) -> std::io::Result<()> {
        if !self.open.contains_key(path) {
            std::fs::create_dir_all(&self.dir)?;
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            if !self.order.iter().any(|(known, _)| known == path) {
                self.order.push_back((path.to_owned(), 0));
                *self.newest.lock().unwrap_or_else(PoisonError::into_inner) = Some(path.to_owned());
            }
            self.open.insert(path.to_owned(), file);
        }
        let line = format!("{text}\n");
        if let Some(file) = self.open.get_mut(path) {
            file.write_all(line.as_bytes())?;
        }
        let written = line.len() as u64;
        if let Some((_, size)) = self.order.iter_mut().find(|(known, _)| known == path) {
            *size += written;
        }
        self.total += written;
        self.trim(path)
    }

    /// Removes the oldest files but `current` while the folder is over the
    /// cap, saying so the first time.
    fn trim(&mut self, current: &Path) -> std::io::Result<()> {
        while self.total > CAP {
            let Some(at) = self.order.iter().position(|(path, _)| path != current) else {
                break;
            };
            let Some((oldest, size)) = self.order.remove(at) else {
                break;
            };
            self.open.remove(&oldest);
            std::fs::remove_file(&oldest)?;
            self.total = self.total.saturating_sub(size);
            if !self.trimmed {
                self.trimmed = true;
                self.reports.send_blocking(WriterEvent::Trimmed).ok();
            }
        }
        Ok(())
    }
}

/// The JSON line of `entry`, its secrets redacted.
fn render(attempt: u32, entry: Entry) -> String {
    let (ms, mut line) = match entry {
        Entry::Request(request) => (None, json!({ "type": "request", "body": request })),
        Entry::Item { ms, item } => (
            Some(ms),
            match item {
                Ok(Relayed::Origin(origin)) => json!({ "type": "origin", "origin": origin }),
                Ok(Relayed::Item(Item::Event(event))) => json!({ "type": "event", "event": event }),
                Ok(Relayed::Item(Item::Unknown(item))) => {
                    json!({ "type": "unknown", "item": item.value() })
                }
                Ok(Relayed::Done(response)) => json!({
                    "type": "done",
                    "request_id": response.provider_request_id,
                    "response": response,
                }),
                Err(error) => json!({ "type": "error", "error": error }),
            },
        ),
        Entry::Answer { ms, outcome } => (
            Some(ms),
            match outcome {
                Ok(outcome) => json!({ "type": "answer", "outcome": outcome }),
                Err(error) => json!({ "type": "error", "error": error }),
            },
        ),
    };
    if let Value::Object(fields) = &mut line {
        if let Some(ms) = ms {
            fields.insert("ms".to_owned(), json!(ms));
        }
        fields.insert("attempt".to_owned(), json!(attempt));
    }
    redact(&mut line);
    line.to_string()
}

/// What a secret looks like.
static SECRETS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{22,})",
        r"\bsk-[A-Za-z0-9_-]{6,}",
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
        r"\bAIza[0-9A-Za-z_-]{30,}",
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}",
        r#"(?i)\b(?:api[_-]?key|token|secret|password|passwd)["']?\s*[:=]\s*["']?[^\s"',;&]{6,}"#,
    ]
    .into_iter()
    .filter_map(|pattern| Regex::new(pattern).ok())
    .collect()
});

/// Every string in `value` with its secrets redacted.
fn redact(value: &mut Value) {
    match value {
        Value::String(text) => {
            for secret in SECRETS.iter() {
                if secret.is_match(text) {
                    *text = secret.replace_all(text, REDACTED).into_owned();
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(redact),
        Value::Object(fields) => fields.values_mut().for_each(redact),
        _ => {}
    }
}
