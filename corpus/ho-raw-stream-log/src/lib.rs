//! Plugin corpus (held-out): H26: Raw provider stream log (spec
//! `H26-raw-stream-log.md`).
//!
//! `/rawlog on|off` (or `--rawlog`) records every model call, an agent's,
//! a subagent's or a plugin's own, to `<session>/raw/<turn>-<call>-<model>.jsonl`:
//! each HTTP send as it went out (the body, no headers), the response's
//! status and headers, then each chunk of the body as it came, with the
//! milliseconds since the send, then the error, if any. Every send is a
//! record of its own in the call's file: a retry of the call, or a resend
//! inside one dispatch.
//!
//! - Every model is reached through a `Tee`: the plugin wraps the
//!   [`Models`] transport once, at build. The tee passes everything on
//!   unchanged and hands a copy of each send of a recorded call (its
//!   [`Serving`] names the call's effect) to a writer thread.
//! - Prepare: each model call admitted while the log is on is recorded by
//!   its effect id, until its entity is gone; a call that supersedes a
//!   recorded one, such as a retry, shares its file.
//! - The writer thread redacts secrets ([`redact`]), writes the
//!   lines, keeps the folder under 50 MB (oldest files first) and reports
//!   back as a stream on an entity: one notice when it first trims, and one
//!   warning when a write fails, which turns the log off.
//! - Whether the log is on is a saved resource.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Instant;

use async_channel::{Receiver, Sender};
use bevy_tasks::{IoTaskPool, TaskPool};
use futures::StreamExt as _;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::http_client::{
    self, BoxedStream, Bytes, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request,
    Response, StreamingResponse,
};
use rig_harness::rig_core::redact;
use rig_harness::rig_core::serve::Serving;
use rig_harness::rig_core::wasm_compat::WasmCompatSend;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// The command's and the flag's name.
const NAME: &str = "rawlog";

/// The most the session's raw folder holds.
const CAP: u64 = 50 * 1024 * 1024;

/// Records every model call's HTTP traffic.
#[derive(Default)]
pub struct RawStreamLogPlugin;

/// The plugin's stage system, for others to order against: it records
/// each admitted model call.
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
            "Record each model call's HTTP traffic in the session's raw/ folder",
        ));
        let flagged = flagged(app.world());
        let recorded = Recorded::default();
        let tee = recorded.clone();
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .wrap_http(move |inner| {
                DynHttpClient::new(Tee {
                    inner,
                    recorded: tee.clone(),
                })
            });
        app.insert_resource(recorded)
            .insert_resource(RawLog {
                on: flagged,
                calls: 0,
            })
            .register_type::<RawLog>()
            .configure_sets(AgentPass, RawLogSystems.in_set(Stage::Prepare))
            .add_systems(Startup, start_writer)
            .add_systems(Update, keep_flag.run_if(resource_added::<SessionRestored>))
            .add_systems(AgentPass, record_model_calls.in_set(RawLogSystems))
            .add_observer(forget_call)
            .add_observer(writer_reported)
            .add_command(
                NAME,
                "Record each model call's HTTP traffic: /rawlog on, off or last",
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
    /// The request as it went out, without its headers.
    Request {
        method: String,
        uri: String,
        body: Bytes,
    },
    /// The response's status and headers, `ms` after the send.
    Response {
        ms: u128,
        status: u16,
        headers: Vec<(String, String)>,
    },
    /// A chunk of the response body, `ms` after the send.
    Chunk { ms: u128, bytes: Bytes },
    /// The send or the body failed, `ms` after the send.
    Error { ms: u128, error: String },
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
        commands.write_message(Notice::warn(None, "The raw log's writer could not start."));
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
            notices.write(Notice::warn(
                None,
                format!("The raw log could not write ({why}); it is off."),
            ))
        }
    };
}

/// A recorded model call and the calls that retry it: their file, named
/// on the first send's model, and how many sends went into it.
struct Record {
    stem: String,
    name: OnceLock<String>,
    sends: AtomicU32,
    lines: Sender<Line>,
}

/// The recorded model calls, by effect id: the tee looks each send's
/// [`Serving`] up here.
#[derive(Resource, Clone, Default)]
struct Recorded(Arc<Mutex<HashMap<EffectId, Arc<Record>>>>);

impl Recorded {
    fn lock(&self) -> MutexGuard<'_, HashMap<EffectId, Arc<Record>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The record of a model call, kept while the call's entity lives.
#[derive(Component, Clone)]
struct RawFile(Arc<Record>);

/// Records each model call admitted while the log is on; a call that
/// supersedes a recorded one keeps its record.
fn record_model_calls(
    calls: Query<
        (Entity, &Effect, Option<&CallOf>, Option<&Supersedes>),
        (Added<Admitted>, ModelCalls),
    >,
    files: Query<&RawFile>,
    turns: Query<&Turn>,
    writer: Option<Res<Writer>>,
    recorded: Res<Recorded>,
    mut log: ResMut<RawLog>,
    mut commands: Commands,
) {
    let Some(writer) = writer.filter(|_| log.on) else {
        return;
    };
    for (call, effect, of, supersedes) in &calls {
        let earlier =
            supersedes.and_then(|of| of.iter().find_map(|earlier| files.get(earlier).ok()));
        let file = match earlier {
            Some(earlier) => earlier.clone(),
            None => {
                log.calls += 1;
                let turn = of.and_then(|&CallOf(turn)| turns.get(turn).ok());
                RawFile(Arc::new(Record {
                    stem: format!("{}-{}", turn.map_or(0, |turn| turn.index), log.calls),
                    name: OnceLock::new(),
                    sends: AtomicU32::new(0),
                    lines: writer.lines.clone(),
                }))
            }
        };
        recorded.lock().insert(effect.id(), file.0.clone());
        commands.entity(call).insert(file);
    }
}

/// A recorded call's entity is gone: its sends are over.
fn forget_call(gone: On<Remove<RawFile>>, effects: Query<&Effect>, recorded: Res<Recorded>) {
    if let Ok(effect) = effects.get(gone.entity) {
        recorded.lock().remove(&effect.id());
    }
}

/// `text` as a part of a file name.
fn file_safe(text: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_');
    text.chars()
        .map(|c| if safe(c) { c } else { '_' })
        .collect()
}

/// One send's lines, when it belongs to a recorded call.
#[derive(Clone)]
struct Tap(Option<Arc<Sending>>);

/// A recorded send: its file, its attempt and when it started.
struct Sending {
    file: String,
    attempt: u32,
    started: Instant,
    lines: Sender<Line>,
}

impl Tap {
    /// Starts recording `request` when the call it serves is recorded,
    /// naming the call's file on the first send's model.
    fn begin(recorded: &Recorded, request: &Request<Bytes>) -> Self {
        let Some(serving) = request.extensions().get::<Serving>() else {
            return Self(None);
        };
        let Some(record) = recorded.lock().get(&serving.effect).cloned() else {
            return Self(None);
        };
        let name = record.name.get_or_init(|| {
            format!(
                "{}-{}.jsonl",
                record.stem,
                file_safe(serving.handler.as_str())
            )
        });
        let attempt = record
            .sends
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        let sending = Sending {
            file: name.clone(),
            attempt,
            started: Instant::now(),
            lines: record.lines.clone(),
        };
        let tap = Self(Some(Arc::new(sending)));
        tap.send(|| Entry::Request {
            method: request.method().to_string(),
            uri: request.uri().to_string(),
            body: request.body().clone(),
        });
        tap
    }

    /// Hands the entry `make` builds to the writer thread, without waiting.
    fn send(&self, make: impl FnOnce() -> Entry) {
        if let Some(sending) = &self.0 {
            let line = Line::Entry {
                file: sending.file.clone(),
                attempt: sending.attempt,
                entry: make(),
            };
            sending.lines.try_send(line).ok();
        }
    }

    /// The milliseconds since the send started.
    fn ms(&self) -> u128 {
        self.0
            .as_ref()
            .map_or(0, |sending| sending.started.elapsed().as_millis())
    }

    /// Records the response's status and headers.
    fn response<B>(&self, response: &Response<B>) {
        self.send(|| Entry::Response {
            ms: self.ms(),
            status: response.status().as_u16(),
            headers: response
                .headers()
                .iter()
                .filter(|(name, _)| *name != "set-cookie")
                .map(|(name, value)| {
                    let value = String::from_utf8_lossy(value.as_bytes()).into_owned();
                    (name.to_string(), value)
                })
                .collect(),
        });
    }

    /// Records a chunk of the body, or its failure.
    fn chunk(&self, chunk: &http_client::Result<Bytes>) {
        self.send(|| match chunk {
            Ok(bytes) => Entry::Chunk {
                ms: self.ms(),
                bytes: bytes.clone(),
            },
            Err(error) => self.error(error),
        });
    }

    /// The failure `error`, as an entry.
    fn error(&self, error: &http_client::Error) -> Entry {
        Entry::Error {
            ms: self.ms(),
            error: error.to_string(),
        }
    }

    /// Records the response, or the failure, a send ended with.
    fn sent<B>(&self, sent: &http_client::Result<Response<B>>) {
        match sent {
            Ok(response) => self.response(response),
            Err(error) => self.send(|| self.error(error)),
        }
    }
}

/// An HTTP client whose sends for recorded calls are copied to the log.
struct Tee {
    inner: DynHttpClient,
    recorded: Recorded,
}

impl HttpClientExt for Tee {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let req = req.map(Into::into);
        let tap = Tap::begin(&self.recorded, &req);
        let sent = self.inner.send::<Bytes, Bytes>(req);
        async move {
            let sent = sent.await;
            tap.sent(&sent);
            sent.map(|response| {
                response.map(|body| -> LazyBody<U> {
                    Box::pin(async move {
                        let body = body.await;
                        tap.chunk(&body);
                        body.map(U::from)
                    })
                })
            })
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let req = req.map(Into::into);
        let tap = Tap::begin(&self.recorded, &req);
        let sent = self.inner.send_streaming(req);
        async move {
            let sent = sent.await;
            tap.sent(&sent);
            sent.map(|response| {
                response.map(|body| -> BoxedStream {
                    Box::pin(body.inspect(move |chunk| tap.chunk(chunk)))
                })
            })
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

/// The JSON line of `entry`, its secrets redacted: a JSON request body as
/// JSON, any other body and every chunk as text.
fn render(attempt: u32, entry: Entry) -> String {
    let text = |bytes: &Bytes| String::from_utf8_lossy(bytes).into_owned();
    let mut line = match entry {
        Entry::Request { method, uri, body } => json!({
            "type": "request",
            "method": method,
            "uri": uri,
            "body": serde_json::from_slice::<Value>(&body).unwrap_or_else(|_| json!(text(&body))),
        }),
        Entry::Response {
            ms,
            status,
            headers,
        } => json!({ "type": "response", "ms": ms, "status": status, "headers": headers }),
        Entry::Chunk { ms, bytes } => json!({ "type": "chunk", "ms": ms, "data": text(&bytes) }),
        Entry::Error { ms, error } => json!({ "type": "error", "ms": ms, "error": error }),
    };
    if let Value::Object(fields) = &mut line {
        fields.insert("attempt".to_owned(), json!(attempt));
    }
    redact::json(&mut line);
    line.to_string()
}
