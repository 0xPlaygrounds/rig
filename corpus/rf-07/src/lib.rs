//! `rerun`: record an agent run with a checkpoint after every turn, replay
//! it offline from any checkpoint, or fork it into a new recording that
//! continues against another model.
//!
//! A run directory holds plain JSON files:
//!
//! - `log.json`: the stamped [`EffectLog`], stream items kept;
//! - `run.json`: the [`RunMeta`] (prompt, run spec, outcome, stream transcript);
//! - `checkpoint-<k>.json`: a [`Checkpoint`] holding the [`AgentRun`] right
//!   after model turn `k` was taken in, its tool calls still pending;
//! - `manifest.json`: a hash of every other file, so an edited or truncated
//!   file is refused.

use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use futures::StreamExt;
use rig_agent::{
    AgentBuilder,
    agent::{AgentHook, DispatchAction, DispatchEvent, HookContext, MultiTurnStreamItem},
    handlers::Handlers,
    run::{
        AgentRun, AgentRunStep, ModelTurn, PendingToolCall, PromptResponse, RunSpec,
        prepare_request,
    },
    tool::{RegisteredTool, Tool, ToolContext, ToolExecutionError, server::ToolServer},
};
use rig_cassette::{
    agent::AgentReplayExt,
    effect_log::{Checkpoint, EffectLog, EffectLogRecorder, EffectLogReplayer, stable_hash},
};
use rig_core::{
    completion::{ProviderCapabilities, Usage},
    effect::{Answered, EffectFamily, EffectKind, EffectRecord, FamilyDescriptor, HandlerKey},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(test)]
mod tests;

/// The owner every key of a `rerun` agent is minted under, so a recording
/// and a replay name the same handler keys.
pub const OWNER: &str = "rerun";

/// The version of the files this program writes and reads.
pub const FORMAT: u32 = 1;

/// Why a record, replay or fork could not go on.
#[derive(Debug, thiserror::Error)]
pub enum RerunError {
    /// A file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: String,
        /// The failure.
        source: std::io::Error,
    },
    /// A file is not what this program wrote: truncated, edited, or of
    /// another version.
    #[error("{path}: {reason}")]
    Corrupt {
        /// The file.
        path: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The program changed in a way the recording cannot answer.
    #[error("replay refused: {0}")]
    Incompatible(String),
    /// No checkpoint at the asked turn.
    #[error("no checkpoint after turn {0}")]
    NoCheckpoint(usize),
    /// Rig refused or failed.
    #[error("{0}")]
    Rig(String),
}

fn rig(error: impl std::fmt::Display) -> RerunError {
    RerunError::Rig(error.to_string())
}

/// The shared row counter the `write_row` tool increments: the side effect
/// a replay must leave alone.
#[derive(Clone, Debug, Default)]
pub struct Rows(Arc<AtomicU64>);

impl Rows {
    /// Rows written so far.
    pub fn count(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// `write_row`'s arguments.
#[derive(Deserialize)]
pub struct RowArgs {
    /// The row to write.
    pub row: String,
}

/// Writes a row: increments [`Rows`] and echoes the row back.
#[derive(Clone, Debug, Default)]
pub struct WriteRow {
    /// Where rows are counted.
    pub rows: Rows,
}

impl Tool for WriteRow {
    const NAME: &'static str = "write_row";
    type Args = RowArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "writes one row to the database".into()
    }

    fn parameters(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {"row": {"type": "string"}}, "required": ["row"]})
    }

    async fn call(&self, _context: &mut ToolContext, args: RowArgs) -> Result<String, Self::Error> {
        let n = self.rows.0.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(format!("wrote row {n}: {}", args.row))
    }
}

/// An observing hook: counts dispatches and never changes one.
#[derive(Clone, Debug, Default)]
pub struct Observer(pub Arc<AtomicU64>);

impl AgentHook for Observer {
    async fn on_dispatch(&self, _ctx: &HookContext, _event: DispatchEvent<'_>) -> DispatchAction {
        self.0.fetch_add(1, Ordering::SeqCst);
        DispatchAction::proceed()
    }
}

/// What the program is: everything a replay must agree with, except the
/// model and the tool bodies.
#[derive(Clone, Debug)]
pub struct Program {
    /// The system prompt.
    pub preamble: String,
    /// The task.
    pub prompt: String,
    /// The model-call budget.
    pub max_turns: usize,
    /// Whether the run streams.
    pub streaming: bool,
    /// An observing hook, when the program installs one.
    pub observer: Option<Observer>,
}

impl Program {
    fn configure(&self, builder: AgentBuilder) -> AgentBuilder {
        let builder = builder.preamble(&self.preamble);
        match &self.observer {
            Some(observer) => builder.add_hook(observer.clone()),
            None => builder,
        }
    }

    /// The tools the program offers, as the log's handler table describes
    /// tools: name, description, argument schema.
    fn tool_descriptors(&self) -> Vec<(String, String, Value)> {
        let tool = WriteRow::default();
        vec![(
            WriteRow::NAME.to_owned(),
            tool.description(),
            tool.parameters(),
        )]
    }
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ending {
    /// The final answer and the run's usage totals.
    Answer {
        /// The final text.
        output: String,
        /// Usage totals.
        usage: Usage,
    },
    /// The error the run ended with.
    Error(String),
}

/// `run.json`: what the log alone does not say about a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunMeta {
    /// [`FORMAT`].
    pub format: u32,
    /// The task.
    pub prompt: String,
    /// The model-call budget.
    pub max_turns: usize,
    /// Whether the run streamed.
    pub streaming: bool,
    /// The agent's run spec (preamble, output mode, sampling...), as JSON.
    pub spec: Value,
    /// How the run ended.
    pub ending: Ending,
    /// The stream items a UI saw, one JSON line each, then the error
    /// that ended the stream, if one did.
    pub transcript: Vec<String>,
}

/// What a replay found.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Every request, tool call, result, stream item, the final answer and
    /// the usage totals match the recording.
    Identical,
    /// The first difference.
    Differs {
        /// The turn it is in (1-based model call).
        turn: usize,
        /// The step within the turn: 0 is the model call, 1.. its tool calls.
        step: usize,
        /// What changed.
        what: String,
    },
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Identical => f.write_str("identical"),
            Self::Differs { turn, step, what } => {
                write!(f, "differs at turn {turn}, step {step}: {what}")
            }
        }
    }
}

/// How a driven run came out: its ending and stream transcript.
struct Driven {
    ending: Ending,
    transcript: Vec<String>,
}

async fn drive(runner: rig_agent::agent::AgentRunner, streaming: bool) -> Driven {
    if !streaming {
        let ending = match runner.run().await {
            Ok(response) => answer(&response),
            Err(error) => Ending::Error(error.to_string()),
        };
        return Driven {
            ending,
            transcript: Vec::new(),
        };
    }
    let mut stream = runner.stream();
    let mut transcript = Vec::new();
    let mut ending = Ending::Error("the stream ended without a final response".into());
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(item)) => {
                transcript
                    .push(serde_json::to_string(&item).unwrap_or_else(|error| error.to_string()));
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => ending = answer(&response),
            Ok(_) => {}
            Err(error) => {
                transcript.push(format!("error: {error}"));
                ending = Ending::Error(error.to_string());
                break;
            }
        }
    }
    Driven { ending, transcript }
}

fn answer(response: &PromptResponse) -> Ending {
    Ending::Answer {
        output: response.output(),
        usage: response.usage,
    }
}

/// Record `program` against `model` into `dir`, with a checkpoint after
/// every turn. `redact` removes secrets rig recognizes from every file
/// before it is written.
pub async fn record(
    program: &Program,
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    rows: Rows,
    dir: &Path,
    redact: bool,
) -> Result<Ending, RerunError> {
    let recorder = EffectLogRecorder::keeping_stream_events();
    let agent = program
        .configure(AgentBuilder::named_model("default", model).owner(OWNER))
        .tool(WriteRow { rows })
        .record_to(recorder.clone())
        .build();
    let driven = drive(
        agent
            .prompt(program.prompt.as_str())
            .max_turns(program.max_turns),
        program.streaming,
    )
    .await;
    let log = agent.stamp(recorder.take());
    let meta = RunMeta {
        format: FORMAT,
        prompt: program.prompt.clone(),
        max_turns: program.max_turns,
        streaming: program.streaming,
        spec: serde_json::to_value(agent.run_spec()).map_err(rig)?,
        ending: driven.ending.clone(),
        transcript: driven.transcript,
    };
    write_run(dir, &meta, &log, redact)?;
    Ok(driven.ending)
}

/// Write a run directory: the log, the meta, a checkpoint per turn, the
/// manifest.
fn write_run(dir: &Path, meta: &RunMeta, log: &EffectLog, redact: bool) -> Result<(), RerunError> {
    fs::create_dir_all(dir).map_err(|source| io(dir, source))?;
    let mut manifest = BTreeMap::new();
    let mut put = |name: String, value: Value| -> Result<(), RerunError> {
        let mut value = value;
        if redact {
            rig_core::redact::json(&mut value);
        }
        let text = serde_json::to_string_pretty(&value).map_err(rig)?;
        let path = dir.join(&name);
        fs::write(&path, &text).map_err(|source| io(&path, source))?;
        manifest.insert(name, stable_hash(&text).map_err(rig)?);
        Ok(())
    };
    put("log.json".into(), serde_json::to_value(log).map_err(rig)?)?;
    put("run.json".into(), serde_json::to_value(meta).map_err(rig)?)?;
    for (turn, checkpoint) in checkpoints(meta, log)? {
        put(
            format!("checkpoint-{turn}.json"),
            serde_json::to_value(&checkpoint).map_err(rig)?,
        )?;
    }
    let manifest = Manifest {
        format: FORMAT,
        files: manifest,
    };
    let path = dir.join("manifest.json");
    let text = serde_json::to_string_pretty(&manifest).map_err(rig)?;
    fs::write(&path, text).map_err(|source| io(&path, source))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: u32,
    files: BTreeMap<String, u64>,
}

fn io(path: &Path, source: std::io::Error) -> RerunError {
    RerunError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// The run state after each model turn, rebuilt by stepping an
/// [`AgentRun`] through the log's records: the classic agent never hands
/// its run state out between turns, so the turn boundaries are refolded
/// from what it recorded.
fn checkpoints(
    meta: &RunMeta,
    log: &EffectLog,
) -> Result<Vec<(usize, Checkpoint<AgentRun>)>, RerunError> {
    let mut spec: RunSpec = serde_json::from_value(meta.spec.clone()).map_err(rig)?;
    spec.max_turns = Some(meta.max_turns);
    let mut run = AgentRun::from_spec(&spec, meta.prompt.as_str(), None);
    let mut records = log.records.iter().enumerate();
    let mut out = Vec::new();
    loop {
        match run.next_step().map_err(rig)? {
            AgentRunStep::CallModel { history, turn, .. } => {
                let Some((at, record)) = records.next() else {
                    return Ok(out);
                };
                let (EffectKind::Completion { request, .. }, Ok(Answered::Completion(response))) =
                    (&record.kind, &record.outcome)
                else {
                    // The turn failed: the run ended here.
                    return Ok(out);
                };
                let capabilities = capabilities(log, &record.key);
                let prepared = prepare_request(
                    &spec,
                    &capabilities,
                    &history,
                    request.tools.clone(),
                    run.output_tool_name(),
                    None,
                )
                .map_err(rig)?;
                run.advertise_tools(turn, prepared.tools.clone());
                run.model_response(ModelTurn::from_policy(response, prepared.policy))
                    .map_err(rig)?;
                out.push((turn, log.checkpoint(at + 1, run.clone()).0));
            }
            AgentRunStep::CallTools { calls } => {
                for call in calls {
                    let PendingToolCall::Execute(call) = call else {
                        return Ok(out);
                    };
                    let Some((_, record)) = records.next() else {
                        return Ok(out);
                    };
                    let Ok(Answered::ToolResult { result }) = &record.outcome else {
                        return Ok(out);
                    };
                    run.answer(call.answer(result.clone())).map_err(rig)?;
                }
            }
            AgentRunStep::Done(_) => return Ok(out),
        }
    }
}

fn capabilities(log: &EffectLog, key: &HandlerKey) -> ProviderCapabilities {
    log.header
        .handlers
        .iter()
        .find_map(|handler| match &handler.family {
            FamilyDescriptor::Completion { capabilities, .. } if &handler.key == key => {
                Some(*capabilities)
            }
            _ => None,
        })
        .unwrap_or_default()
}

/// A run directory read back: the files, checked against the manifest.
pub struct Loaded {
    /// `run.json`.
    pub meta: RunMeta,
    /// `log.json`.
    pub log: EffectLog,
    /// The checkpoint asked for.
    pub checkpoint: Checkpoint<AgentRun>,
    /// The log after the checkpoint, checked to follow it.
    pub tail: EffectLog,
}

fn read(dir: &Path, name: &str) -> Result<String, RerunError> {
    let path = dir.join(name);
    fs::read_to_string(&path).map_err(|source| io(&path, source))
}

fn parse<T: serde::de::DeserializeOwned>(name: &str, text: &str) -> Result<T, RerunError> {
    serde_json::from_str(text).map_err(|error| RerunError::Corrupt {
        path: name.to_owned(),
        reason: error.to_string(),
    })
}

/// Read `dir` and its checkpoint after turn `from`. Any truncated, edited
/// or wrong-version file is an error; nothing is replayed.
pub fn load(dir: &Path, from: usize) -> Result<Loaded, RerunError> {
    let manifest: Manifest = parse("manifest.json", &read(dir, "manifest.json")?)?;
    if manifest.format != FORMAT {
        return Err(RerunError::Corrupt {
            path: "manifest.json".into(),
            reason: format!("format {}, this program reads {FORMAT}", manifest.format),
        });
    }
    let checkpoint_name = format!("checkpoint-{from}.json");
    let mut texts = BTreeMap::new();
    for name in ["log.json", "run.json", checkpoint_name.as_str()] {
        let text = match read(dir, name) {
            Ok(text) => text,
            Err(_) if name == checkpoint_name => return Err(RerunError::NoCheckpoint(from)),
            Err(error) => return Err(error),
        };
        texts.insert(name.to_owned(), text);
    }
    let text = |name: &str| texts.get(name).map(String::as_str).unwrap_or_default();
    let log: EffectLog = parse("log.json", text("log.json"))?;
    let meta: RunMeta = parse("run.json", text("run.json"))?;
    let checkpoint: Checkpoint<AgentRun> = parse(&checkpoint_name, text(&checkpoint_name))?;
    if meta.format != FORMAT {
        return Err(RerunError::Corrupt {
            path: "run.json".into(),
            reason: format!("format {}, this program reads {FORMAT}", meta.format),
        });
    }
    let tail =
        EffectLog::from_checkpoint(&checkpoint, log.tail(checkpoint.at)).map_err(|error| {
            RerunError::Corrupt {
                path: checkpoint_name.clone(),
                reason: error.to_string(),
            }
        })?;
    for (name, text) in &texts {
        let recorded = manifest.files.get(name).copied();
        if recorded != Some(stable_hash(text).map_err(rig)?) {
            return Err(RerunError::Corrupt {
                path: name.clone(),
                reason: "the file differs from the one written (edited or truncated)".into(),
            });
        }
    }
    Ok(Loaded {
        meta,
        log,
        checkpoint,
        tail,
    })
}

/// Refuse a program the recording cannot answer: another run spec
/// (preamble, output mode, sampling...), tool set or tool schema. Added
/// hooks are allowed; a recorded hook that is gone is not.
fn check_compatible(
    program: &Program,
    spec: &Value,
    loaded: &Loaded,
    agent: &rig_agent::Agent,
) -> Result<(), RerunError> {
    if program.streaming != loaded.meta.streaming {
        return Err(RerunError::Incompatible(format!(
            "the recording streamed: {}, this program streams: {}",
            loaded.meta.streaming, program.streaming
        )));
    }
    if let (Value::Object(recorded), Value::Object(mine)) = (&loaded.meta.spec, spec) {
        for field in recorded.keys().chain(mine.keys()) {
            if recorded.get(field) != mine.get(field) {
                return Err(RerunError::Incompatible(format!(
                    "the run spec's `{field}` changed: recorded {}, now {}",
                    recorded.get(field).unwrap_or(&Value::Null),
                    mine.get(field).unwrap_or(&Value::Null)
                )));
            }
        }
    } else if &loaded.meta.spec != spec {
        return Err(RerunError::Incompatible("the run spec changed".into()));
    }
    let recorded: Vec<(String, String, Value)> = loaded
        .log
        .header
        .handlers
        .iter()
        .filter_map(|handler| match &handler.family {
            FamilyDescriptor::Tool {
                name,
                description,
                parameters,
                ..
            } => Some((name.clone(), description.clone(), parameters.clone())),
            _ => None,
        })
        .collect();
    let mine = program.tool_descriptors();
    if recorded != mine {
        return Err(RerunError::Incompatible(format!(
            "the tool set or a tool schema changed: recorded {recorded:?}, now {mine:?}"
        )));
    }
    let now = agent.program_names();
    let mut remaining = now.iter();
    for hook in &loaded.log.header.hooks {
        if !remaining.any(|name| name == hook) {
            return Err(RerunError::Incompatible(format!(
                "the recorded hook `{hook}` is not in this program's hook stack {now:?}"
            )));
        }
    }
    Ok(())
}

/// Replay `dir` from its checkpoint after turn `from`, with no model and
/// no tool body behind the keys: every answer comes from the log.
pub async fn replay(program: &Program, dir: &Path, from: usize) -> Result<Verdict, RerunError> {
    let loaded = load(dir, from)?;
    let tail = &loaded.tail;
    let model_key = loaded
        .log
        .records
        .iter()
        .find(|record| record.kind.family() == EffectFamily::Completion)
        .map(|record| record.key.clone())
        .ok_or_else(|| RerunError::Rig("the log has no model call".into()))?;
    let handlers = Handlers::default();
    rig_cassette::agent::replay::register_all(tail, &handlers).map_err(rig)?;
    let server = ToolServer::new().run();
    for handler in &loaded.log.header.handlers {
        if handler.family.family() == EffectFamily::Tool {
            let replayer = EffectLogReplayer::for_key(tail, &handler.key).map_err(rig)?;
            server.add_registered_tool(RegisteredTool::from_handler(replayer).map_err(rig)?);
        }
    }
    let recorder = EffectLogRecorder::keeping_stream_events();
    handlers.record_to(recorder.clone());
    let agent = program
        .configure(AgentBuilder::over_handlers(handlers, OWNER, model_key))
        .tool_server_handle(server)
        .build();
    let spec = serde_json::to_value(agent.run_spec()).map_err(rig)?;
    check_compatible(program, &spec, &loaded, &agent)?;

    let driven = drive(
        agent.resume(loaded.checkpoint.state.clone()),
        program.streaming,
    )
    .await;
    let replayed = recorder.take();
    Ok(compare(&loaded, &replayed, &driven))
}

/// The turn and step of record `index` of `log`.
fn place(log: &EffectLog, index: usize) -> (usize, usize) {
    let (mut turn, mut step) = (0, 0);
    for record in log.records.iter().take(index + 1) {
        if record.kind.family() == EffectFamily::Completion {
            turn += 1;
            step = 0;
        } else {
            step += 1;
        }
    }
    (turn, step)
}

fn as_data(record: &EffectRecord) -> [(&'static str, Value); 4] {
    let value = |value: Result<Value, serde_json::Error>| {
        rig_cassette::effect_log::canonical_local_ids(value.unwrap_or(Value::Null))
    };
    [
        ("request", value(serde_json::to_value(&record.kind))),
        ("outcome", value(serde_json::to_value(&record.outcome))),
        ("stream items", value(serde_json::to_value(&record.events))),
        ("handler", Value::String(record.key.as_str().to_owned())),
    ]
}

fn compare(loaded: &Loaded, replayed: &EffectLog, driven: &Driven) -> Verdict {
    let at = loaded.checkpoint.at;
    let last = place(&loaded.log, loaded.log.records.len().saturating_sub(1));
    for (offset, original) in loaded.tail.records.iter().enumerate() {
        let (turn, step) = place(&loaded.log, at + offset);
        let Some(new) = replayed.records.get(offset) else {
            return Verdict::Differs {
                turn,
                step,
                what: format!(
                    "the replay stopped before this effect: {}",
                    ending_text(&driven.ending)
                ),
            };
        };
        for ((field, a), (_, b)) in as_data(original).into_iter().zip(as_data(new)) {
            if a != b {
                return Verdict::Differs {
                    turn,
                    step,
                    what: format!("{field} changed: recorded {a}, replayed {b}"),
                };
            }
        }
        let errors = |log: &EffectLog, id| log.header.stream_errors.get(&id).cloned();
        if errors(&loaded.log, original.id) != errors(replayed, new.id) {
            return Verdict::Differs {
                turn,
                step,
                what: "the stream's error items changed".into(),
            };
        }
    }
    if replayed.records.len() > loaded.tail.records.len() {
        return Verdict::Differs {
            turn: last.0 + 1,
            step: 0,
            what: "the replay made effects the recording never did".into(),
        };
    }
    if driven.ending != loaded.meta.ending {
        return Verdict::Differs {
            turn: last.0,
            step: last.1,
            what: format!(
                "the ending changed: recorded {}, replayed {}",
                ending_text(&loaded.meta.ending),
                ending_text(&driven.ending)
            ),
        };
    }
    // The UI saw the same items from the checkpoint on: the recorded
    // transcript ends with the replayed one.
    if !loaded.meta.transcript.ends_with(&driven.transcript) {
        return Verdict::Differs {
            turn: last.0,
            step: last.1,
            what: "the stream items a UI sees changed".into(),
        };
    }
    Verdict::Identical
}

fn ending_text(ending: &Ending) -> String {
    match ending {
        Ending::Answer { output, usage } => format!("answer {output:?} ({usage:?})"),
        Ending::Error(error) => format!("error {error:?}"),
    }
}

/// Restore `dir`'s state after turn `from` and continue live against
/// `model`, writing the head of the original log and the new turns to
/// `new_dir`. The original directory is only read.
pub async fn fork(
    program: &Program,
    dir: &Path,
    from: usize,
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    rows: Rows,
    new_dir: &Path,
) -> Result<Ending, RerunError> {
    let loaded = load(dir, from)?;
    let recorder = EffectLogRecorder::keeping_stream_events();
    let agent = program
        .configure(AgentBuilder::named_model("default", model).owner(OWNER))
        .tool(WriteRow { rows })
        .record_to(recorder.clone())
        .build();
    if let Some(last) = loaded.checkpoint.last {
        agent.handlers().resume_ids_after(last);
    }
    let driven = drive(
        agent.resume(loaded.checkpoint.state.clone()),
        program.streaming,
    )
    .await;
    let continued = agent.stamp(recorder.take());
    let mut log = loaded.log.clone();
    log.records.truncate(loaded.checkpoint.at);
    let kept: std::collections::BTreeSet<_> = log.records.iter().map(|record| record.id).collect();
    log.header.stream_errors.retain(|id, _| kept.contains(id));
    log.header
        .stream_errors
        .extend(continued.header.stream_errors);
    log.header.handlers = continued.header.handlers;
    log.records.extend(continued.records);
    let meta = RunMeta {
        ending: driven.ending.clone(),
        // The items a UI sees from the fork on.
        transcript: driven.transcript,
        ..loaded.meta
    };
    write_run(new_dir, &meta, &log, false)?;
    Ok(driven.ending)
}

/// The tool calls model turn `turn` made, with their recorded results, one
/// line each, read from `dir` with no model and no key.
pub fn tool_calls(dir: &Path, turn: usize) -> Result<Vec<String>, RerunError> {
    let loaded = load(dir, turn)?;
    Ok(loaded
        .log
        .records
        .iter()
        .enumerate()
        .filter(
            |(index, _)| matches!(place(&loaded.log, *index), (t, step) if t == turn && step > 0),
        )
        .map(|(_, record)| match (&record.kind, &record.outcome) {
            (EffectKind::ToolCall { name, args }, Ok(Answered::ToolResult { result })) => {
                format!("{name}({args}) -> {:?}", result.output().as_text())
            }
            (kind, outcome) => format!("{kind:?} -> {outcome:?}"),
        })
        .collect())
}
