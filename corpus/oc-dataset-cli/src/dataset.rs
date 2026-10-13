//! Reading the saved sessions into chat-format lines, and writing them.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

use rig_cassette::effect_log::EffectLog;
use rig_cassette::effect_log::jsonl::{self, SESSION_EFFECT_LOG};
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::prelude::SessionStores;
use rig_harness::rig_core::completion::Message;
use rig_harness::rig_core::conversation::{Change, Conversation, EntryRef, Halt, MessageId};
use rig_harness::rig_core::effect::{EffectKind, EffectRecord, FamilyDescriptor};
use rig_harness::rig_core::message::{
    AssistantContent, StopReason, ToolResultContent, UserContent,
};
use rig_harness::rig_core::redact;
use rig_harness::rig_memory::{Record, SavedAgent, Session, Store, history};
use serde_json::{Map, Value, json};

use crate::Options;

/// What `--redact` puts in place of a secret.
pub const REDACTED: &str = "<REDACTED>";

/// Names whose `NAME=value` value `--redact` replaces, as their ends,
/// upper-cased.
const SECRET_NAMES: [&str; 6] = ["PASSWORD", "PASSWD", "TOKEN", "SECRET", "API_KEY", "APIKEY"];

/// What an export found in the saved sessions.
#[derive(Clone, Debug, Default)]
pub struct Export {
    /// The saved sessions read.
    pub read: usize,
    /// The sessions that matched the filters.
    pub matched: usize,
    /// Each line, with the session it came from, in session order.
    pub lines: Vec<(SessionId, String)>,
    /// The sessions left out, each with why.
    pub skipped: Vec<Skipped>,
    /// The turns `--exclude-errors` dropped.
    pub dropped_turns: usize,
}

/// A session an export left out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    /// Its id.
    pub session: SessionId,
    /// Why it was left out.
    pub why: String,
}

/// How a turn ended, as its agent's log shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    /// The model answered.
    Answered,
    /// It failed, or the run ended before it did.
    Failed,
    /// It was stopped.
    Stopped,
}

impl Export {
    /// The lines of the sessions of `home` that `options` keep, each
    /// session opened by `stores`, read only.
    pub fn collect(options: &Options, home: &Home, stores: &SessionStores) -> Self {
        let mut export = Self::default();
        for (id, dir) in saved_sessions(home) {
            export.read += 1;
            match session_lines(options, &dir, stores) {
                Ok((lines, dropped)) => {
                    export.matched += 1;
                    export.dropped_turns += dropped;
                    export
                        .lines
                        .extend(lines.into_iter().map(|line| (id.clone(), line)));
                }
                Err(why) => export.skipped.push(Skipped { session: id, why }),
            }
        }
        export
    }

    /// Writes the lines under `out`: `train.jsonl` and `valid.jsonl` when
    /// split, else `dataset.jsonl`.
    ///
    /// # Errors
    ///
    /// When a file cannot be written.
    pub fn write(&self, out: &Path, options: &Options) -> io::Result<()> {
        std::fs::create_dir_all(out)?;
        let mut files: BTreeMap<&str, String> = BTreeMap::new();
        for (session, line) in &self.lines {
            let file = match options.split {
                Some(share) if in_train(session, options.seed, share) => "train.jsonl",
                Some(_) => "valid.jsonl",
                None => "dataset.jsonl",
            };
            let text = files.entry(file).or_default();
            text.push_str(line);
            text.push('\n');
        }
        let names: &[&str] = match options.split {
            Some(_) => &["train.jsonl", "valid.jsonl"],
            None => &["dataset.jsonl"],
        };
        for name in names {
            let text = files.get(name).map_or("", String::as_str);
            std::fs::write(out.join(name), text)?;
        }
        Ok(())
    }

    /// What the export read, wrote and skipped.
    pub fn summary(&self) -> String {
        let tokens: usize = self
            .lines
            .iter()
            .map(|(_, line)| line.len().div_ceil(4))
            .sum();
        let mut text = format!(
            "sessions read: {}\nsessions matched: {}\nlines: {}\nturns dropped: {}\n\
             tokens (estimate): {tokens}\nskipped: {}\n",
            self.read,
            self.matched,
            self.lines.len(),
            self.dropped_turns,
            self.skipped.len(),
        );
        for skipped in &self.skipped {
            writeln!(text, "  {}: {}", skipped.session, skipped.why).ok();
        }
        text
    }
}

/// Every saved session of `home`, by id.
fn saved_sessions(home: &Home) -> Vec<(SessionId, SessionDir)> {
    let entries = std::fs::read_dir(home.sessions())
        .into_iter()
        .flatten()
        .flatten();
    let mut sessions: Vec<_> = entries
        .filter_map(|entry| entry.file_name().to_str()?.parse::<SessionId>().ok())
        .map(|id| (id.clone(), home.session(&id)))
        .filter(|(_, dir)| dir.is_saved())
        .collect();
    sessions.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
    sessions
}

/// The lines of the session in `dir` and the turns dropped from them, or
/// why it is left out.
fn session_lines(
    options: &Options,
    dir: &SessionDir,
    stores: &SessionStores,
) -> Result<(Vec<String>, usize), String> {
    if let Some(wanted) = &options.project {
        let project = std::fs::read_to_string(dir.directory()).unwrap_or_default();
        let project = real(Path::new(project.trim_end_matches('\n')));
        if !project.starts_with(real(wanted)) {
            return Err("not in --project".to_owned());
        }
    }
    let store = stores
        .open(dir)
        .map_err(|why| format!("cannot be opened: {why}"))?;
    let (session, unreadable) =
        Session::read_lenient(&*store).map_err(|why| format!("cannot be read: {why}"))?;
    let log = store
        .read_file(SESSION_EFFECT_LOG)
        .ok()
        .and_then(|log| jsonl::read_from(log.as_slice()).ok())
        .unwrap_or_default();
    let records = &log.records;
    let main = session
        .agents
        .iter()
        .find(|agent| agent.parent.is_none() && !agent.conversation.messages().is_empty())
        .ok_or("no conversation")?;
    if let Some(log) = unreadable.first() {
        return Err(format!("log {} cannot be read: {}", log.log, log.error));
    }
    let day = main.started_ms / 86_400_000;
    if options.since.is_some_and(|since| day < since) {
        return Err("started before --since".to_owned());
    }
    if options.until.is_some_and(|until| day > until) {
        return Err("started after --until".to_owned());
    }
    let models = answered_by(&log, &main.id);
    if let Some(wanted) = &options.model
        && !models.iter().any(|model| model == wanted)
    {
        return Err(format!("model {} is not --model", models.join(", ")));
    }
    let session_id = dir.id().map(|id| id.to_string()).unwrap_or_default();
    let mut lines = Vec::new();
    let mut dropped = 0;
    for agent in &session.agents {
        let is_main = agent.id == main.id;
        if !is_main && agent.parent.is_none() || agent.conversation.messages().is_empty() {
            continue;
        }
        let (mut line, kept, gone) = agent_line(&*store, agent, records, options);
        if is_main && kept < options.min_turns {
            return Err(format!("{kept} turns, fewer than --min-turns"));
        }
        let mut metadata = Map::new();
        metadata.insert("session".into(), json!(session_id));
        metadata.insert("agent".into(), json!(agent.id));
        let model = answered_by(&log, &agent.id).into_iter().next();
        metadata.insert("model".into(), json!(model));
        if let Some(parent) = &agent.parent {
            metadata.insert("parent_session".into(), json!(session_id));
            metadata.insert("parent_agent".into(), json!(parent));
        }
        line.insert("metadata".into(), Value::Object(metadata));
        let mut line = Value::Object(line);
        if options.redact {
            redact_json(&mut line);
        }
        dropped += gone;
        lines.push(line.to_string());
    }
    Ok((lines, dropped))
}

/// The models the agent `agent`'s requests went to, as the log's header
/// names their handlers (`provider/name`) and by name alone, in the order
/// first seen.
fn answered_by(log: &EffectLog, agent: &str) -> Vec<String> {
    let mut models = Vec::new();
    for record in log
        .records
        .iter()
        .filter(|r| r.scope.as_deref() == Some(agent))
    {
        let handler = log.header.handlers.iter().find(|h| h.key == record.key);
        let Some(FamilyDescriptor::Completion { model, .. }) = handler.map(|h| &h.family) else {
            continue;
        };
        let reference = model.as_str();
        let name = reference
            .split_once('/')
            .map_or(reference, |(_, name)| name);
        for name in [reference, name] {
            if !models.iter().any(|known| known == name) {
                models.push(name.to_owned());
            }
        }
    }
    models
}

/// The agent's line without its metadata, how many turns it keeps, and
/// how many it dropped.
fn agent_line(
    store: &dyn Store,
    agent: &SavedAgent,
    records: &[EffectRecord],
    options: &Options,
) -> (Map<String, Value>, usize, usize) {
    let conversation = &agent.conversation;
    let entries: Vec<EntryRef<'_>> = conversation.entries().collect();
    let turns = turns(&entries, &halts(store, &agent.id));
    let requests: Vec<_> = records
        .iter()
        .filter(|record| record.scope.as_deref() == Some(agent.id.as_str()))
        .filter_map(|record| match &record.kind {
            EffectKind::Completion { request, .. } => Some((record.purpose.as_deref(), request)),
            _ => None,
        })
        .collect();
    let mut messages = Vec::new();
    // The system prompt the agent's last turn request sent.
    let last = requests
        .iter()
        .rev()
        .find(|(purpose, _)| *purpose == Some("turn"))
        .or(requests.last());
    let system: Vec<&str> = last
        .into_iter()
        .flat_map(|(_, request)| &request.chat_history)
        .filter_map(|message| match message {
            Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    if !system.is_empty() {
        messages.push(json!({ "role": "system", "content": system.join("\n\n") }));
    }
    let (mut kept, mut dropped) = (0, 0);
    for (range, ending) in turns {
        if options.exclude_errors && ending != Ending::Answered {
            dropped += 1;
            continue;
        }
        kept += 1;
        for entry in entries.get(range).unwrap_or_default() {
            chat_messages(entry.message, options.keep_reasoning, &mut messages);
        }
    }
    // Every tool the agent's turn requests offered, by name.
    let mut tools: BTreeMap<String, Value> = BTreeMap::new();
    for (_, request) in requests.iter().filter(|(p, _)| *p == Some("turn")) {
        for tool in &request.tools {
            tools
                .entry(tool.name.as_str().to_owned())
                .or_insert_with(|| {
                    json!({ "type": "function", "function": {
                        "name": tool.name.as_str(),
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }})
                });
        }
    }
    let mut line = Map::new();
    line.insert("messages".into(), Value::Array(messages));
    line.insert("tools".into(), tools.into_values().collect());
    (line, kept, dropped)
}

/// Where the agent's log halted its conversation, by the message it
/// ended with: rig records a halt when a turn ends without an answer, as
/// stopped or kept for later (a failure).
fn halts(store: &dyn Store, agent: &str) -> BTreeMap<MessageId, Halt> {
    let mut halts = BTreeMap::new();
    let mut folded = Conversation::new();
    for (_, record) in history(store, agent).unwrap_or_default() {
        let Record::Change(recorded) = record else {
            continue;
        };
        if folded.apply(&recorded).is_err() {
            break;
        }
        if let Change::Halt { halt: Some(halt) } = recorded.change
            && let Some(last) = folded.entries().next_back()
        {
            halts.insert(last.id, halt);
        }
    }
    halts
}

/// The conversation's turns: where each is in `entries`, and how it ended.
/// A turn begins with a request after an answer, a stopped note or a
/// halted message; the steering and tool results in between are its own.
fn turns(
    entries: &[EntryRef<'_>],
    halts: &BTreeMap<MessageId, Halt>,
) -> Vec<(Range<usize>, Ending)> {
    let mut starts = Vec::new();
    let mut previous: Option<&EntryRef<'_>> = None;
    for (at, entry) in entries.iter().enumerate() {
        let begins = previous.is_none_or(|before| {
            entry.is_request()
                && (before.is_stopped_note() || halts.contains_key(&before.id) || answers(before))
        });
        if begins {
            starts.push(at);
        }
        previous = Some(entry);
    }
    let ends = starts.iter().skip(1).copied().chain([entries.len()]);
    starts
        .iter()
        .zip(ends)
        .map(|(&start, end)| {
            let turn = entries.get(start..end).unwrap_or_default();
            (start..end, ending(turn, halts))
        })
        .collect()
}

/// Whether the entry is a model's answer: an assistant message that calls
/// no tool.
fn answers(entry: &EntryRef<'_>) -> bool {
    matches!(entry.message, Message::Assistant(reply)
        if !reply.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))))
}

/// How the turn of `entries` ended.
fn ending(entries: &[EntryRef<'_>], halts: &BTreeMap<MessageId, Halt>) -> Ending {
    let halt = entries
        .iter()
        .filter_map(|entry| halts.get(&entry.id))
        .next_back();
    if entries.iter().any(EntryRef::is_stopped_note) || halt == Some(&Halt::Stopped) {
        return Ending::Stopped;
    }
    if halt.is_some() {
        return Ending::Failed;
    }
    let Some(last) = entries.last() else {
        return Ending::Failed;
    };
    match last.message {
        Message::Assistant(reply) => match &reply.stop {
            Some(StopReason::Aborted(_)) => Ending::Stopped,
            Some(StopReason::Error(_)) => Ending::Failed,
            _ if answers(last) => Ending::Answered,
            _ => Ending::Failed,
        },
        _ => Ending::Failed,
    }
}

/// `message` in OpenAI chat format, onto `out`: a user message's tool
/// results as `tool` messages, images as `[image]`.
fn chat_messages(message: &Message, keep_reasoning: bool, out: &mut Vec<Value>) {
    match message {
        Message::System { content } => out.push(json!({ "role": "system", "content": content })),
        Message::User { content } => {
            let mut said = Vec::new();
            for item in content {
                match item {
                    UserContent::ToolResult(result) => {
                        let text: Vec<String> = result
                            .content
                            .iter()
                            .map(|part| match part {
                                ToolResultContent::Text(text) => text.text.clone(),
                                ToolResultContent::Image(_) => "[image]".to_owned(),
                                ToolResultContent::Json { value } => value.to_string(),
                            })
                            .collect();
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": result.call.to_string(),
                            "content": text.join("\n"),
                        }));
                    }
                    UserContent::Text(text) => said.push(text.text.clone()),
                    UserContent::Image(_) => said.push("[image]".to_owned()),
                    UserContent::Audio(_) => said.push("[audio]".to_owned()),
                    UserContent::Video(_) => said.push("[video]".to_owned()),
                    UserContent::Document(_) => said.push("[document]".to_owned()),
                }
            }
            if !said.is_empty() {
                out.push(json!({ "role": "user", "content": said.join("\n") }));
            }
        }
        Message::Assistant(reply) => {
            let (mut said, mut thought, mut calls) = (Vec::new(), Vec::new(), Vec::new());
            for block in &reply.content {
                match block {
                    AssistantContent::Text(text) => said.push(text.text.clone()),
                    AssistantContent::Image(_) => said.push("[image]".to_owned()),
                    AssistantContent::Reasoning(reasoning) => thought.push(reasoning.text.clone()),
                    AssistantContent::ToolCall(call) => calls.push(json!({
                        "id": call.id.to_string(),
                        "type": "function",
                        "function": {
                            "name": call.function.name.as_str(),
                            "arguments": call.function.raw_arguments(),
                        },
                    })),
                    _ => {}
                }
            }
            let mut message = Map::new();
            message.insert("role".into(), json!("assistant"));
            let said = (!said.is_empty()).then(|| said.join(""));
            message.insert("content".into(), json!(said));
            if !calls.is_empty() {
                message.insert("tool_calls".into(), json!(calls));
            }
            if keep_reasoning && !thought.is_empty() {
                message.insert("reasoning_content".into(), json!(thought.join("\n")));
            }
            out.push(Value::Object(message));
        }
    }
}

/// Every string of `value` with its secrets replaced by [`REDACTED`].
fn redact_json(value: &mut Value) {
    match value {
        Value::String(text) => *text = redact_text(text),
        Value::Array(items) => items.iter_mut().for_each(redact_json),
        Value::Object(fields) => fields.values_mut().for_each(redact_json),
        _ => {}
    }
}

/// `text` with every key, token and private key block rig-core's
/// [`redact`] finds by its shape, and every `PASSWORD=`, `TOKEN=` (and
/// like) value, replaced by [`REDACTED`].
pub fn redact_text(text: &str) -> String {
    let mut shapes = String::with_capacity(text.len());
    let mut from = 0;
    for secret in redact::find(text) {
        shapes.push_str(text.get(from..secret.range.start).unwrap_or_default());
        shapes.push_str(REDACTED);
        from = secret.range.end;
    }
    shapes.push_str(text.get(from..).unwrap_or_default());
    let mut out = String::with_capacity(shapes.len());
    let mut rest = shapes.as_str();
    while let Some((before, after)) = rest.split_once('=') {
        out.push_str(before);
        out.push('=');
        let name_len: usize = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .map(char::len_utf8)
            .sum();
        let name = before.get(before.len().saturating_sub(name_len)..);
        let name = name.unwrap_or_default().to_ascii_uppercase();
        let secret = SECRET_NAMES.iter().any(|end| name.ends_with(end));
        let len = after
            .find(|c: char| c.is_whitespace() || "\"'`,;&|)\\".contains(c))
            .unwrap_or(after.len());
        rest = after;
        if secret && len > 0 && !after.starts_with(REDACTED) {
            out.push_str(REDACTED);
            rest = after.get(len..).unwrap_or_default();
        }
    }
    out.push_str(rest);
    out
}

/// Whether the session goes to `train.jsonl`: a hash of its id and `seed`
/// (64-bit FNV-1a) below `share` of the range.
fn in_train(session: &SessionId, seed: u64, share: f64) -> bool {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in seed.to_le_bytes().iter().chain(session.as_str().as_bytes()) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    ((hash >> 11) as f64) / ((1_u64 << 53) as f64) < share
}

/// `path` with its links resolved, when it exists.
fn real(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_owned())
}
