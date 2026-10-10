//! Effect logs as JSON lines on disk: `{"header": …}` lines and one
//! resolved record per line. Appending keeps a long-running host's log
//! durable as it goes; [`read`] folds the lines back into one [`EffectLog`]
//! that [`EffectLogReplayer`](super::EffectLogReplayer) replays.
//!
//! The file grows with what is new, not with the length of the
//! conversation:
//!
//! - **Header deltas.** A header line states only what the lines before it
//!   do not: a new handler, a newly touched key. [`read`] merges them.
//! - **Requests as continuations.** A completion request mostly repeats an
//!   earlier one: the agent's previous request plus its reply and a few new
//!   messages, or after a compaction a summary and the messages kept. So
//!   [`Writer`] writes it against a recent request, its base:
//!   `{"id": …, "after": <base id>, "keep": <base messages it starts with>,
//!   "then": [<pieces>], "same_tools": true, "record": …}`, whose record
//!   holds only the messages no piece finds elsewhere, and no tools when
//!   they are the base's. The pieces after the kept prefix are
//!   `{"base": [from, to]}`, a run of the base's messages; `{"new": n}`, the
//!   record's next `n` messages; and `{"reply": id}`, the record's next
//!   message, written without the content and origin it repeats of
//!   completion `id`'s reply; and `{"results": [id, …]}`, the record's next
//!   message, whose tool results without content take those of tool calls
//!   `id`, in order. Without `then`, all the record's messages follow the
//!   prefix. A base is one of the last four requests on the same handler
//!   and scope, or the latest on another.
//! - **Each reply and tool result once.** The model's reply is in its
//!   record's outcome and a tool's result in its call's; the next request's
//!   copies are `reply` and `results` pieces.
//! - **Across restarts.** A [`Writer`] on a file that already has lines
//!   reads it once, on its first append or [`Writer::last_id`], and
//!   continues its header, requests and ids.
//!
//! The writer also leaves out the parts of a completion's raw provider
//! document that echo the request (`instructions`, `tools`) and the
//! per-item usage attribution (`usage.attribution`); nothing reads them
//! back. [`read`] restores every request whole, so a replayer sees what was
//! dispatched.
//!
//! ```no_run
//! use rig_cassette::effect_log::{EffectLogRecorder, jsonl};
//!
//! let recorder = EffectLogRecorder::new();
//! let mut writer = jsonl::Writer::new("effects.jsonl");
//! writer.append(&recorder.take())?;
//! let next = writer.last_id()?.map_or(0, |id| id.as_u64() + 1);
//! let log = jsonl::read("effects.jsonl")?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::effect::{EffectId, EffectKind, EffectRecord, EffectRow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{EffectLog, LogHeader};

/// Appends effect logs, such as successive [`EffectLogRecorder::take`]s,
/// to a JSON-lines file.
///
/// [`EffectLogRecorder::take`]: super::EffectLogRecorder::take
#[derive(Debug)]
pub struct Writer {
    path: PathBuf,
    /// The header last appended, to tell whether the next one changed.
    given: Option<LogHeader>,
    /// What a reader knows after the file's lines: `None` until they are
    /// read and after a failed append, so it never holds what the file
    /// lacks.
    known: Option<Known>,
}

impl Writer {
    /// A writer appending to the file at `path`, created on first write.
    /// When the file has lines, the first append reads them and continues
    /// from them.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            given: None,
            known: None,
        }
    }

    /// Appends `log`'s records, after a header line with what its header
    /// adds to the file's when it differs from the last one appended.
    /// Writes nothing when it has no records: a header alone waits for the
    /// first record it describes, so a session that never made one leaves
    /// no file.
    ///
    /// # Errors
    ///
    /// When the file cannot be read or written. The next append reads it
    /// again, so nothing of a failed append is referred to.
    pub fn append(&mut self, log: &EffectLog) -> io::Result<()> {
        if log.records.is_empty() {
            return Ok(());
        }
        let mut known = self.fold()?;
        let header_due = self.given.as_ref() != Some(&log.header);
        let mut lines = Vec::new();
        if header_due {
            let delta = delta(&known.header, &log.header);
            if !known.has_lines || delta != LogHeader::default() {
                serde_json::to_writer(&mut lines, &HeaderLine { header: &delta })?;
                lines.push(b'\n');
                merge(&mut known.header, delta);
            }
        }
        for record in &log.records {
            Self::encode(&mut known, record, &mut lines)?;
            known.highest = known.highest.max(Some(record.id));
            lines.push(b'\n');
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?
            .write_all(&lines)?;
        known.has_lines = true;
        self.known = Some(known);
        if header_due {
            self.given = Some(log.header.clone());
        }
        Ok(())
    }

    /// The highest effect id in the file, `None` when it has no record.
    /// Reads the file unless an append or an earlier call did, and the
    /// first append then continues from that reading.
    ///
    /// # Errors
    ///
    /// When the file exists but cannot be read.
    pub fn last_id(&mut self) -> io::Result<Option<EffectId>> {
        let known = self.fold()?;
        Ok(self.known.insert(known).highest)
    }

    /// What a reader knows after the file's lines, taken from this writer
    /// or read from the file.
    fn fold(&mut self) -> io::Result<Known> {
        match self.known.take() {
            Some(known) => Ok(known),
            None => Known::of(&self.path),
        }
    }

    /// Writes `record` to `out` and notes it in `known`: a completion as a
    /// continuation of the recent request it shares the most with, when it
    /// shares any, any other record as it is.
    fn encode(known: &mut Known, record: &EffectRecord, out: &mut Vec<u8>) -> io::Result<()> {
        match record.kind {
            EffectKind::Completion { .. } => {}
            EffectKind::ToolCall { .. } => {
                let value = serde_json::to_value(record)?;
                known.note(&value, None)?;
                return Ok(serde_json::to_writer(out, &value)?);
            }
            _ => return Ok(serde_json::to_writer(out, record)?),
        }
        let mut value = serde_json::to_value(record)?;
        for (parent, key) in RAW_ECHOES {
            if let Some(parent) = value.pointer_mut(parent).and_then(Value::as_object_mut) {
                parent.shift_remove(key);
            }
        }
        let request = value
            .pointer("/kind/request")
            .and_then(Value::as_object)
            .ok_or_else(|| io::Error::other("a completion record without a request object"))?;
        let history = request
            .get("chat_history")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let texts: Vec<String> = history.iter().map(Value::to_string).collect();
        let tools = request.get("tools").map(Value::to_string);
        let chain = chain_of(&value);
        let plan = known
            .bases(&chain)
            .filter_map(|base| Plan::of(base, history, &texts, tools.as_deref(), known))
            .min_by_key(|plan| plan.cost);
        let Some(plan) = plan else {
            known.note(&value, None)?;
            return Ok(serde_json::to_writer(out, &value)?);
        };
        if let Some(request) = value
            .pointer_mut("/kind/request")
            .and_then(Value::as_object_mut)
        {
            request.insert("chat_history".to_owned(), Value::Array(plan.written));
            if plan.same_tools {
                request.shift_remove("tools");
            }
        }
        if let Some(fields) = value.as_object_mut() {
            fields.shift_remove("id");
        }
        let line = DeltaLine {
            id: record.id,
            after: plan.after,
            keep: plan.keep,
            then: plan.then,
            same_tools: plan.same_tools,
            record: value,
        };
        serde_json::to_writer(out, &line)?;
        // Noted as a reader notes the line, which restores it while the
        // requests it refers to are still known.
        let (record, texts) = known.restore(line, false)?;
        known.note(&record, Some(texts))
    }
}

/// How many completion requests per handler and scope stay bases for the
/// requests after them: an agent's latest, and those before it, such as
/// its request before a compaction's summarizing one.
const RECENT: usize = 4;

/// How many tool results stay known for the requests after them to
/// repeat: more than an agent's calls in one reply.
const RECENT_RESULTS: usize = 64;

/// What the lines so far establish for the lines after them: the header
/// they merge to, the recent completion requests of each chain (handler
/// key and scope), the recent tool results, and the highest id.
#[derive(Debug, Default)]
struct Known {
    header: LogHeader,
    chains: BTreeMap<String, VecDeque<Head>>,
    /// Tool-call records' successful results as JSON text, latest last.
    results: VecDeque<(EffectId, String)>,
    highest: Option<EffectId>,
    /// Whether there is a line that is not blank.
    has_lines: bool,
}

/// A completion request's history and tools as JSON text.
type Texts = (Vec<Arc<str>>, Option<Arc<str>>);

/// `value` as JSON text.
fn text(value: &Value) -> Arc<str> {
    Arc::from(value.to_string())
}

/// A completion request as written or read: each message and the tools as
/// JSON text, and the reply it got.
#[derive(Debug)]
struct Head {
    id: EffectId,
    history: Vec<Arc<str>>,
    tools: Option<Arc<str>>,
    reply: Option<Reply>,
}

/// A completion's reply as JSON text: its choice and its origin, which the
/// assistant message repeating it carries as `content` and `origin`.
#[derive(Debug)]
struct Reply {
    content: String,
    origin: String,
}

impl Reply {
    /// `message`, written without this reply's content and origin, whole
    /// again, its fields in a message's order.
    fn restore(&self, message: Value) -> io::Result<Value> {
        let Value::Object(mut rest) = message else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a message repeating a reply is not an object",
            ));
        };
        let mut whole = serde_json::Map::new();
        if let Some(role) = rest.shift_remove("role") {
            whole.insert("role".to_owned(), role);
        }
        whole.insert("content".to_owned(), serde_json::from_str(&self.content)?);
        whole.insert("origin".to_owned(), serde_json::from_str(&self.origin)?);
        whole.extend(rest);
        Ok(Value::Object(whole))
    }
}

impl Known {
    /// What the lines of the file at `path` establish; nothing when it is
    /// missing. From a line that cannot be read back on, only ids are read,
    /// so a writer continues with whole records and a whole header and
    /// with ids above every id a line states.
    fn of(path: &Path) -> io::Result<Self> {
        /// A record line's id.
        #[derive(Deserialize)]
        struct IdOnly {
            id: EffectId,
        }
        let file = match File::open(path) {
            Ok(file) => file,
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(failure) => return Err(failure),
        };
        let mut known = Self::default();
        let mut whole = true;
        for line in BufReader::new(file).split(b'\n') {
            let line = line?;
            whole = whole
                && std::str::from_utf8(&line)
                    .map_err(io::Error::other)
                    .and_then(|line| known.line(line, false))
                    .is_ok();
            if !whole {
                known = Self {
                    highest: known.highest.max(
                        serde_json::from_slice::<IdOnly>(&line)
                            .ok()
                            .map(|record| record.id),
                    ),
                    has_lines: true,
                    ..Self::default()
                };
            }
        }
        Ok(known)
    }

    /// Folds one line in: a header merges into the header, a record comes
    /// back, whole when `whole`. A blank line is neither.
    fn line(&mut self, line: &str, whole: bool) -> io::Result<Option<Value>> {
        if line.trim().is_empty() {
            return Ok(None);
        }
        self.has_lines = true;
        let value: Value = serde_json::from_str(line)?;
        if value.get("header").is_some() {
            let HeaderLine { header } = serde_json::from_value::<HeaderLine<LogHeader>>(value)?;
            merge(&mut self.header, header);
            return Ok(None);
        }
        if value.get("after").is_none() {
            self.note(&value, None)?;
            return Ok(Some(value));
        }
        let (record, texts) = self.restore(serde_json::from_value(value)?, whole)?;
        self.note(&record, Some(texts))?;
        Ok(Some(record))
    }

    /// Notes a record: its id may be the highest; a completion becomes its
    /// chain's latest request, with `texts`, its history and tools as
    /// [`Self::restore`] gives them, or else as the record holds them; a
    /// tool call's result becomes the latest.
    fn note(&mut self, record: &Value, texts: Option<Texts>) -> io::Result<()> {
        let id: EffectId = serde_json::from_value(record.get("id").cloned().unwrap_or_default())?;
        self.highest = self.highest.max(Some(id));
        match record.pointer("/kind/effect").and_then(Value::as_str) {
            Some("completion") => {}
            Some("tool_call") => {
                if let Some(value) = record.pointer("/outcome/Ok/result/value") {
                    self.results.push_back((id, value.to_string()));
                    while self.results.len() > RECENT_RESULTS {
                        self.results.pop_front();
                    }
                }
                return Ok(());
            }
            _ => return Ok(()),
        }
        let (history, tools) = texts.unwrap_or_else(|| {
            let request = record.pointer("/kind/request");
            let field = |name: &str| request.and_then(|request| request.get(name));
            let history = match field("chat_history") {
                Some(Value::Array(messages)) => messages.iter().map(text).collect(),
                _ => Vec::new(),
            };
            (history, field("tools").map(text))
        });
        let reply = match (
            record.pointer("/outcome/Ok/choice"),
            record.pointer("/outcome/Ok/origin"),
        ) {
            (Some(content), Some(origin)) => Some(Reply {
                content: content.to_string(),
                origin: origin.to_string(),
            }),
            _ => None,
        };
        let chain = self.chains.entry(chain_of(record)).or_default();
        chain.push_back(Head {
            id,
            history,
            tools,
            reply,
        });
        while chain.len() > RECENT {
            chain.pop_front();
        }
        Ok(())
    }

    /// The recent request `id`, on any chain.
    fn head(&self, id: EffectId) -> Option<&Head> {
        self.chains
            .values()
            .flat_map(VecDeque::iter)
            .find(|head| head.id == id)
    }

    /// The requests a completion on `chain` may be written against: the
    /// chain's recent ones, latest first, then every other chain's latest.
    fn bases<'a>(&'a self, chain: &'a str) -> impl Iterator<Item = &'a Head> {
        let own = self
            .chains
            .get(chain)
            .into_iter()
            .flat_map(|heads| heads.iter().rev());
        let others = self
            .chains
            .iter()
            .filter(move |(name, _)| name.as_str() != chain)
            .filter_map(|(_, heads)| heads.back());
        own.chain(others)
    }

    /// The recent completion whose reply `message` repeats, and `message`
    /// without the content and origin it repeats.
    fn reply_in(&self, message: &Value) -> Option<(EffectId, Value)> {
        let fields = message.as_object()?;
        let origin = fields.get("origin")?.to_string();
        let content = fields.get("content")?.to_string();
        let id = self
            .chains
            .values()
            .flat_map(VecDeque::iter)
            .find(|head| {
                head.reply
                    .as_ref()
                    .is_some_and(|reply| reply.origin == origin && reply.content == content)
            })?
            .id;
        let mut rest = fields.clone();
        rest.shift_remove("content");
        rest.shift_remove("origin");
        Some((id, Value::Object(rest)))
    }

    /// The recent tool calls whose results `message`'s tool results repeat,
    /// in order, and `message` with those results' content left out.
    fn results_in(&self, message: &Value) -> Option<(Vec<EffectId>, Value)> {
        let mut fields = message.as_object()?.clone();
        let mut ids = Vec::new();
        for item in fields.get_mut("content")?.as_array_mut()? {
            let Some(item) = item
                .as_object_mut()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some(TOOL_RESULT))
            else {
                continue;
            };
            let Some(content) = item.get("content").map(Value::to_string) else {
                continue;
            };
            if let Some((id, _)) = self
                .results
                .iter()
                .rev()
                .find(|(_, result)| *result == content)
            {
                ids.push(*id);
                item.shift_remove("content");
            }
        }
        (!ids.is_empty()).then(|| (ids, Value::Object(fields)))
    }

    /// `message`, written without the content of its tool results that
    /// repeat the results of tool calls `ids`, whole again.
    fn restore_results(&self, mut message: Value, ids: Vec<EffectId>) -> io::Result<Value> {
        let invalid = |what: String| io::Error::new(io::ErrorKind::InvalidData, what);
        let mut ids = ids.into_iter();
        let items = message
            .get_mut("content")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| invalid("a message repeating tool results has no content".to_owned()))?;
        for item in items {
            let Some(fields) = item.as_object_mut().filter(|item| {
                item.get("type").and_then(Value::as_str) == Some(TOOL_RESULT)
                    && !item.contains_key("content")
            }) else {
                continue;
            };
            let id = ids.next().ok_or_else(|| {
                invalid("a message leaves out more tool results than it names".to_owned())
            })?;
            let (_, result) = self
                .results
                .iter()
                .find(|(known, _)| *known == id)
                .ok_or_else(|| {
                    invalid(format!(
                        "a message repeats the result of tool call {id}, which is not recent"
                    ))
                })?;
            // A tool result's content comes before its error flag.
            let mut whole = serde_json::Map::new();
            let mut content = Some(serde_json::from_str::<Value>(result)?);
            for (key, value) in std::mem::take(fields) {
                if key == "is_error"
                    && let Some(content) = content.take()
                {
                    whole.insert("content".to_owned(), content);
                }
                whole.insert(key, value);
            }
            if let Some(content) = content {
                whole.insert("content".to_owned(), content);
            }
            *fields = whole;
        }
        if ids.next().is_some() {
            return Err(invalid(
                "a message names more tool results than it leaves out".to_owned(),
            ));
        }
        Ok(message)
    }

    /// The record a continuation line stands for, from its base and the
    /// replies it refers to, and its request's history and tools as text,
    /// which the record holds too only when `whole`: a fold needs the text.
    fn restore(&self, line: DeltaLine<Value>, whole: bool) -> io::Result<(Value, Texts)> {
        let invalid = |what: String| io::Error::new(io::ErrorKind::InvalidData, what);
        let DeltaLine {
            id,
            after,
            keep,
            then,
            same_tools,
            mut record,
        } = line;
        let base = self.head(after).ok_or_else(|| {
            invalid(format!(
                "record {id} continues completion {after}, which is not a recent request"
            ))
        })?;
        let messages = |from: usize, to: usize| {
            base.history.get(from..to).ok_or_else(|| {
                invalid(format!(
                    "record {id} takes messages {from}..{to} of completion {after}, which has {}",
                    base.history.len()
                ))
            })
        };
        let short = || invalid(format!("record {id} places more messages than it holds"));
        let request = record
            .pointer_mut("/kind/request")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| invalid(format!("record {id} continues a request but has none")))?;
        let mut held = match request.shift_remove("chat_history") {
            Some(Value::Array(held)) => held,
            _ => Vec::new(),
        }
        .into_iter();
        let mut history = messages(0, keep)?.to_vec();
        if then.is_empty() {
            history.extend(held.by_ref().map(|message| text(&message)));
        }
        for piece in then {
            match piece {
                Piece::Base(from, to) => history.extend_from_slice(messages(from, to)?),
                Piece::New(count) => {
                    for _ in 0..count {
                        history.push(text(&held.next().ok_or_else(short)?));
                    }
                }
                Piece::Reply(of) => {
                    let reply = self
                        .head(of)
                        .and_then(|head| head.reply.as_ref())
                        .ok_or_else(|| {
                            invalid(format!(
                                "record {id} repeats the reply of completion {of}, which is not a recent request with a reply"
                            ))
                        })?;
                    history.push(text(&reply.restore(held.next().ok_or_else(short)?)?));
                }
                Piece::Results(ids) => {
                    let message = held.next().ok_or_else(short)?;
                    history.push(text(&self.restore_results(message, ids)?));
                }
            }
        }
        if held.next().is_some() {
            return Err(invalid(format!(
                "record {id} holds messages its pieces do not place"
            )));
        }
        let tools = if same_tools {
            let tools = base.tools.as_ref().ok_or_else(|| {
                invalid(format!(
                    "record {id} reuses the tools of completion {after}, which has none"
                ))
            })?;
            Some(Arc::clone(tools))
        } else {
            request.get("tools").map(text)
        };
        if whole {
            let messages = history.iter().map(|message| serde_json::from_str(message));
            let messages = messages.collect::<Result<_, _>>()?;
            request.insert("chat_history".to_owned(), Value::Array(messages));
            if same_tools && let Some(tools) = &tools {
                request.insert("tools".to_owned(), serde_json::from_str(tools)?);
            }
        }
        if let Some(fields) = record.as_object_mut() {
            fields.insert("id".to_owned(), serde_json::to_value(id)?);
        }
        Ok((record, (history, tools)))
    }
}

/// How a completion request is written against a base request.
struct Plan {
    after: EffectId,
    /// How many of the base's first messages it starts with.
    keep: usize,
    /// Where its later messages come from; empty when all are written.
    then: Vec<Piece>,
    /// The messages written in the record.
    written: Vec<Value>,
    same_tools: bool,
    /// About the bytes written: what the choice of base minimizes.
    cost: usize,
}

impl Plan {
    /// How `history` (as `texts`) and `tools` are written against `base`,
    /// or `None` when they share nothing with it.
    fn of(
        base: &Head,
        history: &[Value],
        texts: &[String],
        tools: Option<&str>,
        known: &Known,
    ) -> Option<Self> {
        let keep = base
            .history
            .iter()
            .zip(texts)
            .take_while(|(kept, sent)| ***kept == ***sent)
            .count();
        let mut starts: HashMap<&str, Vec<usize>> = HashMap::new();
        for (at, text) in base.history.iter().enumerate() {
            starts.entry(&**text).or_default().push(at);
        }
        let mut reused = keep > 0;
        let mut then = Vec::new();
        let mut written = Vec::new();
        let mut cost = 0;
        let mut at = keep;
        while let (Some(text), Some(message)) = (texts.get(at), history.get(at)) {
            let run = starts.get(text.as_str()).and_then(|starts| {
                starts
                    .iter()
                    .map(|&from| {
                        let kept = base.history.get(from..).unwrap_or_default();
                        let sent = texts.get(at..).unwrap_or_default();
                        let len = kept
                            .iter()
                            .zip(sent)
                            .take_while(|(kept, sent)| ***kept == ***sent)
                            .count();
                        (from, len)
                    })
                    .max_by_key(|&(_, len)| len)
            });
            if let Some((from, len)) = run.filter(|&(_, len)| len > 0) {
                match then.last_mut() {
                    Some(Piece::Base(_, to)) if *to == from => *to += len,
                    _ => then.push(Piece::Base(from, from + len)),
                }
                reused = true;
                at += len;
                continue;
            }
            let repeated = known
                .reply_in(message)
                .map(|(id, rest)| (Piece::Reply(id), rest))
                .or_else(|| {
                    known
                        .results_in(message)
                        .map(|(ids, rest)| (Piece::Results(ids), rest))
                });
            match repeated {
                Some((piece, rest)) => {
                    cost += rest.to_string().len();
                    then.push(piece);
                    written.push(rest);
                    reused = true;
                }
                None => {
                    cost += text.len();
                    match then.last_mut() {
                        Some(Piece::New(count)) => *count += 1,
                        _ => then.push(Piece::New(1)),
                    }
                    written.push(message.clone());
                }
            }
            at += 1;
        }
        if !reused {
            return None;
        }
        let same_tools = tools.is_some() && base.tools.as_deref() == tools;
        if !same_tools {
            cost += tools.map_or(0, str::len);
        }
        if then.iter().all(|piece| matches!(piece, Piece::New(_))) {
            then.clear();
        }
        // About the bytes of a piece.
        cost += then.len() * 16;
        Some(Self {
            after: base.id,
            keep,
            then,
            written,
            same_tools,
            cost,
        })
    }
}

/// The parts of a completion record's raw provider document a log leaves
/// out, as (parent pointer, key): the request's echoes, which the record's
/// request already holds, and the per-item usage attribution.
const RAW_ECHOES: [(&str, &str); 3] = [
    ("/outcome/Ok/raw", "instructions"),
    ("/outcome/Ok/raw", "tools"),
    ("/outcome/Ok/raw/usage", "attribution"),
];

/// The chain a serialized record belongs to: its handler key and scope.
fn chain_of(record: &Value) -> String {
    let field = |name: &str| record.get(name).map(Value::to_string).unwrap_or_default();
    format!("{}\u{0}{}", field("key"), field("scope"))
}

/// Reads the JSON-lines log at `path` into one [`EffectLog`]: every record
/// in file order, under the headers merged. A handler described again
/// takes its latest description; signatures, required rows, program
/// identities, stream errors and deliveries accumulate; the hook stack,
/// run spec and serving policy are the latest stated. A continued
/// completion request is restored whole. A line that is neither a header
/// nor a record, or a continuation that refers to a request that is not
/// recent or places messages it does not have, is an error.
pub fn read(path: impl AsRef<Path>) -> io::Result<EffectLog> {
    let mut known = Known::default();
    let mut records = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        if let Some(record) = known.line(&line?, true)? {
            records.push(serde_json::from_value(record)?);
        }
    }
    Ok(EffectLog {
        header: known.header,
        records,
    })
}

/// Folds a later header into the merged one.
fn merge(into: &mut LogHeader, header: LogHeader) {
    for handler in header.handlers {
        match into
            .handlers
            .iter_mut()
            .find(|known| known.key == handler.key)
        {
            Some(known) => *known = handler,
            None => into.handlers.push(handler),
        }
    }
    for (key, family) in header.signature.iter() {
        into.signature.insert_if_absent(key.clone(), *family);
    }
    for (key, family) in header.required.iter() {
        into.required.insert_if_absent(key.clone(), *family);
    }
    into.programs.extend(header.programs);
    into.stream_errors.extend(header.stream_errors);
    if let Some(deliveries) = header.deliveries {
        into.deliveries.get_or_insert_default().extend(deliveries);
    }
    for limitation in header.delivery_limitations {
        if !into.delivery_limitations.contains(&limitation) {
            into.delivery_limitations.push(limitation);
        }
    }
    if !header.hooks.is_empty() {
        into.hooks = header.hooks;
    }
    into.run_spec = header.run_spec.or(into.run_spec);
    into.bus = header.bus.or(into.bus.take());
}

/// What a header line must state for a reader that merged `merged` to
/// merge `header`: the parts of `header` that [`merge`] would change.
/// Deliveries accumulate, so they are stated as they are.
fn delta(merged: &LogHeader, header: &LogHeader) -> LogHeader {
    let unknown = |row: &EffectRow, known: &EffectRow| -> EffectRow {
        row.iter()
            .filter(|(key, _)| !known.contains_key(key))
            .map(|(key, family)| (key.clone(), *family))
            .collect()
    };
    LogHeader {
        deliveries: header.deliveries.clone(),
        delivery_limitations: header
            .delivery_limitations
            .iter()
            .filter(|limitation| !merged.delivery_limitations.contains(limitation))
            .cloned()
            .collect(),
        stream_errors: header
            .stream_errors
            .iter()
            .filter(|(id, errors)| merged.stream_errors.get(*id) != Some(*errors))
            .map(|(id, errors)| (*id, errors.clone()))
            .collect(),
        run_spec: header
            .run_spec
            .filter(|spec| merged.run_spec != Some(*spec)),
        handlers: header
            .handlers
            .iter()
            .filter(|handler| !merged.handlers.contains(handler))
            .cloned()
            .collect(),
        signature: unknown(&header.signature, &merged.signature),
        hooks: if header.hooks == merged.hooks {
            Vec::new()
        } else {
            header.hooks.clone()
        },
        required: unknown(&header.required, &merged.required),
        bus: header.bus.filter(|bus| merged.bus != Some(*bus)),
        programs: header
            .programs
            .iter()
            .filter(|(scope, identity)| merged.programs.get(*scope) != Some(*identity))
            .map(|(scope, identity)| (scope.clone(), identity.clone()))
            .collect(),
    }
}

/// A header line, `{"header": …}`; record lines are bare records.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderLine<H> {
    header: H,
}

/// A completion record written as a continuation of request `after`:
/// `record` holds the messages no piece finds elsewhere, no tools when
/// `same_tools`, and no id. Its history is the first `keep` messages of
/// `after`, then `then`'s pieces, or without them every held message.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeltaLine<R> {
    id: EffectId,
    after: EffectId,
    keep: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    then: Vec<Piece>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    same_tools: bool,
    record: R,
}

/// Where a continued request's next messages come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Piece {
    /// The base request's messages `from..to`.
    Base(usize, usize),
    /// The record's next `n` held messages.
    New(usize),
    /// The record's next held message, with the content and origin of the
    /// reply of completion `id`.
    Reply(EffectId),
    /// The record's next held message, whose tool results without content
    /// take, in order, the results of tool calls `ids`.
    Results(Vec<EffectId>),
}

/// The `type` of a tool result in a message's content.
const TOOL_RESULT: &str = "toolresult";

#[cfg(test)]
mod tests;
