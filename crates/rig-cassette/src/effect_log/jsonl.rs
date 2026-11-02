//! Effect logs as JSON lines on disk: `{"header": …}` lines and one
//! resolved record per line. Appending keeps a long-running host's log
//! durable as it goes; [`read_from`] folds the lines back into one [`EffectLog`]
//! that [`EffectLogReplayer`](super::EffectLogReplayer) replays.
//!
//! The file grows with what is new, not with the length of the
//! conversation:
//!
//! - **Header deltas.** A header line states only what the lines before it
//!   do not: a new handler, a newly touched key. [`read_from`] merges them.
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
//! A [`Writer`] appends to a file of a session's store, such as
//! [`SESSION_EFFECT_LOG`] ([`Writer::on`]); [`read_session`] reads it back.
//! A plain file is rig-memory's `DirFiles`. The effect log is a recording for replay that no
//! restore reads; saving and resuming sessions is rig-memory's.
//!
//! The writer also leaves out the parts of a completion's raw provider
//! document that echo the request (`instructions`, `tools`) and the
//! per-item usage attribution (`usage.attribution`); nothing reads them
//! back. [`read_from`] restores every request whole, so a replayer sees what was
//! dispatched.
//!
//! ```
//! use std::sync::Arc;
//!
//! use rig_cassette::effect_log::{EffectLogRecorder, jsonl};
//! use rig_memory::MemoryStore;
//!
//! let store = Arc::new(MemoryStore::default());
//! let recorder = EffectLogRecorder::new();
//! let mut writer = jsonl::Writer::on(store.clone(), jsonl::SESSION_EFFECT_LOG);
//! let log = recorder.take();
//! writer.append(&log.header, &log.records)?;
//! let next = writer.last_id()?.map_or(0, |id| id.as_u64() + 1);
//! let log = jsonl::read_session(&*store)?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{DefaultHasher, Hasher};
use std::io::{self, BufRead};
use std::sync::{Arc, OnceLock};

use rig_core::completion::{CompletionRequest, Message};
use rig_core::effect::{EffectId, EffectKind, EffectRecord, EffectRow};
use rig_core::json_utils::remove_preserving_order;
use rig_memory::Files;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{EffectLog, LogHeader};

/// The file of a session's store ([`Files`]) that holds its effect log.
pub const SESSION_EFFECT_LOG: &str = "effects.jsonl";

/// Appends effect logs, such as successive [`EffectLogRecorder::take`]s,
/// to a JSON-lines file of a store.
///
/// [`EffectLogRecorder::take`]: super::EffectLogRecorder::take
pub struct Writer {
    store: Arc<dyn Files>,
    name: String,
    /// The header last appended, to tell whether the next one changed.
    given: Option<LogHeader>,
    /// What a reader knows after the file's lines: `None` until they are
    /// read and after a failed append, so it never holds what the file
    /// lacks.
    known: Option<Known>,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer").field("name", &self.name).finish()
    }
}

impl Writer {
    /// A writer appending to the file `name` of `store`, such as a
    /// session's [`SESSION_EFFECT_LOG`], created on first write. When the
    /// file has lines, the first append reads them and continues from them.
    pub fn on(store: Arc<dyn Files>, name: impl Into<String>) -> Self {
        Self {
            store,
            name: name.into(),
            given: None,
            known: None,
        }
    }

    /// Appends `records`, after a header line with what `header` adds to
    /// the file's when it differs from the last one appended. Writes
    /// nothing when it has no records: a header alone waits for the
    /// first record it describes, so a session that never made one leaves
    /// no file.
    ///
    /// # Errors
    ///
    /// When the file cannot be read or written. The next append reads it
    /// again, so nothing of a failed append is referred to.
    pub fn append<'a>(
        &mut self,
        header: &LogHeader,
        records: impl IntoIterator<Item = &'a EffectRecord>,
    ) -> io::Result<()> {
        let mut records = records.into_iter().peekable();
        if records.peek().is_none() {
            return Ok(());
        }
        let mut known = self.fold()?;
        let header_due = self.given.as_ref() != Some(header);
        let mut lines = Vec::new();
        if header_due {
            let delta = delta(&known.header, header);
            if !known.has_lines || delta != LogHeader::default() {
                serde_json::to_writer(&mut lines, &HeaderLine { header: &delta })?;
                lines.push(b'\n');
                merge(&mut known.header, delta);
            }
        }
        for record in records {
            Self::encode(&mut known, record, &mut lines)?;
            known.highest = known.highest.max(Some(record.id));
            lines.push(b'\n');
        }
        self.store.append_file(&self.name, &lines)?;
        known.has_lines = true;
        self.known = Some(known);
        if header_due {
            self.given = Some(header.clone());
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
            None => match self.store.read_file(&self.name) {
                Ok(bytes) => Known::of(&bytes[..]),
                Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(Known::default()),
                Err(failure) => Err(failure),
            },
        }
    }

    /// Writes `record` to `out` and notes it in `known`: a completion as a
    /// continuation of the recent request it shares the most with, when it
    /// shares any, any other record as it is. A message its chain's latest
    /// request held too takes that request's text, so only new messages are
    /// serialized.
    fn encode(known: &mut Known, record: &EffectRecord, out: &mut Vec<u8>) -> io::Result<()> {
        let (request, stream) = match &record.kind {
            EffectKind::Completion { request, stream } => (request, *stream),
            EffectKind::ToolCall { .. } => {
                let value = serde_json::to_value(record)?;
                known.note(&value, None, None)?;
                return Ok(serde_json::to_writer(out, &value)?);
            }
            _ => return Ok(serde_json::to_writer(out, record)?),
        };
        // The record without its history, which the plan places, and its
        // tools, which only a record that does not share them states.
        let mut without = CompletionRequest::clone(request);
        without.chat_history.clear();
        without.tools.clear();
        let shallow = EffectRecord {
            kind: EffectKind::Completion {
                request: Arc::new(without),
                stream,
            },
            ..record.clone()
        };
        let mut value = serde_json::to_value(&shallow)?;
        for (parent, key) in RAW_ECHOES {
            if let Some(parent) = value.pointer_mut(parent).and_then(Value::as_object_mut) {
                remove_preserving_order(parent, key);
            }
        }
        let chain = chain_of(&value);
        let latest = known
            .chains
            .get(&chain)
            .and_then(|heads| heads.iter().rev().find(|head| head.sent.is_some()));
        let history = request
            .chat_history
            .iter()
            .enumerate()
            .map(
                |(at, message)| match latest.and_then(|head| head.text_of(message, at)) {
                    Some(text) => Ok(Sent {
                        message: message.as_ref(),
                        text: text.clone(),
                        value: None,
                    }),
                    None => {
                        let value = serde_json::to_value(&**message)?;
                        let text = Text::of(&value);
                        Ok(Sent {
                            message: message.as_ref(),
                            text,
                            value: Some(value),
                        })
                    }
                },
            )
            .collect::<serde_json::Result<Vec<_>>>()?;
        let (tools, mut stated) =
            match latest.and_then(|head| head.sent.as_ref().zip(head.tools.as_ref())) {
                Some((sent, tools)) if sent.tools == request.tools => (tools.clone(), None),
                _ => {
                    let value = serde_json::to_value(&request.tools)?;
                    (Text::of(&value), Some(value))
                }
            };
        let mut plan: Option<Plan> = None;
        for base in known.bases(&chain) {
            let keep = shared(base, &history);
            let under = plan.as_ref().map(|plan| plan.cost);
            if let Some(cheaper) = Plan::of(base, keep, &history, &tools, known, under)? {
                plan = Some(cheaper);
            }
        }
        for sent in history.iter().filter(|sent| sent.value.is_some()) {
            known.seen.entry(sent.text.hash).or_insert(known.noted);
        }
        let written = match &mut plan {
            Some(plan) => std::mem::take(&mut plan.written),
            None => history
                .iter()
                .map(|sent| sent.value().map(Cow::into_owned))
                .collect::<serde_json::Result<_>>()?,
        };
        if let Some(fields) = value
            .pointer_mut("/kind/request")
            .and_then(Value::as_object_mut)
        {
            fields.insert("chat_history".to_owned(), Value::Array(written));
            if plan.as_ref().is_some_and(|plan| plan.same_tools) {
                remove_preserving_order(fields, "tools");
            } else {
                let stated = match stated.take() {
                    Some(value) => value,
                    None => serde_json::to_value(&request.tools)?,
                };
                fields.insert("tools".to_owned(), stated);
            }
        }
        let texts = history.into_iter().map(|sent| sent.text).collect();
        known.note(&value, Some((texts, Some(tools))), Some(request))?;
        let Some(plan) = plan else {
            return Ok(serde_json::to_writer(out, &value)?);
        };
        if let Some(fields) = value.as_object_mut() {
            remove_preserving_order(fields, "id");
        }
        let line = DeltaLine {
            id: record.id,
            after: plan.after,
            keep: plan.keep,
            then: plan.then,
            same_tools: plan.same_tools,
            record: value,
        };
        Ok(serde_json::to_writer(out, &line)?)
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
    /// The hash of every text a request's history held, ever, with the
    /// first request that held it, counted in the order noted: a request
    /// noted before it holds no such text.
    seen: HashMap<u64, usize>,
    /// How many completion requests were noted.
    noted: usize,
    highest: Option<EffectId>,
    /// Whether there is a line that is not blank.
    has_lines: bool,
}

/// A completion request's history and tools as JSON text.
type Texts = (Vec<Text>, Option<Text>);

/// A message's (or the tools') JSON text and a hash of it, to compare and
/// index by.
#[derive(Clone, Debug)]
struct Text {
    text: Arc<str>,
    hash: u64,
}

impl Text {
    /// `value` as JSON text.
    fn of(value: &Value) -> Self {
        let text = value.to_string();
        let mut hasher = DefaultHasher::new();
        hasher.write(text.as_bytes());
        Self {
            hash: hasher.finish(),
            text: text.into(),
        }
    }

    fn len(&self) -> usize {
        self.text.len()
    }
}

impl PartialEq for Text {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && (Arc::ptr_eq(&self.text, &other.text) || self.text == other.text)
    }
}

/// A completion request as written or read: each message and the tools as
/// JSON text, and the reply it got.
#[derive(Debug)]
struct Head {
    id: EffectId,
    history: Vec<Text>,
    tools: Option<Text>,
    reply: Option<Reply>,
    /// The request as this writer wrote it, index-aligned with `history`;
    /// `None` for one read from the file. It keeps its messages alive, so
    /// their addresses stay theirs.
    sent: Option<Arc<CompletionRequest>>,
    /// Each text's hash and place in `history`, in order.
    starts: OnceLock<Vec<(u64, usize)>>,
    /// How many requests were noted before it.
    noted: usize,
}

impl Head {
    /// The text this request gave `message`, when it held that very
    /// message at `at`.
    fn text_of(&self, message: &Arc<Message>, at: usize) -> Option<&Text> {
        let sent = self.sent.as_ref()?;
        sent.chat_history
            .get(at)
            .filter(|held| Arc::ptr_eq(held, message))?;
        self.history.get(at)
    }

    /// The places in `history` of the texts with `hash`, in order.
    fn starts(&self, hash: u64) -> impl Iterator<Item = usize> {
        let starts = self.starts.get_or_init(|| {
            let mut starts: Vec<_> = self.history.iter().map(|text| text.hash).zip(0..).collect();
            starts.sort_unstable();
            starts
        });
        let first = starts.partition_point(|&(other, _)| other < hash);
        let starts = starts.get(first..).unwrap_or_default().iter();
        starts.map_while(move |&(other, at)| (other == hash).then_some(at))
    }
}

/// A message of the request being written: its text, and its JSON value
/// when the text was made from it.
struct Sent<'a> {
    message: &'a Message,
    text: Text,
    value: Option<Value>,
}

impl Sent<'_> {
    fn value(&self) -> serde_json::Result<Cow<'_, Value>> {
        match &self.value {
            Some(value) => Ok(Cow::Borrowed(value)),
            None => serde_json::to_value(self.message).map(Cow::Owned),
        }
    }
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
        if let Some(role) = remove_preserving_order(&mut rest, "role") {
            whole.insert("role".to_owned(), role);
        }
        whole.insert("content".to_owned(), serde_json::from_str(&self.content)?);
        whole.insert("origin".to_owned(), serde_json::from_str(&self.origin)?);
        whole.extend(rest);
        Ok(Value::Object(whole))
    }
}

impl Known {
    /// What the lines `file` reads establish. From a line that cannot be
    /// read back on, only ids are read, so a writer continues with whole
    /// records and a whole header and with ids above every id a line
    /// states.
    fn of(file: impl BufRead) -> io::Result<Self> {
        /// A record line's id.
        #[derive(Deserialize)]
        struct IdOnly {
            id: EffectId,
        }
        let mut known = Self::default();
        let mut whole = true;
        for line in file.split(b'\n') {
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
            self.note(&value, None, None)?;
            return Ok(Some(value));
        }
        let (record, texts) = self.restore(serde_json::from_value(value)?, whole)?;
        self.note(&record, Some(texts), None)?;
        Ok(Some(record))
    }

    /// Notes a record: its id may be the highest; a completion becomes its
    /// chain's latest request, with `texts`, its history and tools as
    /// [`Self::restore`] gives them, or else as the record holds them, and
    /// `sent`, the request this writer wrote; a tool call's result becomes
    /// the latest.
    fn note(
        &mut self,
        record: &Value,
        texts: Option<Texts>,
        sent: Option<&Arc<CompletionRequest>>,
    ) -> io::Result<()> {
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
                Some(Value::Array(messages)) => messages.iter().map(Text::of).collect(),
                _ => Vec::new(),
            };
            (history, field("tools").map(Text::of))
        });
        // A request this writer wrote noted its new texts itself.
        if sent.is_none() {
            for text in &history {
                self.seen.entry(text.hash).or_insert(self.noted);
            }
        }
        let sent = sent.filter(|sent| sent.chat_history.len() == history.len());
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
            sent: sent.cloned(),
            starts: OnceLock::new(),
            noted: self.noted,
        });
        self.noted += 1;
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
        remove_preserving_order(&mut rest, "content");
        remove_preserving_order(&mut rest, "origin");
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
                remove_preserving_order(item, "content");
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
        let mut held = match remove_preserving_order(request, "chat_history") {
            Some(Value::Array(held)) => held,
            _ => Vec::new(),
        }
        .into_iter();
        let mut history = messages(0, keep)?.to_vec();
        if then.is_empty() {
            history.extend(held.by_ref().map(|message| Text::of(&message)));
        }
        for piece in then {
            match piece {
                Piece::Base(from, to) => history.extend_from_slice(messages(from, to)?),
                Piece::New(count) => {
                    for _ in 0..count {
                        history.push(Text::of(&held.next().ok_or_else(short)?));
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
                    history.push(Text::of(&reply.restore(held.next().ok_or_else(short)?)?));
                }
                Piece::Results(ids) => {
                    let message = held.next().ok_or_else(short)?;
                    history.push(Text::of(&self.restore_results(message, ids)?));
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
            Some(tools.clone())
        } else {
            request.get("tools").map(Text::of)
        };
        if whole {
            let messages = history
                .iter()
                .map(|message| serde_json::from_str(&message.text));
            let messages = messages.collect::<Result<_, _>>()?;
            request.insert("chat_history".to_owned(), Value::Array(messages));
            if same_tools && let Some(tools) = &tools {
                request.insert("tools".to_owned(), serde_json::from_str(&tools.text)?);
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
    /// How `history` and `tools` are written against `base`, or `None`
    /// when they share nothing with it or it costs `under` or more.
    fn of(
        base: &Head,
        keep: usize,
        history: &[Sent<'_>],
        tools: &Text,
        known: &Known,
        under: Option<usize>,
    ) -> io::Result<Option<Self>> {
        let mut reused = keep > 0;
        let mut then = Vec::new();
        let mut written = Vec::new();
        let mut cost = 0;
        let mut at = keep;
        while let Some(sent) = history.get(at) {
            if under.is_some_and(|under| cost >= under) {
                return Ok(None);
            }
            let held = known.seen.get(&sent.text.hash);
            let run = (held.is_some_and(|first| *first <= base.noted))
                .then(|| base.starts(sent.text.hash))
                .into_iter()
                .flatten()
                .map(|from| {
                    let kept = base.history.get(from..).unwrap_or_default();
                    let sent = history.get(at..).unwrap_or_default();
                    let len = kept
                        .iter()
                        .zip(sent)
                        .take_while(|(kept, sent)| **kept == sent.text)
                        .count();
                    (from, len)
                })
                .max_by_key(|&(_, len)| len);
            if let Some((from, len)) = run.filter(|&(_, len)| len > 0) {
                match then.last_mut() {
                    Some(Piece::Base(_, to)) if *to == from => *to += len,
                    _ => then.push(Piece::Base(from, from + len)),
                }
                reused = true;
                at += len;
                continue;
            }
            let message = sent.value()?;
            let repeated = known
                .reply_in(&message)
                .map(|(id, rest)| (Piece::Reply(id), rest))
                .or_else(|| {
                    known
                        .results_in(&message)
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
                    cost += sent.text.len();
                    match then.last_mut() {
                        Some(Piece::New(count)) => *count += 1,
                        _ => then.push(Piece::New(1)),
                    }
                    written.push(message.into_owned());
                }
            }
            at += 1;
        }
        if !reused {
            return Ok(None);
        }
        let same_tools = base.tools.as_ref() == Some(tools);
        if !same_tools {
            cost += tools.len();
        }
        if then.iter().all(|piece| matches!(piece, Piece::New(_))) {
            then.clear();
        }
        // About the bytes of a piece.
        cost += then.len() * 16;
        if under.is_some_and(|under| cost >= under) {
            return Ok(None);
        }
        Ok(Some(Self {
            after: base.id,
            keep,
            then,
            written,
            same_tools,
            cost,
        }))
    }
}

/// How many of `base`'s first messages `history` starts with.
fn shared(base: &Head, history: &[Sent<'_>]) -> usize {
    (base.history.iter().zip(history))
        .take_while(|(kept, sent)| **kept == sent.text)
        .count()
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

/// Reads a JSON-lines log from `lines` into one [`EffectLog`], such as the
/// bytes of a session's file in its store ([`Files::read_file`]): every
/// record in file order, under the headers merged. A handler described
/// again takes its latest description; signatures, required rows and
/// stream errors accumulate; the hook stack and run spec are the latest
/// stated. A continued completion request is restored whole.
///
/// # Errors
///
/// When `lines` cannot be read, or holds a line that is neither a header
/// nor a record, or a continuation that refers to a request that is not
/// recent or places messages it does not have.
pub fn read_from(lines: impl BufRead) -> io::Result<EffectLog> {
    let mut known = Known::default();
    let mut records = Vec::new();
    for line in lines.lines() {
        if let Some(record) = known.line(&line?, true)? {
            records.push(serde_json::from_value(record)?);
        }
    }
    Ok(EffectLog {
        header: known.header,
        records,
    })
}

/// Reads the session's effect log, the [`SESSION_EFFECT_LOG`] of its
/// store's `files`, as [`read_from`] does; a store without one holds an
/// empty log.
///
/// # Errors
///
/// When the file cannot be read, or holds a line [`read_from`] refuses.
pub fn read_session(files: &dyn Files) -> io::Result<EffectLog> {
    read_from(session_bytes(files)?.as_slice())
}

/// The records of the session's effect log as [`read_outcomes_from`] reads
/// them; none when its store has no log.
///
/// # Errors
///
/// When the file cannot be read, or holds a line that is neither a header
/// nor a record.
pub fn read_session_outcomes(files: &dyn Files) -> io::Result<Vec<EffectRecord>> {
    read_outcomes_from(session_bytes(files)?.as_slice())
}

/// The bytes of the session's effect log; none when there is none.
fn session_bytes(files: &dyn Files) -> io::Result<Vec<u8>> {
    match files.read_file(SESSION_EFFECT_LOG) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        read => read,
    }
}

/// Reads the records of a JSON-lines log as [`read_from`] does, but with
/// every completion request's history and tools left empty: for a reader
/// of what the effects were and how they ended, such as a call tree, not
/// of what each model call was sent. It rebuilds no continued request, so
/// it reads in time linear in the log's size, where [`read_from`] rebuilds
/// every request whole; and it does not check a continuation against the
/// request it continues.
///
/// # Errors
///
/// When `lines` cannot be read, or holds a line that is neither a header
/// nor a record.
pub fn read_outcomes_from(lines: impl BufRead) -> io::Result<Vec<EffectRecord>> {
    let mut records = Vec::new();
    for line in lines.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut record: Value = serde_json::from_str(&line)?;
        if record.get("header").is_some() {
            continue;
        }
        if record.get("after").is_some() {
            let DeltaLine {
                id, record: held, ..
            } = serde_json::from_value::<DeltaLine<Value>>(record)?;
            record = held;
            if let Some(fields) = record.as_object_mut() {
                fields.insert("id".to_owned(), serde_json::to_value(id)?);
            }
        }
        if record.pointer("/kind/effect").and_then(Value::as_str) == Some("completion")
            && let Some(request) = record
                .pointer_mut("/kind/request")
                .and_then(Value::as_object_mut)
        {
            request.insert("chat_history".to_owned(), Value::Array(Vec::new()));
            request.insert("tools".to_owned(), Value::Array(Vec::new()));
        }
        records.push(serde_json::from_value(record)?);
    }
    Ok(records)
}

/// Folds a later header into the merged one.
fn merge(into: &mut LogHeader, header: LogHeader) {
    into.describe(header.handlers);
    for (key, family) in header.signature.iter() {
        into.signature.insert_if_absent(key.clone(), *family);
    }
    for (key, family) in header.required.iter() {
        into.required.insert_if_absent(key.clone(), *family);
    }
    into.stream_errors.extend(header.stream_errors);
    if !header.hooks.is_empty() {
        into.hooks = header.hooks;
    }
    into.run_spec = header.run_spec.or(into.run_spec);
    into.programs.extend(header.programs);
}

/// What a header line must state for a reader that merged `merged` to
/// merge `header`: the parts of `header` that [`merge`] would change.
fn delta(merged: &LogHeader, header: &LogHeader) -> LogHeader {
    let unknown = |row: &EffectRow, known: &EffectRow| -> EffectRow {
        row.iter()
            .filter(|(key, _)| !known.contains_key(key))
            .map(|(key, family)| (key.clone(), *family))
            .collect()
    };
    LogHeader {
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
        programs: header
            .programs
            .iter()
            .filter(|(scope, program)| merged.programs.get(*scope) != Some(*program))
            .map(|(scope, program)| (scope.clone(), program.clone()))
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
