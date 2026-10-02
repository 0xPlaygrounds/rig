//! `cargo xtask cassette audit [--base REF]`: check the effect goldens.
//!
//! Every golden's streams are checked on their own: an ended text part is
//! the text its fragments carried, and an ended tool call's arguments are
//! the JSON it streamed. Every golden that differs from the base is then
//! classified change by change. A stream in the block shape that finalizes
//! the same content, in the same order, as its typed replacement is a
//! migrated stream; a count that follows the stream items (a delivery
//! batch, a stream error's position, a validated offset) moves only in a
//! migrated golden. A change outside these kinds, or a part that disagrees
//! with its fragments, fails the audit.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde_json::Value;

use crate::support::output;

const EFFECTS: &str = "crates/rig-cassette/fixtures/effects";

/// What one kind of golden change is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Change {
    /// A block-shaped stream rewritten as the typed stream that finalizes
    /// the same content in the same order.
    Migrated,
    /// A part the block stream left open where its run stopped, read to its
    /// end by the typed stream.
    ReadOn,
    /// The tail of a cut stream: a part the block sink closed at the cut is
    /// left open, or, a call, is absent because only closed calls surface.
    CutTail,
    /// A streamed effect's outcome `raw`, now the reply's terminal document
    /// rather than the stream's own summary.
    RawReplaced,
    /// A truncated stream's error, now `ProviderError::Truncated`.
    TruncationReported,
    /// A rejected call a retried request replays as the call the model made.
    CallRestored,
    /// A migrated golden's delivery batches: its effects and their outcome
    /// deliveries are the base's, in order, and no batch delivers past the
    /// stream.
    Rebatched,
    /// A count that follows the number of stream items, in a migrated golden.
    CountShift,
    /// A deleted golden no test names any more.
    Retired,
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Migrated => "stream migrated",
            Self::ReadOn => "open part read on",
            Self::CutTail => "cut tail left open",
            Self::RawReplaced => "raw replaced by the terminal document",
            Self::TruncationReported => "truncation reported as Truncated",
            Self::CallRestored => "rejected call replayed as made",
            Self::Rebatched => "delivery rebatched",
            Self::CountShift => "count shift",
            Self::Retired => "golden retired",
        })
    }
}

/// The findings over a set of goldens.
#[derive(Debug, Default)]
pub(crate) struct Audit {
    pub(crate) files: usize,
    pub(crate) changed: usize,
    /// Ended parts compared with their fragments.
    pub(crate) parts: usize,
    pub(crate) changes: BTreeMap<Change, usize>,
    /// A part that disagrees with its fragments.
    pub(crate) mismatches: Vec<String>,
    /// A change outside the known kinds, delivery churn included.
    pub(crate) other: Vec<String>,
}

impl Audit {
    fn count(&mut self, change: Change, by: usize) {
        *self.changes.entry(change).or_default() += by;
    }

    fn other(&mut self, path: &str, what: impl fmt::Display) {
        self.other.push(format!("{path}: {what}"));
    }

    /// Check one golden's streams, and classify its changes from `base`
    /// when it has one.
    pub(crate) fn file(&mut self, path: &str, base: Option<&Value>, head: &Value) {
        self.files += 1;
        walk_streams(head, &mut |items| self.parts(path, items));
        if let Some(base) = base
            && base != head
        {
            self.changed += 1;
            File::new(self, path).run(base, head);
        }
    }

    /// Check every ended part in `items` against its fragments.
    fn parts(&mut self, path: &str, items: &[Value]) {
        let mut text: BTreeMap<u64, String> = BTreeMap::new();
        let mut arguments: BTreeMap<u64, String> = BTreeMap::new();
        for event in items.iter().filter_map(event_of) {
            let part = event
                .get("part")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let field = |key| event.get(key).and_then(Value::as_str).unwrap_or_default();
            match event.get("event").and_then(Value::as_str) {
                Some("text") => text.entry(part).or_default().push_str(field("text")),
                Some("arguments") => arguments.entry(part).or_default().push_str(field("json")),
                Some("end") => {
                    let content = event.get("content").unwrap_or(&Value::Null);
                    let agrees = match content.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            content.get("text").and_then(Value::as_str)
                                == Some(text.remove(&part).unwrap_or_default().as_str())
                        }
                        Some("toolcall") => {
                            let streamed = arguments.remove(&part).unwrap_or_default();
                            let parsed = match streamed.trim() {
                                "" => Some(Value::Object(serde_json::Map::new())),
                                json => serde_json::from_str(json).ok(),
                            };
                            parsed.as_ref() == content.pointer("/function/arguments")
                        }
                        _ => continue,
                    };
                    self.parts += 1;
                    if !agrees {
                        self.mismatches.push(format!(
                            "{path}: part {part}'s end disagrees with its fragments"
                        ));
                    }
                }
                _ => {}
            }
        }
    }
}

/// The event an item carries, when it is one.
fn event_of(item: &Value) -> Option<&Value> {
    (item.get("item").and_then(Value::as_str) == Some("event"))
        .then(|| item.get("value"))
        .flatten()
}

/// Whether `events` is a typed stream: a list of items.
fn typed(events: &[Value]) -> bool {
    events.iter().all(|item| item.get("item").is_some())
}

/// Whether `events` is a block-shaped stream: a list of block events.
fn block_shaped(events: &[Value]) -> bool {
    events.iter().all(|event| event.get("event").is_some())
}

/// Call `visit` with every typed stream in `value`.
fn walk_streams(value: &Value, visit: &mut impl FnMut(&[Value])) {
    match value {
        Value::Object(object) => {
            if let Some(Value::Array(items)) = object.get("events")
                && typed(items)
            {
                visit(items);
            }
            for child in object.values() {
                walk_streams(child, visit);
            }
        }
        Value::Array(values) => values.iter().for_each(|child| walk_streams(child, visit)),
        _ => {}
    }
}

/// One part of a stream, in start order: its kind, the fragments it
/// streamed, and what its end finalized. `ended` is `None` for a part left
/// open and `Some(Value::Null)` for an end that finalized nothing.
#[derive(Debug)]
struct Part<'a> {
    kind: &'a str,
    fragments: String,
    ended: Option<&'a Value>,
}

impl Part<'_> {
    /// The text a part carried: its ended text, else its fragments.
    fn text(&self) -> &str {
        match self.ended {
            Some(block) if block.get("type").and_then(Value::as_str) == Some("text") => block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            _ => &self.fragments,
        }
    }
}

/// The parts of a block-shaped stream, in the order their ids first appear.
/// An event on an ended id opens a new part, unless it is another end.
fn block_parts(events: &[Value]) -> Vec<Part<'_>> {
    const NOTHING: &Value = &Value::Null;
    let mut parts: Vec<Part<'_>> = Vec::new();
    let mut current: BTreeMap<&str, usize> = BTreeMap::new();
    for event in events {
        let Some(id) = event.get("id").and_then(Value::as_str) else {
            continue;
        };
        let name = event.get("event").and_then(Value::as_str);
        let kind = event
            .pointer("/kind/kind")
            .or_else(|| event.pointer("/end/close"))
            .and_then(Value::as_str)
            .or(
                match event.pointer("/delta/delta").and_then(Value::as_str) {
                    Some("tool_arguments" | "tool_name") => Some("tool_call"),
                    delta => delta,
                },
            );
        let index = match current.get(id) {
            Some(&index)
                if name == Some("block_end")
                    || parts.get(index).is_some_and(|part| part.ended.is_none()) =>
            {
                index
            }
            _ => {
                parts.push(Part {
                    kind: kind.unwrap_or_default(),
                    fragments: String::new(),
                    ended: None,
                });
                current.insert(id, parts.len() - 1);
                parts.len() - 1
            }
        };
        let Some(part) = parts.get_mut(index) else {
            continue;
        };
        if part.kind.is_empty() {
            part.kind = kind.unwrap_or_default();
        }
        match name {
            Some("block_delta") => {
                let fragment = event
                    .pointer("/delta/text")
                    .or_else(|| event.pointer("/delta/arguments"))
                    .and_then(Value::as_str);
                part.fragments.push_str(fragment.unwrap_or_default());
            }
            Some("block_end") => match event.get("block").filter(|block| !block.is_null()) {
                Some(block) => part.ended = Some(block),
                None => {
                    part.ended.get_or_insert(NOTHING);
                }
            },
            _ => {}
        }
    }
    parts
}

/// The parts of a typed stream, in start order; an unknown item is a part
/// of its own that finalized its payload.
fn typed_parts(items: &[Value]) -> Vec<Part<'_>> {
    let mut parts: Vec<Part<'_>> = Vec::new();
    let mut by_index: BTreeMap<u64, usize> = BTreeMap::new();
    for item in items {
        let Some(event) = event_of(item) else {
            parts.push(Part {
                kind: "unknown",
                fragments: String::new(),
                ended: item.get("value"),
            });
            continue;
        };
        let part = event
            .get("part")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let field = |key| event.get(key).and_then(Value::as_str);
        match field("event") {
            Some("start") => {
                by_index.insert(part, parts.len());
                parts.push(Part {
                    kind: field("kind").unwrap_or_default(),
                    fragments: String::new(),
                    ended: None,
                });
            }
            Some(name) => {
                let Some(part) = by_index.get(&part).and_then(|&index| parts.get_mut(index)) else {
                    continue;
                };
                match name {
                    "text" | "reasoning" => {
                        part.fragments.push_str(field("text").unwrap_or_default())
                    }
                    "arguments" => part.fragments.push_str(field("json").unwrap_or_default()),
                    "end" => part.ended = event.get("content"),
                    _ => {}
                }
            }
            None => {}
        }
    }
    parts
}

/// Whether a record's stream was cut: its outcome is an error or a reply
/// the provider ended for length.
fn cut(outcome: Option<&Value>) -> bool {
    outcome.is_some_and(|outcome| {
        outcome.get("Err").is_some()
            || outcome.pointer("/Ok/finish_reason").and_then(Value::as_str) == Some("length")
    })
}

/// Whether the parts from `from` on are the tail of a cut stream: each part
/// the block sink closed at the cut is open in the typed stream with the
/// same content so far, or, a call, is absent because only closed calls
/// surface.
fn cut_tail(base: &[Part<'_>], head: &[Part<'_>], from: usize) -> bool {
    head.len() <= base.len()
        && base
            .iter()
            .enumerate()
            .skip(from)
            .all(|(at, old)| match head.get(at) {
                Some(new) => {
                    new.kind == old.kind
                        && new.ended.is_none()
                        && if old.kind == "tool_call" {
                            old.fragments.starts_with(&new.fragments)
                                || new.fragments.starts_with(&old.fragments)
                        } else {
                            new.fragments == old.text()
                        }
                }
                None => old.kind == "tool_call",
            })
}

/// The comparison of one golden with its base.
struct File<'a> {
    audit: &'a mut Audit,
    path: &'a str,
    /// Whether a stream of this golden migrated: its counts may move.
    migrated: bool,
    /// The records whose streams migrated: their outcome's `raw` may be the
    /// reply's terminal document.
    records: BTreeSet<String>,
    /// Record id to its typed stream's length.
    lengths: BTreeMap<u64, usize>,
    /// Every tool call the base's replies finalized.
    calls: Vec<Value>,
    /// A replayed call's base id to the id of the call it now is.
    restored: BTreeMap<String, Value>,
}

impl<'a> File<'a> {
    fn new(audit: &'a mut Audit, path: &'a str) -> Self {
        Self {
            audit,
            path,
            migrated: false,
            records: BTreeSet::new(),
            lengths: BTreeMap::new(),
            calls: Vec::new(),
            restored: BTreeMap::new(),
        }
    }

    fn run(mut self, base: &Value, head: &Value) {
        for record in base
            .get("records")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let choice = record
                .pointer("/outcome/Ok/choice")
                .and_then(Value::as_array);
            self.calls.extend(
                choice
                    .into_iter()
                    .flatten()
                    .filter(|content| {
                        content.get("type").and_then(Value::as_str) == Some("toolcall")
                    })
                    .cloned(),
            );
        }
        let mut validated = Vec::new();
        self.compare(base, head, "", &mut validated);
        for (at, before, after) in validated {
            if self.migrated {
                self.audit.count(Change::CountShift, 1);
            } else {
                self.audit.other(
                    self.path,
                    format!("{at}: validated offset {before} -> {after}"),
                );
            }
        }
        self.header(base, head);
    }

    fn compare(
        &mut self,
        base: &Value,
        head: &Value,
        at: &str,
        validated: &mut Vec<(String, Value, Value)>,
    ) {
        match (base, head) {
            (Value::Object(old), Value::Object(new)) => {
                if self.restored_call(base, head, at) {
                    return;
                }
                if old.keys().ne(new.keys()) {
                    self.audit.other(self.path, format!("{at}: keys differ"));
                    return;
                }
                if let (Some(Value::Array(old_events)), Some(Value::Array(new_events))) =
                    (old.get("events"), new.get("events"))
                    && typed(new_events)
                {
                    if self.stream(
                        old_events,
                        new_events,
                        new.get("outcome"),
                        &format!("{at}/events"),
                    ) {
                        self.records.insert(at.to_owned());
                    }
                    if let (true, Some(id)) = (
                        at.starts_with("/records["),
                        new.get("id").and_then(Value::as_u64),
                    ) {
                        self.lengths.insert(id, new_events.len());
                    }
                }
                for (key, value) in old {
                    let child = format!("{at}/{key}");
                    if key == "events"
                        && new
                            .get("events")
                            .and_then(Value::as_array)
                            .is_some_and(|events| typed(events))
                        || at == "/header" && (key == "deliveries" || key == "stream_errors")
                    {
                        continue;
                    }
                    if (key == "outcome" || key.ends_with("::EffectOutcome"))
                        && streamed(new)
                        && value.pointer("/Ok/raw") != new[key].pointer("/Ok/raw")
                        && new[key].pointer("/Ok/raw").is_some_and(Value::is_object)
                    {
                        self.audit.count(Change::RawReplaced, 1);
                        let (old, new) = (without_raw(value), without_raw(&new[key]));
                        self.compare(&old, &new, &child, validated);
                        continue;
                    }
                    if key == "stream_validated" && value != &new[key] {
                        validated.push((child, value.clone(), new[key].clone()));
                        continue;
                    }
                    self.compare(value, &new[key], &child, validated);
                }
            }
            (Value::Array(old), Value::Array(new)) => {
                if old.len() != new.len() {
                    self.audit.other(self.path, format!("{at}: length differs"));
                    return;
                }
                for (index, (old, new)) in old.iter().zip(new).enumerate() {
                    self.compare(old, new, &format!("{at}[{index}]"), validated);
                }
            }
            (Value::String(old), Value::String(new))
                if at.ends_with("/Err/message")
                    && old == "the stream ended before its terminal record"
                    && new == "ResponseError: the reply ended before the provider ended it" =>
            {
                self.audit.count(Change::TruncationReported, 1);
            }
            (old, new) if old != new => self.audit.other(self.path, format!("{at}: value differs")),
            _ => {}
        }
    }

    /// A rejected call a retried request replays: the base replayed it with a
    /// local id and no arguments, and now it is the call the model made, one
    /// the base's replies finalized under that name. Its result follows the
    /// call's id.
    fn restored_call(&mut self, base: &Value, head: &Value, at: &str) -> bool {
        if !at.contains("/chat_history[") {
            return false;
        }
        let kind = |value: &Value| value.get("type").and_then(Value::as_str).map(str::to_owned);
        match (kind(base).as_deref(), kind(head).as_deref()) {
            (Some("toolcall"), Some("toolcall")) => {
                let local = base.pointer("/id/local").is_some();
                let name = base.pointer("/function/name");
                let made = self
                    .calls
                    .iter()
                    .any(|call| call == head && call.pointer("/function/name") == name);
                if base != head && local && made {
                    self.restored
                        .insert(base["id"].to_string(), head["id"].clone());
                    self.audit.count(Change::CallRestored, 1);
                    return true;
                }
                false
            }
            (Some("toolresult"), Some("toolresult")) => {
                let Some(call) = self.restored.get(&base["call"].to_string()) else {
                    return false;
                };
                let mut rebased = base.clone();
                if let Some(result) = rebased.as_object_mut() {
                    result.insert("call".to_owned(), call.clone());
                }
                &rebased == head
            }
            _ => false,
        }
    }

    /// A stream that changed: a block-shaped one migrated to the typed
    /// stream whose parts, in start order, finalize the same content. A part
    /// the base left open may read on in the typed stream; a cut stream's
    /// tail may stay open. Returns whether the stream migrated.
    fn stream(&mut self, old: &[Value], new: &[Value], outcome: Option<&Value>, at: &str) -> bool {
        if old == new {
            return false;
        }
        if !block_shaped(old) {
            self.audit
                .other(self.path, format!("{at}: the stream changed"));
            return false;
        }
        let (base, head) = (block_parts(old), typed_parts(new));
        let mut read_on = 0;
        for (index, (before, after)) in base.iter().zip(&head).enumerate() {
            if before.kind == after.kind
                && match before.ended {
                    Some(ended) => after.ended == Some(ended),
                    None => after.fragments.starts_with(&before.fragments),
                }
            {
                read_on += usize::from(before.ended.is_none());
                continue;
            }
            if cut(outcome) && cut_tail(&base, &head, index) {
                self.audit.count(Change::CutTail, 1);
                return self.migrated(read_on);
            }
            self.audit.other(
                self.path,
                format!(
                    "{at}: part {index} finalizes {} in the base, {} now",
                    finalized(before),
                    finalized(after)
                ),
            );
            return false;
        }
        if base.len() > head.len() {
            if cut(outcome) && cut_tail(&base, &head, head.len()) {
                self.audit.count(Change::CutTail, 1);
                return self.migrated(read_on);
            }
            self.audit.other(
                self.path,
                format!("{at}: {} base parts dropped", base.len() - head.len()),
            );
            return false;
        }
        if head.len() > base.len() && base.last().is_none_or(|last| last.ended.is_some()) {
            self.audit.other(
                self.path,
                format!(
                    "{at}: {} parts past the base's end",
                    head.len() - base.len()
                ),
            );
            return false;
        }
        self.migrated(read_on)
    }

    /// Record a migrated stream, `read_on` of whose parts the base left open.
    fn migrated(&mut self, read_on: usize) -> bool {
        self.migrated = true;
        self.audit.count(Change::Migrated, 1);
        self.audit.count(Change::ReadOn, read_on);
        true
    }

    /// Stream error positions and delivery batches: the base's, moved only
    /// in a migrated golden, and never past the stream they count.
    fn header(&mut self, base: &Value, head: &Value) {
        let errors = |log: &Value| log.pointer("/header/stream_errors").cloned();
        match (errors(base), errors(head)) {
            (Some(Value::Object(old)), Some(Value::Object(new))) => {
                if old.keys().ne(new.keys()) {
                    self.audit.other(self.path, "stream error effects differ");
                }
                for (id, old) in &old {
                    self.errors(id, old, new.get(id).unwrap_or(&Value::Null));
                }
            }
            (old, new) if old != new => self.audit.other(self.path, "stream errors differ"),
            _ => {}
        }
        let deliveries = |log: &Value| log.pointer("/header/deliveries").cloned();
        let (old, new) = (deliveries(base), deliveries(head));
        if old == new {
            return;
        }
        let (Some(old), Some(new)) = (
            old.as_ref().and_then(Value::as_array),
            new.as_ref().and_then(Value::as_array),
        ) else {
            self.audit
                .other(self.path, "deliveries appeared or disappeared");
            return;
        };
        let shape = |delivery: &Value| {
            (
                delivery.get("batch").cloned(),
                delivery.get("id").cloned(),
                delivery.pointer("/kind/delivery").cloned(),
            )
        };
        let outcomes = |deliveries: &[Value]| {
            deliveries
                .iter()
                .filter(|delivery| delivery.pointer("/kind/items").is_none())
                .map(|delivery| {
                    (
                        delivery.get("id").cloned(),
                        delivery.pointer("/kind").cloned(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let streams = |deliveries: &[Value]| {
            deliveries
                .iter()
                .filter(|delivery| delivery.pointer("/kind/items").is_some())
                .filter_map(|delivery| delivery.get("id").and_then(Value::as_u64))
                .collect::<BTreeSet<_>>()
        };
        let batches = new
            .iter()
            .map(|delivery| delivery.get("batch").and_then(Value::as_u64));
        if !self.migrated {
            self.audit
                .other(self.path, "delivery batches differ from the base's");
            return;
        }
        if old.len() == new.len()
            && old
                .iter()
                .zip(new)
                .all(|(old, new)| shape(old) == shape(new))
        {
            let shifted = old.iter().zip(new).filter(|(old, new)| old != new).count();
            self.audit.count(Change::CountShift, shifted);
        } else if outcomes(old) == outcomes(new)
            && streams(new).is_subset(&streams(old))
            && batches.clone().is_sorted()
            && batches.clone().all(|batch| batch.is_some())
        {
            self.audit.count(Change::Rebatched, 1);
        } else {
            self.audit
                .other(self.path, "delivery batches differ from the base's");
            return;
        }
        let mut delivered = BTreeMap::<u64, u64>::new();
        for new in new {
            let items = new.pointer("/kind/items").and_then(Value::as_u64);
            if let (Some(id), Some(items)) = (new.get("id").and_then(Value::as_u64), items) {
                *delivered.entry(id).or_default() += items;
            }
        }
        let errors = head
            .pointer("/header/stream_errors")
            .and_then(Value::as_object);
        for (id, delivered) in delivered {
            let Some(length) = self.lengths.get(&id) else {
                continue;
            };
            let errors = errors
                .and_then(|errors| errors.get(&id.to_string()))
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if delivered > (length + errors) as u64 {
                self.audit.other(
                    self.path,
                    format!(
                        "effect {id} delivered {delivered} items of {length} and {errors} errors"
                    ),
                );
            }
        }
    }

    /// The same errors in the same order; their positions move only in a
    /// migrated golden.
    fn errors(&mut self, id: &str, old: &Value, new: &Value) {
        let (Some(old), Some(new)) = (old.as_array(), new.as_array()) else {
            self.audit
                .other(self.path, format!("stream errors of {id} are not lists"));
            return;
        };
        if old.len() != new.len() {
            self.audit.other(
                self.path,
                format!("stream errors of {id} changed in number"),
            );
            return;
        }
        let strip = |error: &Value| {
            let mut error = error.clone();
            if let Some(object) = error.as_object_mut() {
                object.shift_remove("item");
            }
            error
        };
        for (k, (old, new)) in old.iter().zip(new).enumerate() {
            if strip(old) != strip(new) {
                self.audit
                    .other(self.path, format!("stream error {k} of {id} changed"));
            } else if old.get("item") != new.get("item") {
                if self.migrated {
                    self.audit.count(Change::CountShift, 1);
                } else {
                    self.audit
                        .other(self.path, format!("stream error {k} of {id} moved"));
                }
            }
        }
    }
}

/// Whether no test or source names a deleted golden any more, so its
/// producer went with it.
fn retired(root: &Path, path: &str) -> Result<bool, String> {
    let Some(name) = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".effects.json"))
    else {
        return Ok(false);
    };
    let named = output(
        root,
        "git",
        &["grep", "-l", "-F", &format!("\"{name}\""), "--", "*.rs"],
    );
    // `git grep` exits 1, an error here, when nothing matches.
    Ok(named.is_err_and(|error| !error.contains("fatal")))
}

/// Whether an effect streamed: a completion requested as a stream, or one
/// with a stream of its own or on a component.
fn streamed(effect: &serde_json::Map<String, Value>) -> bool {
    effect.get("kind").and_then(|kind| kind.get("stream")) == Some(&Value::Bool(true))
        || std::iter::once(effect)
            .chain(effect.values().filter_map(Value::as_object))
            .any(|object| object.get("events").is_some_and(Value::is_array))
}

/// An outcome without its `raw` document.
fn without_raw(outcome: &Value) -> Value {
    let mut outcome = outcome.clone();
    if let Some(ok) = outcome.get_mut("Ok").and_then(Value::as_object_mut) {
        ok.shift_remove("raw");
    }
    outcome
}

/// How a part reads in a finding.
fn finalized(part: &Part<'_>) -> String {
    let content = match part.ended {
        Some(ended) => ended.to_string(),
        None => format!("open {:?}", part.fragments),
    };
    let mut content = format!("{} {content}", part.kind);
    if content.len() > 160 {
        let cut = (0..=160)
            .rev()
            .find(|&at| content.is_char_boundary(at))
            .unwrap_or(0);
        content.truncate(cut);
        content.push('…');
    }
    content
}

/// Every effect golden in the working tree, by its path from `root`, parsed
/// and in path order. It reads the files, not git, so a golden not yet
/// committed is checked too.
pub(crate) fn goldens(root: &Path) -> Result<Vec<(String, Value)>, String> {
    crate::support::files_under(&root.join(EFFECTS), Some("json"))?
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(root)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{relative}: {e}"))?;
            let golden = serde_json::from_str(&text).map_err(|e| format!("{relative}: {e}"))?;
            Ok((relative, golden))
        })
        .collect()
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let mut base = "HEAD".to_owned();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = args.next().cloned().ok_or("--base needs a ref")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let status = output(
        root,
        "git",
        &["diff", "--name-status", &base, "--", EFFECTS],
    )?;
    let mut changed = Vec::new();
    let mut audit = Audit::default();
    for line in status.lines() {
        let mut fields = line.split('\t');
        match (fields.next(), fields.next()) {
            (Some("M"), Some(path)) => changed.push(path),
            (Some("D"), Some(path)) if retired(root, path)? => audit.count(Change::Retired, 1),
            (Some(kind), Some(path)) => audit.other(path, format!("golden status {kind}")),
            _ => audit.other(line, "unreadable golden status"),
        }
    }
    let cassettes = output(
        root,
        "git",
        &[
            "diff",
            "--name-only",
            &base,
            "--",
            "crates/rig-cassette/fixtures/cassettes",
        ],
    )?;
    for (path, head) in goldens(root)? {
        let path = path.as_str();
        let base = if changed.contains(&path) {
            let text = output(root, "git", &["show", &format!("{base}:{path}")])?;
            match serde_json::from_str::<Value>(&text) {
                Ok(base) => Some(base),
                Err(error) => {
                    audit.other(path, format!("the base golden does not parse: {error}"));
                    continue;
                }
            }
        } else {
            None
        };
        audit.file(path, base.as_ref(), &head);
    }
    println!(
        "{} goldens, {} changed from {base}; {} ended parts checked against their fragments",
        audit.files, audit.changed, audit.parts
    );
    for (change, count) in &audit.changes {
        println!("{change}: {count}");
    }
    println!("other: {}", audit.other.len());
    println!("mismatches: {}", audit.mismatches.len());
    println!("cassettes changed: {}", cassettes.lines().count());
    for problem in audit.mismatches.iter().chain(&audit.other).take(40) {
        println!("  {problem}");
    }
    if audit.other.is_empty() && audit.mismatches.is_empty() && cassettes.trim().is_empty() {
        Ok(())
    } else {
        Err("the golden audit failed".into())
    }
}

#[cfg(test)]
mod tests;
