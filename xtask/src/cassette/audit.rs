//! `cargo xtask cassette audit [--base REF]`: check the effect goldens.
//!
//! Every golden's streams are checked on their own: a text end's block is
//! the text its deltas carried, an unrestated reasoning end's block is the
//! reasoning its deltas carried since the key's last end, and a tool call
//! finalized from its fragments has the arguments they parse to. Every
//! golden that differs from the base is then classified change by change;
//! a change outside the known kinds, or a block that disagrees with its
//! deltas, fails the audit.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde_json::Value;

use super::goldens::{expected_deliveries, git, is_close, same_event};

const EFFECTS: &str = "crates/rig-cassette/fixtures/effects";

/// What one kind of golden change is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Change {
    /// A text or reasoning end the sink inserted to close a block, before
    /// the terminal record or at the end of the stream.
    CloseInserted,
    /// A reasoning start the sink inserted before a sibling part's end.
    StartInserted,
    /// An end that gained the block it finalized.
    BlockAdded,
    /// A count that follows the number of stream items, moved by exactly the
    /// events inserted before it.
    CountShift,
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CloseInserted => "close inserted",
            Self::StartInserted => "start inserted",
            Self::BlockAdded => "block added to an end",
            Self::CountShift => "count shift",
        })
    }
}

/// The findings over a set of goldens.
#[derive(Debug, Default)]
pub(crate) struct Audit {
    pub(crate) files: usize,
    pub(crate) changed: usize,
    pub(crate) blocks: usize,
    pub(crate) changes: BTreeMap<Change, usize>,
    /// A block that disagrees with its deltas.
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
        walk_events(head, &mut |events| self.blocks(path, events));
        if let Some(base) = base
            && base != head
        {
            self.changed += 1;
            File::new(self, path).run(base, head);
        }
    }

    /// Check every block an end in `events` carries against its deltas.
    fn blocks(&mut self, path: &str, events: &[Value]) {
        let mut text: BTreeMap<&str, String> = BTreeMap::new();
        let mut reasoning: BTreeMap<&str, String> = BTreeMap::new();
        let mut arguments: BTreeMap<&str, String> = BTreeMap::new();
        for event in events {
            let id = event.get("id").and_then(Value::as_str).unwrap_or_default();
            match event.get("event").and_then(Value::as_str) {
                Some("block_start")
                    if event.pointer("/kind/kind").and_then(Value::as_str) == Some("tool_call") =>
                {
                    arguments.insert(id, String::new());
                }
                Some("block_delta") => {
                    let fragment = |key| {
                        event
                            .pointer(&format!("/delta/{key}"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                    };
                    match event.pointer("/delta/delta").and_then(Value::as_str) {
                        Some("text") => text.entry(id).or_default().push_str(fragment("text")),
                        Some("reasoning") => {
                            reasoning.entry(id).or_default().push_str(fragment("text"));
                        }
                        Some("tool_arguments") => {
                            arguments
                                .entry(id)
                                .or_default()
                                .push_str(fragment("arguments"));
                        }
                        _ => {}
                    }
                }
                Some("block_end") => {
                    let close = event.pointer("/end/close").and_then(Value::as_str);
                    let block = event.get("block");
                    let deltas = match close {
                        Some("reasoning") => reasoning.remove(id).unwrap_or_default(),
                        Some("tool_call") => arguments.remove(id).unwrap_or_default(),
                        _ => String::new(),
                    };
                    let Some(block) = block else { continue };
                    self.blocks += 1;
                    let agrees = match close {
                        Some("text") => {
                            block.get("text").and_then(Value::as_str)
                                == Some(text.get(id).map_or("", String::as_str))
                        }
                        // A restatement supersedes the deltas.
                        Some("reasoning") if event.pointer("/end/reasoning").is_some() => true,
                        Some("reasoning") => reasoning_text(block) == deltas,
                        // Authoritative arguments supersede the fragments,
                        // and fragments that are not strict JSON went through
                        // the wire's repair policy.
                        Some("tool_call") if event.pointer("/end/arguments").is_some() => true,
                        Some("tool_call") => match deltas.trim() {
                            "" => {
                                block.pointer("/function/arguments")
                                    == Some(&Value::Object(serde_json::Map::new()))
                            }
                            fragments => {
                                serde_json::from_str::<Value>(fragments).map_or(true, |parsed| {
                                    parsed.is_null()
                                        || block.pointer("/function/arguments") == Some(&parsed)
                                })
                            }
                        },
                        _ => true,
                    };
                    if !agrees {
                        self.mismatches.push(format!(
                            "{path}: the block on {id}'s end disagrees with its deltas"
                        ));
                    }
                }
                _ => {}
            }
        }
    }
}

/// The text of a reasoning block's text parts, joined.
fn reasoning_text(block: &Value) -> String {
    block
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|part| part.pointer("/content/text").and_then(Value::as_str))
        .collect()
}

/// Call `visit` with every list of stream events in `value`.
fn walk_events(value: &Value, visit: &mut impl FnMut(&[Value])) {
    match value {
        Value::Object(object) => {
            if let Some(Value::Array(events)) = object.get("events")
                && events.iter().all(|event| event.get("event").is_some())
            {
                visit(events);
            }
            for child in object.values() {
                walk_events(child, visit);
            }
        }
        Value::Array(values) => values.iter().for_each(|child| walk_events(child, visit)),
        _ => {}
    }
}

/// The comparison of one golden with its base.
struct File<'a> {
    audit: &'a mut Audit,
    path: &'a str,
    /// Record id to the event positions its stream inserted.
    inserted: BTreeMap<u64, Vec<usize>>,
    /// World program run to the number of events inserted in it.
    by_run: BTreeMap<String, usize>,
    /// `stream_validated` offsets, checked once every run is counted.
    validated: Vec<(String, u64, u64)>,
}

impl<'a> File<'a> {
    fn new(audit: &'a mut Audit, path: &'a str) -> Self {
        Self {
            audit,
            path,
            inserted: BTreeMap::new(),
            by_run: BTreeMap::new(),
            validated: Vec::new(),
        }
    }

    fn run(mut self, base: &Value, head: &Value) {
        self.compare(base, head, "");
        self.header(base, head);
        for (at, before, after) in std::mem::take(&mut self.validated) {
            let run = run_of(&at);
            let grown = run
                .and_then(|run| self.by_run.get(run))
                .copied()
                .unwrap_or(0);
            if after.checked_sub(before) == Some(grown as u64) && grown > 0 {
                self.audit.count(Change::CountShift, 1);
            } else {
                self.audit.other(
                    self.path,
                    format!("{at}: validated offset {before} -> {after}"),
                );
            }
        }
    }

    fn compare(&mut self, base: &Value, head: &Value, at: &str) {
        match (base, head) {
            (Value::Object(old), Value::Object(new)) => {
                if old.keys().ne(new.keys()) {
                    self.audit.other(self.path, format!("{at}: keys differ"));
                    return;
                }
                let streams = matches!(
                    (old.get("events"), new.get("events")),
                    (Some(Value::Array(old)), Some(Value::Array(new)))
                        if old.iter().chain(new).all(|event| event.get("event").is_some())
                );
                if let (true, Some(Value::Array(old_events)), Some(Value::Array(new_events))) =
                    (streams, old.get("events"), new.get("events"))
                {
                    let inserted = self.align(old_events, new_events, &format!("{at}/events"));
                    if let Some(run) = run_of(at) {
                        *self.by_run.entry(run.to_owned()).or_default() += inserted.len();
                    }
                    if let Some(id) = at
                        .strip_prefix("/records[")
                        .and_then(|_| new.get("id"))
                        .and_then(Value::as_u64)
                    {
                        self.inserted.insert(id, inserted);
                    }
                }
                for (key, value) in old {
                    let child = format!("{at}/{key}");
                    if (key == "events" && streams)
                        || (at == "/header" && (key == "deliveries" || key == "stream_errors"))
                    {
                        continue;
                    }
                    if key == "stream_validated" && value != &new[key] {
                        let (Some(before), Some(after)) = (value.as_u64(), new[key].as_u64())
                        else {
                            self.audit.other(self.path, format!("{child}: not a count"));
                            continue;
                        };
                        self.validated.push((child, before, after));
                        continue;
                    }
                    self.compare(value, &new[key], &child);
                }
            }
            (Value::Array(old), Value::Array(new)) => {
                if old.len() != new.len() {
                    self.audit.other(self.path, format!("{at}: length differs"));
                    return;
                }
                for (index, (old, new)) in old.iter().zip(new).enumerate() {
                    self.compare(old, new, &format!("{at}[{index}]"));
                }
            }
            (old, new) if old != new => self.audit.other(self.path, format!("{at}: value differs")),
            _ => {}
        }
    }

    /// Align a base stream with the regenerated one; return the positions
    /// of the events the change inserted.
    fn align(&mut self, old: &[Value], new: &[Value], at: &str) -> Vec<usize> {
        let mut inserted = Vec::new();
        let (mut i, mut j) = (0, 0);
        while let Some(event) = new.get(j) {
            let base = old.get(i);
            if base == Some(event) {
                i += 1;
                j += 1;
            } else if base.is_some_and(|base| same_event(base, event)) {
                self.audit.count(Change::BlockAdded, 1);
                i += 1;
                j += 1;
            } else if is_close(event) {
                let run = new
                    .iter()
                    .skip(j)
                    .take_while(|event| is_close(event))
                    .count();
                let follows = new.get(j + run);
                let terminal = follows
                    .is_none_or(|next| next.get("event").and_then(Value::as_str) == Some("final"));
                if !terminal {
                    self.audit
                        .other(self.path, format!("{at}[{j}]: a close inserted mid-stream"));
                }
                self.audit.count(Change::CloseInserted, run);
                inserted.extend(j..j + run);
                j += run;
            } else if is_reasoning_start(event)
                && new.get(j + 1).is_some_and(|end| {
                    is_close(end) && end.get("id").is_some() && end.get("id") == event.get("id")
                })
            {
                self.audit.count(Change::StartInserted, 1);
                inserted.push(j);
                j += 1;
            } else {
                self.audit
                    .other(self.path, format!("{at}[{j}]: event differs"));
                return inserted;
            }
        }
        if i != old.len() {
            self.audit.other(
                self.path,
                format!("{at}: {} base events dropped", old.len() - i),
            );
        }
        inserted
    }

    /// Stream error positions and delivery batches, against what the
    /// inserted events predict.
    fn header(&mut self, base: &Value, head: &Value) {
        let errors = |log: &Value| log.pointer("/header/stream_errors").cloned();
        match (errors(base), errors(head)) {
            (Some(Value::Object(old)), Some(Value::Object(new))) => {
                if old.keys().ne(new.keys()) {
                    self.audit.other(self.path, "stream error effects differ");
                }
                for (id, old) in &old {
                    let inserted = id
                        .parse::<u64>()
                        .ok()
                        .and_then(|id| self.inserted.get(&id))
                        .cloned()
                        .unwrap_or_default();
                    self.errors(id, old, new.get(id).unwrap_or(&Value::Null), &inserted);
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
        match expected_deliveries(base, head) {
            Ok(Some(expected)) if Some(&expected) == new.as_ref() => {
                let shifted = old
                    .iter()
                    .flat_map(|old| old.as_array().into_iter().flatten())
                    .zip(expected.as_array().into_iter().flatten())
                    .filter(|(old, new)| old != new)
                    .count();
                self.audit.count(Change::CountShift, shifted);
            }
            Ok(_) => self
                .audit
                .other(self.path, "delivery batches differ from the base's"),
            Err(reason) => self.audit.other(self.path, format!("deliveries: {reason}")),
        }
    }

    fn errors(&mut self, id: &str, old: &Value, new: &Value, inserted: &[usize]) {
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
        for (k, (old, new)) in old.iter().zip(new).enumerate() {
            let position = |error: &Value| error.get("item").and_then(Value::as_u64);
            let (Some(before), Some(after)) = (position(old), position(new)) else {
                self.audit
                    .other(self.path, format!("stream error {k} of {id} has no item"));
                continue;
            };
            let strip = |error: &Value| {
                let mut error = error.clone();
                if let Some(object) = error.as_object_mut() {
                    object.remove("item");
                }
                error
            };
            // Events before this error in the regenerated stream.
            let events = after.saturating_sub(k as u64) as usize;
            let grown = inserted.iter().filter(|&&at| at < events).count() as u64;
            if strip(old) != strip(new) || after != before + grown {
                self.audit.other(
                    self.path,
                    format!("stream error {k} of {id} moved {before} -> {after}"),
                );
            } else if after != before {
                self.audit.count(Change::CountShift, 1);
            }
        }
    }
}

fn is_reasoning_start(event: &Value) -> bool {
    event.get("event").and_then(Value::as_str) == Some("block_start")
        && event.pointer("/kind/kind").and_then(Value::as_str) == Some("reasoning")
}

/// The world program run a path is inside (`/golden/<scope>[<run>]`).
fn run_of(at: &str) -> Option<&str> {
    let rest = at.strip_prefix("/golden/")?;
    let close = rest.find(']')?;
    Some(&at[..("/golden/".len() + close + 1)])
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
    let changed = git(root, &["diff", "--name-only", &base, "--", EFFECTS])?;
    let changed: Vec<&str> = changed.lines().collect();
    let cassettes = git(
        root,
        &[
            "diff",
            "--name-only",
            &base,
            "--",
            "crates/rig-cassette/fixtures/cassettes",
        ],
    )?;
    let mut audit = Audit::default();
    let listed = git(root, &["ls-files", "--", EFFECTS])?;
    for path in listed.lines().filter(|path| path.ends_with(".json")) {
        let text = std::fs::read_to_string(root.join(path)).map_err(|e| format!("{path}: {e}"))?;
        let head: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let base = if changed.contains(&path) {
            git(root, &["show", &format!("{base}:{path}")])
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        } else {
            None
        };
        audit.file(path, base.as_ref(), &head);
    }
    println!(
        "{} goldens, {} changed from {base}, {} blocks checked",
        audit.files, audit.changed, audit.blocks
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
