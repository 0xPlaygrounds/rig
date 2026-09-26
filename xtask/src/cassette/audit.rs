//! `cargo xtask cassette audit`: check what every effect golden's stream
//! ends carry, and classify each golden change against a base ref.
//!
//! Every `block_end` must carry what its block's deltas assemble, unless the
//! end restates the block itself. A text block is the concatenation of its
//! text deltas, a reasoning block the concatenation of its reasoning deltas,
//! and a tool call the parse of its argument fragments. Each
//! golden that differs from the base (`HEAD` unless `--base` names another
//! ref) is classified change by change: a close inserted, a block added to an
//! end, a count shifted by exactly the events inserted before it, delivery
//! churn, or other. The audit fails on other, on delivery churn and on any
//! mismatch, so a golden's delivery batches must be the base's with each
//! batch's `items` grown by the events inserted into it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use serde_json::Value;

use super::goldens::{Stream, align_events, changed_goldens, inserted_items, rebase_deliveries};

/// How the changed goldens differ from the base.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Changes {
    /// Goldens that differ from the base.
    pub(crate) files: usize,
    /// `block_end` events inserted before a `final`, a stream error or the
    /// end of a stream.
    pub(crate) closes_inserted: usize,
    /// Base ends that now carry the block they finalized.
    pub(crate) blocks_added: usize,
    /// Stream error positions, delivery batch sizes and validated offsets
    /// grown by exactly the events inserted before them.
    pub(crate) count_shifts: usize,
    /// Goldens whose delivery batches are not the base's, grown by the
    /// inserted events.
    pub(crate) delivery_churn: usize,
    /// Every other change, by path.
    pub(crate) other: Vec<String>,
}

fn is_stream(value: &Value) -> bool {
    value.as_array().is_some_and(|events| {
        !events.is_empty() && events.iter().all(|event| event.get("event").is_some())
    })
}

fn kind(event: &Value) -> Option<&str> {
    event.get("event").and_then(Value::as_str)
}

struct Walk<'a> {
    changes: &'a mut Changes,
    /// Head positions of the events inserted into each stream, by path.
    inserted: BTreeMap<String, Vec<usize>>,
    /// Changed `stream_validated` offsets: path, base, head.
    validated: Vec<(String, u64, u64)>,
}

impl Walk<'_> {
    fn other(&mut self, path: &str, what: impl std::fmt::Display) {
        self.changes.other.push(format!("{path}: {what}"));
    }

    fn walk(&mut self, base: &Value, head: &Value, path: &str) {
        if base == head {
            return;
        }
        if (is_stream(base) || is_stream(head))
            && let (Some(base), Some(head)) = (base.as_array(), head.as_array())
        {
            self.stream(base, head, path);
            return;
        }
        match (base, head) {
            (Value::Object(base), Value::Object(head)) => {
                if base.keys().ne(head.keys()) {
                    self.other(path, "the keys changed");
                    return;
                }
                for (key, value) in base {
                    if let Some(changed) = head.get(key) {
                        // Paths are JSON pointers, and program keys hold `/`.
                        let key = key.replace('~', "~0").replace('/', "~1");
                        self.walk(value, changed, &format!("{path}/{key}"));
                    }
                }
            }
            (Value::Array(base), Value::Array(head)) if base.len() == head.len() => {
                for (index, (base, head)) in base.iter().zip(head).enumerate() {
                    self.walk(base, head, &format!("{path}/{index}"));
                }
            }
            (Value::Number(before), Value::Number(after))
                if path.ends_with("/stream_validated") =>
            {
                match (before.as_u64(), after.as_u64()) {
                    (Some(before), Some(after)) => {
                        self.validated.push((path.to_owned(), before, after));
                    }
                    _ => self.other(path, "a validated offset is not a count"),
                }
            }
            (Value::Array(_) | Value::Object(_), _) | (_, Value::Array(_) | Value::Object(_)) => {
                self.other(path, "the value changed shape");
            }
            _ => self.other(path, format!("{base} became {head}")),
        }
    }

    fn stream(&mut self, base: &[Value], head: &[Value], path: &str) {
        let aligned = match align_events(base, head) {
            Ok(aligned) => aligned,
            Err(error) => return self.other(path, error),
        };
        let mut inserted = Vec::new();
        for (position, (slot, event)) in aligned.iter().zip(head).enumerate() {
            match slot {
                None => {
                    let next = aligned
                        .iter()
                        .zip(head)
                        .skip(position + 1)
                        .find(|(slot, _)| slot.is_some())
                        .map(|(_, event)| kind(event));
                    if matches!(next, None | Some(Some("final"))) {
                        self.changes.closes_inserted += 1;
                        inserted.push(position);
                    } else {
                        self.other(
                            &format!("{path}/{position}"),
                            "a close inserted before an event other than a final",
                        );
                    }
                }
                Some(index) if base.get(*index) != Some(event) => {
                    self.changes.blocks_added += 1;
                }
                Some(_) => {}
            }
        }
        self.inserted.insert(path.to_owned(), inserted);
    }

    /// A program's validated offset (`…/entities/{turn}/…/stream_validated`)
    /// may grow by the events inserted before it in the stream of one of
    /// the turn's child entities (`bevy_ecs::hierarchy::ChildOf`).
    fn check_validated(&mut self, head: &Value) {
        for (path, before, after) in std::mem::take(&mut self.validated) {
            let grown = after.checked_sub(before).map(|grown| grown as usize);
            let explained = path
                .split_once("/entities/")
                .is_some_and(|(program, rest)| {
                    let turn = rest
                        .split('/')
                        .next()
                        .and_then(|turn| turn.parse::<u64>().ok());
                    let entities = head
                        .pointer(&format!("{program}/entities"))
                        .and_then(Value::as_array);
                    entities
                        .into_iter()
                        .flatten()
                        .enumerate()
                        .filter(|(_, entity)| {
                            entity
                                .get("bevy_ecs::hierarchy::ChildOf")
                                .and_then(Value::as_u64)
                                == turn
                        })
                        .any(|(child, _)| {
                            let prefix = format!("{program}/entities/{child}/");
                            self.inserted
                                .iter()
                                .filter(|(stream, _)| stream.starts_with(&prefix))
                                .any(|(_, inserted)| {
                                    let before_offset = inserted
                                        .iter()
                                        .filter(|position| **position < after as usize)
                                        .count();
                                    Some(before_offset) == grown
                                })
                        })
                });
            if explained {
                self.changes.count_shifts += 1;
            } else {
                self.other(&path, format!("{before} became {after}"));
            }
        }
    }
}

/// Classify the header's delivery batches and stream error positions, then
/// set them to the base's in `head`, so the structural walk compares the
/// rest. Returns why the rebase could not place the streams' changes, if
/// it could not.
fn classify_header(base: &Value, head: &mut Value, changes: &mut Changes) -> Option<String> {
    let base_deliveries = base.pointer("/header/deliveries").cloned();
    let head_deliveries = head.pointer("/header/deliveries");
    let rebased = rebase_deliveries(base, head);
    let refused = rebased.as_ref().err().cloned();
    match rebased {
        Ok(Some(expected)) if head_deliveries == Some(&expected) => {
            changes.count_shifts += expected
                .as_array()
                .into_iter()
                .flatten()
                .zip(base_deliveries.iter().filter_map(Value::as_array).flatten())
                .filter(|(expected, base)| expected != base)
                .count();
        }
        Ok(Some(_)) => changes.delivery_churn += 1,
        _ if head_deliveries == base_deliveries.as_ref() => {}
        _ => changes.delivery_churn += 1,
    }
    if let (Some(header), Some(deliveries)) = (
        head.get_mut("header").and_then(Value::as_object_mut),
        base_deliveries,
    ) {
        header.insert("deliveries".to_owned(), deliveries);
    }

    let records = |log: &Value| -> BTreeMap<String, Value> {
        log.get("records")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|record| Some((record.get("id")?.to_string(), record.clone())))
            .collect()
    };
    let (base_records, head_records) = (records(base), records(head));
    let ids = head
        .pointer("/header/stream_errors")
        .and_then(Value::as_object)
        .map(|errors| errors.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    for id in ids {
        let path = format!("/header/stream_errors/{id}");
        let (Some(base_record), Some(head_record)) = (base_records.get(&id), head_records.get(&id))
        else {
            continue;
        };
        let (Some(base_stream), Some(head_stream)) =
            (Stream::of(base, base_record), Stream::of(head, head_record))
        else {
            continue;
        };
        if base_stream.errors == head_stream.errors {
            continue;
        }
        match inserted_items(&base_stream, &head_stream) {
            Ok(inserted) => {
                for (before, after) in base_stream.errors.iter().zip(&head_stream.errors) {
                    let expected = before + inserted.iter().filter(|next| *next <= before).count();
                    if *after == expected {
                        changes.count_shifts += usize::from(after != before);
                    } else {
                        changes.other.push(format!(
                            "{path}: item {before} became {after}, not {expected}"
                        ));
                    }
                }
            }
            Err(error) => changes.other.push(format!("{path}: {error}")),
        }
        if let Some(errors) = head.pointer_mut(&path).and_then(Value::as_array_mut) {
            for (error, item) in errors.iter_mut().zip(&base_stream.errors) {
                if let Some(error) = error.as_object_mut() {
                    error.insert("item".to_owned(), Value::from(*item));
                }
            }
        }
    }
    refused
}

/// Add how `head` differs from `base` (one golden, as JSON) to `changes`.
pub(crate) fn classify(base: &Value, head: &Value, changes: &mut Changes) {
    changes.files += 1;
    let mut head = head.clone();
    let others = changes.other.len();
    let refused = base
        .get("header")
        .and_then(|_| classify_header(base, &mut head, changes));
    let mut walk = Walk {
        changes,
        inserted: BTreeMap::new(),
        validated: Vec::new(),
    };
    walk.walk(base, &head, "");
    walk.check_validated(&head);
    // A stream change the rebase cannot place is usually reported already;
    // a placement it refuses on its own is reported here.
    if let Some(refused) = refused
        && changes.other.len() == others
    {
        changes.other.push(format!("/header/deliveries: {refused}"));
    }
}

fn text(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or_default()
}

/// The reasoning text a completed reasoning block carries.
fn reasoning_text(block: &Value) -> String {
    block
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|content| text(content.get("type")) == "text")
        .map(|content| text(content.pointer("/content/text")))
        .collect()
}

/// Every `block_end` block in `events` that differs from what its block's
/// deltas assemble, as `position: what`.
pub(crate) fn block_mismatches(events: &[Value]) -> Vec<String> {
    let mut texts = HashMap::<&str, String>::new();
    let mut reasoning = HashMap::<&str, String>::new();
    let mut open_reasoning = HashSet::<&str>::new();
    let mut finished_reasoning = HashMap::<&str, String>::new();
    let mut arguments = HashMap::<&str, String>::new();
    let mut mismatches = Vec::new();
    for (position, event) in events.iter().enumerate() {
        let id = text(event.get("id"));
        match (kind(event), text(event.pointer("/delta/delta"))) {
            (Some("block_start"), _)
                if text(event.pointer("/kind/kind")) == "reasoning"
                    && open_reasoning.insert(id) =>
            {
                reasoning.insert(id, String::new());
            }
            (Some("block_delta"), "text") => {
                texts
                    .entry(id)
                    .or_default()
                    .push_str(text(event.pointer("/delta/text")));
            }
            (Some("block_delta"), "reasoning") => {
                if open_reasoning.insert(id) {
                    reasoning.insert(id, String::new());
                }
                reasoning
                    .entry(id)
                    .or_default()
                    .push_str(text(event.pointer("/delta/text")));
            }
            (Some("block_delta"), "tool_arguments") => {
                arguments
                    .entry(id)
                    .or_default()
                    .push_str(text(event.pointer("/delta/arguments")));
            }
            (Some("block_end"), _) => {
                let block = event.get("block").filter(|block| !block.is_null());
                let mismatch = match (text(event.pointer("/end/close")), block) {
                    ("text", block) => {
                        let expected = texts.get(id).map_or("", String::as_str);
                        match block {
                            Some(block) => (text(block.get("text")) != expected)
                                .then(|| format!("text block is not its deltas {expected:?}")),
                            None => (!expected.is_empty())
                                .then(|| format!("text end carries no block for {expected:?}")),
                        }
                    }
                    ("reasoning", block) => {
                        let closes_open = open_reasoning.remove(id);
                        let expected = if closes_open {
                            let assembled = reasoning.remove(id).unwrap_or_default();
                            finished_reasoning.insert(id, assembled.clone());
                            assembled
                        } else {
                            finished_reasoning.get(id).cloned().unwrap_or_default()
                        };
                        let restated = event
                            .pointer("/end/reasoning")
                            .is_some_and(|reasoning| !reasoning.is_null());
                        match block {
                            Some(block) => (!restated && reasoning_text(block) != expected)
                                .then(|| format!("reasoning block is not its deltas {expected:?}")),
                            None => {
                                closes_open.then(|| "reasoning end carries no block".to_owned())
                            }
                        }
                    }
                    ("tool_call", block) => {
                        let fragments = arguments.remove(id).unwrap_or_default();
                        let restated = event.pointer("/end/arguments").is_some();
                        match block.filter(|_| !restated && !fragments.trim().is_empty()) {
                            Some(block) => {
                                let parsed = serde_json::from_str::<Value>(&fragments).ok();
                                (parsed.as_ref() != block.pointer("/function/arguments")).then(
                                    || {
                                        format!(
                                            "tool call is not its argument fragments {fragments:?}"
                                        )
                                    },
                                )
                            }
                            None => None,
                        }
                    }
                    _ => None,
                };
                mismatches.extend(mismatch.map(|what| format!("{position}: {what}")));
            }
            _ => {}
        }
    }
    mismatches
}

/// Every stream in a golden, by JSON path.
fn streams<'a>(value: &'a Value, path: String, found: &mut Vec<(String, &'a [Value])>) {
    match value {
        Value::Array(items) if is_stream(value) => found.push((path, items)),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                streams(item, format!("{path}/{index}"), found);
            }
        }
        Value::Object(fields) => {
            for (key, field) in fields {
                streams(field, format!("{path}/{key}"), found);
            }
        }
        _ => {}
    }
}

/// Every block/delta mismatch in `golden`, as `path/position: what`.
pub(crate) fn golden_mismatches(golden: &Value) -> Vec<String> {
    let mut found = Vec::new();
    streams(golden, String::new(), &mut found);
    found
        .into_iter()
        .flat_map(|(path, events)| {
            block_mismatches(events)
                .into_iter()
                .map(move |mismatch| format!("{path}/{mismatch}"))
        })
        .collect()
}

fn goldens(dir: &Path, found: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            goldens(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            found.push(path);
        }
    }
    Ok(())
}

/// The number of goldens under the effects tree, and every block/delta
/// mismatch among them as `file/path/position: what`.
pub(crate) fn corpus_mismatches(root: &Path) -> Result<(usize, Vec<String>), String> {
    let mut files = Vec::new();
    goldens(
        &root.join("crates/rig-cassette/fixtures/effects"),
        &mut files,
    )?;
    files.sort();
    let mut mismatches = Vec::new();
    for file in &files {
        let label = file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string();
        let text = std::fs::read_to_string(file).map_err(|error| format!("{label}: {error}"))?;
        let golden =
            serde_json::from_str::<Value>(&text).map_err(|error| format!("{label}: {error}"))?;
        for mismatch in golden_mismatches(&golden) {
            mismatches.push(format!("{label}{mismatch}"));
        }
    }
    Ok((files.len(), mismatches))
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
    let parse = |path: &str, text: &str| {
        serde_json::from_str::<Value>(text).map_err(|error| format!("{path}: {error}"))
    };

    let (audited, mismatches) = corpus_mismatches(root)?;

    let mut changes = Changes::default();
    let mut churned = Vec::new();
    for (path, before) in changed_goldens(root, &base)? {
        let (Some(before), Ok(after)) = (before, std::fs::read_to_string(root.join(&path))) else {
            changes.files += 1;
            changes.other.push(format!("{path}: added or removed"));
            continue;
        };
        let (others, churn) = (changes.other.len(), changes.delivery_churn);
        classify(
            &parse(&path, &before)?,
            &parse(&path, &after)?,
            &mut changes,
        );
        for other in changes.other.iter_mut().skip(others) {
            *other = format!("{path}{other}");
        }
        if changes.delivery_churn > churn {
            churned.push(format!("{path}: delivery churn"));
        }
    }

    for line in changes.other.iter().chain(&churned).chain(&mismatches) {
        println!("{line}");
    }
    println!(
        "audited {} golden(s); {} differ from {base}: {} close(s) inserted, {} block(s) added to an end, {} count shift(s), {} delivery churn, {} other; {} block/delta mismatch(es)",
        audited,
        changes.files,
        changes.closes_inserted,
        changes.blocks_added,
        changes.count_shifts,
        changes.delivery_churn,
        changes.other.len(),
        mismatches.len()
    );
    if changes.other.is_empty() && changes.delivery_churn == 0 && mismatches.is_empty() {
        Ok(())
    } else {
        Err("the golden audit failed".into())
    }
}

#[cfg(test)]
mod tests;
