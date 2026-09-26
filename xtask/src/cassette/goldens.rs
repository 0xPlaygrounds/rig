//! `cargo xtask cassette goldens`: regenerate effect goldens from replay and
//! keep a base's delivery batches.
//!
//! Regenerating replays every selected target with
//! `RIG_REGENERATE_GOLDEN=1`. A world golden's `header.deliveries` records
//! racy delivery order that replay excludes from comparison, so a golden
//! whose only difference from the base (`HEAD` unless `--base` names another
//! ref) is that field is restored. A golden whose content changed keeps the
//! base's batches, with each batch's `items` grown by the stream events the
//! regeneration inserted into it. A golden that does not fit that rule keeps
//! its regenerated deliveries and is reported; `cargo xtask cassette audit`
//! fails on it.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// Whether `before` and `after` (golden JSON) differ only in
/// `header.deliveries`.
pub(crate) fn delivery_only(before: &str, after: &str) -> bool {
    let strip = |text: &str| -> Option<Value> {
        let mut value: Value = serde_json::from_str(text).ok()?;
        if let Some(header) = value.get_mut("header").and_then(Value::as_object_mut) {
            header.remove("deliveries");
        }
        Some(value)
    };
    before != after && strip(before).is_some_and(|before| Some(before) == strip(after))
}

/// Whether `head` restates the base stream event `base`: equal, or a
/// `block_end` that gained the block the base end left out.
pub(crate) fn restates(base: &Value, head: &Value) -> bool {
    if base == head {
        return true;
    }
    let is_end = |event: &Value| event.get("event").and_then(Value::as_str) == Some("block_end");
    if !is_end(base) || !is_end(head) || !base.get("block").is_none_or(Value::is_null) {
        return false;
    }
    let without_block = |event: &Value| {
        let mut event = event.clone();
        if let Some(event) = event.as_object_mut() {
            event.remove("block");
        }
        event
    };
    without_block(base) == without_block(head)
}

/// Where each head event sits against the base events: `Some(i)` restates
/// base event `i`, `None` is an inserted `block_end`. Any other difference is
/// an error.
pub(crate) fn align_events(base: &[Value], head: &[Value]) -> Result<Vec<Option<usize>>, String> {
    let mut aligned = Vec::with_capacity(head.len());
    let mut next = 0;
    for (position, event) in head.iter().enumerate() {
        if base.get(next).is_some_and(|base| restates(base, event)) {
            aligned.push(Some(next));
            next += 1;
        } else if event.get("event").and_then(Value::as_str) == Some("block_end") {
            aligned.push(None);
        } else {
            return Err(format!(
                "event {position} is neither a restated base event nor an inserted block_end"
            ));
        }
    }
    if next != base.len() {
        return Err(format!("base event {next} is missing from the head"));
    }
    Ok(aligned)
}

/// One effect's recorded stream: its events and the item positions of its
/// stream errors.
pub(crate) struct Stream<'a> {
    pub(crate) events: &'a [Value],
    pub(crate) errors: Vec<usize>,
}

impl<'a> Stream<'a> {
    /// The stream `record` keeps in `log`, `None` when it kept no events.
    pub(crate) fn of(log: &'a Value, record: &'a Value) -> Option<Self> {
        let events = record.get("events")?.as_array()?;
        let id = record.get("id")?.to_string();
        let errors = log
            .pointer("/header/stream_errors")
            .and_then(|errors| errors.get(&id))
            .and_then(Value::as_array)
            .map(|errors| {
                errors
                    .iter()
                    .filter_map(|error| error.get("item")?.as_u64())
                    .map(|item| item as usize)
                    .collect()
            })
            .unwrap_or_default();
        Some(Self { events, errors })
    }

    fn len(&self) -> usize {
        self.events.len() + self.errors.len()
    }
}

/// The base item index each event inserted into `head` counts toward: the
/// item after it, or the base's listed item count when nothing follows.
///
/// Every inserted event must be a `block_end` whose next non-inserted item is
/// a `final` event or a stream error, or which ends the stream.
pub(crate) fn inserted_items(base: &Stream, head: &Stream) -> Result<Vec<usize>, String> {
    if base.errors.len() != head.errors.len() {
        return Err("the stream error count changed".into());
    }
    let aligned = align_events(base.events, head.events)?;
    let mut base_items = Vec::with_capacity(base.events.len());
    let mut position = 0;
    while base_items.len() < base.events.len() {
        if !base.errors.contains(&position) {
            base_items.push(position);
        }
        position += 1;
    }
    // Head items as (base item index, whether an inserted event may precede
    // it), `None` for an inserted event.
    let mut items = Vec::with_capacity(head.len());
    let mut events = aligned.iter().zip(head.events);
    for position in 0..head.len() {
        if let Some(error) = head.errors.iter().position(|item| *item == position) {
            items.push(base.errors.get(error).map(|item| (*item, true)));
            continue;
        }
        let (slot, event) = events
            .next()
            .ok_or_else(|| format!("stream error position {position} is past the events"))?;
        items.push(match slot {
            Some(index) => {
                let item = base_items
                    .get(*index)
                    .ok_or_else(|| format!("base event {index} has no item"))?;
                Some((
                    *item,
                    event.get("event").and_then(Value::as_str) == Some("final"),
                ))
            }
            None => None,
        });
    }
    let mut inserted = Vec::new();
    for (position, item) in items.iter().enumerate() {
        if item.is_some() {
            continue;
        }
        let next = items.iter().skip(position + 1).find_map(|item| *item);
        match next {
            Some((item, true)) => inserted.push(item),
            Some((item, false)) => {
                return Err(format!(
                    "an event inserted at item {position} precedes base item {item}, which is neither a final nor a stream error"
                ));
            }
            None => inserted.push(base.len()),
        }
    }
    Ok(inserted)
}

fn records(log: &Value) -> BTreeMap<String, &Value> {
    log.get("records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|record| Some((record.get("id")?.to_string(), record)))
        .collect()
}

/// `base`'s `header.deliveries` with each stream batch's `items` grown by the
/// events `head` inserted into it. `None` when either golden has no
/// deliveries.
///
/// An inserted event counts toward the batch that delivers the item after it,
/// or the effect's last stream batch when nothing follows it. Batch numbers
/// and outcome deliveries are the base's.
pub(crate) fn rebase_deliveries(base: &Value, head: &Value) -> Result<Option<Value>, String> {
    let (Some(deliveries), Some(_)) = (
        base.pointer("/header/deliveries").and_then(Value::as_array),
        head.pointer("/header/deliveries"),
    ) else {
        return Ok(None);
    };
    let (base_records, head_records) = (records(base), records(head));
    if base_records.keys().ne(head_records.keys()) {
        return Err("the record ids changed".into());
    }
    let mut inserted = BTreeMap::new();
    for (id, base_record) in &base_records {
        let head_record = head_records
            .get(id)
            .ok_or_else(|| format!("record {id} is missing from the head"))?;
        match (Stream::of(base, base_record), Stream::of(head, head_record)) {
            (Some(base), Some(head)) => {
                let items = inserted_items(&base, &head)
                    .map_err(|error| format!("record {id}: {error}"))?;
                if !items.is_empty() {
                    inserted.insert(id.clone(), items);
                }
            }
            (None, None) => {}
            _ => return Err(format!("record {id} gained or lost its events")),
        }
    }
    let mut rebased = deliveries.clone();
    let mut delivered = BTreeMap::<String, usize>::new();
    let mut last = BTreeMap::new();
    for (index, delivery) in rebased.iter_mut().enumerate() {
        let id = delivery.get("id").map(Value::to_string).unwrap_or_default();
        let Some(count) = delivery.pointer_mut("/kind/items") else {
            continue;
        };
        let items = count
            .as_u64()
            .ok_or("a stream delivery's items is not a count")? as usize;
        let start = delivered.entry(id.clone()).or_default();
        let range = *start..*start + items;
        *start += items;
        let added = inserted.get(&id).map_or(0, |inserted| {
            inserted.iter().filter(|item| range.contains(item)).count()
        });
        *count = Value::from(items + added);
        last.insert(id, index);
    }
    for (id, items) in &inserted {
        let total = delivered.get(id).copied().unwrap_or_default();
        let listed = base_records
            .get(id)
            .and_then(|record| Stream::of(base, record))
            .map_or(0, |stream| stream.len());
        let trailing = items
            .iter()
            .filter(|item| **item >= total)
            .collect::<Vec<_>>();
        if trailing.is_empty() {
            continue;
        }
        // Nothing may lie between the delivered items and an event counted
        // toward the last batch, or that batch would deliver it instead.
        if total < listed {
            return Err(format!(
                "record {id}: an inserted event follows an item no stream batch delivered"
            ));
        }
        let batch = last
            .get(id)
            .and_then(|index| rebased.get_mut(*index))
            .and_then(|delivery| delivery.pointer_mut("/kind/items"))
            .ok_or_else(|| format!("record {id}: inserted events but no stream batch"))?;
        let items = batch.as_u64().unwrap_or_default() as usize;
        *batch = Value::from(items + trailing.len());
    }
    Ok(Some(Value::Array(rebased)))
}

/// `text` with its `header.deliveries` array replaced by `deliveries`,
/// pretty-printed at the array's indentation. Everything else keeps its
/// bytes.
pub(crate) fn splice_deliveries(text: &str, deliveries: &Value) -> Result<String, String> {
    const KEY: &str = "\"deliveries\": ";
    let key = text.find(KEY).ok_or("no deliveries key")?;
    let start = key + KEY.len();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;
    for (offset, ch) in text.get(start..).unwrap_or_default().char_indices() {
        match (in_string, escaped, ch) {
            (true, true, _) => escaped = false,
            (true, false, '\\') => escaped = true,
            (true, false, '"') => in_string = false,
            (true, false, _) => {}
            (false, _, '"') => in_string = true,
            (false, _, '[' | '{') => depth += 1,
            (false, _, ']' | '}') => {
                depth = depth.checked_sub(1).ok_or("unbalanced deliveries")?;
                if depth == 0 {
                    end = Some(start + offset + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or("unterminated deliveries")?;
    let line = text.get(..key).unwrap_or_default();
    let indent = line
        .get(line.rfind('\n').map_or(0, |newline| newline + 1)..)
        .unwrap_or_default();
    let pretty = serde_json::to_string_pretty(deliveries).map_err(|error| error.to_string())?;
    let pretty = pretty.replace('\n', &format!("\n{indent}"));
    Ok(format!(
        "{}{pretty}{}",
        text.get(..start).unwrap_or_default(),
        text.get(end..).unwrap_or_default()
    ))
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The golden JSON files under the effects tree that differ from `base`,
/// with their text at `base` (`None` when `base` lacks the file).
pub(crate) fn changed_goldens(
    root: &Path,
    base: &str,
) -> Result<Vec<(String, Option<String>)>, String> {
    let changed = git(
        root,
        &[
            "diff",
            "--name-only",
            base,
            "--",
            "crates/rig-cassette/fixtures/effects",
        ],
    )?;
    changed
        .lines()
        .filter(|path| path.ends_with(".json"))
        .map(|path| {
            let spec = format!("{base}:{path}");
            let before = git(root, &["cat-file", "-e", &spec])
                .ok()
                .map(|_| git(root, &["show", &spec]))
                .transpose()?;
            Ok((path.to_owned(), before))
        })
        .collect()
}

/// What [`rebase_goldens`] did.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Rebased {
    /// Goldens restored because only their deliveries changed.
    pub(crate) reverted: usize,
    /// Changed goldens whose deliveries were rewritten to the base's batches.
    pub(crate) rebased: usize,
    /// Changed goldens kept as regenerated, rebased or not.
    pub(crate) kept: usize,
    /// Changed goldens that keep their regenerated deliveries because their
    /// change does not fit the rebase rule, with the reason.
    pub(crate) misfits: Vec<String>,
}

/// Restore every golden whose only change from `base` is
/// `header.deliveries`, and give every other changed golden `base`'s
/// delivery batches (see [`rebase_deliveries`]). A golden whose change does
/// not fit the rule keeps its regenerated deliveries and is named in
/// [`Rebased::misfits`].
pub(crate) fn rebase_goldens(root: &Path, base: &str) -> Result<Rebased, String> {
    let mut summary = Rebased::default();
    for (path, before) in changed_goldens(root, base)? {
        let Some(before) = before else {
            summary.kept += 1;
            continue;
        };
        let Ok(after) = std::fs::read_to_string(root.join(&path)) else {
            continue;
        };
        if delivery_only(&before, &after) {
            std::fs::write(root.join(&path), &before)
                .map_err(|error| format!("{path}: {error}"))?;
            summary.reverted += 1;
            continue;
        }
        summary.kept += 1;
        let (Ok(base_log), Ok(head_log)) = (
            serde_json::from_str::<Value>(&before),
            serde_json::from_str::<Value>(&after),
        ) else {
            continue;
        };
        let deliveries = match rebase_deliveries(&base_log, &head_log) {
            Ok(Some(deliveries)) => deliveries,
            Ok(None) => continue,
            Err(error) => {
                summary.misfits.push(format!("{path}: {error}"));
                continue;
            }
        };
        if head_log.pointer("/header/deliveries") == Some(&deliveries) {
            continue;
        }
        let rebased =
            splice_deliveries(&after, &deliveries).map_err(|error| format!("{path}: {error}"))?;
        std::fs::write(root.join(&path), rebased).map_err(|error| format!("{path}: {error}"))?;
        summary.rebased += 1;
    }
    Ok(summary)
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let mut targets = Vec::new();
    let mut base = "HEAD".to_owned();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--test" => targets.push(args.next().cloned().ok_or("--test needs a target")?),
            "--base" => base = args.next().cloned().ok_or("--base needs a ref")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let mut command = Command::new("cargo");
    command.args([
        "nextest",
        "run",
        "--locked",
        "-p",
        "rig-cassette",
        "--features",
        "http,agent,ecs,bedrock",
        "--retries",
        "0",
        "--no-fail-fast",
    ]);
    for target in &targets {
        command.args(["--test", target]);
    }
    let status = command
        .current_dir(root)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env("RIG_REGENERATE_GOLDEN", "1")
        .status()
        .map_err(|error| format!("cargo nextest: {error}"))?;
    let summary = rebase_goldens(root, &base)?;
    for misfit in &summary.misfits {
        println!("kept the regenerated deliveries of {misfit}");
    }
    println!(
        "against {base}: reverted {} delivery-only golden(s), kept {} (rebased the deliveries of {}, {} misfit(s))",
        summary.reverted,
        summary.kept,
        summary.rebased,
        summary.misfits.len()
    );
    if status.success() {
        Ok(())
    } else {
        Err("the regeneration run failed".into())
    }
}

#[cfg(test)]
mod tests;
