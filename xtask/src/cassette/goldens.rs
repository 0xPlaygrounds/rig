//! `cargo xtask cassette goldens`: regenerate effect goldens from replay and
//! keep their delivery batches from a base.
//!
//! Regenerating replays every selected target with
//! `RIG_REGENERATE_GOLDEN=1`. A world golden's `header.deliveries` records
//! racy delivery batches that replay excludes from comparison. A golden
//! whose only difference from the base (`HEAD` unless `--base` names another
//! ref) is that field is restored. A golden whose content changed keeps the
//! base's batches, their stream counts moved by the events the change
//! inserted, so the diff shows the change and not the race.

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

pub(crate) fn git(root: &Path, args: &[&str]) -> Result<String, String> {
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

/// What happened to the changed goldens under the effects tree.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Rebased {
    /// Restored to the base: their only change was `header.deliveries`.
    pub(crate) reverted: usize,
    /// Kept, with the base's delivery batches.
    pub(crate) rebased: usize,
    /// Kept as regenerated, with the reason the base's batches did not fit.
    pub(crate) kept: Vec<(String, String)>,
}

/// Restore every changed golden whose only change from `base` is
/// `header.deliveries`, and give every other changed golden the base's
/// delivery batches ([`rebase_deliveries`]).
pub(crate) fn revert_delivery_churn(root: &Path, base: &str) -> Result<Rebased, String> {
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
    let mut outcome = Rebased::default();
    for path in changed.lines().filter(|path| path.ends_with(".json")) {
        if git(root, &["cat-file", "-e", &format!("{base}:{path}")]).is_err() {
            // A golden the base does not have has no batches to keep.
            continue;
        }
        let before = git(root, &["show", &format!("{base}:{path}")])?;
        let after = std::fs::read_to_string(root.join(path)).unwrap_or_default();
        if delivery_only(&before, &after) {
            git(root, &["checkout", base, "--", path])?;
            outcome.reverted += 1;
            continue;
        }
        match rebase_deliveries(&before, &after) {
            Ok(Some(rebased)) => {
                std::fs::write(root.join(path), rebased)
                    .map_err(|error| format!("{path}: {error}"))?;
                outcome.rebased += 1;
            }
            Ok(None) => {}
            Err(reason) => outcome.kept.push((path.to_owned(), reason)),
        }
    }
    Ok(outcome)
}

/// `after` with `before`'s `header.deliveries` batches, or `None` when there
/// is nothing to rebase (no deliveries on either side, or already equal).
///
/// The change may only insert events into a stream: every event of `before`
/// appears in `after` in order, a `block_end` possibly gaining its `block`.
/// Each inserted event counts toward the batch that delivers the event after
/// it, or the effect's last stream batch when nothing follows it; batch
/// numbers and outcome deliveries stay as they were. Anything else is an
/// error naming what did not fit.
pub(crate) fn rebase_deliveries(before: &str, after: &str) -> Result<Option<String>, String> {
    let parse = |text: &str| serde_json::from_str::<Value>(text).map_err(|e| e.to_string());
    let (base, head) = (parse(before)?, parse(after)?);
    let Some(expected) = expected_deliveries(&base, &head)? else {
        return Ok(None);
    };
    if head.pointer("/header/deliveries") == Some(&expected) {
        return Ok(None);
    }
    let counts: Vec<u64> = expected
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|delivery| delivery.pointer("/kind/items").and_then(Value::as_u64))
        .collect();
    let rebased = with_deliveries(before, after, &counts)?;
    if parse(&rebased)?.pointer("/header/deliveries") != Some(&expected) {
        return Err("the rebased deliveries did not read back".into());
    }
    Ok(Some(rebased))
}

/// `base`'s `header.deliveries` with each stream count grown by the events
/// `head` inserted into that batch, or `None` when either side has no
/// deliveries. See [`rebase_deliveries`] for the rule and what does not fit.
pub(crate) fn expected_deliveries(base: &Value, head: &Value) -> Result<Option<Value>, String> {
    let deliveries = |log: &Value| log.pointer("/header/deliveries").cloned();
    let (Some(Value::Array(batches)), Some(_)) = (deliveries(base), deliveries(head)) else {
        return Ok(None);
    };
    let ids = |log: &Value| {
        log.get("records")
            .and_then(Value::as_array)
            .map(|records| {
                records
                    .iter()
                    .filter_map(|record| record.get("id").and_then(Value::as_u64))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    if ids(base) != ids(head) {
        return Err("the change adds or drops effects".into());
    }
    let mut counts = Vec::new();
    for delivery in &batches {
        if let Some(items) = delivery.pointer("/kind/items").and_then(Value::as_u64) {
            let id = delivery
                .get("id")
                .and_then(Value::as_u64)
                .ok_or("a stream delivery names no effect id")?;
            counts.push((id, items));
        }
    }
    let mut ids: Vec<u64> = counts.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut grown = counts.clone();
    for id in ids {
        let (placed, total) = inserted_per_batch(base, head, id)?;
        let sizes: Vec<u64> = counts
            .iter()
            .filter(|(of, _)| *of == id)
            .map(|(_, size)| *size)
            .collect();
        let mut extras = place(&sizes, &placed, total).into_iter();
        for (of, count) in grown.iter_mut() {
            if *of == id {
                *count += extras.next().unwrap_or_default();
            }
        }
    }
    let mut grown = grown.into_iter();
    Ok(Some(Value::Array(
        batches
            .into_iter()
            .map(|mut delivery| {
                if let Some(items) = delivery.pointer_mut("/kind/items")
                    && let Some((_, count)) = grown.next()
                {
                    *items = Value::from(count);
                }
                delivery
            })
            .collect(),
    )))
}

/// Where the events `head` inserted into effect `id`'s stream fall, as
/// positions in the base's item sequence: each inserted item is counted at
/// the base index of the item after it (the base's length when none
/// follows). Items interleave events and recorded stream errors.
fn inserted_per_batch(base: &Value, head: &Value, id: u64) -> Result<(Vec<usize>, usize), String> {
    let record = |log: &Value| {
        log.get("records")
            .and_then(Value::as_array)
            .and_then(|records| {
                records
                    .iter()
                    .find(|record| record.get("id").and_then(Value::as_u64) == Some(id))
            })
            .cloned()
    };
    let (Some(old), Some(new)) = (record(base), record(head)) else {
        return Err(format!("effect {id} is missing on one side"));
    };
    let events = |record: &Value| {
        record
            .get("events")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let (old_events, new_events) = (events(&old), events(&new));
    let errors = |log: &Value| {
        log.pointer(&format!("/header/stream_errors/{id}"))
            .and_then(Value::as_array)
            .map(|errors| {
                errors
                    .iter()
                    .filter_map(|error| error.get("item").and_then(Value::as_u64))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    let (old_errors, new_errors) = (errors(base), errors(head));
    if old_errors.len() != new_errors.len() {
        return Err(format!("effect {id}'s stream errors changed in number"));
    }
    // Each new item: `Some(base item)` for one the base had, `None` for one
    // the change inserted.
    let items = |events: usize, errors: &[u64]| -> Vec<bool> {
        let mut items = Vec::new();
        let (mut event, mut error) = (0, 0);
        while event < events || error < errors.len() {
            let at_error = errors
                .get(error)
                .is_some_and(|&at| at as usize == items.len());
            items.push(at_error);
            if at_error {
                error += 1;
            } else {
                event += 1;
            }
        }
        items
    };
    let old_items = items(old_events.len(), &old_errors);
    let new_items = items(new_events.len(), &new_errors);
    let mut mapped: Vec<Option<usize>> = Vec::with_capacity(new_items.len());
    let (mut old_event, mut new_event, mut old_item) = (0, 0, 0);
    for is_error in new_items {
        if is_error {
            // Errors keep their order, and every base event before one must
            // have been matched already.
            if old_items.get(old_item) != Some(&true) {
                return Err(format!(
                    "effect {id}: a base event is missing before an error"
                ));
            }
            mapped.push(Some(old_item));
            old_item += 1;
            continue;
        }
        let event = new_events.get(new_event);
        new_event += 1;
        let matches = old_items.get(old_item) == Some(&false)
            && old_events
                .get(old_event)
                .zip(event)
                .is_some_and(|(old, new)| same_event(old, new));
        if matches {
            mapped.push(Some(old_item));
            old_item += 1;
            old_event += 1;
        } else if event.is_some_and(is_close)
            || event.is_some_and(|event| starts_a_sibling(event, new_events.get(new_event)))
        {
            mapped.push(None);
        } else {
            return Err(format!(
                "effect {id}: event {} is not an inserted close",
                new_event - 1
            ));
        }
    }
    if old_item != old_items.len() {
        return Err(format!(
            "effect {id}: base items are missing from the change"
        ));
    }
    let mut placed = Vec::new();
    for (index, item) in mapped.iter().enumerate() {
        if item.is_none() {
            let next = mapped
                .get(index + 1..)
                .and_then(|rest| rest.iter().flatten().next().copied())
                .unwrap_or(old_items.len());
            placed.push(next);
        }
    }
    Ok((placed, old_items.len()))
}

/// Whether `event` is a reasoning start the sink inserted before the end of
/// a sibling part under the same key (`next`).
pub(crate) fn starts_a_sibling(event: &Value, next: Option<&Value>) -> bool {
    event.get("event").and_then(Value::as_str) == Some("block_start")
        && event.pointer("/kind/kind").and_then(Value::as_str) == Some("reasoning")
        && next.is_some_and(|end| is_close(end) && end.get("id") == event.get("id"))
}

/// How many inserted items each batch of sizes `sizes` takes, given each
/// inserted item's base position ([`inserted_per_batch`]) among the base's
/// `total` items. A position the batches never delivered (a cancelled
/// stream's undelivered tail) is not delivered; one past every base item
/// goes to the last batch when the batches delivered them all.
fn place(sizes: &[u64], placed: &[usize], total: usize) -> Vec<u64> {
    let mut extra = vec![0; sizes.len()];
    let delivered: u64 = sizes.iter().sum();
    for &position in placed {
        let mut start = 0u64;
        let mut target = None;
        for (index, size) in sizes.iter().enumerate() {
            if (position as u64) < start + size {
                target = Some(index);
                break;
            }
            start += size;
        }
        let slot = match target {
            Some(index) => extra.get_mut(index),
            None if position == total && delivered >= total as u64 => extra.last_mut(),
            None => None,
        };
        if let Some(slot) = slot {
            *slot += 1;
        }
    }
    extra
}

/// Whether a base event and a regenerated one are the same, a `block_end`
/// possibly having gained its `block`.
pub(crate) fn same_event(old: &Value, new: &Value) -> bool {
    if old == new {
        return true;
    }
    let strip = |event: &Value| {
        let mut event = event.clone();
        if let Some(object) = event.as_object_mut() {
            object.remove("block");
        }
        event
    };
    is_close(new) && old.get("block").is_none() && strip(old) == strip(new)
}

/// Whether an event is a text or reasoning `block_end`: what canonical
/// streams insert.
pub(crate) fn is_close(event: &Value) -> bool {
    event.get("event").and_then(Value::as_str) == Some("block_end")
        && matches!(
            event.pointer("/end/close").and_then(Value::as_str),
            Some("text" | "reasoning")
        )
}

/// `after`'s text with its `header.deliveries` array replaced by `before`'s,
/// the stream counts set to `counts` in order. The base's text is reused so
/// the formatting matches the golden writer's.
fn with_deliveries(before: &str, after: &str, counts: &[u64]) -> Result<String, String> {
    let base = array_span(before, "\"deliveries\":").ok_or("the base names no deliveries")?;
    let head = array_span(after, "\"deliveries\":").ok_or("the change names no deliveries")?;
    let mut text = String::new();
    let mut rest = &before[base.clone()];
    let mut counts = counts.iter();
    const ITEMS: &str = "\"items\": ";
    while let Some(at) = rest.find(ITEMS) {
        let digits = rest[at + ITEMS.len()..]
            .find(|c: char| !c.is_ascii_digit())
            .ok_or("an unterminated stream count")?;
        let count = counts.next().ok_or("more stream counts than deliveries")?;
        text.push_str(&rest[..at + ITEMS.len()]);
        text.push_str(&count.to_string());
        rest = &rest[at + ITEMS.len() + digits..];
    }
    text.push_str(rest);
    Ok(format!(
        "{}{}{}",
        &after[..head.start],
        text,
        &after[head.end..]
    ))
}

/// The byte range of the array that follows the first `key` in `text`,
/// brackets included.
fn array_span(text: &str, key: &str) -> Option<std::ops::Range<usize>> {
    let start = text.find(key)? + key.len();
    let open = start + text[start..].find('[')?;
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    for (offset, c) in text[open..].char_indices() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open..open + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
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
    let outcome = revert_delivery_churn(root, &base)?;
    println!(
        "against {base}: reverted {} delivery-only golden(s), kept the delivery batches of {}",
        outcome.reverted, outcome.rebased
    );
    for (path, reason) in &outcome.kept {
        println!("kept regenerated deliveries in {path}: {reason}");
    }
    if !status.success() {
        return Err("the regeneration run failed".into());
    }
    if !outcome.kept.is_empty() {
        return Err(format!(
            "{} golden(s) changed in a way the base's delivery batches do not fit",
            outcome.kept.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
