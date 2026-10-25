//! Plugin corpus: 21. Tool output truncation with spill files (spec
//! `21-output-truncate.md`).
//!
//! A tool result longer than [`Truncate::max_lines`] or
//! [`Truncate::max_bytes`] is cut to its first and last lines around a
//! marker naming a spill file, which holds the whole text. A Review system
//! cuts the call's [`Outcome`] before the turn commits it, so the model,
//! the conversation and the transcript all see the same cut result, and an
//! error stays an error. Another result post-processor orders itself
//! `.before(TruncateResults)`; a result it shrank under the limits is left
//! alone.

use std::borrow::Cow;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use bevy_ecs::schedule::SystemSet;
use bevy_log::warn;
use rig_basics::coding_tools::ShellOutput;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::ToolOutput as Output;
use rig_harness::rig_core::{self, effect};
use rig_tools::{Read, Shell};
use serde::Deserialize;

/// The `[truncate]` settings table: what is cut, how, and where the whole
/// text goes.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Truncate {
    /// A result with more lines than this is cut.
    pub max_lines: usize,
    /// A result with more bytes than this is cut.
    pub max_bytes: usize,
    /// The lines kept from the start.
    pub head_lines: usize,
    /// The lines kept from the end.
    pub tail_lines: usize,
    /// Only these tools' results are cut; empty: every tool's.
    pub only: Vec<String>,
    /// These tools' results are never cut.
    pub except: Vec<String>,
    /// Where the whole results go, one `<call-id>.txt` each, in the
    /// project unless absolute.
    pub spill_dir: PathBuf,
    /// Spill files older than this many days are deleted at startup.
    pub max_age_days: u64,
}

impl Default for Truncate {
    fn default() -> Self {
        Self {
            max_lines: 400,
            max_bytes: 40 * 1024,
            head_lines: 150,
            tail_lines: 100,
            only: Vec::new(),
            except: vec![Read::NAME.to_owned()],
            spill_dir: PathBuf::from(".rig/spill"),
            max_age_days: 7,
        }
    }
}

impl Truncate {
    /// Whether `tool`'s results are cut.
    pub fn covers(&self, tool: &str) -> bool {
        (self.only.is_empty() || self.only.iter().any(|only| only == tool))
            && !self.except.iter().any(|skip| skip == tool)
    }
}

/// The Review system that cuts results: another post-processor orders
/// itself before or after it.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TruncateResults;

/// Cuts long tool results and keeps them whole in spill files.
#[derive(Default)]
pub struct OutputTruncatePlugin;

impl Plugin for OutputTruncatePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Truncate>("truncate")
            .init_resource::<ProjectRoot>()
            .configure_sets(AgentPass, TruncateResults.in_set(Stage::Review))
            .add_systems(Startup, sweep_old_spills)
            .add_systems(AgentPass, cut_long_results.in_set(TruncateResults));
    }

    /// The default `shell` keeps only its output's last lines; this
    /// plugin keeps the first ones too, so it asks for all of it.
    fn finish(&self, app: &mut App) {
        let world = app.world();
        let covered = world.get_resource::<Truncate>();
        if covered.is_some_and(|config| config.covers(Shell::NAME))
            && let Some(output) = world.get_resource::<ShellOutput>()
        {
            output.set(None);
        }
    }
}

/// On a call whose result was cut, so it is not cut again: the cut may
/// still be over the byte limit by its marker.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Truncated;

/// Replaces each long result of a covered tool by its cut.
fn cut_long_results(
    mut calls: Query<
        (Entity, &ToolRequest, &mut Outcome),
        (ReadyToSettle, ToolCalls, Without<Truncated>),
    >,
    config: Res<Truncate>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    for (call, ToolRequest { call: request }, mut outcome) in &mut calls {
        let Ok(effect::Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        if !config.covers(request.function.name.as_str()) {
            continue;
        }
        if let Some(cut) = shorten(result, &request.id.wire(), &config, &root.0) {
            *outcome = Outcome::from(cut);
            commands.entity(call).insert(Truncated);
        }
    }
}

/// `result` cut around a marker, with its whole text written to the spill
/// file of the call `id` under `root` first (a JSON result is cut as its
/// pretty-printed text); `None` when it is short enough, is neither text
/// nor JSON (images stay whole), or could not be spilled (then it goes
/// whole, never lost).
pub fn shorten(
    result: &ToolResult,
    id: &str,
    config: &Truncate,
    root: &Path,
) -> Option<ToolResult> {
    let output = result.output();
    let text = match output.as_text() {
        Some(text) => Cow::Borrowed(text),
        None => Cow::Owned(rig_core::serde_json::to_string_pretty(output.as_json()?).ok()?),
    };
    let text = text.as_ref();
    if text.lines().count() <= config.max_lines && text.len() <= config.max_bytes {
        return None;
    }
    let shown = config.spill_dir.join(format!("{}.txt", file_name(id)));
    let path = root.join(&shown);
    if let Err(error) = spill(&path, text) {
        warn!(
            "could not keep a long tool result in {}: {error}",
            path.display()
        );
        return None;
    }
    let cut = cut(text, config, &shown);
    Some(result.clone().with_output(Output::text(cut)))
}

/// Writes `text` whole to `path`.
fn spill(path: &Path, text: &str) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, text)
}

/// `text`'s first and last lines, within the byte budget, around a marker
/// saying what was left out and that `path` has it all.
fn cut(text: &str, config: &Truncate, path: &Path) -> String {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let kept = config.head_lines + config.tail_lines;
    let head_budget = config.max_bytes * config.head_lines / kept.max(1);
    let head = keep(lines.iter().copied(), config.head_lines, head_budget, false);
    let rest = lines.get(head.len()..).unwrap_or_default();
    let tail_budget = config.max_bytes.saturating_sub(head_budget);
    let mut tail = keep(
        rest.iter().rev().copied(),
        config.tail_lines,
        tail_budget,
        true,
    );
    tail.reverse();
    let omitted_lines = lines.len() - head.len() - tail.len();
    let kept_bytes: usize = head.iter().chain(&tail).map(|line| line.len()).sum();
    let omitted = if omitted_lines > 0 {
        format!("{} lines", grouped(omitted_lines))
    } else {
        format!("{} bytes", grouped(text.len().saturating_sub(kept_bytes)))
    };
    let mut out = head.concat();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!(
        "[… {omitted} omitted; full output in `{}` …]\n",
        path.display()
    ));
    out.push_str(&tail.concat());
    out
}

/// Up to `count` of `lines` within `budget` bytes, the last one shortened
/// to fit: its start, or its end when `from_end`.
fn keep<'a>(
    lines: impl Iterator<Item = &'a str>,
    count: usize,
    mut budget: usize,
    from_end: bool,
) -> Vec<&'a str> {
    let mut kept = Vec::new();
    for line in lines.take(count) {
        if line.len() <= budget {
            budget -= line.len();
            kept.push(line);
            continue;
        }
        let part = if from_end {
            line.get(line.ceil_char_boundary(line.len() - budget)..)
        } else {
            line.get(..line.floor_char_boundary(budget))
        };
        if let Some(part) = part.filter(|part| !part.is_empty()) {
            kept.push(part);
        }
        break;
    }
    kept
}

/// `n` with a comma between each group of three digits, as `3,214`.
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A call id as a file name: anything but letters, digits, `-` and `_`
/// becomes `_`.
fn file_name(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Deletes the spill files older than [`Truncate::max_age_days`].
fn sweep_old_spills(config: Res<Truncate>, root: Res<ProjectRoot>) {
    let Ok(entries) = fs::read_dir(root.0.join(&config.spill_dir)) else {
        return;
    };
    let max_age = Duration::from_secs(config.max_age_days * 24 * 60 * 60);
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let old = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > max_age);
        if old && entry.path().is_file() {
            fs::remove_file(entry.path()).ok();
        }
    }
}

#[cfg(test)]
mod tests;
