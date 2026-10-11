//! Plugin corpus: 21. Tool output truncation with spill files (spec
//! `21-output-truncate.md`).
//!
//! A tool result longer than [`TruncateConfig::max_lines`] or
//! [`TruncateConfig::max_bytes`] is cut to its first and last lines around
//! a marker naming a spill file, which holds the whole text. A Review
//! system cuts the call's [`Outcome`] before the turn commits it, so the
//! model, the conversation and the transcript all see the same cut
//! result, and an error stays an error.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use bevy_log::warn;
use rig_core::effect;
use rig_core::tool::ToolOutput as Output;
use rig_ecs::prelude::*;

/// What is cut, how, and where the whole text goes.
#[derive(Resource, Clone, Debug)]
pub struct TruncateConfig {
    /// A result with more lines than this is cut.
    pub max_lines: usize,
    /// A result with more bytes than this is cut.
    pub max_bytes: usize,
    /// The lines kept from the start.
    pub head_lines: usize,
    /// The lines kept from the end.
    pub tail_lines: usize,
    /// Whose results are cut.
    pub tools: Tools,
    /// Where the whole results go, one `<call-id>.txt` each.
    pub spill_dir: PathBuf,
    /// Spill files older than this are deleted at startup.
    pub max_age: Duration,
}

impl Default for TruncateConfig {
    fn default() -> Self {
        Self {
            max_lines: 400,
            max_bytes: 40 * 1024,
            head_lines: 150,
            tail_lines: 100,
            tools: Tools::AllExcept(vec!["read".to_owned()]),
            spill_dir: PathBuf::from(".rig/spill"),
            max_age: Duration::from_secs(7 * 24 * 60 * 60),
        }
    }
}

/// The tools whose results are cut.
#[derive(Clone, Debug)]
pub enum Tools {
    /// Every tool but these.
    AllExcept(Vec<String>),
    /// Only these.
    Only(Vec<String>),
}

impl Tools {
    fn covers(&self, name: &str) -> bool {
        match self {
            Self::AllExcept(names) => !names.iter().any(|skip| skip == name),
            Self::Only(names) => names.iter().any(|only| only == name),
        }
    }
}

/// Cuts long tool results and keeps them whole in spill files.
#[derive(Default)]
pub struct OutputTruncatePlugin;

impl Plugin for OutputTruncatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TruncateConfig>()
            .add_systems(Startup, sweep_old_spills)
            .add_systems(AgentPass, cut_long_results.in_set(Stage::Review));
    }
}

/// On a call whose result was cut, so it is not cut again: the cut may
/// still be over the byte limit by its marker.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Cut;

/// Replaces each long result of a covered tool by its cut.
fn cut_long_results(
    calls: Query<(Entity, &ToolRequest, &Outcome), (Unreviewed, Without<Cut>)>,
    config: Res<TruncateConfig>,
    mut commands: Commands,
) {
    for (call, ToolRequest { call: request }, outcome) in &calls {
        let Ok(effect::Outcome::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        if !config.tools.covers(request.function.name.as_str()) {
            continue;
        }
        if let Some(cut) = shorten(result, &request.id.wire(), &config) {
            commands.entity(call).insert((Outcome::from(cut), Cut));
        }
    }
}

/// `result` cut around a marker, with its whole text written to the spill
/// file of the call `id` first; `None` when it is short enough, is not
/// plain text, or could not be spilled (then it goes whole, never lost).
pub fn shorten(result: &ToolResult, id: &str, config: &TruncateConfig) -> Option<ToolResult> {
    let text = result.output().as_text()?;
    if text.lines().count() <= config.max_lines && text.len() <= config.max_bytes {
        return None;
    }
    let path = config.spill_dir.join(format!("{}.txt", file_name(id)));
    if let Err(error) = spill(&path, text) {
        warn!(
            "could not keep a long tool result in {}: {error}",
            path.display()
        );
        return None;
    }
    let cut = cut(text, config, &path);
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
fn cut(text: &str, config: &TruncateConfig, path: &Path) -> String {
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

/// Deletes the spill files older than [`TruncateConfig::max_age`].
fn sweep_old_spills(config: Res<TruncateConfig>) {
    let Ok(entries) = fs::read_dir(&config.spill_dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let old = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > config.max_age);
        if old && entry.path().is_file() {
            fs::remove_file(entry.path()).ok();
        }
    }
}

#[cfg(test)]
mod tests;
