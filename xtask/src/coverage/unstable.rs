//! The regions whose coverage depends on scheduling, kept in
//! `crates/rig-cassette/coverage/unstable.tsv` with a reason each.
//!
//! A test reaches such a region in some instrumented runs and not in others,
//! for example when it wins a race with a worker thread. The line gate leaves
//! the region out of the baseline and out of every measurement, so neither
//! outcome of the race can fail the gate. Each row names the file's source
//! hash, the region (`line` or `line.block.branch`), the trimmed source line
//! and the reason. A row applies while the file's source and its baseline row
//! carry the row's hash. Writing the baseline from several runs adds every
//! region the runs disagree on, with an empty reason that `--check` refuses
//! until someone writes one.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::lines::{Branch, FileCoverage, parse_branch};
use super::{Result, invalid};

/// The file's name in the baseline directory.
pub(crate) const FILE: &str = "unstable.tsv";

/// The file's column header.
pub(crate) const HEADER: &str = "file\tsource\tregion\tcode\treason";

/// A line, or one outcome of a branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Region {
    Line(u32),
    Branch(Branch),
}

impl Region {
    fn parse(text: &str) -> Option<Self> {
        match text.parse() {
            Ok(line) => Some(Self::Line(line)),
            Err(_) => parse_branch(text).map(Self::Branch),
        }
    }

    pub(crate) fn line(self) -> u32 {
        match self {
            Self::Line(line) | Self::Branch((line, _, _)) => line,
        }
    }

    fn at(self, line: u32) -> Self {
        match self {
            Self::Line(_) => Self::Line(line),
            Self::Branch((_, block, branch)) => Self::Branch((line, block, branch)),
        }
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Line(line) => write!(f, "{line}"),
            Self::Branch((line, block, branch)) => write!(f, "{line}.{block}.{branch}"),
        }
    }
}

/// One row of the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Unstable {
    pub(crate) file: String,
    pub(crate) source: String,
    pub(crate) region: Region,
    pub(crate) code: String,
    pub(crate) reason: String,
}

/// The rows of `text`; an empty text has none.
pub(crate) fn parse(text: &str) -> Result<Vec<Unstable>> {
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate().skip(1) {
        let columns: Vec<&str> = line.split('\t').collect();
        let row = match columns.as_slice() {
            [file, source, region, code, reason] => Region::parse(region).map(|region| Unstable {
                file: (*file).to_owned(),
                source: (*source).to_owned(),
                region,
                code: (*code).to_owned(),
                reason: (*reason).to_owned(),
            }),
            _ => None,
        };
        rows.push(row.ok_or_else(|| invalid(format!("{FILE} line {}: {line:?}", number + 1)))?);
    }
    Ok(rows)
}

/// The file's text, rows sorted by file and region.
pub(crate) fn render(rows: &[Unstable]) -> String {
    let mut rows = rows.to_vec();
    rows.sort_by(|a, b| (&a.file, a.region).cmp(&(&b.file, b.region)));
    let mut out = format!("{HEADER}\n");
    for row in &rows {
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}",
            row.file, row.source, row.region, row.code, row.reason
        );
    }
    out
}

/// A source line as the `code` column holds it.
pub(crate) fn code(line: &str) -> String {
    line.trim().replace('\t', " ")
}

/// Remove from `files` every region a row names whose hash is the file's
/// current `source`. Only those rows apply: a stale row's line numbers belong
/// to another version of the file.
pub(crate) fn exclude(
    files: &mut BTreeMap<String, FileCoverage>,
    rows: &[Unstable],
    source: impl Fn(&str) -> Option<String>,
) {
    for row in rows {
        if source(&row.file).as_deref() != Some(row.source.as_str()) {
            continue;
        }
        if let Some(coverage) = files.get_mut(&row.file) {
            match row.region {
                Region::Line(line) => {
                    coverage.lines.remove(&line);
                }
                Region::Branch(branch) => {
                    coverage.branches.remove(&branch);
                }
            }
        }
    }
}

/// How many line and branch rows each file has, for the ratio of a file
/// whose source changed: its baseline counts already leave those regions out.
pub(crate) fn per_file(rows: &[Unstable]) -> BTreeMap<String, (usize, usize)> {
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for row in rows {
        let entry = counts.entry(row.file.clone()).or_default();
        match row.region {
            Region::Line(_) => entry.0 += 1,
            Region::Branch(_) => entry.1 += 1,
        }
    }
    counts
}

/// What a `--check` refuses in the file itself: a row without a reason, and
/// a row written for another baseline than the one `baseline` gives the
/// hash of.
pub(crate) fn problems(
    rows: &[Unstable],
    baseline: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    let mut found = Vec::new();
    for row in rows {
        if row.reason.trim().is_empty() {
            found.push(format!(
                "{FILE}: {} {} has no reason; say why its coverage depends on scheduling",
                row.file, row.region
            ));
        }
        if baseline(&row.file).as_deref() != Some(row.source.as_str()) {
            found.push(format!(
                "{FILE}: {} {} was written for another baseline; rewrite it with \
                 `cargo xtask coverage --only lines`",
                row.file, row.region
            ));
        }
    }
    found
}

/// The regions some of the runs covered and others did not: every run's
/// union less their intersection.
pub(crate) fn disagreements(
    union: &BTreeMap<String, FileCoverage>,
    intersection: &BTreeMap<String, FileCoverage>,
) -> Vec<(String, Region)> {
    let mut found = Vec::new();
    for (file, all) in union {
        let every = intersection.get(file);
        let covered_by_every = |region: Region| {
            every.is_some_and(|every| match region {
                Region::Line(line) => every.lines.get(&line) == Some(&true),
                Region::Branch(branch) => every.branches.get(&branch) == Some(&true),
            })
        };
        let some = all
            .lines
            .iter()
            .filter(|(_, covered)| **covered)
            .map(|(line, _)| Region::Line(*line))
            .chain(
                all.branches
                    .iter()
                    .filter(|(_, covered)| **covered)
                    .map(|(branch, _)| Region::Branch(*branch)),
            );
        for region in some {
            if !covered_by_every(region) {
                found.push((file.clone(), region));
            }
        }
    }
    found
}

/// The rows for the baseline about to be written. Kept rows move to their
/// file's current `source`: a row whose file changed follows its code to the
/// nearest line holding the same text, and goes when no line does. Every
/// region in `found` without a row gets one with an empty reason.
/// `text_of` gives a file's current source text and hash.
pub(crate) fn refresh(
    rows: &[Unstable],
    found: &[(String, Region)],
    text_of: impl Fn(&str) -> Option<(String, String)>,
) -> (Vec<Unstable>, Vec<String>) {
    let mut notes = Vec::new();
    let mut kept: Vec<Unstable> = Vec::new();
    for row in rows {
        let Some((text, source)) = text_of(&row.file) else {
            notes.push(format!(
                "{FILE}: dropped {} {}: the file is gone",
                row.file, row.region
            ));
            continue;
        };
        let line = if row.source == source {
            Some(row.region.line())
        } else {
            nearest(&text, &row.code, row.region.line())
        };
        match line {
            Some(line) => kept.push(Unstable {
                source,
                region: row.region.at(line),
                ..row.clone()
            }),
            None => notes.push(format!(
                "{FILE}: dropped {} {}: no line reads {:?} any more",
                row.file, row.region, row.code
            )),
        }
    }
    let held: BTreeSet<(String, Region)> = kept
        .iter()
        .map(|row| (row.file.clone(), row.region))
        .collect();
    for (file, region) in found {
        if held.contains(&(file.clone(), *region)) {
            continue;
        }
        let Some((text, source)) = text_of(file) else {
            continue;
        };
        let code = text
            .lines()
            .nth(region.line().saturating_sub(1) as usize)
            .map(code)
            .unwrap_or_default();
        notes.push(format!(
            "{FILE}: added {file} {region} ({code}); write its reason before `--check` passes"
        ));
        kept.push(Unstable {
            file: file.clone(),
            source,
            region: *region,
            code,
            reason: String::new(),
        });
    }
    (kept, notes)
}

/// The 1-based number of the line nearest `from` whose trimmed text is
/// `code`, the earlier one on a tie.
fn nearest(text: &str, code_text: &str, from: u32) -> Option<u32> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| code(line) == code_text)
        .filter_map(|(index, _)| u32::try_from(index + 1).ok())
        .min_by_key(|line| (line.abs_diff(from), *line))
}
