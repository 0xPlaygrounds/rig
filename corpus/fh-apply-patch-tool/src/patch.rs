//! The `*** Begin Patch` format: parsing a patch, and checking it against
//! the files it names into a [`Plan`] of whole-file writes and deletes,
//! or the list of everything that keeps it from applying.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rig_tools::fs::file_change;
use rig_tools::workspace::{Stat, Workspace};

/// One file operation of a patch.
enum Op {
    /// `*** Add File:`, with its lines.
    Add { path: String, lines: Vec<String> },
    /// `*** Delete File:`.
    Delete { path: String },
    /// `*** Update File:`, maybe `*** Move to:`, with its hunks.
    Update {
        path: String,
        move_to: Option<String>,
        hunks: Vec<Hunk>,
    },
}

/// An `@@` hunk: the line its header names, and the lines it expects and
/// leaves.
#[derive(Default)]
struct Hunk {
    anchor: Option<String>,
    old: Vec<String>,
    new: Vec<String>,
}

/// A patch that applies: the files to write whole and the files to
/// delete, by the paths the patch names, and a line per file for the
/// result.
#[derive(Default)]
pub(crate) struct Plan {
    /// Each file written whole, with its text after.
    pub writes: Vec<(String, String)>,
    /// Each file deleted.
    pub deletes: Vec<String>,
    /// `A path (+n)`, `M path (+n -m)`, `D path (-m)`, `R from -> to (+n -m)`.
    pub summary: Vec<String>,
}

/// Parses `input` and checks it against the files of `workspace`, placing
/// each path from `base` as the file tools do (refused outside it when
/// `confine`): the plan, or every reason it does not apply.
pub(crate) fn check(
    input: &str,
    workspace: &dyn Workspace,
    base: &Path,
    confine: bool,
) -> Result<Plan, String> {
    let ops = parse(input)?;
    let mut plan = Plan::default();
    let mut problems = Vec::new();
    let mut touched: Vec<PathBuf> = Vec::new();
    let mut place_once =
        |path: &str, problems: &mut Vec<String>| match workspace.place(Some(base), confine, path) {
            Ok(resolved) if touched.contains(&resolved) => {
                problems.push(format!("{path}: named twice in the patch"));
                None
            }
            Ok(resolved) => {
                touched.push(resolved.clone());
                Some(resolved)
            }
            Err(error) => {
                problems.push(error.to_string());
                None
            }
        };
    let exists = |path: &Path| !matches!(workspace.stat(path), Ok(Stat::Missing));
    for op in ops {
        match op {
            Op::Add { path, lines } => {
                let Some(resolved) = place_once(&path, &mut problems) else {
                    continue;
                };
                if exists(&resolved) {
                    problems.push(format!("{path}: the file to add exists"));
                    continue;
                }
                let after = join(&lines, true);
                let change = file_change(&path, None, Some(&after));
                plan.summary.push(format!("A {path} (+{})", change.added));
                plan.writes.push((path, after));
            }
            Op::Delete { path } => {
                let Some(resolved) = place_once(&path, &mut problems) else {
                    continue;
                };
                match workspace.read_text(&resolved) {
                    Ok(before) => {
                        let change = file_change(&path, Some(&before), Some(""));
                        plan.summary.push(format!("D {path} (-{})", change.removed));
                        plan.deletes.push(path);
                    }
                    Err(error) => problems.push(format!("{path}: the file to delete: {error}")),
                }
            }
            Op::Update {
                path,
                move_to,
                hunks,
            } => {
                let Some(resolved) = place_once(&path, &mut problems) else {
                    continue;
                };
                let before = match workspace.read_text(&resolved) {
                    Ok(before) => before,
                    Err(error) => {
                        problems.push(format!("{path}: the file to update: {error}"));
                        continue;
                    }
                };
                let after = match apply(&before, &hunks) {
                    Ok(after) => after,
                    Err(failures) => {
                        problems.extend(
                            failures
                                .into_iter()
                                .map(|failure| format!("{path}, {failure}")),
                        );
                        continue;
                    }
                };
                let change = file_change(&path, Some(&before), Some(&after));
                let counts = format!("(+{} -{})", change.added, change.removed);
                let Some(to) = move_to else {
                    plan.summary.push(format!("M {path} {counts}"));
                    plan.writes.push((path, after));
                    continue;
                };
                let Some(target) = place_once(&to, &mut problems) else {
                    continue;
                };
                if exists(&target) {
                    problems.push(format!("{to}: the file to move to exists"));
                    continue;
                }
                plan.summary.push(format!("R {path} -> {to} {counts}"));
                plan.writes.push((to, after));
                plan.deletes.push(path);
            }
        }
    }
    if problems.is_empty() {
        Ok(plan)
    } else {
        let mut text = "The patch was not applied; no file changed.".to_owned();
        for problem in problems {
            write!(text, "\n\n- {problem}").ok();
        }
        Err(text)
    }
}

/// The operations of `input`, or why it is no patch.
fn parse(input: &str) -> Result<Vec<Op>, String> {
    let mut lines = input
        .lines()
        .skip_while(|line| line.trim().is_empty())
        .peekable();
    if lines.next().map(str::trim) != Some("*** Begin Patch") {
        return Err("A patch starts with `*** Begin Patch`.".to_owned());
    }
    let mut ops = Vec::new();
    while let Some(line) = lines.next() {
        let line = line.trim_end();
        if line == "*** End Patch" {
            return Ok(ops);
        }
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let mut added = Vec::new();
            while let Some(next) = lines.next_if(|next| !next.starts_with("*** ")) {
                match next.strip_prefix('+') {
                    Some(text) => added.push(text.to_owned()),
                    None => {
                        return Err(format!(
                            "{path}: each line of an added file starts with `+`"
                        ));
                    }
                }
            }
            ops.push(Op::Add {
                path: path.trim().to_owned(),
                lines: added,
            });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            ops.push(Op::Delete {
                path: path.trim().to_owned(),
            });
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            let move_to = lines
                .next_if(|next| next.starts_with("*** Move to: "))
                .and_then(|next| next.strip_prefix("*** Move to: "))
                .map(|to| to.trim().to_owned());
            let mut hunks: Vec<Hunk> = Vec::new();
            while let Some(next) = lines
                .next_if(|next| !next.starts_with("*** ") || next.trim_end() == "*** End of File")
            {
                if next.trim_end() == "*** End of File" {
                    continue;
                }
                if let Some(anchor) = next.strip_prefix("@@") {
                    let anchor = anchor.trim();
                    hunks.push(Hunk {
                        anchor: (!anchor.is_empty()).then(|| anchor.to_owned()),
                        ..Hunk::default()
                    });
                    continue;
                }
                if hunks.is_empty() {
                    hunks.push(Hunk::default());
                }
                let Some(hunk) = hunks.last_mut() else {
                    continue;
                };
                // An empty line is an empty context line, as models often
                // drop its leading space.
                let (mark, text) = next.split_at_checked(1).unwrap_or((" ", ""));
                match mark {
                    " " => {
                        hunk.old.push(text.to_owned());
                        hunk.new.push(text.to_owned());
                    }
                    "-" => hunk.old.push(text.to_owned()),
                    "+" => hunk.new.push(text.to_owned()),
                    _ => {
                        return Err(format!(
                            "{path}: a hunk line starts with ` `, `-` or `+`: `{next}`"
                        ));
                    }
                }
            }
            ops.push(Op::Update {
                path: path.trim().to_owned(),
                move_to,
                hunks,
            });
        } else if !line.trim().is_empty() {
            return Err(format!(
                "Expected a file header or `*** End Patch`, got `{line}`."
            ));
        }
    }
    Err("A patch ends with `*** End Patch`.".to_owned())
}

/// `text` with `hunks` applied in order, each found by its lines after the
/// one before (trailing whitespace aside), or a line about each hunk that
/// does not apply.
fn apply(text: &str, hunks: &[Hunk]) -> Result<String, Vec<String>> {
    let lines: Vec<&str> = text.lines().collect();
    let same = |a: &str, b: &str| a.trim_end() == b.trim_end();
    let mut cursor = 0;
    let mut replacements: Vec<(usize, usize, &[String])> = Vec::new();
    let mut failures = Vec::new();
    for (number, hunk) in (1..).zip(hunks) {
        let from = match &hunk.anchor {
            None => cursor,
            Some(anchor) => {
                let found = lines
                    .iter()
                    .enumerate()
                    .skip(cursor)
                    .find(|(_, line)| line.contains(anchor.as_str()));
                match found {
                    Some((at, _)) => at.saturating_add(1),
                    None => {
                        failures.push(format!(
                            "hunk {number}: no line containing `{anchor}` after line {cursor}"
                        ));
                        continue;
                    }
                }
            }
        };
        if hunk.old.is_empty() {
            let at = if hunk.anchor.is_some() {
                from
            } else {
                lines.len()
            };
            replacements.push((at, 0, &hunk.new));
            cursor = at;
            continue;
        }
        let fits = |start: usize| {
            lines
                .get(start..start.saturating_add(hunk.old.len()))
                .is_some_and(|window| window.iter().zip(&hunk.old).all(|(a, b)| same(a, b)))
        };
        match (from..lines.len()).find(|&start| fits(start)) {
            Some(start) => {
                replacements.push((start, hunk.old.len(), &hunk.new));
                cursor = start.saturating_add(hunk.old.len());
            }
            None => failures.push(not_found(number, &hunk.old, &lines)),
        }
    }
    if !failures.is_empty() {
        return Err(failures);
    }
    let mut out: Vec<String> = lines.iter().map(|line| (*line).to_owned()).collect();
    for (start, len, new) in replacements.into_iter().rev() {
        let end = start.saturating_add(len).min(out.len());
        out.splice(start.min(end)..end, new.iter().cloned());
    }
    Ok(join(&out, text.is_empty() || text.ends_with('\n')))
}

/// Says that hunk `number`, expecting `old`, was not found in `lines`, and
/// shows the region most like it.
fn not_found(number: usize, old: &[String], lines: &[&str]) -> String {
    let score = |start: usize| {
        old.iter()
            .enumerate()
            .filter(|(at, want)| {
                lines
                    .get(start.saturating_add(*at))
                    .is_some_and(|line| line.trim_end() == want.trim_end())
            })
            .count()
    };
    let mut text = format!("hunk {number}: its context was not found. It looked for:");
    for line in old {
        write!(text, "\n    {line}").ok();
    }
    let best = (0..lines.len()).max_by_key(|&start| (score(start), std::cmp::Reverse(start)));
    match best.filter(|&start| score(start) > 0) {
        Some(start) => {
            let end = start.saturating_add(old.len()).min(lines.len());
            write!(
                text,
                "\nThe closest region, lines {}-{end}:",
                start.saturating_add(1)
            )
            .ok();
            for (at, line) in
                (start.saturating_add(1)..).zip(lines.get(start..end).unwrap_or_default())
            {
                write!(text, "\n{at:>6}  {line}").ok();
            }
        }
        None => text.push_str("\nNo region of the file is like it."),
    }
    text
}

/// `lines` as a text, ending with a newline when `newline`.
fn join(lines: &[String], newline: bool) -> String {
    let mut text = lines.join("\n");
    if newline && !lines.is_empty() {
        text.push('\n');
    }
    text
}
