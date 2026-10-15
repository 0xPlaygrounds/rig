//! Plugin corpus (held-out): H13: Hunk-by-hunk review after each turn (spec `H13-hunk-review.md`).
//!
//! A Gate system snapshots each file an `edit` or `write` call is about to
//! change, the first time one does, and notes each `shell` command as not
//! reviewable, on the top-level agent the call works for (a subagent's go
//! to its top-level parent). When that agent's turn ends, each file's
//! snapshot is diffed with the file as the turn left it, and a [`Picker`]
//! steps through the hunks: keep, revert, revert with a comment (typed),
//! keep the rest; Esc defers it and `/review` reopens it. `/review keep
//! <n>`, `/review revert <n> [comment]` and `/review done` do the same.
//! A revert applies the reverse hunk unless the lines it covers changed
//! since the turn ended (a conflict, skipped and reported). Once done, a
//! queued message tells the agent what the user reverted and why. When no
//! front answers, as with `--print`, everything is kept and a summary goes
//! to stderr.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use rig_harness::prelude::*;
use rig_tools::fs::write_atomic;
use rig_tools::{Edit, Shell, Write};
use similar::{Algorithm, DiffTag, TextDiff, capture_diff_slices};

/// Adds the snapshots, the review and `/review`.
#[derive(Default)]
pub struct HunkReviewPlugin;

impl Plugin for HunkReviewPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, HunkReview>()
            .configure_sets(AgentPass, HunkReviewSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, snapshot.in_set(HunkReviewSystems))
            .add_observer(turn_ended)
            .add_command(
                "review",
                "Review the last turn's changes: /review, /review keep N, \
                 /review revert N [comment], /review done",
                review,
            );
    }
}

/// The Gate system that snapshots files before a call changes them, for
/// others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HunkReviewSystems;

/// A top-level agent's changes under review: the files its turns (and its
/// subagents') changed since the last review was done.
#[derive(Component, Default, Debug)]
pub struct HunkReview {
    /// Each file changed, by absolute path.
    pub files: BTreeMap<String, FileReview>,
    /// The `shell` commands run meanwhile, which are not reviewable.
    pub shell: Vec<String>,
    /// What the user reverted, as the agent is told: each hunk's file and
    /// lines, with the user's comment.
    pub reverted: Vec<String>,
    /// The picker showing the review, while one does.
    pub picker: Option<Entity>,
}

/// One file under review.
#[derive(Default, Debug)]
pub struct FileReview {
    /// The file before the first call that changed it; `None` when there
    /// was none.
    pub snapshot: Option<String>,
    /// The file as the last turn left it, with the hunks reverted since.
    pub expected: String,
    /// Its hunks, snapshot against the file as the last turn left it.
    pub hunks: Vec<Hunk>,
}

/// A run of changed lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hunk {
    /// Where its lines start in the file as the turn left it.
    pub start: usize,
    /// The lines it replaced, with their line ends.
    pub old: Vec<String>,
    /// The lines it put there.
    pub new: Vec<String>,
    /// What the user decided.
    pub decision: Option<Decision>,
}

/// What became of a hunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Kept as the agent left it.
    Keep,
    /// Put back as it was.
    Reverted,
    /// Its lines changed since the turn ended: left alone.
    Conflict,
}

/// On a call whose file was snapshotted, or whose command was noted.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Noted;

/// The top-level agent `agent` works for.
fn top(mut agent: Entity, parents: &Query<&SpawnedBy>) -> Entity {
    while let Ok(&SpawnedBy(parent)) = parents.get(agent) {
        agent = parent;
    }
    agent
}

/// Snapshots the file of each `edit` and `write` call about to run, the
/// first time its file is changed since the last review, and notes each
/// `shell` command. The file is the one the call's footprint names, as the
/// tool resolves it in its agent's directory.
fn snapshot(
    calls: Query<
        (Entity, &ToolRequest, &ForAgent, &Footprint),
        (ReadyToStart, ToolCalls, Without<Noted>),
    >,
    parents: Query<&SpawnedBy>,
    mut reviews: Query<&mut HunkReview>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), Footprint(touches)) in &calls {
        let function = &request.call.function;
        let Ok(mut review) = reviews.get_mut(top(agent, &parents)) else {
            continue;
        };
        match (function.name.as_str(), touches) {
            (Edit::NAME | Write::NAME, Touches::Writes(Scope::Paths(paths))) => {
                for path in paths {
                    let path = path.to_string_lossy().into_owned();
                    review
                        .files
                        .entry(path)
                        .or_insert_with_key(|path| FileReview {
                            snapshot: std::fs::read_to_string(path).ok(),
                            ..FileReview::default()
                        });
                }
            }
            (Shell::NAME, _) => {
                let command = function.arguments.get("command").and_then(|c| c.as_str());
                review.shell.push(command.unwrap_or_default().to_owned());
            }
            _ => continue,
        }
        commands.entity(call).insert(Noted);
    }
}

/// The lines of `text`, each with its line end.
fn lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

/// The hunks turning `before` into `after`: each run of changed lines.
fn diff(before: &str, after: &str) -> Vec<Hunk> {
    let diff = TextDiff::from_lines(before, after);
    let (old, new) = (lines(before), lines(after));
    let take = |lines: &[&str], range: std::ops::Range<usize>| -> Vec<String> {
        lines
            .get(range)
            .unwrap_or_default()
            .iter()
            .map(|line| (*line).to_owned())
            .collect()
    };
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut last_end = None;
    for op in diff.ops() {
        let (tag, old_range, new_range) = op.as_tag_tuple();
        if tag == DiffTag::Equal {
            continue;
        }
        let joins = last_end == Some(new_range.start);
        last_end = Some(new_range.end);
        match hunks.last_mut() {
            Some(hunk) if joins => {
                hunk.old.extend(take(&old, old_range));
                hunk.new.extend(take(&new, new_range));
            }
            _ => hunks.push(Hunk {
                start: new_range.start,
                old: take(&old, old_range),
                new: take(&new, new_range),
                decision: None,
            }),
        }
    }
    hunks
}

/// Where hunk `index` (from 1) is: its file and its position there.
fn find(review: &HunkReview, index: usize) -> Option<(&str, usize)> {
    review
        .files
        .iter()
        .flat_map(|(path, file)| (0..file.hunks.len()).map(move |at| (path.as_str(), at)))
        .nth(index.checked_sub(1)?)
}

/// Every hunk with its number (from 1) and its file.
fn numbered(review: &HunkReview) -> impl Iterator<Item = (usize, &str, &Hunk)> {
    review
        .files
        .iter()
        .flat_map(|(path, file)| file.hunks.iter().map(move |hunk| (path.as_str(), hunk)))
        .enumerate()
        .map(|(at, (path, hunk))| (at + 1, path, hunk))
}

/// `path` as the user reads it: from the project's root when under it.
fn shown(path: &str, root: &Path) -> String {
    Path::new(path)
        .strip_prefix(root)
        .map_or_else(|_| path.to_owned(), |path| path.display().to_string())
}

/// The lines a hunk covers in the file as the agent left it, from 1, or
/// where its removed lines were.
fn line_range(hunk: &Hunk) -> String {
    let first = hunk.start + 1;
    match hunk.new.len() {
        0 | 1 => first.to_string(),
        n => format!("{first}-{}", first + n - 1),
    }
}

/// The hunk as diff lines.
fn hunk_text(hunk: &Hunk) -> String {
    let mut text = String::new();
    let mut put = |sign: char, line: &String| {
        let _ = writeln!(text, "{sign}{}", line.trim_end_matches(['\n', '\r']));
    };
    hunk.old.iter().for_each(|line| put('-', line));
    hunk.new.iter().for_each(|line| put('+', line));
    text
}

/// The shell commands' line, empty without one.
fn shell_note(review: &HunkReview) -> String {
    if review.shell.is_empty() {
        return String::new();
    }
    let commands: Vec<String> = review.shell.iter().map(|c| format!("`{c}`")).collect();
    format!("\nNot reviewable (shell): {}", commands.join(", "))
}

/// Recomputes each file's hunks from its snapshot to the file now, as a
/// turn left it, keeping what was decided on a hunk that is still there;
/// one put back is no hunk any more.
fn refresh(review: &mut HunkReview) {
    for (path, file) in &mut review.files {
        let now = std::fs::read_to_string(path).unwrap_or_default();
        let before = file.snapshot.clone().unwrap_or_default();
        let mut hunks = diff(&before, &now);
        for hunk in &mut hunks {
            hunk.decision = file
                .hunks
                .iter()
                .find(|old| old.old == hunk.old && old.new == hunk.new)
                .and_then(|old| old.decision.clone())
                .filter(|decision| *decision != Decision::Reverted);
        }
        file.expected = now;
        file.hunks = hunks;
    }
    review.files.retain(|_, file| !file.hunks.is_empty());
}

/// Opens the review over the transcript as the top-level agent's turn
/// ends with changes, or keeps everything when nobody can answer.
fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, (With<Turn>, With<TurnOutcome>)>,
    tops: Query<(), Without<SpawnedBy>>,
    mut reviews: Query<&mut HunkReview>,
    answerers: Answerers,
    root: Option<Res<ProjectRoot>>,
    mut commands: Commands,
) {
    let Ok(&ForAgent(agent)) = turns.get(settled.entity) else {
        return;
    };
    let (true, Ok(mut review)) = (tops.contains(agent), reviews.get_mut(agent)) else {
        return;
    };
    refresh(&mut review);
    if review.files.is_empty() && review.shell.is_empty() {
        return;
    }
    let root = root.map(|root| root.0.clone()).unwrap_or_default();
    if answerers.can_answer(Some(agent)) {
        open(&mut review, agent, &root, &mut commands);
        return;
    }
    let mut summary = format!(
        "rig: kept {} change(s), unreviewed:",
        numbered(&review).count()
    );
    for (path, file) in &review.files {
        let _ = write!(
            summary,
            " {} ({} hunks)",
            shown(path, &root),
            file.hunks.len()
        );
    }
    eprintln!("{summary}{}", shell_note(&review));
    *review = HunkReview::default();
}

/// Shows the first hunk not decided yet, or the review's end, in a picker
/// in place of the one shown.
fn open(review: &mut HunkReview, agent: Entity, root: &Path, commands: &mut Commands) {
    if let Some(picker) = review.picker.take() {
        commands.entity(picker).try_despawn();
    }
    let total = numbered(review).count();
    let next = numbered(review).find(|(_, _, hunk)| hunk.decision.is_none());
    let picker = match next {
        Some((index, path, hunk)) => {
            let title = format!(
                "Review {index}/{total}: {}:{}",
                shown(path, root),
                line_range(hunk)
            );
            let body = format!(
                "{}Type a comment to revert it with that comment.{}",
                hunk_text(hunk),
                shell_note(review)
            );
            Picker::new(title)
                .body(body)
                .item("Keep", format!("review keep {index}"))
                .item("Revert", format!("review revert {index}"))
                .item("Keep the rest", "review done")
                .free_text()
        }
        None => Picker::new("Review: every hunk decided")
            .body(shell_note(review).trim_start().to_owned())
            .item("Done", "review done"),
    };
    let index = next.map(|(index, ..)| index);
    let picker = commands
        .spawn(picker.for_agent(agent))
        .observe(
            move |picked: On<Picked>,
                  mut commands: Commands,
                  mut reviews: Query<&mut HunkReview>,
                  mut notices: MessageWriter<Notice>| {
                let comment = picked.chosen.as_ref().map(|chosen| &chosen.text);
                match (comment, index) {
                    (Some(Some(comment)), Some(index)) => {
                        let line = format!("review revert {index} {comment}");
                        commands.trigger(RunCommand::new(agent, line));
                    }
                    (None, _) => {
                        if let Ok(mut review) = reviews.get_mut(agent)
                            && review.picker == Some(picked.entity)
                        {
                            review.picker = None;
                            notices
                                .write(Notice::info(agent, "Review deferred: /review reopens it."));
                        }
                    }
                    _ => {}
                }
            },
        )
        .id();
    review.picker = Some(picker);
}

/// Puts hunk `at` of `path` back as the snapshot had it, unless its lines
/// changed since the turn ended.
fn revert(file: &mut FileReview, path: &str, at: usize) -> Decision {
    let Some(hunk) = file.hunks.get(at).cloned() else {
        return Decision::Conflict;
    };
    // Where the hunk is in the file as expected, after the hunks before it
    // that were reverted.
    let start = file
        .hunks
        .iter()
        .filter(|other| other.start < hunk.start)
        .filter(|other| other.decision == Some(Decision::Reverted))
        .fold(hunk.start as isize, |at, other| {
            at + other.old.len() as isize - other.new.len() as isize
        });
    let Ok(start) = usize::try_from(start) else {
        return Decision::Conflict;
    };
    let end = start + hunk.new.len();
    let Ok(current) = std::fs::read_to_string(path) else {
        return Decision::Conflict;
    };
    let (expected, now) = (lines(&file.expected), lines(&current));
    // How the file changed since: the hunk's lines must be untouched, and
    // the lines added or removed before them move it.
    let mut moved = 0isize;
    for op in capture_diff_slices(Algorithm::Myers, &expected, &now) {
        let (tag, old, new) = op.as_tag_tuple();
        if tag == DiffTag::Equal {
            continue;
        }
        let touches = if start == end {
            old.start <= start && start <= old.end
        } else if old.is_empty() {
            start < old.start && old.start < end
        } else {
            old.start < end && old.end > start
        };
        if touches {
            return Decision::Conflict;
        }
        if old.end <= start {
            moved += new.len() as isize - old.len() as isize;
        }
    }
    let Ok(at_now) = usize::try_from(start as isize + moved) else {
        return Decision::Conflict;
    };
    let splice = |lines: &[&str], from: usize| -> Option<String> {
        let (head, tail) = (lines.get(..from)?, lines.get(from + hunk.new.len()..)?);
        let middle = lines.get(from..from + hunk.new.len())?;
        (middle.iter().zip(&hunk.new).all(|(a, b)| a == b)).then(|| {
            let old = hunk.old.iter().map(String::as_str);
            head.iter()
                .copied()
                .chain(old)
                .chain(tail.iter().copied())
                .collect()
        })
    };
    let (Some(reverted), Some(expected)) = (splice(&now, at_now), splice(&expected, start)) else {
        return Decision::Conflict;
    };
    let written = if reverted.is_empty() && file.snapshot.is_none() {
        std::fs::remove_file(path)
    } else {
        write_atomic(Path::new(path), reverted.as_bytes())
    };
    if written.is_err() {
        return Decision::Conflict;
    }
    file.expected = expected;
    Decision::Reverted
}

/// `/review`: lists the hunks and reopens the review; `keep N`,
/// `revert N [comment]` and `done` decide.
fn review(
    In(args): In<CommandArgs>,
    mut reviews: Query<&mut HunkReview>,
    root: Option<Res<ProjectRoot>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let root = root.map(|root| root.0.clone()).unwrap_or_default();
    let Ok(mut review) = reviews.get_mut(agent) else {
        notices.write(Notice::error(
            agent,
            "Only a top-level agent's turns are reviewed.",
        ));
        return;
    };
    if numbered(&review).next().is_none() {
        notices.write(Notice::info(agent, "Nothing to review."));
        return;
    }
    let mut words = args.args.splitn(3, char::is_whitespace);
    let verb = words.next().unwrap_or_default();
    let index: Option<usize> = words.next().and_then(|n| n.parse().ok());
    let comment = words.next().map(str::trim).filter(|c| !c.is_empty());
    match (verb, index) {
        ("", _) => {
            let mut text = String::from("Changes under review:");
            for (index, path, hunk) in numbered(&review) {
                let state = match &hunk.decision {
                    None => "undecided".to_owned(),
                    Some(Decision::Keep) => "kept".to_owned(),
                    Some(Decision::Reverted) => "reverted".to_owned(),
                    Some(Decision::Conflict) => "conflict".to_owned(),
                };
                let at = format!("{}:{}", shown(path, &root), line_range(hunk));
                let _ = write!(text, "\n{index}. {at} ({state})\n{}", hunk_text(hunk));
            }
            text.push_str(&shell_note(&review));
            notices.write(Notice::info(agent, text));
        }
        ("keep" | "revert", Some(index)) => {
            let review = &mut *review;
            let Some((path, at)) = find(review, index).map(|(path, at)| (path.to_owned(), at))
            else {
                notices.write(Notice::error(agent, format!("There is no hunk {index}.")));
                return;
            };
            let Some(file) = review.files.get_mut(&path) else {
                return;
            };
            if file
                .hunks
                .get(at)
                .is_some_and(|hunk| hunk.decision.is_some())
            {
                notices.write(Notice::error(agent, format!("Hunk {index} is decided.")));
                return;
            }
            let decision = match verb {
                "keep" => Decision::Keep,
                _ => revert(file, &path, at),
            };
            let Some(hunk) = file.hunks.get_mut(at) else {
                return;
            };
            let place = format!("{}:{}", shown(&path, &root), line_range(hunk));
            match (&decision, comment) {
                (Decision::Conflict, _) => {
                    let text =
                        format!("Hunk {index} ({place}) changed since the turn ended: skipped.");
                    notices.write(Notice::info(agent, text));
                }
                (Decision::Reverted, Some(comment)) => {
                    review.reverted.push(format!("{place} ({comment})"));
                }
                (Decision::Reverted, None) => review.reverted.push(place),
                (Decision::Keep, _) => {}
            }
            hunk.decision = Some(decision);
        }
        ("done", _) => {
            finish(&mut review, agent, &mut commands, &mut notices);
            return;
        }
        _ => {
            notices.write(Notice::error(
                agent,
                "Use /review, /review keep N, /review revert N [comment] or /review done.",
            ));
            return;
        }
    }
    open(&mut review, agent, &root, &mut commands);
}

/// Ends the review: what is undecided is kept, and the agent is told what
/// the user reverted, if anything.
fn finish(
    review: &mut HunkReview,
    agent: Entity,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    if let Some(picker) = review.picker.take() {
        commands.entity(picker).try_despawn();
    }
    let conflicts = numbered(review)
        .filter(|(.., hunk)| hunk.decision == Some(Decision::Conflict))
        .count();
    let kept = numbered(review)
        .filter(|(.., hunk)| matches!(hunk.decision, None | Some(Decision::Keep)))
        .count();
    let reverted = &review.reverted;
    let text = format!(
        "Review done: {kept} kept, {} reverted, {conflicts} skipped as conflicts.",
        reverted.len()
    );
    notices.write(Notice::info(agent, text));
    if !reverted.is_empty() {
        let text = format!("The user reverted: {}", reverted.join("; "));
        let message = Deliver::new(agent, text, DeliveryMode::Queue);
        commands.trigger(message.with_author(Author::plugin("hunk-review")));
    }
    *review = HunkReview::default();
}

#[cfg(test)]
mod tests;
