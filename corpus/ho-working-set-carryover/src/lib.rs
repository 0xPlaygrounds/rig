//! The working set, kept across compaction: each agent's files read and
//! changed and its last test or build commands, tracked from the facts its
//! tool calls publish ([`FilesRead`], [`FilesChanged`], [`ExitStatus`]).
//! Whenever a compaction splices a summary into an agent's conversation
//! ([`Committed::Splice`] with an item of the origin `compaction`, by
//! rig-compaction or `/compact`), a "Working set" block goes into the
//! conversation right after the summary, by a splice that keeps the
//! message's id, and into the request already built from it, so it goes
//! with the next request and a resumed session has it. The tracking is saved with the
//! session ([`WorkingSet`]).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use rig_compaction::summary_origin;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ExitStatus, FilesChanged, FilesRead};
use serde::{Deserialize, Serialize};

/// How many read files the working set message names.
const SHOWN_READS: usize = 15;
/// How many read files are kept between condensations.
const KEPT_READS: usize = 50;
/// How many test or build runs are kept.
const KEPT_RUNS: usize = 3;
/// How many of a run's last output lines are kept.
const TAIL_LINES: usize = 10;

/// Tracks each agent's working set and carries it across condensations.
#[derive(Default)]
pub struct WorkingSetPlugin;

impl Plugin for WorkingSetPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<WorkingSetSettings>("working_set")
            .init_resource::<ProjectRoot>()
            .register_required_components::<Agent, WorkingSet>()
            .add_observer(track)
            .configure_sets(AgentPass, WorkingSetSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, carry_over.in_set(WorkingSetSystems));
    }
}

/// The Prepare system that carries the working set over a compaction, for
/// another plugin to order against. It runs after the default prompt went
/// in, and before compaction measures the request it adds to.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkingSetSystems;

/// The `[working_set]` table: `test_pattern`, the regex a command that
/// tests or builds matches.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct WorkingSetSettings {
    pub test_pattern: String,
}

impl Default for WorkingSetSettings {
    fn default() -> Self {
        Self {
            test_pattern: r"cargo (test|check|build)|npm test|pytest".to_owned(),
        }
    }
}

/// What an agent worked on: the files it read since its conversation was
/// last condensed, the files it changed this session and its last test or
/// build runs. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct WorkingSet {
    /// Read files, oldest first, each once.
    pub read: Vec<ReadFile>,
    /// Changed files, in the order first changed, each once.
    pub changed: Vec<ChangedFile>,
    /// Test or build runs, oldest first.
    pub runs: Vec<TestRun>,
}

/// A file the agent read.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadFile {
    /// Its path, as the tool published it.
    pub path: String,
    /// When it was last read, in milliseconds since the Unix epoch.
    pub at_ms: u64,
}

/// A file the agent changed, over all its changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangedFile {
    /// Its path, as the tool published it.
    pub path: String,
    /// Lines added.
    pub added: usize,
    /// Lines removed.
    pub removed: usize,
    /// The tools that changed it, each once.
    pub tools: Vec<String>,
}

/// A command matching the test pattern, and how it ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestRun {
    /// The command line.
    pub command: String,
    /// How it ended.
    pub status: ExitStatus,
    /// Its last output lines, as the model read them.
    pub tail: Vec<String>,
}

/// Adds what each settled tool call published to its agent's working set.
fn track(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Published, &Outcome)>,
    mut agents: Query<&mut WorkingSet>,
    settings: Res<WorkingSetSettings>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((request, &ForAgent(agent), published, outcome)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut set) = agents.get_mut(agent) else {
        return;
    };
    let tool = request.call.function.name.as_str();
    let at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        });
    if let Ok(Some(FilesRead(files))) = published.0.get::<FilesRead>() {
        for path in files.into_iter().map(|file| file.path) {
            set.read.retain(|read| read.path != path);
            set.read.push(ReadFile { path, at_ms });
        }
        let over = set.read.len().saturating_sub(KEPT_READS);
        set.read.drain(..over);
    }
    if let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() {
        for change in changes {
            let at = set.changed.iter().position(|file| file.path == change.path);
            let file = match at.and_then(|at| set.changed.get_mut(at)) {
                Some(file) => file,
                None => {
                    set.changed.push(ChangedFile {
                        path: change.path.clone(),
                        added: 0,
                        removed: 0,
                        tools: Vec::new(),
                    });
                    let Some(file) = set.changed.last_mut() else {
                        continue;
                    };
                    file
                }
            };
            file.added += change.added;
            file.removed += change.removed;
            if !file.tools.iter().any(|known| known == tool) {
                file.tools.push(tool.to_owned());
            }
        }
    }
    let command = request.call.function.arguments.get("command");
    if let (Ok(Some(status)), Some(command)) = (
        published.0.get::<ExitStatus>(),
        command.and_then(|command| command.as_str()),
    ) {
        let pattern = match Regex::new(&settings.test_pattern) {
            Ok(pattern) => pattern,
            Err(error) => {
                notices.write(Notice::error(
                    agent,
                    format!("working_set.test_pattern is no regex: {error}"),
                ));
                return;
            }
        };
        if pattern.is_match(command) {
            let output = outcome.tool_result().output().render();
            let lines: Vec<&str> = output.lines().collect();
            let tail = lines.get(lines.len().saturating_sub(TAIL_LINES)..);
            let tail = tail
                .unwrap_or_default()
                .iter()
                .map(|line| (*line).to_owned());
            set.runs.push(TestRun {
                command: command.to_owned(),
                status,
                tail: tail.collect(),
            });
            let over = set.runs.len().saturating_sub(KEPT_RUNS);
            set.runs.drain(..over);
        }
    }
}

/// Puts the working set right after the summary each time a compaction
/// splices one into an agent's conversation, in the conversation and in
/// the agent's turn request built this pass, if any, and starts the read
/// list over. A restore commits nothing, so it carries nothing over again.
fn carry_over(
    // What was committed is read before the splice commits more.
    mut committing: ParamSet<(MessageReader<Committed>, Commit)>,
    mut agents: Query<(&mut Conversation, &mut WorkingSet)>,
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    root: Res<ProjectRoot>,
) {
    let summary = summary_origin();
    let compactions: Vec<Entity> = committing
        .p0()
        .read()
        .filter_map(|change| match change {
            Committed::Splice { agent, removed, .. } if !removed.is_empty() => Some(*agent),
            _ => None,
        })
        .collect();
    for agent in compactions {
        let Ok((mut conversation, mut set)) = agents.get_mut(agent) else {
            continue;
        };
        let Some(mut entry) = conversation.entry(0) else {
            continue;
        };
        let Some(text) = render(&set, &root.0) else {
            continue;
        };
        // The summary's items lead the first message.
        let origins = &entry.origins;
        let leads = |item: &usize| {
            let summarized = |(at, origin): &(usize, Origin)| at == item && *origin == summary;
            origins.iter().any(summarized)
        };
        let after = (0..usize::MAX).take_while(leads).count();
        let message::Message::User { content } = &mut entry.message else {
            continue;
        };
        if after == 0 {
            continue;
        }
        let block = UserContent::text(text);
        content.insert(after, block.clone());
        for (item, _) in &mut entry.origins {
            if *item >= after {
                *item += 1;
            }
        }
        entry.origins.push((after, Origin::plugin("working-set")));
        committing
            .p1()
            .splice(agent, &mut conversation, 0..1, [entry]);
        for (mut call, _) in calls.iter_mut().filter(|(_, to)| to.0 == agent) {
            let history = &mut call.request.chat_history;
            let first = history.iter_mut().find_map(|message| match message {
                message::Message::User { content } => Some(content),
                _ => None,
            });
            if let Some(content) = first {
                content.insert(after.min(content.len()), block.clone());
            }
        }
        set.read.clear();
    }
}

/// The working set message: changed files that exist with their line
/// counts now, the newest read files that exist, and the last test run;
/// `None` when there is nothing to say.
fn render(set: &WorkingSet, root: &Path) -> Option<String> {
    let mut sections = Vec::new();
    let changed: Vec<String> = set
        .changed
        .iter()
        .filter_map(|file| {
            let lines = line_count(&file.path)?;
            let tools = file.tools.join(", ");
            Some(format!(
                "- {}: {lines} lines now (+{} -{} by {tools})",
                relative(&file.path, root),
                file.added,
                file.removed
            ))
        })
        .collect();
    if !changed.is_empty() {
        sections.push(format!(
            "Files changed this session:\n{}",
            changed.join("\n")
        ));
    }
    let read: Vec<String> = set
        .read
        .iter()
        .rev()
        .filter(|file| Path::new(&file.path).is_file())
        .take(SHOWN_READS)
        .map(|file| format!("- {}", relative(&file.path, root)))
        .collect();
    if !read.is_empty() {
        sections.push(format!("Files read, newest first:\n{}", read.join("\n")));
    }
    if let Some(run) = set.runs.last() {
        let how = match run.status {
            ExitStatus::Exited(0) => "passed (exit 0)".to_owned(),
            ExitStatus::Exited(code) => format!("failed (exit {code})"),
            ExitStatus::Signaled => "was killed by a signal".to_owned(),
            ExitStatus::TimedOut => "timed out".to_owned(),
            ExitStatus::Killed => "was killed".to_owned(),
        };
        let tail: Vec<String> = run.tail.iter().map(|line| format!("    {line}")).collect();
        sections.push(format!(
            "Last test/build command: `{}` {how}. Its last output:\n{}",
            run.command,
            tail.join("\n")
        ));
    }
    if sections.is_empty() {
        return None;
    }
    Some(format!(
        "Working set, carried over from before the summary (no need to re-read these to \
         know they exist):\n\n{}",
        sections.join("\n\n")
    ))
}

/// The lines of the file at `path`, if it is one.
fn line_count(path: &str) -> Option<usize> {
    let bytes = std::fs::read(path).ok()?;
    let newlines = bytes.iter().filter(|&&byte| byte == b'\n').count();
    let unterminated = bytes.last().is_some_and(|&byte| byte != b'\n');
    Some(newlines + usize::from(unterminated))
}

/// `path` relative to the project, when it is in it.
fn relative(path: &str, root: &Path) -> String {
    Path::new(path)
        .strip_prefix(root)
        .map_or_else(|_| path.to_owned(), |inner| inner.display().to_string())
}

#[cfg(test)]
mod tests;
