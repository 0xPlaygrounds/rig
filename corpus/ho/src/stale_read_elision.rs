//! Plugin corpus (held-out): H06: Elide superseded file reads from requests
//! (spec `H06-stale-read-elision.md`).
//!
//! Each turn's model request, never the stored conversation, sends a
//! one-line stub in place of a `read` result from an earlier turn that a
//! later read of the same file covers, or that a later successful `edit` or
//! `write` of the file made stale. The calls stay, so every call keeps its
//! result. A Prepare system edits the request as it is admitted, before
//! compaction measures it; `/elided` says what the agent's last request
//! stubbed.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use corpus_shared::few_shot_examples::ExampleSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::tokens_label;
use rig_harness::rig_core::message::{CallId, Message, ToolResult, ToolResultContent};
use rig_harness::rig_core::tool::Tool;
use rig_memory::{HeuristicTokenCounter, TokenCounter};
use rig_tools::fs::resolve;
use rig_tools::{Edit, EditArgs, Read, ReadArgs, Write, WriteArgs};
use serde_json::{Map, Value};

#[cfg(test)]
mod tests;

/// Stubs superseded file reads in model requests.
#[derive(Default)]
pub struct StaleReadElisionPlugin;

/// The plugin's Prepare system, which edits turn requests' `read` results,
/// for others to order against. It runs in [`PrepareSystems::Edit`], after
/// few-shot-examples put its examples in.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElisionSystems;

/// On an agent: what its last turn request stubbed.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Default)]
pub struct Elided {
    /// `read` results stubbed.
    pub results: usize,
    /// About how many tokens that left out.
    pub tokens: usize,
}

impl Plugin for StaleReadElisionPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Elided>()
            .configure_sets(
                AgentPass,
                ElisionSystems
                    .in_set(PrepareSystems::Edit)
                    .after(ExampleSystems),
            )
            .add_systems(AgentPass, elide.in_set(ElisionSystems))
            .add_command(
                "elided",
                "How many superseded file reads the last request left out",
                show,
            );
    }
}

/// The lines of a file a `read` covers, from 1: `end` is `None` to the end.
#[derive(Clone, Copy)]
struct Lines {
    start: usize,
    end: Option<usize>,
}

impl Lines {
    fn of(args: &ReadArgs) -> Self {
        let start = args.offset.unwrap_or(1).max(1);
        let end = args.limit.map(|limit| start.saturating_add(limit));
        Self { start, end }
    }

    /// Whether these lines include all of `other`.
    fn covers(self, other: Self) -> bool {
        let ends_later = match (self.end, other.end) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(end), Some(other)) => other <= end,
        };
        self.start <= other.start && ends_later
    }
}

/// What a successful file tool call did to a file.
enum Access {
    Read(Lines),
    Changed,
}

/// One successful file tool result of a request, in order.
struct FileResult {
    call: CallId,
    /// The path as the model wrote it.
    path: String,
    /// The path as the tool resolved it, in the agent's directory.
    file: PathBuf,
    access: Access,
}

/// Parses `arguments` as the arguments of tool `T`.
fn args<T: serde::de::DeserializeOwned>(arguments: &Map<String, Value>) -> Option<T> {
    serde_json::from_value(Value::Object(arguments.clone())).ok()
}

/// What the successful `result` of the call with `arguments` did to which
/// file, when it is a built-in file tool's working in `dir`.
fn file_result(
    result: &ToolResult,
    arguments: &Map<String, Value>,
    dir: Option<&Path>,
) -> Option<FileResult> {
    if result.is_error {
        return None;
    }
    let (path, access) = match result.name.as_str() {
        Read::NAME => {
            let read: ReadArgs = args(arguments)?;
            let lines = Lines::of(&read);
            (read.path, Access::Read(lines))
        }
        Edit::NAME => (args::<EditArgs>(arguments)?.path, Access::Changed),
        Write::NAME => (args::<WriteArgs>(arguments)?.path, Access::Changed),
        _ => return None,
    };
    Some(FileResult {
        call: result.call.clone(),
        file: resolve(dir, &path),
        path,
        access,
    })
}

/// The stub that replaces `earlier`, a read, when one of `later` supersedes
/// it.
fn stub(earlier: &FileResult, later: &[FileResult]) -> Option<String> {
    let Access::Read(lines) = earlier.access else {
        return None;
    };
    let path = &earlier.path;
    later
        .iter()
        .filter(|later| later.file == earlier.file)
        .find_map(|later| match later.access {
            Access::Read(again) if again.covers(lines) => {
                Some(format!("[superseded: {path} was read again later]"))
            }
            Access::Changed => Some(format!("[superseded: {path} was changed after this read]")),
            Access::Read(_) => None,
        })
}

/// The stub for each `read` result of `messages` answering a call of
/// `eligible` that a later result supersedes, the file tools working in
/// `dir`.
fn stubs(
    messages: &[std::sync::Arc<Message>],
    eligible: &HashSet<CallId>,
    dir: Option<&Path>,
) -> HashMap<CallId, String> {
    let mut arguments = HashMap::new();
    let mut results = Vec::new();
    for message in messages {
        match &**message {
            Message::Assistant(reply) => {
                for block in &reply.content {
                    if let AssistantContent::ToolCall(call) = block {
                        arguments.insert(&call.id, &call.function.arguments);
                    }
                }
            }
            Message::User { content } => {
                for item in content {
                    if let UserContent::ToolResult(result) = item
                        && let Some(called) = arguments.get(&result.call)
                    {
                        results.extend(file_result(result, called, dir));
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
    (results.iter().enumerate())
        .filter(|(_, earlier)| eligible.contains(&earlier.call))
        .filter_map(|(at, earlier)| {
            let stub = stub(earlier, results.get(at + 1..)?)?;
            Some((earlier.call.clone(), stub))
        })
        .collect()
}

/// The ids of the calls whose results `messages` hold.
fn answered(messages: &[std::sync::Arc<Message>]) -> HashSet<CallId> {
    let content = messages.iter().filter_map(|message| match &**message {
        Message::User { content } => Some(content),
        _ => None,
    });
    let results = content.flatten().filter_map(|item| match item {
        UserContent::ToolResult(result) => Some(result.call.clone()),
        _ => None,
    });
    results.collect()
}

/// About how many tokens `result` takes.
fn tokens(result: &ToolResult) -> usize {
    let content = vec![UserContent::ToolResult(result.clone())];
    HeuristicTokenCounter::default().count(&Message::User { content })
}

/// Stubs each turn request's superseded reads from before the turn began,
/// each changed entry an edit by this plugin that keeps its id, and keeps
/// what it stubbed on the agent.
fn elide(
    mut calls: Query<
        (&mut ModelRequest, &CallOf, &ForAgent),
        (Added<Admitted>, ModelCalls, With<TurnCall>),
    >,
    turns: Query<&Turn>,
    agents: Query<(&Conversation, Option<&WorkDir>)>,
    mut commands: Commands,
) {
    for (mut call, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let (Ok(turn), Ok((conversation, dir))) = (turns.get(turn), agents.get(agent)) else {
            continue;
        };
        let Some(history) = &mut call.history else {
            continue;
        };
        let first = conversation.start_of(turn.start);
        let eligible = answered(conversation.messages().get(..first).unwrap_or_default());
        let stubs = stubs(
            history.messages(),
            &eligible,
            dir.map(|WorkDir(dir)| dir.as_path()),
        );
        let mut elided = Elided::default();
        for at in 0..history.messages().len() {
            let Some(mut entry) = history.entry(at) else {
                continue;
            };
            let Message::User { content } = std::sync::Arc::make_mut(&mut entry.message) else {
                continue;
            };
            let mut changed = false;
            for item in content {
                if let UserContent::ToolResult(result) = item
                    && let Some(stub) = stubs.get(&result.call)
                {
                    let before = tokens(result);
                    result.content = vec![ToolResultContent::text(stub.clone())];
                    elided.results += 1;
                    elided.tokens += before.saturating_sub(tokens(result));
                    changed = true;
                }
            }
            if changed {
                history.splice(at..=at, [entry], Author::plugin("stale-read-elision"));
            }
        }
        commands.entity(agent).try_insert(elided);
    }
}

/// `/elided`.
fn show(In(args): In<CommandArgs>, agents: Query<&Elided>, mut notices: MessageWriter<Notice>) {
    let text = match agents.get(args.agent) {
        Ok(elided) => format!(
            "The last request left out {} superseded file reads, about {} tokens.",
            elided.results,
            tokens_label(u64::try_from(elided.tokens).unwrap_or(u64::MAX))
        ),
        Err(_) => "No request was sent yet.".to_owned(),
    };
    notices.write(Notice::info(args.agent, text));
}
