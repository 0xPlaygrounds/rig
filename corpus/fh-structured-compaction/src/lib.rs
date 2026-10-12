//! FH-10: structured compaction. Every compaction rig-compaction makes,
//! near the window, after an overflow or on `/compact <focus>`, is
//! rewritten before its summary call is sent: the summary is asked of the
//! model `RIG_HOME/compact.toml` names (else the agent's own), in a fixed
//! template ([`HEADINGS`]), from the previous summary plus the messages
//! since, focused on what `/compact` was given. Everything before the
//! user's last message is summarized, so tool calls and results before it
//! are dropped and the most recent exchange stays whole; the user's last
//! [`StructuredCompaction::keep_user_messages`] messages stay word for
//! word after the summary. A summary that fails or lacks a heading is
//! asked for once more, then the compaction rig-compaction planned runs
//! as it would have, with a notice.
//!
//! It is a Prepare system on rig-compaction's summary calls
//! ([`Summarizing`]), a Review system that checks their answers and sends
//! them again ([`resend`]), and an observer that puts the kept messages
//! back once the summary went in.

use std::borrow::Cow;
use std::path::Path;

use rig_compaction::{
    Compaction, CompactionPlugin, Summarized, Summarizing, Until, summary_origin,
};
use rig_core::completion::{CompletionRequest, Message};
use rig_core::message::UserContent;
use rig_harness::prelude::*;
use rig_memory::{CompactReason, Summarizer, SummaryLimits, SummaryPrompts, SummaryState};
use rig_tools::attach::attached_file;
use serde::Deserialize;

/// The headings every structured summary has, in order.
pub const HEADINGS: [&str; 6] = [
    "Goal",
    "Constraints the user gave",
    "Decisions and why",
    "Files touched",
    "Current state",
    "Open questions / next steps",
];

/// Summaries asked for in the template before falling back: the first
/// and one retry.
const TRIES: u8 = 2;

/// The summarizer of the template.
const TEMPLATE: Summarizer = Summarizer {
    prompts: SummaryPrompts {
        system: Cow::Borrowed(
            "You condense a conversation between a user and a coding agent into a summary the \
             agent will continue the work from. Keep the user's instructions and the reasons \
             behind what was done. Do not continue the conversation and do not answer \
             questions in it. Output only the summary.",
        ),
        initial: Cow::Borrowed("Summarize the conversation above."),
        update: Cow::Borrowed(
            "The conversation above is the NEW part of a conversation whose earlier part is \
             summarized in <previous-summary>. Write one summary of both: keep everything in \
             the previous summary that still holds, above all the user's constraints and the \
             reasons for decisions, add what is new, and drop only what no longer matters.",
        ),
        format: Cow::Borrowed(
            "\n\nUse exactly these headings, every one of them, writing \"(none)\" under a \
             heading with nothing to say:\n\n\
             ## Goal\n\
             [What the user is trying to get done.]\n\n\
             ## Constraints the user gave\n\
             - [Each instruction, preference or prohibition, in the user's words where short.]\n\n\
             ## Decisions and why\n\
             - **[Decision]**: [Why it was made, and what was ruled out.]\n\n\
             ## Files touched\n\
             - `path`: [one line: what was read or changed there, and why]\n\n\
             ## Current state\n\
             [What works, what is half done, what was last run and how it ended.]\n\n\
             ## Open questions / next steps\n\
             1. [What is unanswered or comes next, in order.]\n\n\
             Keep exact paths, names, commands and error messages.",
        ),
    },
    limits: SummaryLimits::DEFAULT,
};

/// Restructures rig-compaction's compactions; adds rig-compaction unless
/// it is there.
#[derive(Default)]
pub struct StructuredCompactionPlugin;

/// This plugin's Prepare system, which edits rig-compaction's summary
/// calls in [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuredCompactionSystems;

impl Plugin for StructuredCompactionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CompactionPlugin>() {
            app.add_plugins(CompactionPlugin);
        }
        if !app.world().contains_resource::<StructuredCompaction>() {
            let path = RigHome::of(app.world()).root().join("compact.toml");
            let (config, problem) = StructuredCompaction::read(&path);
            app.insert_resource(config);
            if let Some(problem) = problem {
                app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                    notices.write(Notice::warn(None, problem.clone()));
                });
            }
        }
        app.configure_sets(
            AgentPass,
            StructuredCompactionSystems.in_set(PrepareSystems::Edit),
        )
        .add_systems(
            AgentPass,
            (
                structure.in_set(StructuredCompactionSystems),
                check.in_set(ReviewSystems::Act),
            ),
        )
        .add_observer(follow_successor::<Structured>)
        .add_observer(keep_verbatim);
    }
}

/// `RIG_HOME/compact.toml`: `model = "vendor/name"` writes the summaries
/// (the agent's own model when left out), and `keep_user_messages` of the
/// user's last messages stay word for word (3 when left out).
#[derive(Resource, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StructuredCompaction {
    /// The model that writes the summaries, as `vendor/name`.
    pub model: Option<String>,
    /// How many of the user's last messages stay word for word, the last
    /// one with its whole exchange.
    pub keep_user_messages: usize,
}

impl Default for StructuredCompaction {
    fn default() -> Self {
        Self {
            model: None,
            keep_user_messages: 3,
        }
    }
}

impl StructuredCompaction {
    /// The file at `path`, the default when there is none, and what was
    /// wrong with it.
    pub fn read(path: &Path) -> (Self, Option<String>) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return (Self::default(), None);
        };
        match toml::from_str(&text) {
            Ok(config) => (config, None),
            Err(why) => (Self::default(), Some(format!("{}: {why}", path.display()))),
        }
    }
}

/// On a summary call this plugin restructured: how many summaries were
/// asked for in the template, what rig-compaction planned (the fallback),
/// and the user's messages to keep after the summary.
#[derive(Component, Clone)]
pub struct Structured {
    /// Summaries asked for in the template so far; more than `TRIES`
    /// means the fallback runs.
    tries: u8,
    /// rig-compaction's own request, cut and state.
    planned: (ModelRequest, Until, SummaryState),
    /// The user's messages kept word for word, oldest first.
    verbatim: Vec<String>,
}

/// On an agent whose structured summary was accepted: the user's messages
/// to put after it once it is in the conversation.
#[derive(Component, Clone, Debug)]
struct KeepVerbatim(Vec<String>);

/// What a structured compaction asks and keeps.
struct Plan {
    request: CompletionRequest,
    kept: MessageId,
    state: SummaryState,
    verbatim: Vec<String>,
}

/// Whether `text` has every heading of [`HEADINGS`] as a heading line.
pub fn has_headings(text: &str) -> bool {
    HEADINGS.iter().all(|heading| {
        text.lines().map(str::trim).any(|line| {
            line.starts_with('#')
                && line
                    .trim_start_matches('#')
                    .trim()
                    .eq_ignore_ascii_case(heading)
        })
    })
}

/// The messages of `conversation` without its summary items, and how many
/// of the first held nothing else (as rig-compaction reads them, privately).
fn unsummarized(conversation: &Conversation) -> (Vec<Message>, usize) {
    let summary = summary_origin();
    let mut from = 0;
    let mut messages = conversation.messages().to_vec();
    for (at, message) in messages.iter_mut().enumerate() {
        let Message::User { content } = message else {
            continue;
        };
        let mut item = 0;
        content.retain(|_| {
            item += 1;
            conversation.origin(at, item - 1) != Some(&summary)
        });
        if content.is_empty() && from == at {
            from += 1;
        }
    }
    (messages, from)
}

/// Whether message `at` is one the user typed that answers no tool call,
/// so a request may start with it.
fn starts_exchange(conversation: &Conversation, at: usize) -> bool {
    let answers_a_call = match conversation.messages().get(at) {
        Some(Message::User { content }) => content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        _ => true,
    };
    !answers_a_call && typed(conversation, at).is_some()
}

/// The text items the user typed before message `before`, the newest
/// `count`, oldest first, as they were typed.
fn user_texts(conversation: &Conversation, before: usize, count: usize) -> Vec<String> {
    let messages = conversation.messages().get(..before).unwrap_or_default();
    let mut texts = Vec::new();
    for (at, message) in messages.iter().enumerate().rev() {
        let Message::User { content } = message else {
            continue;
        };
        for (item, content) in content.iter().enumerate().rev() {
            let from_user = conversation
                .origin(at, item)
                .is_none_or(|origin| origin.kind == OriginKind::User);
            if let UserContent::Text(text) = content
                && from_user
                && attached_file(&text.text).is_none()
                && !text.text.trim().is_empty()
                && texts.len() < count
            {
                texts.push(text.text.clone());
            }
        }
    }
    texts.reverse();
    texts
}

/// The structured compaction of `conversation` for `reason`, written by
/// a model of `spec`: everything before the user's last message, from the
/// previous summary on. `None` when nothing before it is unsummarized.
fn plan(
    conversation: &Conversation,
    summarized: Option<&Summarized>,
    (reason, spec): (&CompactReason, &rig_core::catalog::ModelSpec),
    (config, policy): (&StructuredCompaction, &Compaction),
) -> Option<Plan> {
    let (messages, from) = unsummarized(conversation);
    let cut = (from + 1..messages.len())
        .rev()
        .find(|&at| starts_exchange(conversation, at))?;
    let older = messages.get(from..cut)?;
    let previous = summarized.map_or("", |summarized| summarized.0.summary.as_str());
    let focus = match reason {
        CompactReason::Asked { focus } => focus.trim(),
        CompactReason::Threshold | CompactReason::Overflow => "",
    };
    let request = TEMPLATE.request(older, previous, focus, spec).ok()?;
    let mut state = summarized
        .map(|summarized| summarized.0.clone())
        .unwrap_or_default();
    state.track(older, &policy.0.tracked);
    let keep = config.keep_user_messages.saturating_sub(1);
    Some(Plan {
        request,
        kept: conversation.id(cut)?,
        state,
        verbatim: user_texts(conversation, cut, keep),
    })
}

/// Rewrites each summary call rig-compaction admits into the template, on
/// the configured model, cut before the user's last message. A call sent
/// again keeps its request, unless it is the fallback, which gets back
/// what rig-compaction planned.
fn structure(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &mut Summarizing,
            &ForAgent,
            Option<&Structured>,
        ),
        (Added<Admitted>, ModelCalls, Without<TurnCall>),
    >,
    agents: Query<(&Conversation, Option<&Summarized>, Option<&Connection>)>,
    (config, policy, models): (Res<StructuredCompaction>, Res<Compaction>, Res<Models>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, mut summarizing, &ForAgent(agent), structured) in &mut calls {
        if let Some(structured) = structured {
            if structured.tries > TRIES {
                let (request, until, state) = structured.planned.clone();
                *model = request;
                summarizing.until = until;
                summarizing.state = state;
                commands.entity(agent).remove::<KeepVerbatim>();
            }
            continue;
        }
        let Ok((conversation, summarized, own)) = agents.get(agent) else {
            continue;
        };
        let connection = config.model.as_deref().and_then(|name| {
            models
                .connect(name)
                .inspect_err(|why| {
                    let text =
                        format!("Cannot reach the summary model {name}: {why}; using the agent's.");
                    notices.write(Notice::warn(agent, text));
                })
                .ok()
        });
        let Some(spec) = connection.as_ref().or(own).map(|to| to.spec.clone()) else {
            continue;
        };
        let reason = (&summarizing.reason, &*spec);
        let Some(plan) = plan(conversation, summarized, reason, (&config, &policy)) else {
            continue;
        };
        let planned = (model.clone(), summarizing.until, summarizing.state.clone());
        model.request = plan.request;
        model.connection = connection;
        summarizing.until = Until::Before(plan.kept);
        summarizing.state = plan.state;
        commands.entity(agent).remove::<KeepVerbatim>();
        commands.entity(call).insert(Structured {
            tries: 1,
            planned,
            verbatim: plan.verbatim,
        });
    }
}

/// Checks each structured summary as it is answered: one with every
/// heading goes in, with the user's messages to keep; a failed one or one
/// missing a heading is asked for again, once, then rig-compaction's own
/// compaction runs in its place, with a notice.
fn check(
    mut calls: Query<
        (Entity, &Outcome, &ForAgent, &mut Structured),
        (
            Unreviewed,
            ModelCalls,
            With<Summarizing>,
            Without<SupersededBy>,
        ),
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, outcome, &ForAgent(agent), mut structured) in &mut calls {
        if structured.tries > TRIES {
            continue;
        }
        let summary = outcome.completion().map(Summarizer::summary_text);
        if summary.is_some_and(|summary| summary.is_ok_and(|text| has_headings(&text))) {
            let verbatim = KeepVerbatim(structured.verbatim.clone());
            commands.entity(agent).insert(verbatim);
            continue;
        }
        structured.tries += 1;
        if structured.tries > TRIES {
            notices.write(Notice::warn(
                agent,
                "The summary came back without its structure twice; compacting the default way.",
            ));
        }
        commands.queue(resend(call));
    }
}

/// Once rig-compaction put an accepted structured summary in the
/// conversation, puts the user's kept messages right after it, as they
/// were typed. rig-compaction inserts [`Summarized`] after its splice.
fn keep_verbatim(
    summarized: On<Insert<Summarized>>,
    mut agents: Query<(&mut Conversation, &KeepVerbatim)>,
    mut commit: Commit,
    mut commands: Commands,
) {
    let agent = summarized.entity;
    let Ok((mut conversation, KeepVerbatim(texts))) = agents.get_mut(agent) else {
        return;
    };
    let texts = texts.clone();
    commands.entity(agent).remove::<KeepVerbatim>();
    let summary = summary_origin();
    let found = (0..conversation.messages().len()).find_map(|at| {
        let entry = conversation.entry(at)?;
        let (item, _) = entry
            .origins
            .iter()
            .find(|(_, origin)| *origin == summary)?;
        Some((at, *item + 1, entry))
    });
    let Some((at, after, mut entry)) = found else {
        return;
    };
    let Message::User { content } = &mut entry.message else {
        return;
    };
    if after > content.len() {
        return;
    }
    let added = texts.len();
    content.splice(after..after, texts.into_iter().map(UserContent::text));
    for (item, _) in &mut entry.origins {
        if *item >= after {
            *item += added;
        }
    }
    commit.splice(agent, &mut conversation, at..=at, [entry]);
}

#[cfg(test)]
mod tests;
