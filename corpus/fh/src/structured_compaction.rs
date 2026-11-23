//! Structured compaction. Every compaction `rig_basics::compaction` makes, near the window, after
//! an overflow or on `/compact <focus>`, summarizes in a fixed template ([`HEADINGS`]), from the
//! previous summary plus the messages since, focused on what `/compact` was given, on the model the
//! `[compact]` table of `RIG_HOME/settings.toml` names (else the agent's own). Everything before
//! the oldest of the user's last [`StructuredCompaction::keep_user_messages`] messages is
//! summarized; those messages and all that follows them stay in the conversation as they are,
//! attachments, tool calls and results included. With nothing before them, the compaction is
//! refused with a notice. A summary that fails or lacks a heading is asked for once more, then
//! the agent's model writes the default summary of the same messages, with a notice.
//!
//! It replaces the requests of `rig_basics::compaction`'s summary calls ([`Summarizing`]) with its
//! own. In the terminal view the summary is a "Summary" entry.

use rig_basics::compaction::{CompactReason, CompactionPlugin, Summarizing};
use rig_core::catalog::ModelSpec;
use rig_core::completion::{CompletionRequest, Message, UnsupportedOption};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::prelude::*;
use rig_memory::{SummaryState, serialize_conversation, summarization_request, summary_text};
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

/// What the template's summarizer is told.
const SYSTEM: &str = "You condense a conversation between a user and a coding agent into a \
    summary the agent will continue the work from. Keep the user's instructions and the reasons \
    behind what was done. Do not continue the conversation and do not answer questions in it. \
    Output only the summary.";

/// What the template's summarizer is asked without a previous summary.
const INITIAL: &str = "Summarize the conversation.";

/// What the template's summarizer is asked with a previous summary.
const UPDATE: &str = "Summarize the conversation. An earlier part of it is summarized already, in \
    <previous-summary>. Write one summary of both: keep everything in the previous summary that \
    still holds, above all the user's constraints and the reasons for decisions, add what is new, \
    and drop only what no longer matters.";

/// The template's format.
const FORMAT: &str = "\n\nUse exactly these headings, every one of them, writing \"(none)\" \
    under a heading with nothing to say:\n\n\
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
    Keep exact paths, names, commands and error messages.";

/// The template's request to `spec` for `summarizing`: its messages as text, the previous summary
/// and the focus.
fn template_request(
    summarizing: &Summarizing,
    spec: &ModelSpec,
) -> Result<CompletionRequest, UnsupportedOption> {
    let mut ask = format!(
        "<conversation>\n{}\n</conversation>",
        serialize_conversation(&summarizing.messages)
    );
    let previous = summarizing.previous.as_ref();
    let previous = previous.and_then(SummaryState::previous_summary);
    if let Some(previous) = &previous {
        ask.push_str(&format!(
            "\n\n<previous-summary>\n{previous}\n</previous-summary>"
        ));
    }
    ask.push_str("\n\n");
    ask.push_str(match previous {
        Some(_) => UPDATE,
        None => INITIAL,
    });
    let focus = summarizing.reason.focus();
    if !focus.is_empty() {
        ask.push_str(&format!(
            "\n\nThe user asked the summary to keep, above all: {focus}"
        ));
    }
    ask.push_str(FORMAT);
    summarization_request((SYSTEM, ask), spec, None, None)
}

/// Restructures `rig_basics::compaction`'s compactions; adds `rig_basics::compaction` unless
/// it is there.
#[derive(Default)]
pub struct StructuredCompactionPlugin;

/// This plugin's Prepare system, which picks the model of `rig_basics::compaction`'s
/// summary calls in [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuredCompactionSystems;

impl Plugin for StructuredCompactionPlugin {
    fn build(&self, app: &mut App) {
        rig_harness::load::<CompactionPlugin>(app, "rig-basics", "");
        app.add_setting::<StructuredCompaction>("compact")
            .configure_sets(
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
            .add_observer(follow_successor::<Structured>);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// The `[compact]` table of `RIG_HOME/settings.toml`: `model =
/// "vendor/name"` writes the summaries (the agent's own model when left
/// out), and `keep_user_messages` of the user's last messages stay as they
/// are, with all that follows them (3 when left out).
#[derive(Resource, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StructuredCompaction {
    /// The model that writes the summaries, as `vendor/name`.
    pub model: Option<String>,
    /// How many of the user's last messages stay, each with its whole
    /// exchange; the last one always does, so 0 keeps as 1 does.
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

/// On a summary call in the template: how many summaries were asked for
/// so far (more than [`TRIES`] means the fallback runs).
#[derive(Component, Clone)]
struct Structured {
    tries: u8,
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

/// The first summary call of a compaction of `conversation` for `reason` that keeps the user's
/// last `keep` messages, each with all that follows it; where a tool exchange spans the oldest,
/// from the nearest of the user's messages before it. `None` when nothing comes before them.
fn plan_kept(
    conversation: &Conversation,
    reason: &CompactReason,
    keep: usize,
) -> Option<Summarizing> {
    let typed = conversation
        .entries()
        .filter(|entry| entry.is_users_own() && matches!(entry.message, Message::User { .. }));
    let typed: Vec<usize> = typed.map(|entry| entry.index).collect();
    let oldest = typed.len().saturating_sub(keep.max(1));
    let mut cuts = typed.get(..=oldest)?.iter().rev();
    cuts.find_map(|&at| Summarizing::plan_at(conversation, at, reason.clone()))
}

/// Plans each summary call `rig_basics::compaction` admits again, cut before the user's last
/// messages ([`plan_kept`]), in the template, for the configured model's window, and sends it
/// there; one it cannot plan, of a compaction that had something to summarize, is refused, so
/// none of the kept messages is summarized. The fallback asks the agent's model for the default
/// summary of the same messages.
fn structure(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &mut Summarizing,
            &ForAgent,
            Option<&Structured>,
        ),
        (Added<Admitted>, ModelCalls),
    >,
    agents: Query<(&Conversation, Option<&Connection>)>,
    (config, models): (Res<StructuredCompaction>, Res<Models>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, mut summarizing, &ForAgent(agent), structured) in &mut calls {
        let Ok((conversation, own)) = agents.get(agent) else {
            continue;
        };
        match structured {
            None => {
                let routed = config.model.as_deref().and_then(|name| {
                    let connected = models
                        .connect(name)
                        .map_err(|why| Models::cannot_use(name, why));
                    connected
                        .map_err(|why| {
                            let text = format!(
                                "Cannot use the summary model {name}: {why}; using the agent's."
                            );
                            notices.write(Notice::warn(agent, text));
                        })
                        .ok()
                });
                let Some(spec) = routed.as_ref().or(own).map(|it| &it.spec) else {
                    continue;
                };
                let keep = config.keep_user_messages;
                let planned = plan_kept(conversation, &summarizing.reason, keep);
                let planned = planned.ok_or_else(|| {
                    let keep = keep.max(1);
                    format!("nothing comes before the user's last {keep} messages")
                });
                let request = planned.and_then(|planned| {
                    let request = template_request(&planned, spec);
                    request
                        .map(|request| (planned, request))
                        .map_err(|why| why.to_string())
                });
                match request {
                    Ok((planned, request)) => {
                        *summarizing = planned;
                        *model = ModelRequest::new(request);
                        model.connection = routed;
                        commands.entity(call).insert(Structured { tries: 1 });
                    }
                    // A `/compact` `rig_basics::compaction` found nothing in, which it refuses.
                    Err(_) if summarizing.messages.is_empty() => {}
                    Err(why) => {
                        let why = ErrorReport::new(ErrorKind::Cancelled, why);
                        commands.entity(call).try_insert_if_new(Outcome(Err(why)));
                    }
                }
            }
            Some(structured) if structured.tries > TRIES => {
                let Some(own) = own else {
                    continue;
                };
                match summarizing.request(&own.spec, None) {
                    Ok(request) => *model = ModelRequest::new(request),
                    Err(why) => {
                        notices.write(Notice::warn(agent, format!("Cannot compact: {why}.")));
                    }
                }
            }
            Some(_) => {}
        }
    }
}

/// Checks each structured summary as it is answered: one with every
/// heading goes in; a failed one or one
/// missing a heading is asked for again, once, then the fallback runs in
/// its place, with a notice.
fn check(
    mut calls: Query<
        (Entity, &Outcome, &ForAgent, &mut Structured),
        (ReadyToSettle, ModelCalls, Without<SupersededBy>),
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, outcome, &ForAgent(agent), mut structured) in &mut calls {
        if structured.tries > TRIES {
            continue;
        }
        let summary = outcome
            .completion()
            .map(|reply| summary_text(reply, "Summarization"));
        if summary.is_some_and(|summary| summary.is_ok_and(|text| has_headings(&text))) {
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

#[cfg(feature = "tui")]
mod tui;

#[cfg(test)]
mod tests;
