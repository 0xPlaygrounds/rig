//! FH-10: structured compaction. Every compaction rig-compaction makes,
//! near the window, after an overflow or on `/compact <focus>`, summarizes
//! in a fixed template ([`HEADINGS`]), from the previous summary plus the
//! messages since, focused on what `/compact` was given, on the model the
//! `[compact]` table of `RIG_HOME/settings.toml` names (else the agent's
//! own). Everything before the user's last message is summarized, so tool
//! calls and results before it are dropped and the most recent exchange
//! stays whole; the user's last [`StructuredCompaction::keep_user_messages`]
//! messages stay word for word after the summary. A summary that fails or
//! lacks a heading is asked for once more, then the compaction runs the
//! way it would have without this plugin, with a notice.
//!
//! It sets rig-compaction's policy (its summary prompts and its [`Cut`]),
//! and adds a Prepare system that sends rig-compaction's summary calls to
//! the configured model, a Review system that checks their answers and
//! sends them again ([`resend`]), and an observer that puts the kept
//! messages after the summary once it is in. In the terminal view the
//! summary is a collapsible "Summary" entry.

use std::borrow::Cow;
use std::sync::Arc;

use rig_compaction::{
    Compaction, CompactionPlugin, Summarized, Summarizing, Until, summary_origin, unsummarized,
};
use rig_core::catalog::ModelSpec;
use rig_core::completion::{CompletionRequest, Message};
use rig_core::message::UserContent;
use rig_harness::prelude::*;
use rig_memory::{CompactReason, CompactionPolicy, Cut, Summarizer, SummaryPrompts, SummaryState};
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

/// What the template's summarizer asks.
const TEMPLATE: SummaryPrompts = SummaryPrompts {
    system: Cow::Borrowed(
        "You condense a conversation between a user and a coding agent into a summary the \
         agent will continue the work from. Keep the user's instructions and the reasons \
         behind what was done. Do not continue the conversation and do not answer questions \
         in it. Output only the summary.",
    ),
    initial: Cow::Borrowed("Summarize the conversation above."),
    update: Cow::Borrowed(
        "The conversation above is the NEW part of a conversation whose earlier part is \
         summarized in <previous-summary>. Write one summary of both: keep everything in the \
         previous summary that still holds, above all the user's constraints and the reasons \
         for decisions, add what is new, and drop only what no longer matters.",
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
};

/// Restructures rig-compaction's compactions; adds rig-compaction unless
/// it is there.
#[derive(Default)]
pub struct StructuredCompactionPlugin;

/// This plugin's Prepare system, which picks the model of rig-compaction's
/// summary calls in [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuredCompactionSystems;

impl Plugin for StructuredCompactionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CompactionPlugin>() {
            app.add_plugins(CompactionPlugin);
        }
        app.add_setting::<StructuredCompaction>("compact")
            .insert_resource(Fallback {
                prompts: SummaryPrompts::DEFAULT,
                cut: Cut::Budget,
            })
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
            .add_observer(follow_successor::<Structured>)
            .add_observer(keep_verbatim);
    }

    /// Puts the template and the cut in rig-compaction's policy, once every
    /// plugin built, keeping what they replace for the fallback.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let Some(mut compaction) = world.get_resource_mut::<Compaction>() else {
            return;
        };
        let policy = &mut compaction.0;
        let prompts = std::mem::replace(&mut policy.summarizer.prompts, TEMPLATE);
        let cut = std::mem::replace(&mut policy.cut, Cut::With(Arc::new(before_last_user)));
        world.insert_resource(Fallback { prompts, cut });
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// The `[compact]` table of `RIG_HOME/settings.toml`: `model =
/// "vendor/name"` writes the summaries (the agent's own model when left
/// out), and `keep_user_messages` of the user's last messages stay word
/// for word (3 when left out).
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

/// The summary prompts and cut rig-compaction's policy had before this
/// plugin, which the fallback compacts with.
#[derive(Resource)]
struct Fallback {
    prompts: SummaryPrompts,
    cut: Cut,
}

/// On a summary call in the template: how many summaries were asked for
/// so far (more than [`TRIES`] means the fallback runs), and the user's
/// messages to keep after it.
#[derive(Component, Clone)]
struct Structured {
    tries: u8,
    verbatim: Vec<String>,
}

/// On an agent whose structured summary was accepted: the user's messages
/// to put after it once it is in the conversation.
#[derive(Component, Clone, Debug)]
struct KeepVerbatim(Vec<String>);

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

/// The cut: before the last user message that answers no tool call, so
/// the most recent exchange stays whole.
fn before_last_user(
    messages: &[Message],
    _: usize,
    _: &CompactReason,
    _: Option<&ModelSpec>,
) -> Option<usize> {
    messages.iter().rposition(|message| match message {
        Message::User { content } => !content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        Message::Assistant { .. } | Message::System { .. } => false,
    })
}

/// What the user typed in the `count` messages before message `before`,
/// one text a message, oldest first.
fn last_typed(conversation: &Conversation, before: usize, count: usize) -> Vec<String> {
    let mut texts: Vec<String> = (0..before)
        .rev()
        .filter_map(|at| typed(conversation, at))
        .take(count)
        .collect();
    texts.reverse();
    texts
}

/// The compaction `policy` with the [`Fallback`]'s prompts and cut would
/// make: its request, where it ends and the state it goes into.
fn fallback(
    policy: CompactionPolicy,
    conversation: &Conversation,
    summarized: Option<&Summarized>,
    (reason, spec): (&CompactReason, &ModelSpec),
) -> Option<(CompletionRequest, Until, SummaryState)> {
    let (messages, from) = unsummarized(conversation);
    let upto = policy.cut(&messages, from, reason, Some(spec))?;
    let none = SummaryState::default();
    let state = summarized.map_or(&none, |summarized| &summarized.0);
    let planned = policy.plan(state, &messages, from, upto, spec, reason);
    let (state, request) = planned.ok()?;
    let until = conversation.id(upto).map_or(Until::All, Until::Before);
    Some((request, until, state))
}

/// Sends each summary call rig-compaction admits to the configured model;
/// the fallback gets back the agent's model, the default prompts and cut.
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
    agents: Query<(&Conversation, Option<&Summarized>, Option<&Connection>)>,
    (config, compaction, before, models): (
        Res<StructuredCompaction>,
        Res<Compaction>,
        Res<Fallback>,
        Res<Models>,
    ),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, mut summarizing, &ForAgent(agent), structured) in &mut calls {
        match structured {
            None => {
                let cut = match summarizing.until {
                    Until::Before(kept) => agents.get(agent).ok().and_then(|(conversation, ..)| {
                        Some((conversation, conversation.index_of(kept)?))
                    }),
                    Until::All => None,
                };
                let count = config.keep_user_messages.saturating_sub(1);
                let verbatim = cut.map(|(conversation, cut)| last_typed(conversation, cut, count));
                let verbatim = verbatim.unwrap_or_default();
                commands
                    .entity(call)
                    .insert(Structured { tries: 1, verbatim });
                commands.entity(agent).remove::<KeepVerbatim>();
                let Some(name) = config.model.as_deref() else {
                    continue;
                };
                match models.connect(name) {
                    Ok(connection) => model.connection = Some(connection),
                    Err(why) => {
                        let text = format!(
                            "Cannot reach the summary model {name}: {why}; using the agent's."
                        );
                        notices.write(Notice::warn(agent, text));
                    }
                }
            }
            Some(structured) if structured.tries > TRIES => {
                let Ok((conversation, summarized, Some(own))) = agents.get(agent) else {
                    continue;
                };
                let policy = CompactionPolicy {
                    summarizer: Summarizer {
                        prompts: before.prompts.clone(),
                        ..compaction.0.summarizer.clone()
                    },
                    cut: before.cut.clone(),
                    ..compaction.0.clone()
                };
                let reason = (&summarizing.reason, &*own.spec);
                if let Some((request, until, state)) =
                    fallback(policy, conversation, summarized, reason)
                {
                    *model = ModelRequest::new(request);
                    summarizing.until = until;
                    summarizing.state = state;
                }
            }
            Some(_) => {}
        }
    }
}

/// Checks each structured summary as it is answered: one with every
/// heading goes in, with the user's messages to keep; a failed one or one
/// missing a heading is asked for again, once, then the fallback runs in
/// its place, with a notice.
fn check(
    mut calls: Query<
        (Entity, &Outcome, &ForAgent, &mut Structured),
        (Unreviewed, ModelCalls, Without<SupersededBy>),
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
/// conversation (it inserts [`Summarized`] after its splice), puts the
/// user's kept messages right after it, as they were typed.
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
        let &(item, _) = entry
            .origins
            .iter()
            .find(|(_, origin)| *origin == summary)?;
        Some((at, item + 1, entry))
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

#[cfg(feature = "tui")]
mod tui;

#[cfg(test)]
mod tests;
