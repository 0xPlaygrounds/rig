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
//! It sets rig-compaction's default policy ([`CompactionDefault`]: its
//! summary prompts and its cut, [`CUT`], which it adds as a named cut),
//! under each agent's own knobs, and adds a Prepare system that sends
//! rig-compaction's summary calls to the configured model, a Review system
//! that checks their answers and sends them again ([`resend`]), and an
//! observer that puts the kept messages after the summary once it is in. In the terminal view the
//! summary is a collapsible "Summary" entry.

use std::borrow::Cow;

use rig_compaction::{
    AppCompactionExt, Compaction, CompactionCuts, CompactionDefault, CompactionPlugin, Summarizing,
    plan, policy_of, summarized_prefix,
};
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::message::UserContent;
use rig_harness::prelude::*;
use rig_memory::{CompactReason, Compacted, CompactionPolicy, Cut, Summarizer, SummaryPrompts};
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

/// The name of the cut this plugin adds: before the user's last message.
pub const CUT: &str = "before-last-user";

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
            .add_compaction_cut(CUT, before_last_user)
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

    /// Puts the template and the cut in rig-compaction's default policy,
    /// once every plugin built, keeping the default as it was for the
    /// fallback.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let Some(mut default) = world.get_resource_mut::<CompactionDefault>() else {
            return;
        };
        let fallback = Fallback(default.clone());
        default.0.summarizer.prompts = TEMPLATE;
        default.0.cut = Cut::Named(CUT.to_owned());
        world.insert_resource(fallback);
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

/// rig-compaction's default policy as it was before this plugin, under
/// each agent's own knobs the fallback compacts with.
#[derive(Resource)]
struct Fallback(CompactionDefault);

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

/// Sends each summary call rig-compaction admits to the configured model,
/// planned again for that model's window; the fallback gets back the
/// agent's model, the default prompts and cut.
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
    agents: Query<(&Conversation, Option<&Compaction>, Option<&Connection>)>,
    (config, default, cuts, fallback, models): (
        Res<StructuredCompaction>,
        Res<CompactionDefault>,
        Res<CompactionCuts>,
        Res<Fallback>,
        Res<Models>,
    ),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, mut summarizing, &ForAgent(agent), structured) in &mut calls {
        let Ok((conversation, knobs, own)) = agents.get(agent) else {
            continue;
        };
        let replan = |policy: &CompactionPolicy, spec: &ModelSpec| {
            let reason = summarizing.reason.clone();
            plan(conversation, policy, &cuts.0, reason, spec)
        };
        match structured {
            None => {
                if let Some(name) = config.model.as_deref() {
                    let connected = models.connect(name).map_err(|why| why.to_string());
                    let routed = connected.and_then(|connection| {
                        let planned = replan(&policy_of(knobs, &default), &connection.spec)?;
                        Ok((connection, planned))
                    });
                    match routed {
                        Ok((connection, planned)) => {
                            if let Some((planned, request)) = planned {
                                *summarizing = planned;
                                *model = ModelRequest::new(request);
                            }
                            model.connection = Some(connection);
                        }
                        Err(why) => {
                            let text = format!(
                                "Cannot use the summary model {name}: {why}; using the agent's."
                            );
                            notices.write(Notice::warn(agent, text));
                        }
                    }
                }
                let count = config.keep_user_messages.saturating_sub(1);
                let cut = summarizing
                    .upto
                    .and_then(|kept| conversation.index_of(kept));
                let verbatim = cut.map(|cut| last_typed(conversation, cut, count));
                let verbatim = verbatim.unwrap_or_default();
                commands
                    .entity(call)
                    .insert(Structured { tries: 1, verbatim });
                commands.entity(agent).remove::<KeepVerbatim>();
            }
            Some(structured) if structured.tries > TRIES => {
                let Some(own) = own else {
                    continue;
                };
                match replan(&policy_of(knobs, &fallback.0), &own.spec) {
                    Ok(Some((planned, request))) => {
                        *summarizing = planned;
                        *model = ModelRequest::new(request);
                    }
                    Ok(None) => {}
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
/// conversation, which it does as the summary call settles, puts the
/// user's kept messages right after it, each its own message, as they
/// were typed.
fn keep_verbatim(
    settled: On<Add<Settled>>,
    calls: Query<(&ForAgent, &Summarizing), With<Structured>>,
    mut commands: Commands,
) {
    let Ok((&ForAgent(agent), summarizing)) = calls.get(settled.entity) else {
        return;
    };
    let Some(summary) = summarizing.summary.clone() else {
        return;
    };
    // Queued, so it runs once every observer of the settled call did.
    commands.queue(move |world: &mut World| put_verbatim(world, agent, &summary));
}

/// Puts the agent's [`KeepVerbatim`] messages after its summaries, when its
/// newest compaction took `summary`.
fn put_verbatim(world: &mut World, agent: Entity, summary: &str) {
    let Ok(mut agent) = world.get_entity_mut(agent) else {
        return;
    };
    let Some(KeepVerbatim(texts)) = agent.take::<KeepVerbatim>() else {
        return;
    };
    let Some(mut conversation) = agent.get_mut::<Conversation>() else {
        return;
    };
    let newest = Compacted::of(&conversation).last();
    if newest.is_none_or(|(_, compacted)| compacted.state.summary != summary) {
        return;
    }
    let after = summarized_prefix(&conversation);
    if after == 0 {
        return;
    }
    let kept = texts
        .into_iter()
        .map(|text| Entry::new(Message::user(text)));
    let by = Author::plugin("fh-structured-compaction");
    conversation.splice(after..after, kept, by);
}

#[cfg(feature = "tui")]
mod tui;

#[cfg(test)]
mod tests;
