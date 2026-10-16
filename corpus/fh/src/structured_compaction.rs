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
//! It sets the summary prompts of rig-compaction's default policy
//! ([`CompactionDefault`]). It adds a Prepare system that plans
//! rig-compaction's summary calls again, cut before the user's last
//! message, and sends them to the configured model, a
//! Review system that checks their answers and sends them again
//! ([`resend`]), and an observer that puts the kept messages after the
//! summary as it goes in. In the terminal view the summary is a
//! "Summary" entry.

use std::borrow::Cow;

use rig_compaction::{
    CompactionDefault, CompactionPlugin, Summarizing, plan, summarized_prefix, summary_state,
};
use rig_core::catalog::ModelSpec;
use rig_core::completion::{CompletionRequest, Message};
use rig_core::message::UserContent;
use rig_harness::prelude::*;
use rig_memory::{CompactReason, Compacted, CompactionPolicy, Summarizer, SummaryPrompts};
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
    ..SummaryPrompts::DEFAULT
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
        let world = app.world_mut();
        if let Some(mut default) = world.get_resource_mut::<CompactionDefault>() {
            let policy = &mut default.0;
            let fallback = Fallback {
                prompts: std::mem::replace(&mut policy.summarizer.prompts, TEMPLATE),
            };
            world.insert_resource(fallback);
        }
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
            .add_observer(follow_successor::<Structured>)
            .add_observer(keep_verbatim);
    }

    /// Draws summaries as a "Summary" entry once rig-tui's renderers are in.
    #[cfg(feature = "tui")]
    fn finish(&self, app: &mut App) {
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

/// The default policy's prompts as they were before this plugin: the
/// fallback compacts with them, cut as rig-compaction cuts.
#[derive(Resource)]
struct Fallback {
    prompts: SummaryPrompts,
}

impl Fallback {
    /// `default` with these prompts.
    fn over(&self, default: &CompactionDefault) -> CompactionPolicy {
        let mut policy = default.0.clone();
        policy.summarizer.prompts = self.prompts.clone();
        policy
    }
}

/// On a summary call in the template: how many summaries were asked for
/// so far (more than [`TRIES`] means the fallback runs), and the user's
/// messages to keep after it.
#[derive(Component, Clone)]
struct Structured {
    tries: u8,
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

/// The cut: before the last user message that answers no tool call, so
/// the most recent exchange stays whole.
fn before_last_user(messages: &[std::sync::Arc<Message>]) -> Option<usize> {
    messages.iter().rposition(|message| match &**message {
        Message::User { content } => !content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        Message::Assistant { .. } | Message::System { .. } => false,
    })
}

/// Plans a summary of `conversation` for `reason` by `policy`, written by
/// `spec`, cut [`before_last_user`]: `None` when nothing new would be
/// summarized.
fn plan_structured(
    conversation: &Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    spec: &ModelSpec,
) -> Result<Option<(Summarizing, CompletionRequest)>, String> {
    let messages = conversation.messages();
    let from = summarized_prefix(conversation);
    let upto = before_last_user(messages).filter(|&upto| upto > from && upto <= messages.len());
    let Some(upto) = upto else {
        return Ok(None);
    };
    let state = summary_state(conversation);
    let planned = policy.plan(&state, messages, from, upto, spec, &reason);
    let (state, request) = planned.map_err(|why| why.to_string())?;
    let keep_from = conversation.id(upto);
    let summarizing = Summarizing {
        reason,
        keep_from,
        state,
        summary: None,
    };
    Ok(Some((summarizing, request)))
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

/// Plans each summary call rig-compaction admits again, cut before the
/// user's last message, for the configured model's window, and sends it
/// there; the fallback gets back the agent's model, the default prompts
/// and rig-compaction's cut.
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
    (config, default, fallback, models): (
        Res<StructuredCompaction>,
        Res<CompactionDefault>,
        Res<Fallback>,
        Res<Models>,
    ),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, mut summarizing, &ForAgent(agent), structured) in &mut calls {
        let Ok((conversation, own)) = agents.get(agent) else {
            continue;
        };
        let reason = summarizing.reason.clone();
        match structured {
            None => {
                let routed = config.model.as_deref().and_then(|name| {
                    let connected = models.connect(name);
                    connected
                        .map_err(|why| {
                            let text = format!(
                                "Cannot use the summary model {name}: {why}; using the agent's."
                            );
                            notices.write(Notice::warn(agent, text));
                        })
                        .ok()
                });
                let spec = routed.as_ref().or(own).map(|connection| &connection.spec);
                let planned =
                    spec.map(|spec| plan_structured(conversation, &default.0, reason, spec));
                match planned {
                    Some(Ok(Some((planned, request)))) => {
                        *summarizing = planned;
                        *model = ModelRequest::new(request);
                    }
                    Some(Err(why)) => {
                        notices.write(Notice::warn(agent, format!("Cannot compact: {why}.")));
                    }
                    Some(Ok(None)) | None => {}
                }
                if let Some(connection) = routed {
                    model.connection = Some(connection);
                }
                let count = config.keep_user_messages.saturating_sub(1);
                let cut = summarizing
                    .keep_from
                    .and_then(|kept| conversation.index_of(kept));
                let verbatim = cut.map(|cut| last_typed(conversation, cut, count));
                let verbatim = verbatim.unwrap_or_default();
                commands
                    .entity(call)
                    .insert(Structured { tries: 1, verbatim });
            }
            Some(structured) if structured.tries > TRIES => {
                let Some(own) = own else {
                    continue;
                };
                match plan(conversation, &fallback.over(&default), reason, &own.spec) {
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
        let summary = outcome.completion().map(Summarizer::summary_text);
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

/// As rig-compaction puts an accepted structured summary in the
/// conversation, which it does as the summary call settles, puts the
/// user's kept messages right after it, each its own message, as they
/// were typed.
fn keep_verbatim(
    settled: On<Add<Settled>>,
    calls: Query<(&ForAgent, &Summarizing, &Structured)>,
    mut commands: Commands,
) {
    let Ok((&ForAgent(agent), summarizing, structured)) = calls.get(settled.entity) else {
        return;
    };
    let Some(summary) = summarizing.summary.clone() else {
        return;
    };
    if structured.tries > TRIES {
        return;
    }
    let texts = structured.verbatim.clone();
    // Queued, so it runs once every observer of the settled call did.
    commands.queue(move |world: &mut World| put_verbatim(world, agent, &summary, texts));
}

/// Puts `texts` after the agent's summaries, when its newest compaction
/// took `summary`.
fn put_verbatim(world: &mut World, agent: Entity, summary: &str, texts: Vec<String>) {
    let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
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
