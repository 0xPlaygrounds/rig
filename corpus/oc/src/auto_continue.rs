//! OC-22: continue replies cut off by the output limit (one-conversation
//! fresh held-out spec `22-auto-continue.md`).
//!
//! A turn's model reply whose provider says it stopped at the output-token
//! limit ([`FinishReason::Length`]) is taken over in Review before the
//! kernel commits it: the plugin keeps its text and its complete tool
//! calls in a [`Continuing`] marker and sends the call again ([`resend`]),
//! so the cut reply never reaches the conversation and nothing it called
//! runs yet. The new call is built from the conversation as usual; in
//! Prepare the plugin adds the partial reply to its `history`, as an
//! unfinished assistant message where the catalog says the model continues
//! one ([`Compat::prefill`]), else followed by a short user note asking to
//! continue (or to issue the calls the provider cut
//! ([`CompletionResponse::unfinished_calls`]) again, a `write` in several
//! edits). The new call's [`Partial`] starts with the text so far, so the
//! view keeps the one reply growing, a dim "continued" [`TextMark`] at
//! each seam. When a reply ends without the limit,
//! the plugin joins the text so far in front of its text in Review, the
//! words the model repeated at the seam taken out, and the kept calls
//! after it, so the conversation gets one assistant message. Each
//! request's usage counts for the turn as every model call's does
//! (`rig_basics::telemetry`'s `TurnUsage`). Once the joined reply is committed,
//! its marks point at the saved message's text. After `max_continuations`,
//! or when its turn is ending, a cut reply is kept as it is, its cut calls
//! dropped, with a notice. A provider that reports no finish reason is
//! never continued.
//!
//! [`Compat::prefill`]: rig_harness::rig_core::catalog::Compat::prefill

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ReasoningSupport;
use rig_harness::rig_core::completion::{CompletionResponse, FinishReason, Reasoning};
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::message::{Message, Text};
use rig_harness::rig_tools::Write;
use rig_tui::ratatui::style::Style;
use rig_tui::{MarkIn, TextMark, TranscriptView};
use serde::Deserialize;

/// The plugin's name, as an author, and its settings table.
const NAME: &str = "auto_continue";

/// The shortest repeat at a seam taken for an overlap, in bytes: a
/// shorter one is as likely a chance match.
const MIN_OVERLAP: usize = 4;

/// The `[auto_continue]` table of the settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct AutoContinueSettings {
    /// How many continuations one reply may take.
    pub max_continuations: u32,
}

impl Default for AutoContinueSettings {
    fn default() -> Self {
        Self {
            max_continuations: 3,
        }
    }
}

/// The stage systems of the plugin, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutoContinueSystems {
    /// Joins the reply so far in front of the reply that ends a
    /// continuation (`ReviewSystems::Edit`).
    Join,
    /// Sends a reply cut at the output limit again as a continuation
    /// (`ReviewSystems::Act`).
    Continue,
    /// Adds the partial reply to a continuation's request
    /// (`PrepareSystems::Edit`).
    Prepare,
    /// Points the seams' marks at the joined reply once it is committed
    /// (`PrepareSystems::Input`, the first after the commit).
    Mark,
}

/// Continues turn replies cut off by the output-token limit.
#[derive(Default)]
pub struct AutoContinuePlugin;

impl Plugin for AutoContinuePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<AutoContinueSettings>(NAME)
            .configure_sets(
                AgentPass,
                (
                    AutoContinueSystems::Join.in_set(ReviewSystems::Edit),
                    AutoContinueSystems::Continue.in_set(ReviewSystems::Act),
                    AutoContinueSystems::Prepare.in_set(PrepareSystems::Edit),
                    AutoContinueSystems::Mark.in_set(PrepareSystems::Input),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    join.in_set(AutoContinueSystems::Join),
                    continue_cut.in_set(AutoContinueSystems::Continue),
                    add_partial.in_set(AutoContinueSystems::Prepare),
                    mark_seams.in_set(AutoContinueSystems::Mark),
                ),
            )
            .add_observer(follow_successor::<Continuing>);
    }
}

/// On a turn's model call that continues a reply cut at the output limit:
/// the reply so far.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Continuing {
    /// The text so far, joined.
    pub text: String,
    /// Where in `text` each later piece begins.
    pub seams: Vec<usize>,
    /// The tool calls the pieces so far finished, to run with the reply.
    pub kept: Vec<ToolCall>,
    /// How many continuations the reply took, this one included.
    pub count: u32,
    /// The tools the last piece was calling when the provider cut it.
    pub cut: Vec<String>,
}

/// On a reply joined once (Review sees it each pass): the text block that
/// holds the joined text, and where in it each later piece begins.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Joined {
    block: Option<usize>,
    seams: Vec<usize>,
}

/// Whether the provider said the reply stopped at the output limit.
fn cut(response: &CompletionResponse) -> bool {
    response.finish_reason() == Some(FinishReason::Length)
}

/// `before` and `after` joined, the longest end of `before` that `after`
/// starts with again (at least four bytes) said once, and where `after`
/// begins in it.
fn join_at_seam(before: &str, after: &str) -> (String, usize) {
    let longest = before.len().min(after.len());
    let overlap = (MIN_OVERLAP..=longest)
        .rev()
        .filter(|&len| before.is_char_boundary(before.len() - len))
        .find(|&len| {
            let tail = before.get(before.len() - len..);
            tail.is_some_and(|tail| after.starts_with(tail))
        })
        .unwrap_or(0);
    let rest = after.get(overlap..).unwrap_or(after);
    (format!("{before}{rest}"), before.len() - overlap)
}

/// The text so far, if any, joined with a piece's `text`, and the seams.
fn joined(so_far: Option<&Continuing>, text: &str) -> (String, Vec<usize>) {
    let Some(so_far) = so_far else {
        return (text.to_owned(), Vec::new());
    };
    let (joined, seam) = join_at_seam(&so_far.text, text);
    let mut seams = so_far.seams.clone();
    if !so_far.text.is_empty() && !text.is_empty() {
        seams.push(seam);
    }
    (joined, seams)
}

/// The mark drawn at a seam.
fn seam(r#in: MarkIn, at: usize) -> TextMark {
    TextMark {
        r#in,
        at,
        label: " (continued) ".to_owned(),
        style: Style::new().dim().italic(),
    }
}

/// Takes over each turn reply cut at the output limit, while its reply has
/// continuations left: marks it with the reply so far and sends it again.
/// The cut reply is superseded, so it never reaches the conversation and
/// none of its tool calls runs.
fn continue_cut(
    calls: Query<
        (Entity, &Outcome, &CallOf, Option<&Continuing>),
        (
            ReadyToSettle,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<Joined>,
        ),
    >,
    ending: Query<(), With<TurnOutcome>>,
    settings: Res<AutoContinueSettings>,
    mut commands: Commands,
) {
    for (call, outcome, &CallOf(turn), so_far) in &calls {
        let Ok(Answered::Completion(response)) = &outcome.0 else {
            continue;
        };
        let count = so_far.map_or(0, |so_far| so_far.count);
        if !cut(response) || count >= settings.max_continuations || ending.contains(turn) {
            continue;
        }
        let (text, seams) = joined(so_far, &response.text());
        let cut: Vec<&ToolCall> = response.unfinished_calls().collect();
        let finished = response.tool_calls().filter(|call| !cut.contains(call));
        let next = Continuing {
            text,
            seams,
            kept: so_far
                .into_iter()
                .flat_map(|so_far| so_far.kept.iter())
                .chain(finished)
                .cloned()
                .collect(),
            count: count + 1,
            cut: cut
                .iter()
                .map(|call| call.function.name.to_string())
                .collect(),
        };
        commands.entity(call).insert(next);
        commands.queue(resend(call));
    }
}

/// Puts the reply so far in each continuation's request as it is built:
/// an unfinished assistant message where its model continues one, else the
/// partial text and a note on how to go on. Its [`Partial`] starts with
/// the text so far, marked at each seam.
fn add_partial(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &Continuing,
            &mut Partial,
            Option<&ForAgent>,
        ),
        (Added<Admitted>, With<TurnCall>),
    >,
    agents: Query<&Connection>,
    mut views: Query<&mut TranscriptView>,
) {
    for (entity, mut call, so_far, mut partial, agent) in &mut calls {
        let own = agent.and_then(|agent| agents.get(agent.0).ok());
        let spec = call.connection.as_ref().or(own).map(|it| it.spec.clone());
        // The catalog's prefill fact holds while reasoning is off: asked
        // off, or left to a model that does not reason.
        let reasoning = &call.request.options.reasoning;
        let prefill = spec.is_some_and(|spec| {
            let off = match reasoning {
                Some(reasoning) => matches!(reasoning, Reasoning::Off),
                None => matches!(spec.reasoning, ReasoningSupport::None),
            };
            spec.compat.prefill && off
        }) && so_far.cut.is_empty()
            && so_far.kept.is_empty();
        let Some(history) = call.history.as_mut() else {
            continue;
        };
        let by = Author::plugin(NAME);
        if !so_far.text.is_empty() {
            history.push(Entry::new(Message::assistant(so_far.text.clone())).by(by.clone()));
        }
        if !prefill {
            history.push(Entry::new(Message::user(note(so_far))).by(by));
        }
        partial.text.clone_from(&so_far.text);
        if !so_far.text.is_empty() {
            let at = so_far.seams.iter().copied().chain([so_far.text.len()]);
            let marks: Vec<TextMark> = at.map(|at| seam(MarkIn::Streaming(entity), at)).collect();
            for mut view in &mut views {
                view.marks.extend(marks.iter().cloned());
            }
        }
    }
}

/// The note after a cut reply, by what was cut and what was kept.
fn note(so_far: &Continuing) -> String {
    let mut note = "Your reply was cut off by the output limit.".to_owned();
    if !so_far.kept.is_empty() {
        let kept: Vec<&str> = so_far
            .kept
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        note.push_str(&format!(
            " Your complete tool calls ({}) were kept and run after your reply; do not issue \
             them again.",
            kept.join(", ")
        ));
    }
    if so_far.cut.is_empty() {
        note.push_str(" Continue exactly where you stopped, without repeating anything.");
        return note;
    }
    note.push_str(&format!(
        " It was cut in the middle of the tool call {}, which was discarded and not run. Issue \
         it again, complete.",
        so_far.cut.join(", ")
    ));
    if so_far.cut.iter().any(|tool| tool == Write::NAME) {
        note.push_str(
            " The file is too long for one reply: write its first part with `write`, then add \
             the rest in several `edit` calls.",
        );
    }
    note
}

/// Joins the reply so far in front of the reply that ends a continuation,
/// in place, once: the text so far in its first text block, the seam's
/// repeat said once, and the kept calls after it. A reply cut with no
/// continuation left, or as its turn ends, is kept as it is, without the
/// calls the provider cut, and a notice says so.
fn join(
    mut calls: Query<
        (
            Entity,
            &mut Outcome,
            &CallOf,
            Option<&Continuing>,
            Option<&ForAgent>,
        ),
        (
            ReadyToSettle,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<Joined>,
        ),
    >,
    ending: Query<(), With<TurnOutcome>>,
    settings: Res<AutoContinueSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut outcome, &CallOf(turn), so_far, agent) in &mut calls {
        let Ok(Answered::Completion(response)) = &mut outcome.0 else {
            continue;
        };
        let given_up = cut(response);
        let count = so_far.map_or(0, |so_far| so_far.count);
        let continues = count < settings.max_continuations && !ending.contains(turn);
        if (given_up && continues) || (!given_up && so_far.is_none()) {
            continue;
        }
        let response = std::sync::Arc::make_mut(response);
        let dropped: Vec<ToolCall> = response.unfinished_calls().cloned().collect();
        response.choice.retain(|block| match block {
            AssistantContent::ToolCall(call) => !dropped.contains(call),
            _ => true,
        });
        // The reply's first text, else where it would go: after any
        // reasoning, before its calls.
        let first_text = response
            .choice
            .iter()
            .position(|block| matches!(block, AssistantContent::Text(_)));
        let at = first_text.unwrap_or_else(|| {
            let mut choice = response.choice.iter();
            choice
                .position(|block| !matches!(block, AssistantContent::Reasoning(_)))
                .unwrap_or(response.choice.len())
        });
        let current = match response.choice.get(at) {
            Some(AssistantContent::Text(text)) => text.text.as_str(),
            _ => "",
        };
        let (text, seams) = joined(so_far, current);
        let block = (!text.is_empty()).then_some(at);
        match (first_text, response.choice.get_mut(at)) {
            (Some(_), Some(AssistantContent::Text(old))) => *old = Text::new(text),
            _ if block.is_some() => response
                .choice
                .insert(at, AssistantContent::Text(Text::new(text))),
            _ => {}
        }
        let after = block.map_or(at, |at| at + 1);
        let kept = so_far
            .into_iter()
            .flat_map(|so_far| so_far.kept.iter().cloned());
        response
            .choice
            .splice(after..after, kept.map(AssistantContent::ToolCall));
        commands.entity(call).insert(Joined { block, seams });
        if given_up {
            let text = format!(
                "The reply hit the output limit {} times; it is kept as it is.",
                count + 1
            );
            notices.write(Notice::warn(agent.map(|agent| agent.0), text));
        }
    }
}

/// As turn calls settle: takes their seams' marks off their streaming
/// text, and puts a joined reply's on the message it was committed as.
fn mark_seams(
    settled: Query<
        (
            Entity,
            Option<&Outcome>,
            Option<&Joined>,
            Option<&ForAgent>,
            Has<SupersededBy>,
        ),
        (With<TurnCall>, Added<Settled>),
    >,
    agents: Query<&Conversation>,
    mut views: Query<&mut TranscriptView>,
) {
    for (call, outcome, joined, agent, superseded) in &settled {
        let streaming = MarkIn::Streaming(call);
        for mut view in &mut views {
            if view.marks.iter().any(|mark| mark.r#in == streaming) {
                view.marks.retain(|mark| mark.r#in != streaming);
            }
        }
        let (Some(Outcome(Ok(Answered::Completion(reply)))), Some(joined), Some(agent), false) =
            (outcome, joined, agent, superseded)
        else {
            continue;
        };
        let Some((block, message)) = joined.block.zip(reply.message()) else {
            continue;
        };
        let conversation = agents.get(agent.0).ok();
        let mut entries = conversation.into_iter().flat_map(|it| it.entries().rev());
        let Some(entry) = entries.find(|entry| *entry.message == message) else {
            continue;
        };
        let id = entry.id;
        let marks = joined.seams.iter();
        let marks: Vec<TextMark> = marks
            .map(|&at| seam(MarkIn::Message { id, block }, at))
            .collect();
        for mut view in &mut views {
            view.marks.extend(marks.iter().cloned());
        }
    }
}

#[cfg(test)]
mod tests;
