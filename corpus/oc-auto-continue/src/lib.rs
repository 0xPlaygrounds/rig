//! OC-22: continue replies cut off by the output limit (one-conversation
//! fresh held-out spec `22-auto-continue.md`).
//!
//! A turn's model reply whose provider says it stopped at the output-token
//! limit ([`FinishReason::Length`]) is taken over in Review before the
//! kernel commits it: the plugin keeps its text in a [`Continuing`] marker
//! and sends the call again ([`resend`]), so the cut reply never reaches the
//! conversation and a tool call it cut off never runs. The new call is
//! built from the conversation as usual; in Prepare the plugin adds the
//! partial reply to its `history`, as an unfinished assistant message on a
//! provider listed for prefilling, else followed by a short user note
//! asking to continue (or to issue a cut tool call again, a large `write`
//! in several edits). While it streams, its [`Partial`] starts with the
//! text so far and a `[continued]` mark, so the view keeps the one reply
//! growing. When a reply ends without the limit, the plugin joins the text
//! so far in front of its text in Review, the words the model repeated at
//! the seam taken out, so the conversation gets one assistant message.
//! Each request's usage counts for the turn as every model call's does
//! (rig-telemetry's `TurnSpending`). After `max_continuations` the reply
//! is kept as it is, its cut tool call dropped, with a notice. A provider
//! that reports no finish reason is never continued.

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, FinishReason};
use rig_harness::rig_core::effect::Outcome as Answered;
use rig_harness::rig_core::message::{Message, Text};
use serde::Deserialize;

/// The plugin's name, as an author and a decider, and its settings table.
const NAME: &str = "auto_continue";

/// What the view shows at a seam while the continuation streams.
const MARK: &str = " [continued] ";

/// The shortest repeat at a seam taken for an overlap, in bytes: a
/// shorter one is as likely a chance match.
const MIN_OVERLAP: usize = 4;

/// The `[auto_continue]` table of the settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct AutoContinueSettings {
    /// How many continuations one reply may take.
    pub max_continuations: u32,
    /// The vendors (`anthropic`) whose models take an unfinished assistant
    /// message as the start of their reply. The model catalog does not
    /// say which do, so none by default.
    pub prefill_vendors: Vec<String>,
}

impl Default for AutoContinueSettings {
    fn default() -> Self {
        Self {
            max_continuations: 3,
            prefill_vendors: Vec::new(),
        }
    }
}

/// The stage systems of the plugin, in `ReviewSystems::Edit` (joining a
/// continued reply) and `ReviewSystems::Act` (taking a cut reply over),
/// for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutoContinueSystems {
    /// Joins the text so far in front of the reply that ends a
    /// continuation.
    Join,
    /// Sends a reply cut at the output limit again as a continuation.
    Continue,
    /// Adds the partial reply to a continuation's request.
    Prepare,
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
                ),
            )
            .add_systems(
                AgentPass,
                (
                    join.in_set(AutoContinueSystems::Join),
                    continue_cut.in_set(AutoContinueSystems::Continue),
                    add_partial.in_set(AutoContinueSystems::Prepare),
                ),
            )
            .add_observer(follow_successor::<Continuing>);
    }
}

/// On a turn's model call that continues a reply cut at the output limit:
/// the reply so far.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Continuing {
    /// The text so far, joined.
    pub text: String,
    /// How many continuations the reply took, this one included.
    pub count: u32,
    /// The tool the last cut piece was calling when it was cut, if any.
    pub cut_call: Option<String>,
}

/// On a continued reply joined once: Review sees it each pass.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Joined;

/// The reply a call answered with, if a model answered it.
fn completion(outcome: &Outcome) -> Option<&CompletionResponse> {
    match &outcome.0 {
        Ok(Answered::Completion(response)) => Some(response),
        _ => None,
    }
}

/// Whether the provider said the reply stopped at the output limit.
fn cut(response: &CompletionResponse) -> bool {
    response.finish_reason() == Some(FinishReason::Length)
}

/// The tool a cut reply was calling: the limit stops the item being
/// written, so only a call that is its last item is cut.
fn cut_call(response: &CompletionResponse) -> Option<String> {
    match response.choice.last() {
        Some(AssistantContent::ToolCall(call)) => Some(call.function.name.as_str().to_owned()),
        _ => None,
    }
}

/// `before` and `after` joined, the longest end of `before` that `after`
/// starts with again (at least four bytes) said once.
pub fn join_at_seam(before: &str, after: &str) -> String {
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
    format!("{before}{rest}")
}

/// Takes over each turn reply cut at the output limit, while its reply has
/// continuations left: marks it with the text so far and sends it again.
/// The cut reply is superseded, so it never reaches the conversation and
/// none of its tool calls runs.
fn continue_cut(
    calls: Query<
        (Entity, &Outcome, &CallOf, Option<&Continuing>),
        (
            Unreviewed,
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
        let Some(response) = completion(outcome).filter(|response| cut(response)) else {
            continue;
        };
        let count = so_far.map_or(0, |so_far| so_far.count);
        if count >= settings.max_continuations || ending.contains(turn) {
            continue;
        }
        let text = response.text();
        let joined =
            so_far.map_or_else(|| text.clone(), |so_far| join_at_seam(&so_far.text, &text));
        let next = Continuing {
            text: joined,
            count: count + 1,
            cut_call: cut_call(response),
        };
        commands.entity(call).insert(next);
        commands.queue(resend(call));
    }
}

/// Puts the reply so far in each continuation's request as it is built:
/// an unfinished assistant message where its model's vendor takes one,
/// else the partial text and a note on how to go on. Its [`Partial`]
/// starts with the text so far and the seam's mark.
fn add_partial(
    mut calls: Query<
        (&mut ModelRequest, &Continuing, &mut Partial),
        (Added<Admitted>, With<TurnCall>),
    >,
    settings: Res<AutoContinueSettings>,
) {
    for (mut call, so_far, mut partial) in &mut calls {
        let vendor = call
            .connection
            .as_ref()
            .map(|connection| connection.spec.provider.vendor().to_owned());
        let prefill = so_far.cut_call.is_none()
            && vendor.is_some_and(|vendor| settings.prefill_vendors.contains(&vendor));
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
        partial.text = format!("{}{MARK}", so_far.text);
    }
}

/// The note after a cut reply, by what was cut.
fn note(so_far: &Continuing) -> String {
    match so_far.cut_call.as_deref() {
        None => "Your reply was cut off by the output limit. Continue exactly where you \
                 stopped, without repeating anything."
            .to_owned(),
        Some(tool) => {
            let mut note = format!(
                "Your reply was cut off by the output limit in the middle of a `{tool}` tool \
                 call, which was discarded and not run. Issue the tool call again, complete."
            );
            if tool == "write" {
                note.push_str(
                    " The file is too long for one reply: write its first part with `write`, \
                     then add the rest in several `edit` calls.",
                );
            }
            note
        }
    }
}

/// Joins the text so far in front of the reply that ends a continuation,
/// in place, once: its first text block, the seam's repeat said once. A
/// reply cut again with no continuation left is kept as it is, without
/// the tool call it was cut in, and a notice says so.
fn join(
    mut calls: Query<
        (Entity, &mut Outcome, &Continuing, Option<&ForAgent>),
        (
            Unreviewed,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<Joined>,
        ),
    >,
    settings: Res<AutoContinueSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut outcome, so_far, agent) in &mut calls {
        let Ok(Answered::Completion(response)) = &mut outcome.0 else {
            continue;
        };
        let given_up = cut(response);
        if given_up && so_far.count < settings.max_continuations {
            continue;
        }
        if given_up && cut_call(response).is_some() {
            response.choice.pop();
        }
        let first_text = response
            .choice
            .iter()
            .position(|block| matches!(block, AssistantContent::Text(_)));
        match first_text.and_then(|at| response.choice.get_mut(at)) {
            Some(AssistantContent::Text(text)) => {
                let joined = join_at_seam(&so_far.text, &text.text);
                *text = Text::new(joined);
            }
            _ if !so_far.text.is_empty() => {
                // Before its calls, after any reasoning.
                let at = response
                    .choice
                    .iter()
                    .position(|block| !matches!(block, AssistantContent::Reasoning(_)))
                    .unwrap_or(response.choice.len());
                let text = AssistantContent::Text(Text::new(so_far.text.clone()));
                response.choice.insert(at, text);
            }
            _ => {}
        }
        commands.entity(call).insert(Joined);
        if given_up {
            let text = format!(
                "The reply hit the output limit {} times; it is kept as it is.",
                so_far.count + 1
            );
            notices.write(Notice::warn(agent.map(|agent| agent.0), text));
        }
    }
}

#[cfg(test)]
mod tests;
