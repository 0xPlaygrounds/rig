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
//! view keeps the one reply growing. When a reply ends without the limit,
//! the plugin joins the text so far in front of its text in Review, the
//! words the model repeated at the seam taken out, and the kept calls
//! after it, so the conversation gets one assistant message. Each
//! request's usage counts for the turn as every model call's does
//! (rig-telemetry's `TurnSpending`). After `max_continuations` the reply
//! is kept as it is, its cut calls dropped, with a notice. A provider that
//! reports no finish reason is never continued.
//!
//! [`Compat::prefill`]: rig_harness::rig_core::catalog::Compat::prefill

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, FinishReason, Reasoning};
use rig_harness::rig_core::effect::Outcome as Answered;
use rig_harness::rig_core::message::{Message, Text};
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
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Continuing {
    /// The text so far, joined.
    pub text: String,
    /// The tool calls the pieces so far finished, to run with the reply.
    pub kept: Vec<ToolCall>,
    /// How many continuations the reply took, this one included.
    pub count: u32,
    /// The tools the last piece was calling when the provider cut it.
    pub cut: Vec<String>,
}

/// On a continued reply joined once: Review sees it each pass.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Joined;

/// Whether the provider said the reply stopped at the output limit.
fn cut(response: &CompletionResponse) -> bool {
    response.finish_reason() == Some(FinishReason::Length)
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
/// continuations left: marks it with the reply so far and sends it again.
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
        let Ok(Answered::Completion(response)) = &outcome.0 else {
            continue;
        };
        let count = so_far.map_or(0, |so_far| so_far.count);
        if !cut(response) || count >= settings.max_continuations || ending.contains(turn) {
            continue;
        }
        let text = response.text();
        let cut: Vec<&ToolCall> = response.unfinished_calls().collect();
        let finished = response.tool_calls().filter(|call| !cut.contains(call));
        let next = Continuing {
            text: so_far.map_or_else(|| text.clone(), |so_far| join_at_seam(&so_far.text, &text)),
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
/// the text so far.
fn add_partial(
    mut calls: Query<
        (
            &mut ModelRequest,
            &Continuing,
            &mut Partial,
            Option<&ForAgent>,
        ),
        (Added<Admitted>, With<TurnCall>),
    >,
    agents: Query<&Connection>,
) {
    for (mut call, so_far, mut partial, agent) in &mut calls {
        let own = agent.and_then(|agent| agents.get(agent.0).ok());
        let spec = call.connection.as_ref().or(own).map(|it| it.spec.clone());
        // The catalog's prefill fact holds while reasoning is off.
        let thinking = matches!(
            call.request.options.reasoning,
            Some(Reasoning::Effort(_) | Reasoning::Budget { .. })
        );
        let prefill = spec.is_some_and(|spec| spec.compat.prefill)
            && !thinking
            && so_far.cut.is_empty()
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
    if so_far.cut.iter().any(|tool| tool == "write") {
        note.push_str(
            " The file is too long for one reply: write its first part with `write`, then add \
             the rest in several `edit` calls.",
        );
    }
    note
}

/// Joins the reply so far in front of the reply that ends a continuation,
/// in place, once: the text so far in its first text block, the seam's
/// repeat said once, and the kept calls after it. A reply cut again with
/// no continuation left is kept as it is, without the calls the provider
/// cut, and a notice says so.
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
        let after = if let Some(at) = first_text {
            if let Some(AssistantContent::Text(text)) = response.choice.get_mut(at) {
                *text = Text::new(join_at_seam(&so_far.text, &text.text));
            }
            at + 1
        } else {
            let at = response
                .choice
                .iter()
                .position(|block| !matches!(block, AssistantContent::Reasoning(_)))
                .unwrap_or(response.choice.len());
            if so_far.text.is_empty() {
                at
            } else {
                let text = AssistantContent::Text(Text::new(so_far.text.clone()));
                response.choice.insert(at, text);
                at + 1
            }
        };
        let kept = so_far.kept.iter().cloned().map(AssistantContent::ToolCall);
        response.choice.splice(after..after, kept);
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
