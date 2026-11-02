//! Redaction ([`RedactPlugin`]): the API keys, tokens, private keys and
//! values of secret-named variables that [`rig_core::redact`] finds become
//! `[REDACTED]` where they come in: the model's reply as it streams, each
//! tool's output and answer, and each message. The conversation, the logs,
//! every front, subagents and a resumed session then hold them redacted,
//! and so does the model. Without this plugin nothing is redacted.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_basics::redact::RedactPlugin);
//! ```

use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::redact::{self, Streamed};
use rig_harness::rig_core::streaming::{Item, Relayed};
use rig_harness::rig_core::tool::{ContextValue, FilesChanged, Hunk, HunkLine};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_tools::LineFilter;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// Redacts every secret where it comes in, so nothing downstream holds it.
#[derive(Default)]
pub struct RedactPlugin;

/// After every [`ReviewSystems`] phase, in [`Stage::Review`]: each tool
/// call's answer redacted again, as a reviewer may have changed it in
/// place. A system that reads answers there to keep them goes after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedactSystems;

impl Plugin for RedactPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            AgentPass,
            RedactSystems
                .in_set(Stage::Review)
                .after(ReviewSystems::Act),
        )
        .add_observer(redact_outcome)
        .add_observer(redact_if_stopped)
        .add_observer(redact_facts)
        .add_systems(
            AgentPass,
            // Each after every other plugin that edits the same data.
            (
                adapt_model_calls
                    .in_set(Stage::Prepare)
                    .after(PrepareSystems::AdaptOutermost)
                    .before(PrepareSystems::Fit),
                filter_output
                    .after(PerCallSystems::Place)
                    .before(PerCallSystems::Build),
                redact_reviewed.in_set(RedactSystems),
                redact_messages
                    .after(GateSystems::After)
                    .after(crate::approval::ApprovalSystems)
                    .before(GateSystems::Fallback),
                redact_recalled.in_set(Stage::Start),
            ),
        )
        .add_systems(
            Last,
            redact_records
                .run_if(resource_exists::<crate::telemetry::effect_log::EffectLog>)
                .after(StopTurnsOnExit)
                .before(WriteLogs),
        );
        #[cfg(feature = "tui")]
        app.add_systems(
            Update,
            redact_history
                .run_if(resource_exists_and_changed::<rig_tui::PromptHistory>)
                .after(rig_tui::HistorySystems::Record)
                .before(rig_tui::HistorySystems::Save),
        );
    }
}

/// `data` with every string in it redacted, through its JSON form; `None`
/// when it holds no secret, or when its redacted form does not read back.
fn redacted<T: Serialize + DeserializeOwned>(data: &T) -> Option<T> {
    let mut json = serde_json::to_value(data).ok()?;
    redact::json(&mut json)
        .then(|| serde_json::from_value(json).ok())
        .flatten()
}

/// A tool call's result or an error, redacted; `None` when it holds no
/// secret. A plugin's own answer, such as the line that answers its
/// question, reaches it as given.
fn redacted_answer(answer: &Outcome) -> Option<Result<Answered, ErrorReport>> {
    match &answer.0 {
        Ok(Answered::ToolResult { .. }) | Err(_) => redacted(&answer.0),
        Ok(_) => None,
    }
}

/// An answer as it is inserted: a result, a stop's answer, one kept at
/// exit, a model call's error.
fn redact_outcome(inserted: On<Insert<Outcome>>, mut answers: Query<&mut Outcome>) {
    if let Ok(mut answer) = answers.get_mut(inserted.entity)
        && let Some(redacted) = redacted_answer(&answer)
    {
        answer.0 = redacted;
    }
}

/// What a running call answers if stopped, which a stop reads as it is.
fn redact_if_stopped(inserted: On<Insert<IfStopped>>, mut answers: Query<&mut IfStopped>) {
    if let Ok(mut answer) = answers.get_mut(inserted.entity)
        && let Some(redacted) = redacted_answer(&answer.0)
    {
        answer.0.0 = redacted;
    }
}

/// Each tool call's answer after every reviewer, which may have changed it
/// in place, and before it is committed.
fn redact_reviewed(
    mut answers: Query<&mut Outcome, (ToolCalls, Changed<Outcome>, Without<Settled>)>,
) {
    for mut answer in &mut answers {
        if let Some(redacted) = redacted_answer(&answer) {
            answer.0 = redacted;
        }
    }
}

/// What a call published, which the conversation keeps with its result.
/// A changed file's old text, which it never keeps, stays whole.
fn redact_facts(inserted: On<Insert<Published>>, mut published: Query<&mut Published>) {
    let Ok(mut published) = published.get_mut(inserted.entity) else {
        return;
    };
    let Ok(Value::Object(mut slots)) = serde_json::to_value(&published.0) else {
        return;
    };
    let mut changed = false;
    for (key, value) in &mut slots {
        let redacted = match key == FilesChanged::KEY {
            true => serde_json::from_value::<FilesChanged>(value.clone())
                .ok()
                .and_then(|mut files| {
                    let hunks = files.0.iter_mut().map(|change| &mut change.hunks);
                    let changed = hunks.fold(false, |any, hunks| redact_hunks(hunks) | any);
                    changed.then(|| serde_json::to_value(files).ok()).flatten()
                }),
            false => redacted(value),
        };
        if let Some(redacted) = redacted {
            *value = redacted;
            changed = true;
        }
    }
    if changed && let Ok(redacted) = serde_json::from_value(Value::Object(slots)) {
        published.0 = redacted;
    }
}

/// Redacts each line of `hunks` alone, as its JSON form's diff sign would
/// hide a secret at its start; a secret across lines takes every hunk.
/// Whether any changed.
fn redact_hunks(hunks: &mut Vec<Hunk>) -> bool {
    // Each line ends in a break, so one without its ending (the file's
    // last) does not run into the next.
    let mut joined = String::new();
    for line in hunks.iter().flat_map(|hunk| &hunk.lines) {
        joined.push_str(line.text());
        if !line.text().ends_with(['\n', '\r']) {
            joined.push('\n');
        }
    }
    let across = redact::find(&joined).any(|found| {
        let text = joined.get(found.range).unwrap_or_default();
        text.trim_end().contains(['\n', '\r'])
    });
    if across {
        hunks.clear();
        return true;
    }
    let mut changed = false;
    for line in hunks.iter_mut().flat_map(|hunk| &mut hunk.lines) {
        let (HunkLine::Same(text) | HunkLine::Removed(text) | HunkLine::Added(text)) = line;
        if let Cow::Owned(redacted) = redact::text(text) {
            *text = redacted;
            changed = true;
        }
    }
    changed
}

/// Each message that goes to its agent, its files attached, once every
/// gate took the lines it takes as typed (a slash command, an answer), and
/// before the fallback keeps or hands back one a turn did not send.
fn redact_messages(
    mut messages: Query<(&mut Incoming, &Decision), (IncomingMessages, Without<Started>)>,
) {
    for (mut message, decision) in &mut messages {
        if decision.is_decided() {
            continue;
        }
        let mut changed = false;
        for part in &mut message.bypass_change_detection().parts {
            let text = match part {
                MessagePart::Text(text) => text,
                MessagePart::Attachment(Attachment { content, .. }) => match content {
                    UserContent::Text(text) => &mut text.text,
                    UserContent::Image(_) | UserContent::Audio(_) | UserContent::Video(_) => {
                        continue;
                    }
                    // A document's or a result's text, as JSON.
                    content => {
                        if let Some(redacted) = redacted(&*content) {
                            *content = redacted;
                            changed = true;
                        }
                        continue;
                    }
                },
            };
            if let Cow::Owned(redacted) = redact::text(text) {
                *text = redacted;
                changed = true;
            }
        }
        if changed {
            message.set_changed();
        }
    }
}

/// A refused command's line, which the view that hands it back shows.
fn redact_recalled(mut recalled: MessageMutator<Recalled>) {
    for recalled in recalled.read() {
        if let Cow::Owned(redacted) = redact::text(&recalled.text) {
            recalled.text = redacted;
        }
    }
}

/// Every model call streams its reply redacted, wrapping the outermost
/// adapters too, such as a replay.
fn adapt_model_calls(mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>) {
    for mut adapted in &mut calls {
        let handler = adapted.0.handler.0.clone();
        adapted.0.handler = Handler(handler.map_stream("redact", |_| redact_stream()));
    }
}

/// What one part of a reply holds back; a call's argument JSON is not
/// text, and is redacted whole at its end.
enum Held {
    Text(Streamed),
    Reasoning(Streamed),
    Arguments(String),
}

impl Held {
    /// Takes `piece`; what can go out now.
    fn push(&mut self, piece: &str) -> String {
        match self {
            Self::Text(text) | Self::Reasoning(text) => text.push(piece),
            Self::Arguments(json) => {
                json.push_str(piece);
                String::new()
            }
        }
    }

    /// What it still holds, redacted: arguments that are no JSON, of a call
    /// that fails, as text.
    fn finish(&mut self) -> String {
        match self {
            Self::Text(text) | Self::Reasoning(text) => text.finish(),
            Self::Arguments(json) => match serde_json::from_str::<Value>(json) {
                Ok(value) => {
                    redacted(&value).map_or_else(|| json.clone(), |value| value.to_string())
                }
                Err(_) => redact::text(json).into_owned(),
            },
        }
    }

    /// `text` as an event of `part`, of this kind; none when empty.
    fn event(&self, part: Part, text: String) -> Option<Relayed> {
        let event = match self {
            _ if text.is_empty() => return None,
            Self::Text(_) => StreamEvent::Text { part, text },
            Self::Reasoning(_) => StreamEvent::Reasoning { part, text },
            Self::Arguments(_) => StreamEvent::Arguments { part, json: text },
        };
        Some(Relayed::Item(Item::Event(event)))
    }
}

/// One reply's stream redacted: text and reasoning as they stream
/// ([`Streamed`]), a call's arguments at its end, the rest as JSON. What a
/// part holds goes out before its end, and all of it before the response.
fn redact_stream() -> impl FnMut(Relayed) -> Vec<Relayed> {
    let mut held: Vec<(Part, Held)> = Vec::new();
    move |item| {
        let mut out = Vec::new();
        match item {
            Relayed::Item(Item::Event(StreamEvent::Text { part, text })) => {
                let new = || Held::Text(Streamed::default());
                grow(&mut held, (part, new), &text, &mut out);
            }
            Relayed::Item(Item::Event(StreamEvent::Reasoning { part, text })) => {
                let new = || Held::Reasoning(Streamed::default());
                grow(&mut held, (part, new), &text, &mut out);
            }
            Relayed::Item(Item::Event(StreamEvent::Arguments { part, json })) => {
                let new = || Held::Arguments(String::new());
                grow(&mut held, (part, new), &json, &mut out);
            }
            Relayed::Item(item) => {
                if let Item::Event(StreamEvent::End { part, .. }) = &item {
                    let_go(&mut held, Some(*part), &mut out);
                }
                out.push(Relayed::Item(redacted(&item).unwrap_or(item)));
            }
            Relayed::Done(mut response) => {
                let_go(&mut held, None, &mut out);
                if let Some(redacted) = redacted(&*response) {
                    *Arc::make_mut(&mut response) = redacted;
                }
                out.push(Relayed::Done(response));
            }
            origin @ Relayed::Origin(_) => out.push(origin),
        }
        out
    }
}

/// Takes the next `piece` of `part`, held as `new` makes it when it is the
/// part's first, putting what can go out in `out`.
fn grow(
    held: &mut Vec<(Part, Held)>,
    (part, new): (Part, fn() -> Held),
    piece: &str,
    out: &mut Vec<Relayed>,
) {
    let at = held
        .iter()
        .position(|(at, _)| *at == part)
        .unwrap_or_else(|| {
            held.push((part, new()));
            held.len() - 1
        });
    if let Some((_, held)) = held.get_mut(at) {
        let shown = held.push(piece);
        out.extend(held.event(part, shown));
    }
}

/// Puts in `out` what `part`, or every part, holds.
fn let_go(held: &mut Vec<(Part, Held)>, part: Option<Part>, out: &mut Vec<Relayed>) {
    let (going, staying) = std::mem::take(held)
        .into_iter()
        .partition(|(at, _)| part.is_none_or(|part| part == *at));
    *held = staying;
    for (part, mut going) in going {
        let text = going.finish();
        out.extend(going.event(part, text));
    }
}

/// Each line of every call's output, before the tool reports, spills or
/// keeps it ([`redact_line`]).
fn filter_output(mut places: Query<&mut CallPlace, (ToolCalls, Added<CallPlace>)>) {
    for mut place in &mut places {
        let in_key = Mutex::new(false);
        place.filter = Some(LineFilter::new(move |line| {
            let mut in_key = in_key.lock().unwrap_or_else(PoisonError::into_inner);
            redact_line(line, &mut in_key)
        }));
    }
}

/// `line` redacted; a private key block's lines after its first, through
/// its end line, are left out (`in_key`).
fn redact_line(line: &str, in_key: &mut bool) -> String {
    if *in_key {
        *in_key = !line.contains("-----END ");
        return String::new();
    }
    let Cow::Owned(mut redacted) = redact::text(line) else {
        return line.to_owned();
    };
    *in_key = redact::find(line).any(|secret| {
        secret.kind == redact::Kind::PrivateKey
            && line
                .get(secret.range)
                .is_some_and(|key| !key.contains("-----END "))
    });
    // A key redacted to the line's end took its newline.
    if line.ends_with('\n') && !redacted.ends_with('\n') {
        redacted.push('\n');
    }
    redacted
}

/// Every record reaches the effect log, when it is written, redacted: the
/// request as sent, a call's first answer, what it published.
fn redact_records(mut records: MessageMutator<EffectRecorded>) {
    for EffectRecorded(record) in records.read() {
        if let Some(redacted) = redacted(&*record) {
            *record = redacted;
        }
    }
}

/// A prompt holding a secret is left out of the terminal view's history,
/// so Up never shows it and it is never saved.
#[cfg(feature = "tui")]
fn redact_history(mut history: ResMut<rig_tui::PromptHistory>) {
    if history
        .entries
        .iter()
        .any(|prompt| redact::has_secret(prompt))
    {
        history.entries.retain(|prompt| !redact::has_secret(prompt));
    }
}
