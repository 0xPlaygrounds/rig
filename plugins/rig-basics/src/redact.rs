//! Redaction ([`RedactPlugin`]): the API keys, tokens, private keys and
//! values of secret-named variables that [`rig_core::redact`] finds become
//! `[REDACTED]` where they come in: the model's reply as it streams, each
//! tool's output and answer, and each message. The conversation, the logs,
//! every front, subagents and a resumed session then hold them redacted,
//! and so does the model. Without this plugin nothing is redacted.
//!
//! A slash command and a line that answers a question reach the plugin
//! that takes them as typed, and a plugin's own saved values are saved as
//! it gives them.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_basics::redact::RedactPlugin);
//! ```

use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::prelude::*;
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

impl Plugin for RedactPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(redact_outcome)
            .add_observer(redact_if_stopped)
            .add_observer(redact_facts)
            .add_systems(
                AgentPass,
                (
                    adapt_model_calls.in_set(PrepareSystems::AdaptOutermost),
                    filter_output
                        .after(PerCallSystems::Place)
                        .before(PerCallSystems::Build),
                    redact_reviewed
                        .in_set(Stage::Review)
                        .after(ReviewSystems::Act),
                    redact_messages
                        .after(GateSystems::After)
                        .after(crate::approval::ApprovalSystems)
                        .before(GateSystems::Fallback),
                    redact_recalled.in_set(Stage::Start),
                ),
            )
            .add_systems(
                Last,
                redact_records.after(StopTurnsOnExit).before(WriteLogs),
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

/// An answer as it is inserted: a result, a stop's answer, one kept at
/// exit, a model call's error.
fn redact_outcome(inserted: On<Insert<Outcome>>, mut answers: Query<&mut Outcome>) {
    if let Ok(mut answer) = answers.get_mut(inserted.entity)
        && let Some(redacted) = redacted(&answer.0)
    {
        answer.0 = redacted;
    }
}

/// What a running call answers if it is stopped, which a stop and the
/// exit read as it is.
fn redact_if_stopped(inserted: On<Insert<IfStopped>>, mut answers: Query<&mut IfStopped>) {
    if let Ok(mut answer) = answers.get_mut(inserted.entity)
        && let Some(redacted) = redacted(&answer.0.0)
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
        if let Some(redacted) = redacted(&answer.0) {
            answer.0 = redacted;
        }
    }
}

/// What a call published, which the conversation keeps with its result
/// and a view draws it from. A changed file's whole old text, which the
/// conversation never keeps and a plugin may restore the file from, is
/// left as it is.
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

/// Redacts each line of `hunks` on its own: its JSON form starts with the
/// line's diff sign, which would hide a secret at the line's start. A
/// secret across lines, such as a private key block, is found in the lines
/// together first, and takes every hunk with it. Whether any changed.
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
/// gate decided the lines it takes: a slash command's, or one answering a
/// question, reaches the plugin that took it as typed. The kernel's
/// fallback after it puts a message a turn did not send in the
/// conversation or hands it back ([`Recalled`]), redacted.
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
                MessagePart::Attachment(Attachment {
                    content: UserContent::Text(text),
                    ..
                }) => &mut text.text,
                MessagePart::Attachment(_) => continue,
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

/// Every model call streams its reply redacted, whichever model it goes
/// to. It adapts outermost, so what any other adapter streams is redacted.
fn adapt_model_calls(mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>) {
    for mut adapted in &mut calls {
        let handler = adapted.0.handler.0.clone();
        adapted.0.handler = Handler(handler.map_stream("redact", |_| redact_stream()));
    }
}

/// What one part of a reply holds back: its text or reasoning (`true`), or
/// a tool call's argument JSON, which is not text and is redacted whole.
enum Held {
    Text(Streamed, bool),
    Arguments(String),
}

/// Rewrites one reply's stream redacted: each text or reasoning part as it
/// streams, holding only what may still become part of a secret
/// ([`Streamed`]); each tool call's arguments at its end; every other event
/// and the response it ends with as JSON. What a part holds goes out
/// before its end, and what every part holds before the response.
fn redact_stream() -> impl FnMut(Relayed) -> Vec<Relayed> {
    let mut held: Vec<(Part, Held)> = Vec::new();
    move |item| {
        let mut out = Vec::new();
        match item {
            Relayed::Item(Item::Event(StreamEvent::Text { part, text })) => {
                grow(&mut held, (part, Some(false)), &text, &mut out);
            }
            Relayed::Item(Item::Event(StreamEvent::Reasoning { part, text })) => {
                grow(&mut held, (part, Some(true)), &text, &mut out);
            }
            Relayed::Item(Item::Event(StreamEvent::Arguments { part, json })) => {
                grow(&mut held, (part, None), &json, &mut out);
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

/// Takes the next `piece` of `part`'s text, or reasoning (`Some(true)`),
/// putting what can be shown in `out`, or of its tool call's arguments
/// (`None`).
fn grow(
    held: &mut Vec<(Part, Held)>,
    (part, reasoning): (Part, Option<bool>),
    piece: &str,
    out: &mut Vec<Relayed>,
) {
    let at = held
        .iter()
        .position(|(at, _)| *at == part)
        .unwrap_or_else(|| {
            let new = match reasoning {
                Some(reasoning) => Held::Text(Streamed::default(), reasoning),
                None => Held::Arguments(String::new()),
            };
            held.push((part, new));
            held.len() - 1
        });
    match held.get_mut(at) {
        Some((_, Held::Text(streamed, reasoning))) => {
            out.extend(event(part, *reasoning, streamed.push(piece)));
        }
        Some((_, Held::Arguments(json))) => json.push_str(piece),
        None => {}
    }
}

/// `text` of `part` as an event, if any.
fn event(part: Part, reasoning: bool, text: String) -> Option<Relayed> {
    let event = match reasoning {
        _ if text.is_empty() => return None,
        true => StreamEvent::Reasoning { part, text },
        false => StreamEvent::Text { part, text },
    };
    Some(Relayed::Item(Item::Event(event)))
}

/// Puts in `out` what `part` holds, or every part when `None`, and forgets
/// it.
fn let_go(held: &mut Vec<(Part, Held)>, part: Option<Part>, out: &mut Vec<Relayed>) {
    let (going, staying) = std::mem::take(held)
        .into_iter()
        .partition(|(at, _)| part.is_none_or(|part| part == *at));
    *held = staying;
    for (part, going) in going {
        out.extend(match going {
            Held::Text(mut streamed, reasoning) => event(part, reasoning, streamed.finish()),
            Held::Arguments(json) => (!json.is_empty()).then(|| {
                // Arguments that are no JSON, a call that fails, as text.
                let json = match serde_json::from_str::<Value>(&json) {
                    Ok(value) => redacted(&value).map_or(json, |value| value.to_string()),
                    Err(_) => redact::text(&json).into_owned(),
                };
                Relayed::Item(Item::Event(StreamEvent::Arguments { part, json }))
            }),
        });
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

/// `line` redacted, given and keeping whether it is inside a private key
/// block, which spans lines: its first is redacted to its end and the lines
/// after it, through its end line, are left out.
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

/// Every record reaches the effect log redacted: the request as sent, a
/// call's first answer, what it published.
fn redact_records(mut records: MessageMutator<EffectRecorded>) {
    for EffectRecorded(record) in records.read() {
        if let Some(redacted) = redacted(&*record) {
            *record = redacted;
        }
    }
}

/// Every prompt the terminal view keeps for Up and saves.
#[cfg(feature = "tui")]
fn redact_history(mut history: ResMut<rig_tui::PromptHistory>) {
    let rewrite = |prompt: &str| match redact::text(prompt) {
        Cow::Owned(redacted) => Some(redacted),
        Cow::Borrowed(_) => None,
    };
    history.bypass_change_detection().rewrite(rewrite);
}
