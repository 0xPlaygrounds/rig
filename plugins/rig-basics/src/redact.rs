//! Redaction ([`RedactPlugin`]): each distinct API key, token, private key
//! and value of a secret-named variable that [`rig_core::redact`] finds
//! becomes a placeholder, `‹SECRET_1›`, where it comes in: the model's reply
//! as it streams, each tool's output and answer, and each message. The
//! model, the conversation, the logs, every front, subagents and a resumed
//! session hold only placeholders. Without this plugin nothing is redacted.
//!
//! A tool call that names a placeholder runs with its secret: the secrets
//! are kept in memory for the run, never saved or sent, and a call naming
//! one this run does not know, such as one from before a restart, is
//! refused.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_basics::redact::RedactPlugin);
//! ```

use std::borrow::Cow;
use std::future::ready;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::redact::{self, Streamed};
use rig_harness::rig_core::serve::{AfterServe, BeforeServe, Intercept};
use rig_harness::rig_core::streaming::{Item, Relayed};
use rig_harness::rig_core::tool::{ContextValue, FilesChanged, Hunk, HunkLine};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::tools::ResultFilter;
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_tools::LineFilter;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// The layers' and the decider's name.
const NAME: &str = "redact";

/// What the model is told of the placeholders.
const PROMPT: &str = "A secret, such as an API key, shows as a placeholder such as \
‹SECRET_1›. Write the placeholder where its value is needed in a tool call: the tool gets \
the value, which you never see.";

/// Redacts every secret where it comes in, so nothing downstream holds it.
#[derive(Default)]
pub struct RedactPlugin;

/// After every [`ReviewSystems`] phase, in [`Stage::Review`]: each tool
/// call's answer redacted again, as a reviewer may have changed it in
/// place. A system that reads answers there to keep them goes after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedactSystems;

/// The run's secrets, `‹SECRET_n›` being the n-th, shared with the calls
/// that stream and run off the main thread; `None` for a number an earlier
/// run handed out. Never saved.
#[derive(Resource, Clone, Default)]
struct Secrets(Arc<Mutex<Vec<Option<String>>>>);

/// How many placeholders the session handed out, saved with it, so a later
/// run never gives an old number to a new secret.
#[derive(Resource, Reflect, Default, PartialEq)]
#[reflect(Resource, Saved)]
struct Issued(usize);

impl Plugin for RedactPlugin {
    fn build(&self, app: &mut App) {
        let secrets = Secrets::default();
        let filter = secrets.clone();
        app.insert_resource(secrets)
            .insert_resource(ResultFilter(Arc::new(move |result| {
                filter.redacted(result)
            })))
            .init_resource::<Issued>()
            .register_type::<Issued>()
            .configure_sets(
                AgentPass,
                RedactSystems
                    .in_set(Stage::Review)
                    .after(ReviewSystems::Act),
            )
            .add_observer(redact_outcome)
            .add_observer(redact_if_stopped)
            .add_observer(redact_facts)
            .add_systems(Startup, (keep_count, tell_the_model))
            // After the frame's agent work, so the count saved is what it handed out.
            .add_systems(PostUpdate, keep_count)
            .add_systems(
                AgentPass,
                // Each after every other plugin that edits the same data.
                (
                    adapt_model_calls
                        .in_set(Stage::Prepare)
                        .after(PrepareSystems::AdaptOutermost)
                        .before(PrepareSystems::Fit),
                    reveal_to_tools
                        .in_set(Stage::Prepare)
                        .after(PrepareSystems::Edit)
                        .before(PrepareSystems::Adapt),
                    filter_output
                        .after(PerCallSystems::Place)
                        .before(PerCallSystems::Build),
                    redact_reviewed.in_set(RedactSystems),
                    refuse_unknown.in_set(Stage::Gate),
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

/// The placeholder of the n-th secret.
fn placeholder(number: usize) -> String {
    format!("‹SECRET_{number}›")
}

/// Each placeholder in `text`: where it is, and its number.
fn placeholders(text: &str) -> impl Iterator<Item = (Range<usize>, usize)> + '_ {
    text.match_indices("‹SECRET_")
        .filter_map(move |(at, open)| {
            let digits = text.get(at + open.len()..)?;
            let len = digits.find(|ch: char| !ch.is_ascii_digit())?;
            let number = digits.get(..len)?.parse().ok()?;
            let end = at + open.len() + len + '›'.len_utf8();
            digits
                .get(len..)?
                .starts_with('›')
                .then_some((at..end, number))
        })
}

impl Secrets {
    fn vault(&self) -> MutexGuard<'_, Vec<Option<String>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `input` with each secret replaced by its placeholder, learned when new.
    fn text<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut vault = self.vault();
        redact::text_with(input, |secret| name(&mut vault, secret))
    }

    /// `data` with every string in it redacted, through its JSON form;
    /// `None` when it holds no secret or its redacted form does not read
    /// back.
    fn redacted<T: Serialize + DeserializeOwned>(&self, data: &T) -> Option<T> {
        let mut json = serde_json::to_value(data).ok()?;
        let mut vault = self.vault();
        let changed = redact::json_with(&mut json, &mut |secret| name(&mut vault, secret));
        drop(vault);
        changed.then(|| serde_json::from_value(json).ok()).flatten()
    }

    /// `text` with each placeholder this run knows replaced by its secret.
    fn reveal<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let vault = self.vault();
        let mut out = String::new();
        let mut from = 0;
        for (at, number) in placeholders(text) {
            let known = number
                .checked_sub(1)
                .and_then(|at| vault.get(at)?.as_deref());
            if let Some(secret) = known {
                out.push_str(text.get(from..at.start).unwrap_or_default());
                out.push_str(secret);
                from = at.end;
            }
        }
        if from == 0 {
            return Cow::Borrowed(text);
        }
        out.push_str(text.get(from..).unwrap_or_default());
        Cow::Owned(out)
    }
}

/// The placeholder that names `secret`, learning it when new.
fn name(vault: &mut Vec<Option<String>>, secret: &str) -> String {
    let known = vault
        .iter()
        .position(|known| known.as_deref() == Some(secret));
    let at = known.unwrap_or_else(|| {
        vault.push(Some(secret.to_owned()));
        vault.len() - 1
    });
    placeholder(at + 1)
}

/// The numbers earlier runs handed out stay taken; the ones this run hands
/// out are saved as taken.
fn keep_count(mut issued: ResMut<Issued>, secrets: Res<Secrets>) {
    let mut vault = secrets.vault();
    while vault.len() < issued.0 {
        vault.push(None);
    }
    issued.set_if_neq(Issued(vault.len()));
}

/// Tells every agent's model how to use a placeholder.
fn tell_the_model(mut commands: Commands) {
    commands.spawn(PromptSection::new(900, "secrets", PROMPT));
}

/// A tool call's result or an error, redacted; `None` when it holds no
/// secret. A plugin's own answer, such as the line that answers its
/// question, reaches it as given.
fn redacted_answer(answer: &Outcome, secrets: &Secrets) -> Option<Result<Answered, ErrorReport>> {
    match &answer.0 {
        Ok(Answered::ToolResult { .. }) | Err(_) => secrets.redacted(&answer.0),
        Ok(_) => None,
    }
}

/// An answer as it is inserted: a result, a stop's answer, one kept at
/// exit, a model call's error.
fn redact_outcome(
    inserted: On<Insert<Outcome>>,
    mut answers: Query<&mut Outcome>,
    secrets: Res<Secrets>,
) {
    if let Ok(mut answer) = answers.get_mut(inserted.entity)
        && let Some(redacted) = redacted_answer(&answer, &secrets)
    {
        answer.0 = redacted;
    }
}

/// What a running call answers if stopped, which a stop reads as it is.
fn redact_if_stopped(
    inserted: On<Insert<IfStopped>>,
    mut answers: Query<&mut IfStopped>,
    secrets: Res<Secrets>,
) {
    if let Ok(mut answer) = answers.get_mut(inserted.entity)
        && let Some(redacted) = redacted_answer(&answer.0, &secrets)
    {
        answer.0.0 = redacted;
    }
}

/// Each tool call's answer after every reviewer, which may have changed it
/// in place, and before it is committed.
fn redact_reviewed(
    mut answers: Query<&mut Outcome, (ToolCalls, Changed<Outcome>, Without<Settled>)>,
    secrets: Res<Secrets>,
) {
    for mut answer in &mut answers {
        if let Some(redacted) = redacted_answer(&answer, &secrets) {
            answer.0 = redacted;
        }
    }
}

/// What a call published, which the conversation keeps with its result.
/// A changed file's old text, which it never keeps, stays whole.
fn redact_facts(
    inserted: On<Insert<Published>>,
    mut published: Query<&mut Published>,
    secrets: Res<Secrets>,
) {
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
                    let changed =
                        hunks.fold(false, |any, hunks| redact_hunks(hunks, &secrets) | any);
                    changed.then(|| serde_json::to_value(files).ok()).flatten()
                }),
            false => secrets.redacted(value),
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
fn redact_hunks(hunks: &mut Vec<Hunk>, secrets: &Secrets) -> bool {
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
        if let Cow::Owned(redacted) = secrets.text(text) {
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
    secrets: Res<Secrets>,
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
                        if let Some(redacted) = secrets.redacted(&*content) {
                            *content = redacted;
                            changed = true;
                        }
                        continue;
                    }
                },
            };
            if let Cow::Owned(redacted) = secrets.text(text) {
                *text = redacted;
                changed = true;
            }
        }
        if changed {
            message.set_changed();
        }
    }
}

/// A refused command's line, and why it was refused, which may quote it.
fn redact_recalled(mut recalled: MessageMutator<Recalled>, secrets: Res<Secrets>) {
    for recalled in recalled.read() {
        let texts = std::iter::once(&mut recalled.text).chain(&mut recalled.why);
        for text in texts {
            if let Cow::Owned(redacted) = secrets.text(text) {
                *text = redacted;
            }
        }
    }
}

/// Each tool call runs with the secrets its placeholders name: only the
/// call as it is sent to its tool holds them, not the call every view and
/// log reads, and its answer is redacted again.
fn reveal_to_tools(
    mut calls: Query<&mut CallHandler, (ToolCalls, Added<Admitted>)>,
    secrets: Res<Secrets>,
) {
    for mut handler in &mut calls {
        let revealing = handler.0.0.clone().layered(Reveal(secrets.clone()));
        handler.0 = Handler(revealing);
    }
}

/// Puts the secrets back in a tool call's arguments as it is sent.
struct Reveal(Secrets);

impl Intercept for Reveal {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    fn before(&self, _: EffectId, kind: &EffectKind) -> impl Future<Output = BeforeServe> + Send {
        let revealed = match kind {
            EffectKind::ToolCall { name, args } => serde_json::from_str::<Value>(args)
                .ok()
                .and_then(|mut args| {
                    let changed = each_string(&mut args, &mut |text| match self.0.reveal(text) {
                        Cow::Owned(revealed) => Some(revealed),
                        Cow::Borrowed(_) => None,
                    });
                    changed.then(|| EffectKind::ToolCall {
                        name: name.clone(),
                        args: args.to_string(),
                    })
                }),
            _ => None,
        };
        ready(revealed.map_or(BeforeServe::Proceed, BeforeServe::Patch))
    }

    fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Answered, ErrorReport>,
    ) -> impl Future<Output = AfterServe> + Send {
        ready(AfterServe::Keep)
    }
}

/// Every string in `value` that `change` gives a new text for; whether any.
fn each_string(value: &mut Value, change: &mut impl FnMut(&str) -> Option<String>) -> bool {
    match value {
        Value::String(text) => change(text).map(|changed| *text = changed).is_some(),
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |any, item| each_string(item, change) | any),
        Value::Object(fields) => fields
            .values_mut()
            .fold(false, |any, field| each_string(field, change) | any),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// A tool call naming a placeholder this run does not know, such as one
/// from before a restart, is refused: nothing runs with the placeholder
/// as its value.
fn refuse_unknown(
    mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>,
    secrets: Res<Secrets>,
) {
    for (request, mut decision) in &mut calls {
        let args = Value::Object(request.call.function.arguments.clone()).to_string();
        let vault = secrets.vault();
        let unknown = placeholders(&args).find(|(_, number)| {
            let at = number.checked_sub(1);
            at.and_then(|at| vault.get(at)?.as_ref()).is_none()
        });
        if let Some((at, _)) = unknown {
            let why = format!(
                "{} is not a secret this run knows: secrets are kept in memory only, so one \
                 from before a restart is gone. Nothing ran; ask the user for the value again.",
                args.get(at).unwrap_or_default()
            );
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
        }
    }
}

/// Every model call streams its reply redacted, wrapping the outermost
/// adapters too, such as a replay.
fn adapt_model_calls(
    mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>,
    secrets: Res<Secrets>,
) {
    for mut adapted in &mut calls {
        let handler = adapted.0.handler.0.clone();
        let secrets = secrets.clone();
        let redacting = handler.map_stream(NAME, move |_| redact_stream(secrets.clone()));
        adapted.0.handler = Handler(redacting);
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
    /// Takes `piece`; what can go out now, redacted.
    fn push(&mut self, piece: &str, secrets: &Secrets) -> String {
        match self {
            Self::Text(text) | Self::Reasoning(text) => {
                secrets.text(&text.push(piece)).into_owned()
            }
            Self::Arguments(json) => {
                json.push_str(piece);
                String::new()
            }
        }
    }

    /// What it still holds, redacted: arguments that are no JSON, of a call
    /// that fails, as text.
    fn finish(&mut self, secrets: &Secrets) -> String {
        match self {
            Self::Text(text) | Self::Reasoning(text) => secrets.text(&text.finish()).into_owned(),
            Self::Arguments(json) => match serde_json::from_str::<Value>(json) {
                Ok(value) => secrets
                    .redacted(&value)
                    .map_or_else(|| json.clone(), |value| value.to_string()),
                Err(_) => secrets.text(json).into_owned(),
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
fn redact_stream(secrets: Secrets) -> impl FnMut(Relayed) -> Vec<Relayed> {
    let mut held: Vec<(Part, Held)> = Vec::new();
    move |item| {
        let mut out = Vec::new();
        match item {
            Relayed::Item(Item::Event(StreamEvent::Text { part, text })) => {
                let new = || Held::Text(Streamed::default());
                grow(&mut held, (part, new), &text, (&secrets, &mut out));
            }
            Relayed::Item(Item::Event(StreamEvent::Reasoning { part, text })) => {
                let new = || Held::Reasoning(Streamed::default());
                grow(&mut held, (part, new), &text, (&secrets, &mut out));
            }
            Relayed::Item(Item::Event(StreamEvent::Arguments { part, json })) => {
                let new = || Held::Arguments(String::new());
                grow(&mut held, (part, new), &json, (&secrets, &mut out));
            }
            Relayed::Item(item) => {
                if let Item::Event(StreamEvent::End { part, .. }) = &item {
                    let_go(&mut held, Some(*part), (&secrets, &mut out));
                }
                out.push(Relayed::Item(secrets.redacted(&item).unwrap_or(item)));
            }
            Relayed::Done(mut response) => {
                let_go(&mut held, None, (&secrets, &mut out));
                if let Some(redacted) = secrets.redacted(&*response) {
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
    (secrets, out): (&Secrets, &mut Vec<Relayed>),
) {
    let at = held
        .iter()
        .position(|(at, _)| *at == part)
        .unwrap_or_else(|| {
            held.push((part, new()));
            held.len() - 1
        });
    if let Some((_, held)) = held.get_mut(at) {
        let shown = held.push(piece, secrets);
        out.extend(held.event(part, shown));
    }
}

/// Puts in `out` what `part`, or every part, holds.
fn let_go(
    held: &mut Vec<(Part, Held)>,
    part: Option<Part>,
    (secrets, out): (&Secrets, &mut Vec<Relayed>),
) {
    let (going, staying) = std::mem::take(held)
        .into_iter()
        .partition(|(at, _)| part.is_none_or(|part| part == *at));
    *held = staying;
    for (part, mut going) in going {
        let text = going.finish(secrets);
        out.extend(going.event(part, text));
    }
}

/// Each line of every call's output, before the tool reports, spills or
/// keeps it ([`redact_line`]).
fn filter_output(
    mut places: Query<&mut CallPlace, (ToolCalls, Added<CallPlace>)>,
    secrets: Res<Secrets>,
) {
    for mut place in &mut places {
        let in_key = Mutex::new(false);
        let secrets = secrets.clone();
        place.filter = Some(LineFilter::new(move |line| {
            let mut in_key = in_key.lock().unwrap_or_else(PoisonError::into_inner);
            redact_line(line, &mut in_key, &secrets)
        }));
    }
}

/// `line` redacted; a private key block's lines after its first, through
/// its end line, are left out (`in_key`).
fn redact_line(line: &str, in_key: &mut bool, secrets: &Secrets) -> String {
    if *in_key {
        *in_key = !line.contains("-----END ");
        return String::new();
    }
    let Cow::Owned(mut redacted) = secrets.text(line) else {
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
fn redact_records(mut records: MessageMutator<EffectRecorded>, secrets: Res<Secrets>) {
    for EffectRecorded(record) in records.read() {
        if let Some(redacted) = secrets.redacted(&*record) {
            *record = redacted;
        }
    }
}

/// Every prompt the terminal view keeps for Up and saves.
#[cfg(feature = "tui")]
fn redact_history(mut history: ResMut<rig_tui::PromptHistory>, secrets: Res<Secrets>) {
    let rewrite = |prompt: &str| match secrets.text(prompt) {
        Cow::Owned(redacted) => Some(redacted),
        Cow::Borrowed(_) => None,
    };
    history.bypass_change_detection().rewrite(rewrite);
}
