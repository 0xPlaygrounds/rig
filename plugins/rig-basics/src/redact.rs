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
use std::cmp::Reverse;
use std::future::ready;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::message::{MessageCursor, Messages};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message;
use rig_harness::rig_core::redact::{self, Streamed};
use rig_harness::rig_core::serve::{AfterServe, BeforeServe, Intercept};
use rig_harness::rig_core::streaming::{Item, Relayed};
use rig_harness::rig_core::tool::{ContextValue, FilesChanged, Hunk, HunkLine};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::tools::ResultFilter;
use rig_harness::rig_ecs::{SaveSystems, StopTurnsOnExit};
use rig_harness::run::RunExitSystems;
use rig_tools::LineFilter;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// The layers' and the decider's name.
const NAME: &str = "redact";

/// The shortest original found again where it shows bare, as short as a
/// secret [`redact::find`] finds: shorter text is likely something else.
const SHORTEST: usize = 8;

/// The most of a private key block a command's output holds for it.
const BLOCK_MOST: usize = 16 * 1024;

/// What a secret revealed into a shell command may not hold: the shell would
/// read it as code, not the value the call was approved with.
const SHELL_SPECIAL: &[char] = &[
    '\'', '"', '`', '$', '\\', ';', '&', '|', '<', '>', '(', ')', '\n', '\r', ' ', '\t', '*', '?',
    '!', '#', '~', '{', '}', '[', ']',
];

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

/// The run's secrets, shared with the calls that stream and run off the
/// main thread. Never saved.
#[derive(Resource, Clone, Default)]
struct Secrets(Arc<Mutex<Vault>>);

/// The secrets, `‹SECRET_n›` being the n-th, `None` for a number an earlier
/// run handed out; and the names a model request gives, its model's and its
/// tools', which a secret the run knows is never found in, a name being the
/// program's text.
#[derive(Default)]
struct Vault {
    secrets: Vec<Option<String>>,
    names: Vec<String>,
}

impl Vault {
    /// Each secret found again wherever it shows, with its number: one of
    /// at least [`SHORTEST`] characters that is no name, nor part of one.
    fn findable(&self) -> impl Iterator<Item = (usize, &str)> {
        (1..)
            .zip(&self.secrets)
            .filter_map(|(number, secret)| Some((number, secret.as_deref()?)))
            .filter(|(_, secret)| {
                secret.len() >= SHORTEST && !self.names.iter().any(|name| name.contains(secret))
            })
    }
}

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
            .insert_resource(ResultFilter(Arc::new(move |answer| {
                redacted_answer(answer, &filter).map(Outcome)
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
                ),
            )
            .add_systems(BetweenStages, (redact_recalled, redact_notices))
            .add_systems(
                Last,
                (
                    // On the exit frame: what is kept and printed as it is.
                    (redact_messages, redact_stopped)
                        .run_if(on_message::<AppExit>)
                        .before(RunExitSystems)
                        .before(StopTurnsOnExit),
                    (
                        redact_records
                            .run_if(resource_exists::<crate::telemetry::effect_log::EffectLog>),
                        // Every number handed out, the exit's too, is saved.
                        keep_count,
                    )
                        .chain()
                        .after(StopTurnsOnExit)
                        .before(SaveSystems),
                ),
            );
        // In the key schedule too, so a prompt seeded from a conversation
        // shown is redacted before the key after it recalls it.
        #[cfg(feature = "tui")]
        app.add_systems(
            Update,
            redact_history
                .run_if(resource_exists_and_changed::<rig_tui::PromptHistory>)
                .after(rig_tui::HistorySystems::Record)
                .before(rig_tui::HistorySystems::Save),
        )
        .add_systems(
            rig_tui::TuiKeys,
            redact_history
                .run_if(resource_exists_and_changed::<rig_tui::PromptHistory>)
                .after(rig_tui::HistorySystems::Seed)
                .before(rig_tui::InputSystems::Filter),
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
    fn vault(&self) -> MutexGuard<'_, Vault> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `input` with each secret replaced by its placeholder, learned when
    /// new, and each one this run knows wherever it shows bare.
    fn text<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut vault = self.vault();
        let mut redacted = redact::text_with(input, |secret| name(&mut vault.secrets, secret));
        if let Some(bare) = known(&vault, &redacted) {
            redacted = Cow::Owned(bare);
        }
        redacted
    }

    /// `data` with every string in it redacted, through its JSON form;
    /// `None` when it holds no secret or its redacted form does not read
    /// back.
    ///
    /// A secret that is, or holds, a field's tag would change the tag, so a
    /// string whose change alone keeps `data` from reading back is kept as
    /// given: a tag is the program's text, not data.
    fn redacted<T: Serialize + DeserializeOwned>(&self, data: &T) -> Option<T> {
        let given = serde_json::to_value(data).ok()?;
        let mut redacted = given.clone();
        if !json(self, &mut redacted) {
            return None;
        }
        // Again, for a secret a later string taught to show bare in an
        // earlier one.
        json(self, &mut redacted);
        let reads = |json: &Value| T::deserialize(json).is_ok();
        if !reads(&redacted) {
            let mut strings = Vec::new();
            changed(&given, &redacted, String::new(), &mut strings);
            for at in strings {
                let mut alone = given.clone();
                if let (Some(string), Some(new)) = (alone.pointer_mut(&at), redacted.pointer(&at)) {
                    string.clone_from(new);
                }
                if !reads(&alone)
                    && let (Some(string), Some(old)) =
                        (redacted.pointer_mut(&at), given.pointer(&at))
                {
                    string.clone_from(old);
                }
            }
        }
        T::deserialize(&redacted).ok()
    }

    /// `request` redacted in its own text: its system prompt, its documents
    /// and each user message's text, such as a plugin's request-only ask.
    /// Its other messages were redacted as they came in, and the rest
    /// (tools, schemas, media, settings) is the program's text or data a
    /// key's shape could match by chance. `None` when none held a secret.
    fn request(&self, request: &CompletionRequest) -> Option<CompletionRequest> {
        let asked = self.request_once(request)?;
        // Again, for a secret a later field taught to show bare in an
        // earlier one.
        Some(self.request_once(&asked).unwrap_or(asked))
    }

    /// One pass of [`Self::request`].
    fn request_once(&self, request: &CompletionRequest) -> Option<CompletionRequest> {
        let mut asked = request.clone();
        let mut changed = false;
        for entry in &mut asked.chat_history {
            let redacted = match &**entry {
                message::Message::System { content } => {
                    owned(self.text(content)).map(|content| message::Message::System { content })
                }
                message::Message::User { content } => {
                    let texts: Vec<_> = content
                        .iter()
                        .map(|part| match part {
                            UserContent::Text(text) => owned(self.text(&text.text)),
                            _ => None,
                        })
                        .collect();
                    texts.iter().any(Option::is_some).then(|| {
                        let parts = content.iter().zip(texts).map(|(part, text)| {
                            let mut part = part.clone();
                            if let (UserContent::Text(was), Some(text)) = (&mut part, text) {
                                was.text = text;
                            }
                            part
                        });
                        message::Message::User {
                            content: parts.collect(),
                        }
                    })
                }
                _ => None,
            };
            if let Some(redacted) = redacted {
                *entry = Arc::new(redacted);
                changed = true;
            }
        }
        for document in &mut asked.documents {
            let props = document.additional_props.values_mut();
            for text in [&mut document.id, &mut document.text]
                .into_iter()
                .chain(props)
            {
                if let Some(redacted) = owned(self.text(text)) {
                    *text = redacted;
                    changed = true;
                }
            }
        }
        changed.then_some(asked)
    }

    /// `text` with each placeholder this run knows replaced by its secret.
    fn reveal<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let vault = self.vault();
        let mut out = String::new();
        let mut from = 0;
        for (at, number) in placeholders(text) {
            let known = number
                .checked_sub(1)
                .and_then(|at| vault.secrets.get(at)?.as_deref());
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

/// `text` with each secret `vault` finds again ([`Vault::findable`])
/// replaced by its placeholder, longest first, so one inside another goes
/// with it, such as a password a tool printed without the name that made it
/// a secret, but never inside a placeholder; `None` when none shows.
fn known(vault: &Vault, text: &str) -> Option<String> {
    known_as(vault, text, placeholder)
}

/// [`known`], each secret replaced by what `with` gives for its number.
fn known_as(vault: &Vault, text: &str, with: impl Fn(usize) -> String) -> Option<String> {
    let mut shown: Vec<(usize, &str)> = vault
        .findable()
        .filter(|(_, secret)| text.contains(secret))
        .collect();
    shown.sort_by_key(|(_, secret)| Reverse(secret.len()));
    let mut out = Cow::Borrowed(text);
    for (number, secret) in shown {
        // Between the placeholders already there, `SECRET_1` is not found in
        // `‹SECRET_1›`.
        let mut replaced = String::with_capacity(out.len());
        let mut from = 0;
        let spans = placeholders(&out).map(|(at, _)| at);
        for span in spans.chain(std::iter::once(out.len()..out.len())) {
            let gap = out.get(from..span.start).unwrap_or_default();
            replaced.push_str(&gap.replace(secret, &with(number)));
            replaced.push_str(out.get(span.clone()).unwrap_or_default());
            from = span.end;
        }
        if replaced != *out {
            out = Cow::Owned(replaced);
        }
    }
    match out {
        Cow::Owned(out) => Some(out),
        Cow::Borrowed(_) => None,
    }
}

/// The JSON pointer of each string in `before` that `after`, the same JSON
/// but for its strings, changes.
fn changed(before: &Value, after: &Value, at: String, out: &mut Vec<String>) {
    match (before, after) {
        (Value::String(before), Value::String(after)) if before != after => out.push(at),
        (Value::Array(before), Value::Array(after)) => {
            for (index, (before, after)) in before.iter().zip(after).enumerate() {
                changed(before, after, format!("{at}/{index}"), out);
            }
        }
        (Value::Object(before), Value::Object(after)) => {
            for (key, before) in before {
                if let Some(after) = after.get(key) {
                    let key = key.replace('~', "~0").replace('/', "~1");
                    changed(before, after, format!("{at}/{key}"), out);
                }
            }
        }
        _ => {}
    }
}

/// Every string in `value` redacted as [`Secrets::text`] redacts it, and
/// one that holds JSON, such as a call's arguments, as that JSON too, where
/// an escape would hide a secret; whether any was. A file's encoded bytes,
/// such as an image's base64, are left whole: a key's shape or a known
/// secret could match them by chance and break them.
fn json(secrets: &Secrets, value: &mut Value) -> bool {
    match value {
        Value::String(text) => match json_text(secrets, text) {
            Some(changed) => {
                *text = changed;
                true
            }
            None => false,
        },
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |any, item| json(secrets, item) | any),
        Value::Object(fields) => {
            // How a media source's base64 serializes: its value stays whole
            // when it is base64.
            let encoded = fields.get("type").and_then(Value::as_str) == Some("base64")
                && fields
                    .get("value")
                    .and_then(Value::as_str)
                    .is_some_and(is_base64);
            fields
                .iter_mut()
                .filter(|(key, _)| !(encoded && key.as_str() == "value"))
                .fold(false, |any, (_, field)| json(secrets, field) | any)
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// One string of [`json`]: as text, then as the JSON it holds; `None` when
/// unchanged. A `data:` URL of base64 is left whole.
fn json_text(secrets: &Secrets, text: &str) -> Option<String> {
    let data_url = text
        .strip_prefix("data:")
        .and_then(|url| url.split_once(";base64,"));
    if data_url.is_some_and(|(_, encoded)| is_base64(encoded)) {
        return None;
    }
    let mut changed = owned(secrets.text(text));
    let current = changed.as_deref().unwrap_or(text);
    if current.trim_start().starts_with(['{', '['])
        && let Ok(mut inner) = serde_json::from_str::<Value>(current)
        && json(secrets, &mut inner)
    {
        changed = Some(inner.to_string());
    }
    changed
}

/// Whether `text` is made of base64's characters only.
fn is_base64(text: &str) -> bool {
    text.bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
}

/// The text a [`Cow`] holds when it was changed.
fn owned(text: Cow<'_, str>) -> Option<String> {
    match text {
        Cow::Owned(text) => Some(text),
        Cow::Borrowed(_) => None,
    }
}

/// The longest end of `text` that starts a secret `vault` knows, which a
/// later piece may finish: held back from a stream until it does.
fn starts_known(vault: &Vault, text: &str) -> usize {
    vault
        .findable()
        .map(|(_, secret)| secret)
        .flat_map(|secret| {
            (1..secret.len().min(text.len() + 1))
                .rev()
                .filter(|&len| secret.is_char_boundary(len))
                .find(|&len| secret.get(..len).is_some_and(|start| text.ends_with(start)))
        })
        .max()
        .unwrap_or(0)
}

/// The numbers earlier runs handed out stay taken; the ones this run hands
/// out are saved as taken.
fn keep_count(mut issued: ResMut<Issued>, secrets: Res<Secrets>) {
    let mut vault = secrets.vault();
    while vault.secrets.len() < issued.0 {
        vault.secrets.push(None);
    }
    issued.set_if_neq(Issued(vault.secrets.len()));
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

/// What a running call answers if stopped, changed in place since it was
/// inserted, as the exit prints and keeps it.
fn redact_stopped(mut answers: Query<&mut IfStopped, Changed<IfStopped>>, secrets: Res<Secrets>) {
    for mut answer in &mut answers {
        if let Some(redacted) = redacted_answer(&answer.0, &secrets) {
            answer.0.0 = redacted;
        }
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
    let pass = |slots: &mut serde_json::Map<String, Value>| {
        let mut changed = false;
        for (key, value) in slots {
            let redacted = match key == FilesChanged::KEY {
                true => serde_json::from_value::<FilesChanged>(value.clone())
                    .ok()
                    .and_then(|mut files| {
                        let changed = files.0.iter_mut().fold(false, |any, change| {
                            // A path can hold a secret, as one a call wrote a
                            // file named after.
                            let renamed = match secrets.text(&change.path) {
                                Cow::Owned(path) => {
                                    change.path = path;
                                    true
                                }
                                Cow::Borrowed(_) => false,
                            };
                            redact_hunks(&mut change.hunks, &secrets) | renamed | any
                        });
                        changed.then(|| serde_json::to_value(files).ok()).flatten()
                    }),
                false => secrets.redacted(value),
            };
            if let Some(redacted) = redacted {
                *value = redacted;
                changed = true;
            }
        }
        changed
    };
    // Again, for a secret a later slot taught to show bare in an earlier one.
    let changed = pass(&mut slots);
    if changed {
        pass(&mut slots);
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
        let parts = &mut message.bypass_change_detection().parts;
        // Again, for a secret a later part taught to show bare in an earlier
        // one.
        let changed = redact_parts(parts, &secrets);
        if changed {
            redact_parts(parts, &secrets);
        }
        if changed {
            message.set_changed();
        }
    }
}

/// Each of a message's parts redacted: its text, an attachment's label,
/// text and link, a document's or a result's text; whether any was.
fn redact_parts(parts: &mut [MessagePart], secrets: &Secrets) -> bool {
    let mut changed = false;
    for part in parts {
        let text = match part {
            MessagePart::Text(text) => text,
            MessagePart::Attachment(Attachment { label, content }) => {
                // A label is a path or a URI, which may hold a key too.
                if let Cow::Owned(redacted) = secrets.text(label) {
                    *label = redacted;
                    changed = true;
                }
                match content {
                    UserContent::Text(text) => &mut text.text,
                    // A medium's link, not its data, which a key's shape
                    // could match by chance.
                    UserContent::Image(message::Image { data, .. })
                    | UserContent::Audio(message::Audio { data, .. })
                    | UserContent::Video(message::Video { data, .. }) => match data {
                        message::DocumentSourceKind::Url(url) => url,
                        _ => continue,
                    },
                    // A document's or a result's text, as JSON.
                    content => {
                        if let Some(redacted) = secrets.redacted(&*content) {
                            *content = redacted;
                            changed = true;
                        }
                        continue;
                    }
                }
            }
        };
        if let Cow::Owned(redacted) = secrets.text(text) {
            *text = redacted;
            changed = true;
        }
    }
    changed
}

/// A notice, which may quote what the user typed, such as a file a message
/// names that is not there; between the stages, before a later stage, a
/// front's observer of it or a front after the agent loop reads it. Every
/// one kept, each time, so a secret learned since is found in it too.
fn redact_notices(mut notices: ResMut<Messages<Notice>>, secrets: Res<Secrets>) {
    for notice in MessageCursor::default().read_mut(&mut notices) {
        if let Some(text) = owned(secrets.text(&notice.text)) {
            notice.text = text;
        }
    }
}

/// A refused command's line, and why it was refused, which may quote it;
/// every one kept, as [`redact_notices`] does.
fn redact_recalled(mut recalled: ResMut<Messages<Recalled>>, secrets: Res<Secrets>) {
    for recalled in MessageCursor::default().read_mut(&mut recalled) {
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
        let revealing = handler.0.0.clone().layered(AsSent(secrets.clone()));
        handler.0 = Handler(revealing);
    }
}

/// What a call sends: a tool call's arguments with their secrets put back,
/// a model request redacted in its own text ([`Secrets::request`]), and any
/// other effect redacted whole.
struct AsSent(Secrets);

impl Intercept for AsSent {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    fn before(&self, _: EffectId, kind: &EffectKind) -> impl Future<Output = BeforeServe> + Send {
        let sent = match kind {
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
            EffectKind::Completion { request, stream } => {
                let mut vault = self.0.vault();
                let tools = request.tools.iter().map(|tool| tool.name.as_str());
                for name in request.model.as_deref().into_iter().chain(tools) {
                    if !vault.names.iter().any(|known| known == name) {
                        vault.names.push(name.to_owned());
                    }
                }
                drop(vault);
                self.0.request(request).map(|asked| EffectKind::Completion {
                    request: Arc::new(asked),
                    stream: *stream,
                })
            }
            kind => self.0.redacted(kind),
        };
        ready(sent.map_or(BeforeServe::Proceed, BeforeServe::Patch))
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
/// as its value. So is a shell command naming one whose secret holds what
/// the shell reads as code: the command that ran would not be the one
/// approved.
fn refuse_unknown(
    mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>,
    secrets: Res<Secrets>,
) {
    for (request, mut decision) in &mut calls {
        let args = Value::Object(request.call.function.arguments.clone()).to_string();
        let vault = secrets.vault();
        let unknown = placeholders(&args).find(|(_, number)| {
            let at = number.checked_sub(1);
            at.and_then(|at| vault.secrets.get(at)?.as_ref()).is_none()
        });
        let shell = request.call.function.name.as_str() == rig_tools::Shell::NAME;
        let special = placeholders(&args).filter(|_| shell).find(|(_, number)| {
            let at = number.checked_sub(1);
            let secret = at.and_then(|at| vault.secrets.get(at)?.as_deref());
            secret.is_some_and(|secret| secret.contains(SHELL_SPECIAL))
        });
        if let Some((at, _)) = special {
            let why = format!(
                "{} holds characters a shell reads as code, such as a quote, a space or `$`, \
                 so it is not put in a command. Nothing ran; ask the user to run the command \
                 with the value themselves.",
                args.get(at).unwrap_or_default()
            );
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
        } else if let Some((at, _)) = unknown {
            let why = format!(
                "{} is not a secret this run knows: secrets are kept in memory only, so one \
                 from before a restart, a /reload or a /resume is gone. Nothing ran; ask the \
                 user for the value again.",
                args.get(at).unwrap_or_default()
            );
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
        }
    }
}

/// Every model call is sent redacted and streams its reply redacted,
/// wrapping the outermost adapters too, such as a replay.
fn adapt_model_calls(
    mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>,
    secrets: Res<Secrets>,
) {
    for mut adapted in &mut calls {
        let handler = adapted.0.handler.0.clone().layered(AsSent(secrets.clone()));
        let secrets = secrets.clone();
        let redacting = handler.map_stream(NAME, move |_| redact_stream(secrets.clone()));
        adapted.0.handler = Handler(redacting);
    }
}

/// What one part of a reply holds back: text or reasoning, and the end of
/// what it settled, redacted, that may start a secret the run knows; a
/// call's argument JSON, which is not text and is redacted whole at its end.
enum Held {
    Text(Streamed, String),
    Reasoning(Streamed, String),
    Arguments(String),
}

impl Held {
    /// Takes `piece`; what can go out now, redacted.
    fn push(&mut self, piece: &str, secrets: &Secrets) -> String {
        match self {
            Self::Text(text, start) | Self::Reasoning(text, start) => {
                let mut settled = std::mem::take(start);
                settled.push_str(&text.push(piece));
                // Redacted first, so a whole secret the run knows goes as a
                // placeholder, and only the end that may start one waits.
                let mut redacted = secrets.text(&settled).into_owned();
                let cut = redacted.len() - starts_known(&secrets.vault(), &redacted);
                *start = redacted.split_off(cut);
                redacted
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
            Self::Text(text, start) | Self::Reasoning(text, start) => {
                let mut rest = std::mem::take(start);
                rest.push_str(&text.finish());
                secrets.text(&rest).into_owned()
            }
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
            Self::Text(..) => StreamEvent::Text { part, text },
            Self::Reasoning(..) => StreamEvent::Reasoning { part, text },
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
                let new = || Held::Text(Streamed::default(), String::new());
                grow(&mut held, (part, new), &text, (&secrets, &mut out));
            }
            Relayed::Item(Item::Event(StreamEvent::Reasoning { part, text })) => {
                let new = || Held::Reasoning(Streamed::default(), String::new());
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
        let block = Mutex::new(None);
        let secrets = secrets.clone();
        place.filter = Some(LineFilter::new(move |line| {
            let mut block = block.lock().unwrap_or_else(PoisonError::into_inner);
            redact_line(line, &mut block, &secrets)
        }));
    }
}

/// `line` redacted. A private key block spans lines: its placeholder goes
/// out with its first line, which stands for the whole block, and the lines
/// after it, through its end line or past [`BLOCK_MOST`], are added to that
/// secret and left out, so a call naming it gets the whole key. `block` is
/// the secret's number while one is open.
fn redact_line(line: &str, block: &mut Option<usize>, secrets: &Secrets) -> String {
    if let Some(number) = *block {
        let mut vault = secrets.vault();
        let held = number
            .checked_sub(1)
            .and_then(|at| vault.secrets.get_mut(at)?.as_mut());
        let full = held.is_none_or(|held| {
            held.push_str(line);
            held.len() > BLOCK_MOST
        });
        if full || line.contains("-----END ") {
            *block = None;
        }
        return String::new();
    }
    let opened = redact::find(line).find(|secret| {
        secret.kind == redact::Kind::PrivateKey
            && line
                .get(secret.range.clone())
                .is_some_and(|key| !key.contains("-----END "))
    });
    let Some(opened) = opened else {
        return secrets.text(line).into_owned();
    };
    let before = secrets
        .text(line.get(..opened.range.start).unwrap_or_default())
        .into_owned();
    let mut vault = secrets.vault();
    vault
        .secrets
        .push(Some(line.get(opened.range).unwrap_or_default().to_owned()));
    *block = Some(vault.secrets.len());
    format!("{before}{}\n", placeholder(vault.secrets.len()))
}

/// Every record reaches the effect log, when it is written, redacted: the
/// record whole, but for a model request, redacted in its own text
/// ([`Secrets::request`]) as [`AsSent`] sends it, which also covers one a
/// gate refused before it was sent.
fn redact_records(mut records: MessageMutator<EffectRecorded>, secrets: Res<Secrets>) {
    for EffectRecorded(record) in records.read() {
        let asked = match &mut record.kind {
            EffectKind::Completion { request, .. } => Some(std::mem::replace(
                request,
                Arc::new(CompletionRequest::new("")),
            )),
            _ => None,
        };
        if let Some(redacted) = secrets.redacted(&*record) {
            *record = redacted;
        }
        if let (EffectKind::Completion { request, .. }, Some(asked)) = (&mut record.kind, asked) {
            *request = secrets.request(&asked).map_or(asked, Arc::new);
        }
    }
}

/// Every prompt the terminal view keeps for Up and saves across sessions,
/// each secret, and each placeholder, as [`redact::REDACTED`]: a
/// placeholder there would name another session's secret of that number.
#[cfg(feature = "tui")]
fn redact_history(mut history: ResMut<rig_tui::PromptHistory>, secrets: Res<Secrets>) {
    let vault = secrets.vault();
    let rewrite = |prompt: &str| {
        let shaped = redact::text(prompt);
        let bare = known_as(&vault, &shaped, |_| redact::REDACTED.to_owned());
        let bare = bare.map_or(shaped, Cow::Owned);
        let mut out = String::new();
        let mut from = 0;
        for (at, _) in placeholders(&bare) {
            out.push_str(bare.get(from..at.start).unwrap_or_default());
            out.push_str(redact::REDACTED);
            from = at.end;
        }
        out.push_str(bare.get(from..).unwrap_or_default());
        (out != prompt).then_some(out)
    };
    history.bypass_change_detection().rewrite(rewrite);
}
