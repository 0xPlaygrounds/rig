//! Plugin corpus: 10. Secret redaction (spec `10-secret-redaction.md`).
//!
//! Secrets (the API keys, tokens and private keys `rig_core::redact`
//! finds, `PASSWORD=…` values and the values of the env vars the
//! `[secret-redaction]` table of the user's `settings.toml` names,
//! `env_vars = ["DB_PASSWORD"]`) never leave the machine: each distinct
//! one becomes a stable placeholder `‹SECRET_n›` in what is sent and kept.
//! A tool call that names a placeholder runs with the real value. The
//! secrets live in memory only; `/secrets` lists them.
//!
//! - Prepare: messages are redacted once their files are attached; every
//!   model call gets an adapter that redacts its request as it is sent and
//!   its reply as it streams; a tool call's placeholders are replaced by
//!   their secrets.
//! - Gate: a tool call still naming a placeholder, one this run does not
//!   know, is refused.
//! - Review: every tool result is redacted before it is committed.
//! - `Last`: every effect record is redacted before it is written.

use std::future::ready;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use bevy_log::warn;
use regex::{Captures, Regex};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::redact;
use rig_harness::rig_core::serve::{self, AfterServe as Answer, Intercept};
use rig_harness::rig_core::streaming::{Item, Part, Relayed, StreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// Values shorter than this are not taken as secrets.
const SHORTEST: usize = 6;

/// The adapter's and the layer's name.
const NAME: &str = "secret-redaction";

/// Redacts secrets from everything that leaves the machine.
#[derive(Default)]
pub struct SecretRedactionPlugin;

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RedactionSystems {
    /// Redacts messages and reveals tool arguments; in
    /// [`PrepareSystems::Edit`], once the files a message names are
    /// attached and the calls have their handlers.
    Prepare,
    /// Adapts model calls, in [`PrepareSystems::AdaptOutermost`].
    Adapt,
    /// Refuses a tool call naming an unknown placeholder.
    Gate,
    /// Redacts tool results.
    Review,
}

/// The `[secret-redaction]` table of `settings.toml`.
#[derive(Resource, serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct RedactionSettings {
    /// The env vars whose exact values are secrets.
    pub env_vars: Vec<String>,
}

impl Plugin for SecretRedactionPlugin {
    fn build(&self, app: &mut App) {
        use RedactionSystems as Set;
        app.add_setting::<RedactionSettings>(NAME)
            .init_resource::<Secrets>()
            .init_resource::<Issued>()
            .register_type::<Issued>()
            .configure_sets(
                AgentPass,
                (
                    Set::Prepare.in_set(PrepareSystems::Edit),
                    Set::Adapt.in_set(PrepareSystems::AdaptOutermost),
                    Set::Gate.in_set(Stage::Gate),
                    Set::Review.in_set(Stage::Review),
                ),
            )
            .add_systems(Startup, (keep_count, learn_env_vars).chain())
            // After the frame's agent work, so `Last` saves what it handed out.
            .add_systems(PostUpdate, keep_count)
            .add_systems(
                AgentPass,
                (
                    (redact_messages, reveal_placeholders).in_set(Set::Prepare),
                    adapt_model_calls.in_set(Set::Adapt),
                    refuse_forgotten.in_set(Set::Gate),
                    redact_results.in_set(Set::Review),
                ),
            )
            .add_systems(
                Last,
                redact_records.after(StopTurnsOnExit).before(WriteLogs),
            )
            .add_command("secrets", "List the placeholders secrets are sent as", list);
    }
}

/// One secret seen this run.
struct Secret {
    value: String,
    hint: String,
}

/// The secrets, `‹SECRET_n›` being the n-th; `None` for a number an
/// earlier run of the session handed out, which is forgotten.
#[derive(Default)]
struct Vault(Vec<Option<Secret>>);

/// The run's secrets, shared with the model calls that stream off the main
/// thread. Never saved.
#[derive(Resource, Clone, Default)]
pub struct Secrets(Arc<Mutex<Vault>>);

/// How many placeholder numbers the session handed out, saved with it, so
/// a run after a restart never gives an old number to a new secret.
#[derive(Resource, Reflect, Default, PartialEq)]
#[reflect(Resource, Saved)]
struct Issued(usize);

/// A password-like assignment, `PASSWORD=…` or `api_key="…"`, its value
/// the first group: a secret with no key shape, so not one
/// [`redact`] finds.
static PASSWORD: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b[a-z0-9_]*(?:password|passwd|secret|token|api_?key)=["']?([^\s"'‹›,;&]{6,})"#,
    )
    .ok()
});

/// A placeholder, its number the first group.
static PLACEHOLDER: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"‹SECRET_(\d+)›").ok());

fn placeholder(number: usize) -> String {
    format!("‹SECRET_{number}›")
}

/// Every string in `value`, changed by `change`; whether any changed.
fn each_string(value: &mut Value, change: &mut impl FnMut(&mut String) -> bool) -> bool {
    match value {
        Value::String(text) => change(text),
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |changed, item| each_string(item, change) | changed),
        Value::Object(fields) => fields
            .values_mut()
            .fold(false, |changed, field| each_string(field, change) | changed),
        _ => false,
    }
}

impl Vault {
    /// The number of `value`, learned from `hint` when new.
    fn learn(&mut self, value: &str, hint: &str) -> usize {
        let known = self
            .0
            .iter()
            .position(|secret| secret.as_ref().is_some_and(|secret| secret.value == value));
        if let Some(index) = known {
            return index + 1;
        }
        self.0.push(Some(Secret {
            value: value.to_owned(),
            hint: hint.to_owned(),
        }));
        self.0.len()
    }

    /// The secret `‹SECRET_number›` stands for, if this run knows it.
    fn value(&self, number: usize) -> Option<&str> {
        let secret = self.0.get(number.checked_sub(1)?)?.as_ref()?;
        Some(&secret.value)
    }

    /// `text` with every secret in it replaced by its placeholder, the
    /// ones it shows first learned as seen in `source`; whether any was.
    fn redact(&mut self, text: &mut String, source: &str) -> bool {
        for secret in redact::find(text) {
            if let Some(value) = text.get(secret.range) {
                self.learn(value, &format!("{} in {source}", secret.kind));
            }
        }
        let passwords = PASSWORD
            .iter()
            .flat_map(|pattern| pattern.captures_iter(text));
        for value in passwords.filter_map(|found| found.get(1)) {
            self.learn(value.as_str(), &format!("password in {source}"));
        }
        // The longest first, so one inside another goes with it.
        let mut known: Vec<(usize, &str)> = (1..)
            .zip(&self.0)
            .filter_map(|(number, secret)| Some((number, secret.as_ref()?.value.as_str())))
            .filter(|(_, value)| value.len() >= SHORTEST && text.contains(value))
            .collect();
        known.sort_by_key(|(_, value)| std::cmp::Reverse(value.len()));
        for &(number, value) in &known {
            *text = text.replace(value, &placeholder(number));
        }
        !known.is_empty()
    }

    /// `text` with each placeholder this run knows replaced by its secret;
    /// whether any was.
    fn reveal(&self, text: &mut String) -> bool {
        let Some(pattern) = PLACEHOLDER.as_ref() else {
            return false;
        };
        let mut revealed = false;
        let replaced = pattern.replace_all(text, |found: &Captures| {
            let number = found.get(1).and_then(|number| number.as_str().parse().ok());
            let whole = found.get(0).map_or("", |whole| whole.as_str());
            match number.and_then(|number| self.value(number)) {
                Some(secret) => {
                    revealed = true;
                    secret.to_owned()
                }
                None => whole.to_owned(),
            }
        });
        let replaced = replaced.into_owned();
        *text = replaced;
        revealed
    }
}

impl Secrets {
    fn vault(&self) -> MutexGuard<'_, Vault> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Each placeholder this run knows and where its secret was first seen.
    pub fn hints(&self) -> Vec<(String, String)> {
        (1..)
            .zip(&self.vault().0)
            .filter_map(|(number, secret)| {
                Some((placeholder(number), secret.as_ref()?.hint.clone()))
            })
            .collect()
    }

    /// Redacts every string of `data`, through its JSON form, as seen in
    /// `source`; whether any changed. Data whose redacted form does not
    /// read back is kept, with a warning.
    fn redact_data<T: Serialize + DeserializeOwned>(&self, data: &mut T, source: &str) -> bool {
        let Ok(mut value) = serde_json::to_value(&*data) else {
            return false;
        };
        let mut vault = self.vault();
        if !each_string(&mut value, &mut |text| vault.redact(text, source)) {
            return false;
        }
        drop(vault);
        match serde_json::from_value(value) {
            Ok(redacted) => {
                *data = redacted;
                true
            }
            Err(error) => {
                warn!("secret redaction: could not redact {source}: {error}");
                false
            }
        }
    }
}

/// The placeholders of earlier runs stay taken; the ones this run hands
/// out are saved as taken. A number first handed out in a record, which
/// only the effect log shows, is saved a frame later, or not at all on
/// exit, which is harmless: no model saw it.
fn keep_count(mut issued: ResMut<Issued>, secrets: Res<Secrets>) {
    let mut vault = secrets.vault();
    while vault.0.len() < issued.0 {
        vault.0.push(None);
    }
    issued.set_if_neq(Issued(vault.0.len()));
}

/// The values of the env vars the settings name are secrets from the start.
fn learn_env_vars(settings: Res<RedactionSettings>, secrets: Res<Secrets>) {
    let mut vault = secrets.vault();
    for name in &settings.env_vars {
        if let Ok(value) = std::env::var(name)
            && value.len() >= SHORTEST
        {
            vault.learn(&value, &format!("the value of ${name}"));
        }
    }
}

/// Every message goes to its agent with placeholders, text and attached
/// files, but a slash command, which never reaches the model.
fn redact_messages(
    mut messages: Query<&mut Incoming, (Added<Admitted>, IncomingMessages)>,
    secrets: Res<Secrets>,
) {
    for mut message in &mut messages {
        if message.command().is_some() {
            continue;
        }
        for part in &mut message.parts {
            match part {
                MessagePart::Text(text) => {
                    secrets.vault().redact(text, "a message");
                }
                MessagePart::Attachment(attachment) => {
                    let source = format!("the attachment {}", attachment.label);
                    secrets.redact_data(&mut attachment.content, &source);
                }
            }
        }
    }
}

/// Every model call, whichever model it goes to, is sent with placeholders
/// and streams its reply with them. It adapts outermost, so every other
/// adapter sees the request with placeholders, and what any of them
/// streams is redacted.
fn adapt_model_calls(
    mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>,
    secrets: Res<Secrets>,
) {
    for mut adapted in &mut calls {
        let sent = adapted
            .0
            .handler
            .0
            .clone()
            .layered(RedactRequest(secrets.clone()));
        let secrets = secrets.clone();
        let streamed = sent.map_stream(NAME, move |_| redact_stream(secrets.clone()));
        adapted.0.handler = Handler(streamed);
    }
}

/// Every tool call runs with the secrets its placeholders name. The
/// conversation keeps the call as the model sent it, and its record is
/// redacted again before it is written.
fn reveal_placeholders(
    mut calls: Query<&mut ToolRequest, (Added<Admitted>, ToolCalls)>,
    secrets: Res<Secrets>,
) {
    for mut call in &mut calls {
        let mut args = Value::Object(call.call.function.arguments.clone());
        let vault = secrets.vault();
        if each_string(&mut args, &mut |text| vault.reveal(text))
            && let Value::Object(args) = args
        {
            call.call.function.arguments = args;
        }
    }
}

/// A tool call that still names a placeholder names one this run does not
/// know, such as one from before a restart: it is refused.
fn refuse_forgotten(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let args = Value::Object(request.call.function.arguments.clone()).to_string();
        let unknown = PLACEHOLDER.as_ref().and_then(|pattern| pattern.find(&args));
        if let Some(unknown) = unknown {
            let why = format!(
                "{} is not a known secret: placeholders are kept in memory only and are \
                 forgotten when the agent restarts. Nothing ran; ask the user for the value again.",
                unknown.as_str()
            );
            decision.decide(Decider::plugin("secret-redaction"), Outcome::refused(why));
        }
    }
}

/// On a call whose result was redacted: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Redacted;

/// Every tool result reaches the conversation, the session log and the
/// views with placeholders.
fn redact_results(
    mut answered: Query<
        (Entity, &mut Outcome, &ToolRequest),
        (ReadyToSettle, ToolCalls, Without<Redacted>),
    >,
    secrets: Res<Secrets>,
    mut commands: Commands,
) {
    for (call, mut outcome, request) in &mut answered {
        let source = format!("the result of `{}`", request.call.function.name.as_str());
        let mut redacted = outcome.0.clone();
        if secrets.redact_data(&mut redacted, &source) {
            outcome.0 = redacted;
        }
        // Once: Review sees the outcome each pass until it settles.
        commands.entity(call).insert(Redacted);
    }
}

/// Every record reaches the effect log with placeholders: the revealed
/// arguments of a tool call, its raw result, the request as built.
fn redact_records(mut records: MessageMutator<EffectRecorded>, secrets: Res<Secrets>) {
    for EffectRecorded(record) in records.read() {
        secrets.redact_data(record, "the effect log");
    }
}

/// `/secrets`.
fn list(In(args): In<CommandArgs>, secrets: Res<Secrets>, mut notices: MessageWriter<Notice>) {
    let lines: Vec<String> = secrets
        .hints()
        .into_iter()
        .map(|(placeholder, hint)| format!("{placeholder}  {hint}"))
        .collect();
    let text = match lines.is_empty() {
        true => "No secrets seen in this run.".to_owned(),
        false => format!("Secrets, sent as placeholders:\n{}", lines.join("\n")),
    };
    notices.write(Notice::info(args.agent, text));
}

/// Sends each model request with its secrets replaced by placeholders,
/// whatever put them there: the system prompt, another plugin's edit.
struct RedactRequest(Secrets);

impl Intercept for RedactRequest {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    fn before(
        &self,
        _: EffectId,
        kind: &EffectKind,
    ) -> impl Future<Output = serve::BeforeServe> + Send {
        let mut kind = kind.clone();
        ready(match self.0.redact_data(&mut kind, "a model request") {
            true => serve::BeforeServe::Patch(kind),
            false => serve::BeforeServe::Proceed,
        })
    }

    fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Answered, ErrorReport>,
    ) -> impl Future<Output = Answer> + Send {
        ready(Answer::Keep)
    }
}

/// Text or reasoning of one part held back until it ends at whitespace, so
/// a secret streamed in pieces is redacted whole.
struct Held {
    part: Part,
    reasoning: bool,
    text: String,
}

impl Held {
    /// The held text, redacted, as one fragment.
    fn release(mut self, secrets: &Secrets) -> Option<Relayed> {
        if self.text.is_empty() {
            return None;
        }
        secrets.vault().redact(&mut self.text, "a model reply");
        let (part, text) = (self.part, self.text);
        let event = match self.reasoning {
            true => StreamEvent::Reasoning { part, text },
            false => StreamEvent::Text { part, text },
        };
        Some(Relayed::Item(Item::Event(event)))
    }
}

/// Rewrites one reply's stream with placeholders: its text as it streams,
/// every other event, and the response it ends with.
fn redact_stream(secrets: Secrets) -> impl FnMut(Relayed) -> Vec<Relayed> {
    let mut held: Option<Held> = None;
    move |item| {
        let grown = match &item {
            Relayed::Item(Item::Event(StreamEvent::Text { part, text })) => {
                Some((*part, false, text))
            }
            Relayed::Item(Item::Event(StreamEvent::Reasoning { part, text })) => {
                Some((*part, true, text))
            }
            _ => None,
        };
        let mut out = Vec::new();
        let Some((part, reasoning, text)) = grown else {
            out.extend(held.take().and_then(|held| held.release(&secrets)));
            let mut item = item;
            match &mut item {
                Relayed::Item(event) => {
                    secrets.redact_data(event, "a model reply");
                }
                Relayed::Done(response) => {
                    secrets.redact_data(Arc::make_mut(response), "a model reply");
                }
                Relayed::Origin(_) => {}
            }
            out.push(item);
            return out;
        };
        let mut pending = match held.take() {
            Some(earlier) if earlier.part == part && earlier.reasoning == reasoning => earlier,
            earlier => {
                out.extend(earlier.and_then(|earlier| earlier.release(&secrets)));
                Held {
                    part,
                    reasoning,
                    text: String::new(),
                }
            }
        };
        pending.text.push_str(text);
        let cut = pending
            .text
            .char_indices()
            .rev()
            .find(|(_, char)| char.is_whitespace())
            .map(|(at, char)| at + char.len_utf8());
        if let Some(cut) = cut {
            let rest = pending.text.split_off(cut);
            out.extend(pending.release(&secrets));
            pending = Held {
                part,
                reasoning,
                text: rest,
            };
        }
        held = Some(pending);
        out
    }
}
