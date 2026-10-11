//! Plugin corpus: 10. Secret redaction (spec `10-secret-redaction.md`).
//!
//! Secrets (API keys, tokens, private keys, `PASSWORD=…` values and the
//! values of the env vars the `[secret-redaction]` table of the user's
//! `settings.toml` names, `env_vars = ["DB_PASSWORD"]`) never leave the
//! machine: each distinct one becomes a stable placeholder `‹SECRET_n›` in
//! what is sent and kept. A tool call that names a placeholder runs with
//! the real value. The placeholders live in memory only; `/secrets` lists
//! them.
//!
//! - Gate: every message and model request about to start is redacted,
//!   once every Prepare system added to it; a tool call naming a
//!   placeholder this run does not know is refused.
//! - Prepare: every tool call's handler is wrapped ([`Unmasked`]): it runs
//!   with the secrets and answers with placeholders, so the effect log
//!   records the call as the model sent it and the redacted result.
//! - Review: every tool result and model reply is redacted before it is
//!   committed (and so before the session log and the views see it).

use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use bevy_log::warn;
use regex::{Captures, Regex};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{EffectKind, HandlerDescriptor, family};
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve, serve_inline_with};
use rig_harness::rig_ecs::effects::Handler;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// Values shorter than this are not taken as secrets.
const SHORTEST: usize = 6;

/// Redacts secrets from everything that leaves the machine.
#[derive(Default)]
pub struct SecretRedactionPlugin;

/// The `[secret-redaction]` table of `settings.toml`.
#[derive(Resource, serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct RedactionSettings {
    /// The env vars whose exact values are secrets.
    pub env_vars: Vec<String>,
}

impl Plugin for SecretRedactionPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<RedactionSettings>("secret-redaction")
            .init_resource::<Secrets>()
            .add_systems(Startup, learn_env_vars)
            .add_systems(
                AgentPass,
                (
                    unmask_calls.in_set(Stage::Prepare),
                    (redact_messages, redact_requests, refuse_forgotten).in_set(Stage::Gate),
                    redact_outcomes.in_set(Stage::Review),
                ),
            )
            .add_observer(forget_restored)
            .add_command("secrets", "List the placeholders secrets are sent as", list);
    }
}

/// One secret seen this run.
struct Secret {
    value: String,
    hint: String,
}

/// The secrets, `‹SECRET_n›` being the n-th; `None` for a number from
/// before a restart, which is forgotten. Never saved.
#[derive(Default)]
struct Vault(Vec<Option<Secret>>);

/// The run's secrets, shared with the tools that run off the main thread.
#[derive(Resource, Clone, Default)]
pub struct Secrets(Arc<Mutex<Vault>>);

/// What each kind of secret looks like: its name, and a pattern whose
/// first group, or else whole match, is the secret.
static PATTERNS: LazyLock<Vec<(&str, Regex)>> = LazyLock::new(|| {
    [
        (
            "a private key",
            r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        ),
        ("an AWS access key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        (
            "a GitHub token",
            r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b",
        ),
        ("an API key", r"\bsk-[A-Za-z0-9_-]{20,}"),
        (
            "a password",
            r#"(?i)\b[a-z0-9_]*(?:password|passwd|secret|token|api_?key)=["']?([^\s"'‹›,;&]{6,})"#,
        ),
    ]
    .into_iter()
    .filter_map(|(kind, pattern)| Some((kind, Regex::new(pattern).ok()?)))
    .collect()
});

/// A placeholder, its number the first group.
static PLACEHOLDER: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"‹SECRET_(\d+)›").ok());

fn placeholder(number: usize) -> String {
    format!("‹SECRET_{number}›")
}

/// The placeholders in `text`: each match and its number.
fn placeholders(text: &str) -> impl Iterator<Item = (&str, Option<usize>)> {
    PLACEHOLDER.iter().flat_map(move |pattern| {
        pattern.captures_iter(text).map(|found: Captures| {
            let whole = found.get(0).map_or("", |whole| whole.as_str());
            let number = found.get(1).and_then(|number| number.as_str().parse().ok());
            (whole, number)
        })
    })
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
    /// ones it shows first learned as seen in `source`; `None` when it has
    /// none.
    fn redact(&mut self, text: &str, source: &str) -> Option<String> {
        for (kind, pattern) in PATTERNS.iter() {
            for found in pattern.captures_iter(text) {
                if let Some(secret) = found.get(1).or_else(|| found.get(0)) {
                    self.learn(secret.as_str(), &format!("{kind} in {source}"));
                }
            }
        }
        // The longest first, so one inside another goes with it.
        let mut known: Vec<(usize, &str)> = (1..)
            .zip(&self.0)
            .filter_map(|(number, secret)| Some((number, secret.as_ref()?.value.as_str())))
            .filter(|(_, value)| value.len() >= SHORTEST && text.contains(value))
            .collect();
        if known.is_empty() {
            return None;
        }
        known.sort_by_key(|(_, value)| std::cmp::Reverse(value.len()));
        let redacted = known
            .into_iter()
            .fold(text.to_owned(), |text, (number, value)| {
                text.replace(value, &placeholder(number))
            });
        Some(redacted)
    }

    /// Redacts every string in `value`; whether any changed.
    fn redact_value(&mut self, value: &mut Value, source: &str) -> bool {
        match value {
            Value::String(text) => self
                .redact(text, source)
                .map(|redacted| *text = redacted)
                .is_some(),
            Value::Array(items) => items.iter_mut().fold(false, |changed, item| {
                self.redact_value(item, source) | changed
            }),
            Value::Object(fields) => fields.values_mut().fold(false, |changed, field| {
                self.redact_value(field, source) | changed
            }),
            _ => false,
        }
    }

    /// Replaces each placeholder in every string of `value` by its secret,
    /// or names the first one this run does not know.
    fn reveal_value(&self, value: &mut Value) -> Result<(), String> {
        match value {
            Value::String(text) => {
                for (whole, number) in placeholders(text) {
                    if number.and_then(|number| self.value(number)).is_none() {
                        return Err(whole.to_owned());
                    }
                }
                if let Some(pattern) = PLACEHOLDER.as_ref() {
                    let revealed = pattern.replace_all(text, |found: &Captures| {
                        let number = found.get(1).and_then(|number| number.as_str().parse().ok());
                        number
                            .and_then(|number| self.value(number))
                            .unwrap_or_default()
                            .to_owned()
                    });
                    *text = revealed.into_owned();
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.reveal_value(item)?;
                }
            }
            Value::Object(fields) => {
                for field in fields.values_mut() {
                    self.reveal_value(field)?;
                }
            }
            _ => {}
        }
        Ok(())
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
        if !self.vault().redact_value(&mut value, source) {
            return false;
        }
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

    /// The JSON arguments `args` with their placeholders revealed, or the
    /// first placeholder this run does not know.
    fn reveal_args(&self, args: &str) -> Result<String, String> {
        let Ok(mut value) = serde_json::from_str::<Value>(args) else {
            return Ok(args.to_owned());
        };
        self.vault().reveal_value(&mut value)?;
        Ok(value.to_string())
    }
}

/// Why a call naming `unknown` does not run.
fn forgotten(unknown: &str) -> String {
    format!(
        "{unknown} is not a known secret: placeholders are kept in memory only and are \
         forgotten when the agent restarts. Nothing ran; ask the user for the value again."
    )
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

/// A conversation restored from the session names placeholders of an
/// earlier run: their numbers are forgotten, never given to new secrets.
fn forget_restored(
    added: On<Add<Conversation>>,
    agents: Query<&Conversation>,
    secrets: Res<Secrets>,
) {
    let Ok(text) = agents
        .get(added.entity)
        .map(|conversation| serde_json::to_string(conversation).unwrap_or_default())
    else {
        return;
    };
    let highest = placeholders(&text).filter_map(|(_, number)| number).max();
    let mut vault = secrets.vault();
    while vault.0.len() < highest.unwrap_or_default() {
        vault.0.push(None);
    }
}

/// Every tool call with a handler runs with the secrets its placeholders
/// name, and answers with placeholders.
fn unmask_calls(
    mut calls: Query<&mut CallHandler, (Added<Admitted>, With<ToolRequest>)>,
    secrets: Res<Secrets>,
) {
    for mut handler in &mut calls {
        let inner = handler.0.0.clone();
        let secrets = secrets.clone();
        handler.0 = Handler(ErasedHandler::new(Unmasked { inner, secrets }));
    }
}

/// Every message goes to its agent with placeholders, text and
/// attachments, but a slash command, which never reaches the model.
fn redact_messages(mut messages: Query<&mut Incoming, Undecided>, secrets: Res<Secrets>) {
    for mut message in &mut messages {
        if message.command().is_some() {
            continue;
        }
        let mut text = message.text.clone();
        if secrets.redact_data(&mut text, "a message") {
            message.text = text;
        }
        for attachment in &mut message.attachments {
            let source = format!("the attachment {}", attachment.label);
            secrets.redact_data(&mut attachment.content, &source);
        }
    }
}

/// Every model request goes out with placeholders: its system prompt and
/// messages, whatever put the secrets there.
fn redact_requests(mut calls: Query<&mut ModelRequest, Undecided>, secrets: Res<Secrets>) {
    for mut call in &mut calls {
        for (index, message) in call.request.chat_history.iter_mut().enumerate() {
            let source = if index == 0 {
                "the system prompt"
            } else {
                "a request"
            };
            secrets.redact_data(message, source);
        }
    }
}

/// A tool call naming a placeholder this run does not know, such as one
/// from before a restart, is refused.
fn refuse_forgotten(
    calls: Query<(Entity, &ToolRequest), Undecided>,
    secrets: Res<Secrets>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        let args = serde_json::to_string(&request.call.function.arguments).unwrap_or_default();
        let vault = secrets.vault();
        let unknown = placeholders(&args)
            .find(|(_, number)| number.and_then(|number| vault.value(number)).is_none());
        if let Some((unknown, _)) = unknown {
            commands
                .entity(call)
                .insert_if_new(Outcome::refused(forgotten(unknown)));
        }
    }
}

/// On an effect whose outcome was redacted: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Redacted;

/// Every tool result and model reply reaches the conversation, the
/// session log and the views with placeholders.
fn redact_outcomes(
    answered: Query<
        (Entity, &Outcome, Option<&ToolRequest>),
        (
            Unreviewed,
            Or<(With<ToolRequest>, With<ModelRequest>)>,
            Without<Redacted>,
        ),
    >,
    secrets: Res<Secrets>,
    mut commands: Commands,
) {
    for (effect, outcome, tool) in &answered {
        let source = tool.map_or_else(
            || "a model reply".to_owned(),
            |tool| format!("the result of `{}`", tool.call.function.name.as_str()),
        );
        let mut redacted = outcome.0.clone();
        let mut done = commands.entity(effect);
        if secrets.redact_data(&mut redacted, &source) {
            done.insert(Outcome(redacted));
        }
        done.insert(Redacted);
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

/// A tool handler that runs its calls with the placeholders in their
/// arguments replaced by the secrets, and answers with the secrets in its
/// result replaced by placeholders. The kernel records the call as the
/// model sent it and this answer: the inner handler runs inline, with the
/// same scopes (what the tool publishes) but unrecorded, since a rig-core
/// `Layer` would record the revealed arguments and the raw result.
pub struct Unmasked {
    inner: ErasedHandler,
    secrets: Secrets,
}

impl Serve for Unmasked {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let EffectKind::ToolCall { name, args } = kind else {
            return self.inner.serve(kind, dispatch).await;
        };
        let source = format!("the result of `{name}`");
        let mut outcome = match self.secrets.reveal_args(&args) {
            Ok(args) => {
                let kind = EffectKind::ToolCall { name, args };
                serve_inline_with(&self.inner, kind, dispatch.scopes()).await
            }
            Err(unknown) => Outcome::refused(forgotten(&unknown)).0,
        };
        self.secrets.redact_data(&mut outcome, &source);
        Reply::Outcome(outcome)
    }
}
