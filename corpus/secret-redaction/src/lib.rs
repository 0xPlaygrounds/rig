//! Plugin corpus: 10. Secret redaction (spec `10-secret-redaction.md`).
//!
//! Secrets (API keys, tokens, private keys, `PASSWORD=…` values and the
//! values of the env vars named in `RIG_REDACT_ENV`) never leave the
//! machine: each distinct one becomes a stable placeholder `‹SECRET_n›` in
//! every request, in the session log and in the results of the coding
//! tools. A tool call that names a placeholder gets the real value. The
//! placeholders live in memory only; `/secrets` lists them.

use std::io;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::world::DeferredWorld;
use bevy_log::warn;
use regex::Regex;
use rig_cassette::journal::JournalStore;
use rig_core::effect::family;
use rig_core::effect::{EffectKind, HandlerDescriptor, Outcome};
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_ecs::journal::SessionStore;
use rig_harness::harness_protocol::env::AGENT_ONLY;
use rig_harness::prelude::*;
use rig_tools::{Edit, Read, Search, Shell, Write};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// The env var naming, comma-separated, the env vars whose values are
/// secrets.
pub const ENV: &str = "RIG_REDACT_ENV";

/// Values shorter than this are not taken as secrets.
const SHORTEST: usize = 6;

/// Redacts secrets from everything that leaves the machine.
pub struct SecretRedactionPlugin {
    /// The env vars whose exact values are secrets.
    pub env_vars: Vec<String>,
}

impl Default for SecretRedactionPlugin {
    fn default() -> Self {
        let named = std::env::var(ENV).unwrap_or_default();
        let env_vars = named
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect();
        Self { env_vars }
    }
}

impl Plugin for SecretRedactionPlugin {
    fn build(&self, app: &mut App) {
        let secrets = Secrets::default();
        for name in &self.env_vars {
            if let Ok(value) = std::env::var(name)
                && value.len() >= SHORTEST
            {
                secrets
                    .vault()
                    .learn(&value, &format!("the value of ${name}"));
            }
        }
        app.insert_resource(secrets)
            .add_observer(redact_request)
            .add_command("secrets", "List the placeholders secrets are sent as", list);
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let Some(secrets) = world.get_resource::<Secrets>().cloned() else {
            return;
        };
        if let Some(store) = world.get_resource::<SessionStore>() {
            let inner = store.0.clone();
            let store = RedactingStore {
                inner,
                secrets: secrets.clone(),
            };
            world.insert_resource(SessionStore::new(store));
        }
        let spill = world
            .get_resource::<SessionPaths>()
            .map(SessionPaths::spill);
        wrap(world, &secrets, Read);
        wrap(world, &secrets, Write);
        wrap(world, &secrets, Edit);
        wrap(world, &secrets, Search);
        let shell = Shell {
            launcher: rig_tools::process::Launcher::default().without_env(AGENT_ONLY),
            spill,
            ..Shell::default()
        };
        wrap(world, &secrets, shell);
        let hooks = world.register_component_hooks::<Done<ModelReply>>();
        if hooks.try_on_insert(redact_reply).is_none() {
            warn!("secret redaction: model replies already have an insert hook; not redacted");
        }
    }
}

/// One secret seen this session.
struct Secret {
    value: String,
    hint: String,
}

/// The secrets seen, `‹SECRET_n›` being the n-th. Never saved.
#[derive(Default)]
struct Vault(Vec<Secret>);

/// The session's secrets, shared with the tools that run off the main
/// thread.
#[derive(Resource, Clone, Default)]
pub struct Secrets(Arc<Mutex<Vault>>);

/// What a placeholder starts and ends with.
const OPEN: &str = "‹SECRET_";
const CLOSE: char = '›';

fn placeholder(number: usize) -> String {
    format!("{OPEN}{number}{CLOSE}")
}

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

impl Vault {
    /// The number of `value`, learned from `hint` when new.
    fn learn(&mut self, value: &str, hint: &str) -> usize {
        if let Some(index) = self.0.iter().position(|secret| secret.value == value) {
            return index + 1;
        }
        self.0.push(Secret {
            value: value.to_owned(),
            hint: hint.to_owned(),
        });
        self.0.len()
    }

    /// `text` with every secret in it replaced by its placeholder, the
    /// ones it shows first learned as seen in `source`; `None` when it has
    /// none.
    fn redact(&mut self, text: &str, source: &str) -> Option<String> {
        for (kind, pattern) in PATTERNS.iter() {
            for found in pattern.captures_iter(text) {
                let secret = found.get(1).or_else(|| found.get(0));
                if let Some(secret) = secret.map(|secret| secret.as_str())
                    && !secret.contains(OPEN)
                {
                    self.learn(secret, &format!("{kind} in {source}"));
                }
            }
        }
        // The longest first, so one inside another goes with it.
        let mut known: Vec<(usize, &str)> = (1..)
            .zip(&self.0)
            .filter(|(_, secret)| secret.value.len() >= SHORTEST)
            .map(|(number, secret)| (number, secret.value.as_str()))
            .filter(|(_, value)| text.contains(value))
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
            Value::String(text) => match self.redact(text, source) {
                Some(redacted) => {
                    *text = redacted;
                    true
                }
                None => false,
            },
            Value::Array(items) => items.iter_mut().fold(false, |changed, item| {
                self.redact_value(item, source) | changed
            }),
            Value::Object(fields) => fields.values_mut().fold(false, |changed, field| {
                self.redact_value(field, source) | changed
            }),
            _ => false,
        }
    }

    /// `text` with each placeholder in it replaced by its secret, or the
    /// first placeholder that is not known.
    fn reveal(&self, text: &str) -> Result<String, String> {
        let mut revealed = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find(OPEN) {
            let (before, from) = rest.split_at(start);
            revealed.push_str(before);
            let after = from.get(OPEN.len()..).unwrap_or_default();
            let digits = after.len() - after.trim_start_matches(|c: char| c.is_ascii_digit()).len();
            let (number, tail) = after.split_at(digits);
            let Some(tail) = tail.strip_prefix(CLOSE) else {
                revealed.push_str(OPEN);
                rest = after;
                continue;
            };
            let secret = number
                .parse::<usize>()
                .ok()
                .and_then(|number| self.0.get(number.checked_sub(1)?));
            match secret {
                Some(secret) => revealed.push_str(&secret.value),
                None => return Err(format!("{OPEN}{number}{CLOSE}")),
            }
            rest = tail;
        }
        revealed.push_str(rest);
        Ok(revealed)
    }

    /// Reveals every string in `value`.
    fn reveal_value(&self, value: &mut Value) -> Result<(), String> {
        match value {
            Value::String(text) if text.contains(OPEN) => *text = self.reveal(text)?,
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

    /// Each placeholder and where its secret was first seen.
    pub fn hints(&self) -> Vec<(String, String)> {
        let vault = self.vault();
        (1..)
            .zip(&vault.0)
            .map(|(number, secret)| (placeholder(number), secret.hint.clone()))
            .collect()
    }

    /// Redacts every string of `data`, through its JSON form, as seen in
    /// `source`. Data the redacted form does not read back as is kept.
    fn redact_data<T: Serialize + DeserializeOwned>(&self, data: &mut T, source: &str) {
        let Ok(mut value) = serde_json::to_value(&*data) else {
            return;
        };
        if self.vault().redact_value(&mut value, source) {
            match serde_json::from_value(value) {
                Ok(redacted) => *data = redacted,
                Err(error) => warn!("secret redaction: could not redact {source}: {error}"),
            }
        }
    }
}

/// Every request goes out with placeholders: its system prompt and
/// messages, whatever put the secrets there.
fn redact_request(mut prepare: On<PrepareRequest>, secrets: Res<Secrets>) {
    let redacted = secrets
        .vault()
        .redact(&prepare.preamble, "the system prompt");
    if let Some(preamble) = redacted {
        prepare.preamble = preamble;
    }
    for message in &mut prepare.messages {
        secrets.redact_data(message, "the conversation");
    }
}

/// A model reply shows placeholders, to the user and in the conversation:
/// redacted as it is inserted, before the turn's observer takes it.
fn redact_reply(mut world: DeferredWorld, context: HookContext) {
    let Some(secrets) = world.get_resource::<Secrets>().cloned() else {
        return;
    };
    if let Some(mut done) = world.get_mut::<Done<ModelReply>>(context.entity)
        && let Ok(response) = &mut done.0
    {
        for content in &mut response.choice {
            secrets.redact_data(content, "a model reply");
        }
    }
}

/// `/secrets`.
fn list(In(args): In<CommandArgs>, secrets: Res<Secrets>, mut notices: MessageWriter<Notice>) {
    let hints = secrets.hints();
    let text = if hints.is_empty() {
        "No secrets seen in this session.".to_owned()
    } else {
        let lines: Vec<String> = hints
            .into_iter()
            .map(|(placeholder, hint)| format!("{placeholder}  {hint}"))
            .collect();
        format!("Secrets, sent as placeholders:\n{}", lines.join("\n"))
    };
    notices.write(Notice::info(args.agent, text));
}

/// Registers `tool` again behind [`Unmasked`] in place of the same tool
/// another plugin registered, with its rules and footprint, if it did.
fn wrap<T: Tool + 'static>(world: &mut World, secrets: &Secrets, tool: T) {
    let (description, parameters) = (tool.description(), tool.parameters());
    let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
    let found = tools.iter(world).find(|(_, def, ..)| {
        def.0.name.as_str() == T::NAME
            && def.0.description == description
            && def.0.parameters == parameters
    });
    let Some((entity, def, rules, &footprint)) = found else {
        return;
    };
    let (def, rules) = (def.0.clone(), rules.0.clone());
    world.entity_mut(entity).insert(Disabled);
    let handler = ErasedHandler::new(Unmasked {
        inner: ToolAdapter::new(tool),
        secrets: secrets.clone(),
    });
    let rules: Vec<&str> = rules.iter().map(String::as_str).collect();
    let options = ToolOptions {
        rules: &rules,
        footprint,
    };
    world.spawn_tool(def, handler, options);
}

/// A tool handler that runs its calls with the placeholders in their
/// arguments replaced by the secrets, and answers with the secrets in its
/// result replaced by placeholders. The call is recorded as the model
/// sent it, the result as redacted.
struct Unmasked<S> {
    inner: S,
    secrets: Secrets,
}

impl<S: Serve<Family = family::Tool>> Serve for Unmasked<S> {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let EffectKind::ToolCall { name, args } = kind else {
            return self.inner.serve(kind, dispatch).await;
        };
        let args = match self.reveal(&args) {
            Ok(args) => args,
            Err(unknown) => {
                let why = format!(
                    "{unknown} is not a known secret: placeholders are kept in memory only and \
                     are forgotten when the agent restarts. Nothing ran; ask the user for the \
                     value again."
                );
                let result = ToolResult::failed(ToolExecutionError::other(why));
                return Reply::Outcome(Ok(Outcome::ToolResult { result }));
            }
        };
        let source = format!("the result of `{name}`");
        let reply = self
            .inner
            .serve(EffectKind::ToolCall { name, args }, dispatch);
        let outcome = reply.await.into_outcome().await;
        Reply::Outcome(outcome.map(|outcome| match outcome {
            Outcome::ToolResult { mut result } => {
                self.secrets.redact_data(&mut result, &source);
                Outcome::ToolResult { result }
            }
            other => other,
        }))
    }
}

impl<S> Unmasked<S> {
    /// The JSON arguments `args` with their placeholders revealed.
    fn reveal(&self, args: &str) -> Result<String, String> {
        if !args.contains(OPEN) {
            return Ok(args.to_owned());
        }
        let Ok(mut value) = serde_json::from_str::<Value>(args) else {
            return Ok(args.to_owned());
        };
        self.secrets.vault().reveal_value(&mut value)?;
        Ok(value.to_string())
    }
}

/// The session store, with every line written to it redacted.
struct RedactingStore {
    inner: Arc<dyn JournalStore>,
    secrets: Secrets,
}

impl JournalStore for RedactingStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        self.inner.agents()
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        self.inner.read(agent)
    }

    fn truncate(&self, agent: &str, len: u64) -> io::Result<()> {
        self.inner.truncate(agent, len)
    }

    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()> {
        let text = String::from_utf8_lossy(bytes);
        let mut vault = self.secrets.vault();
        let mut redacted = String::with_capacity(text.len());
        for line in text.split_inclusive('\n') {
            let (body, end) = line
                .strip_suffix('\n')
                .map_or((line, ""), |body| (body, "\n"));
            match serde_json::from_str::<Value>(body) {
                Ok(mut value) => {
                    if vault.redact_value(&mut value, "the session log") {
                        redacted.push_str(&value.to_string());
                    } else {
                        redacted.push_str(body);
                    }
                }
                Err(_) => match vault.redact(body, "the session log") {
                    Some(body) => redacted.push_str(&body),
                    None => redacted.push_str(body),
                },
            }
            redacted.push_str(end);
        }
        drop(vault);
        self.inner.append(agent, redacted.as_bytes())
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.inner.put_blob(name, bytes)
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.inner.blob(name)
    }
}
