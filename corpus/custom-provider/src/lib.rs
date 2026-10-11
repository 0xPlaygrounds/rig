//! Plugin corpus: 16. Custom provider registration (spec `16-custom-provider.md`).
//!
//! Providers listed in `RIG_HOME/providers.toml` (a company gateway, say)
//! become models the agent can use as `<name>/<model>`: `/model corp/big`,
//! or the `/model` picker, which lists them beside the catalog's. Each is
//! reached on the wire of its `api`, at its `base_url`, with its extra
//! `headers` and its key from `key_env` or `key_command` (run again once
//! the token it printed expired), and priced and windowed by its own
//! entry. The choice is the saved [`CustomModel`] on the agent, which a
//! restart or `/reload` connects again; `/reload` restarts the process, so
//! it also re-reads the file.
//!
//! The kernel's catalog cannot hold a vendor it does not know, so a
//! configured model is not a [`ModelChoice`]: this plugin replaces
//! `/model`, connects the agent itself and removes its `ModelChoice`.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use http::{HeaderMap, HeaderName, HeaderValue, Method, Uri};
use rig_core::catalog::{Catalog, ModelSpec, Pricing, ReasoningSupport};
use rig_core::http_client::{self, HttpMiddleware};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectOptions, Format, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_ecs::effects::Handler;
use rig_harness::prelude::*;
use serde::Deserialize;

/// The file under `RIG_HOME`.
pub const FILE: &str = "providers.toml";

/// How long a token a `key_command` printed is used when it says nothing
/// of its expiry.
const TOKEN_LIFETIME: Duration = Duration::from_secs(300);

/// Adds the providers of `RIG_HOME/providers.toml` (or of a
/// [`CustomProviders`] inserted before it) and replaces `/model` with one
/// that also takes their models.
#[derive(Default)]
pub struct CustomProviderPlugin;

impl Plugin for CustomProviderPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<CustomProviders>() {
            let providers = CustomProviders::load(&Home::from_env().root().join(FILE));
            app.insert_resource(providers);
        }
        app.init_resource::<Settled>()
            .add_observer(reconnect_restored)
            .add_observer(leave_custom)
            .add_systems(Startup, report_problems)
            .add_systems(First, settle);
        #[cfg(feature = "tui")]
        app.add_message::<rig_tui::PickRequest>();
    }

    /// Replaces `/model`, whichever plugin added it, once every plugin's
    /// `build` ran.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let builtin = world
            .query_filtered::<(Entity, &Name), With<SlashCommand>>()
            .iter(world)
            .find(|(_, name)| name.as_str() == "/model")
            .map(|(entity, _)| entity);
        if let Some(builtin) = builtin {
            world.entity_mut(builtin).insert(Disabled);
        }
        app.add_command(
            "model",
            "Pick the model, or set it with /model vendor/model or /model <provider>/<model>",
            model,
        );
    }
}

/// The agent's model is `<provider>/<model>` of `providers.toml`. Saved
/// with the session; an agent with one has no [`ModelChoice`].
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(Component, Saved)]
pub struct CustomModel(pub String);

/// The configured providers, read once at startup, and what in the file
/// could not be used.
#[derive(Resource, Clone, Default)]
pub struct CustomProviders {
    /// The providers.
    pub providers: Vec<Arc<Provider>>,
    /// Each mistake found in the file.
    pub problems: Vec<String>,
}

/// One provider of `providers.toml`.
pub struct Provider {
    /// What its models' references start with.
    pub name: String,
    base_url: String,
    api: Api,
    key: KeySource,
    headers: HeaderMap,
    /// Its models, as specs on the built-in provider whose wire its `api`
    /// speaks, named `<name>/<id>`.
    pub models: Vec<Arc<ModelSpec>>,
}

/// Where a provider's key comes from.
enum KeySource {
    /// An environment variable.
    Env(String),
    /// A command's output, kept until it expires.
    Command {
        command: String,
        cached: Mutex<Option<(String, Instant)>>,
    },
}

/// The wire a provider speaks.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Api {
    OpenaiChat,
    OpenaiResponses,
    Anthropic,
}

impl Api {
    /// The built-in provider whose wire this is: a configured model is
    /// encoded as one of its models, at the configured URL. Groq stands for
    /// a plain Chat Completions server.
    fn carrier(self) -> Option<ProviderId> {
        match self {
            Api::OpenaiChat => ProviderId::new("groq", Format::OpenAi),
            Api::OpenaiResponses => ProviderId::new("openai", Format::OpenAi),
            Api::Anthropic => ProviderId::new("anthropic", Format::Anthropic),
        }
    }

    /// The header the key goes in.
    fn auth(self, key: &str) -> Result<(HeaderName, HeaderValue), String> {
        let (name, value) = match self {
            Api::OpenaiChat | Api::OpenaiResponses => {
                (http::header::AUTHORIZATION, format!("Bearer {key}"))
            }
            Api::Anthropic => (HeaderName::from_static("x-api-key"), key.to_owned()),
        };
        let mut value = HeaderValue::from_str(&value).map_err(|why| why.to_string())?;
        value.set_sensitive(true);
        Ok((name, value))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    provider: Vec<ProviderEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderEntry {
    name: String,
    base_url: String,
    api: Api,
    key_env: Option<String>,
    key_command: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    id: String,
    name: Option<String>,
    context_window: Option<u32>,
    max_output: Option<u32>,
    price: Option<Price>,
    #[serde(default)]
    reasoning: bool,
    #[serde(default = "yes")]
    tools: bool,
}

/// USD per million tokens.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Price {
    input: f64,
    output: f64,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

fn yes() -> bool {
    true
}

impl CustomProviders {
    /// The providers of the file at `path`; none when there is no file.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text),
            Err(why) if why.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(why) => Self {
                problems: vec![format!("Could not read {}: {why}.", path.display())],
                ..Self::default()
            },
        }
    }

    /// The providers `text`, in the file's shape, lists.
    pub fn parse(text: &str) -> Self {
        let file: File = match toml::from_str(text) {
            Ok(file) => file,
            Err(why) => {
                return Self {
                    problems: vec![format!("{FILE}: {why}")],
                    ..Self::default()
                };
            }
        };
        let mut providers = Self::default();
        for entry in file.provider {
            let name = entry.name.clone();
            match Provider::new(entry) {
                Ok(provider) => providers.providers.push(Arc::new(provider)),
                Err(why) => providers
                    .problems
                    .push(format!("{FILE}: provider `{name}` skipped: {why}.")),
            }
        }
        providers
    }

    /// The provider and model `reference` (`<provider>/<model>`) names.
    pub fn find(&self, reference: &str) -> Option<(&Arc<Provider>, &Arc<ModelSpec>)> {
        let (name, id) = reference.trim().split_once('/')?;
        let provider = self
            .providers
            .iter()
            .find(|provider| provider.name == name)?;
        let spec = provider.models.iter().find(|spec| spec.id == id)?;
        Some((provider, spec))
    }
}

impl Provider {
    fn new(entry: ProviderEntry) -> Result<Self, String> {
        if entry.name.is_empty() || entry.name.contains('/') {
            return Err("a name is a word without `/`".to_owned());
        }
        if ProviderId::catalog(&entry.name).is_some() {
            return Err("the name is a built-in provider's".to_owned());
        }
        let carrier = entry
            .api
            .carrier()
            .ok_or_else(|| format!("this build cannot speak {:?}", entry.api))?;
        let key = match (entry.key_env, entry.key_command) {
            (Some(env), None) => KeySource::Env(env),
            (None, Some(command)) => KeySource::Command {
                command,
                cached: Mutex::new(None),
            },
            _ => return Err("it needs one of `key_env` and `key_command`".to_owned()),
        };
        let mut headers = HeaderMap::new();
        for (name, value) in entry.headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|why| format!("header `{name}`: {why}"))?;
            let value =
                HeaderValue::from_str(&value).map_err(|why| format!("header `{name}`: {why}"))?;
            headers.insert(name, value);
        }
        let models = entry
            .models
            .into_iter()
            .map(|model| {
                let mut spec = ModelSpec::new(carrier, &model.id)
                    .with_display_name(model.name.unwrap_or_else(|| model.id.clone()))
                    .with_tools(model.tools);
                if let Some(tokens) = model.context_window {
                    spec = spec.with_context_window(tokens);
                }
                if let Some(tokens) = model.max_output {
                    spec = spec.with_max_output_tokens(tokens);
                }
                if let Some(price) = model.price {
                    let mut pricing = Pricing::new(price.input, price.output);
                    if let Some(read) = price.cache_read {
                        pricing = pricing.with_cache_read(read);
                    }
                    if let Some(write) = price.cache_write {
                        pricing = pricing.with_cache_write(write);
                    }
                    spec = spec.with_pricing(pricing);
                }
                if model.reasoning {
                    spec = spec.with_reasoning(ReasoningSupport::Unknown { default: None });
                }
                Arc::new(spec)
            })
            .collect();
        Ok(Self {
            name: entry.name,
            base_url: entry.base_url,
            api: entry.api,
            key,
            headers,
            models,
        })
    }

    /// `<name>/<id>`.
    pub fn reference(&self, spec: &ModelSpec) -> String {
        format!("{}/{}", self.name, spec.id)
    }

    /// Whether a key is there: `None` when only running its command tells.
    pub fn has_key(&self) -> Option<bool> {
        match &self.key {
            KeySource::Env(name) => Some(std::env::var(name).is_ok_and(|key| !key.is_empty())),
            KeySource::Command { .. } => None,
        }
    }

    /// The key: the variable's value, or the command's token, run again
    /// once the last one expired. A failed command is reported with its
    /// stderr.
    fn key(&self) -> Result<String, String> {
        match &self.key {
            KeySource::Env(name) => std::env::var(name)
                .ok()
                .filter(|key| !key.is_empty())
                .ok_or_else(|| format!("no key: set `{name}`")),
            KeySource::Command { command, cached } => {
                let mut cached = cached.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some((token, until)) = cached.as_ref()
                    && Instant::now() < *until
                {
                    return Ok(token.clone());
                }
                let (token, lifetime) = run_key_command(command)?;
                *cached = Some((token.clone(), Instant::now() + lifetime));
                Ok(token)
            }
        }
    }

    /// `spec` connected: its wire at the configured URL, with the headers
    /// and key on every request.
    fn connect(self: &Arc<Self>, spec: &Arc<ModelSpec>) -> Result<Connection, String> {
        let reference = self.reference(spec);
        let cannot = |why: String| format!("Cannot use {reference}: {why}.");
        let key = self.key().map_err(cannot)?;
        let http = rig_reqwest::shared().with_middleware(Gateway(Arc::clone(self)));
        let options = ConnectOptions::new()
            .api_key(key)
            .base_url(self.base_url.clone())
            .http(http);
        let model = Catalog::builtin()
            .connect_with(&**spec, options)
            .map_err(|why| cannot(why.to_string()))?;
        let handler = ModelAdapter::<Completion>::new(reference, model);
        Ok(Connection {
            spec: Arc::clone(spec),
            handler: Handler(ErasedHandler::new(handler)),
        })
    }
}

/// Runs a `key_command` with `sh -c`: its output is the token, or JSON
/// with `token` and `expires_in` seconds.
fn run_key_command(command: &str) -> Result<(String, Duration), String> {
    let output = Command::new("sh")
        .args(["-c", command])
        .output()
        .map_err(|why| format!("`{command}` did not run: {why}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`{command}` failed ({}): {}",
            output.status,
            stderr.trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    #[derive(Deserialize)]
    struct Token {
        token: String,
        expires_in: Option<u64>,
    }
    let (token, lifetime) = match serde_json::from_str::<Token>(&stdout) {
        Ok(Token { token, expires_in }) => (
            token,
            expires_in.map_or(TOKEN_LIFETIME, Duration::from_secs),
        ),
        Err(_) => (stdout, TOKEN_LIFETIME),
    };
    if token.is_empty() {
        return Err(format!("`{command}` printed no token"));
    }
    Ok((token, lifetime))
}

/// Puts a provider's headers and current key on each of its requests.
struct Gateway(Arc<Provider>);

impl HttpMiddleware for Gateway {
    fn before_request_headers<'a>(
        &'a self,
        _method: &'a Method,
        _uri: &'a Uri,
        headers: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, http_client::Result<()>> {
        Box::pin(async move {
            for (name, value) in &self.0.headers {
                headers.insert(name.clone(), value.clone());
            }
            let failed = |why: String| http_client::Error::Instance(why.into());
            let key = self.0.key().map_err(failed)?;
            let (name, value) = self.0.api.auth(&key).map_err(failed)?;
            headers.insert(name, value);
            Ok(())
        })
    }
}

/// Set once the first frame's [`settle`] ran: from then on a
/// [`ModelChoice`] inserted is the user's.
#[derive(Resource, Default)]
struct Settled(bool);

/// `/model <provider>/<model>` connects a configured model; any other
/// reference is the catalog's, which [`SetModel`] sets; without arguments,
/// the terminal view lets the user pick one of either.
fn model(
    In(args): In<CommandArgs>,
    providers: Res<CustomProviders>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    #[cfg(feature = "tui")] models: Res<Models>,
    #[cfg(feature = "tui")] mut picks: MessageWriter<rig_tui::PickRequest>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if args.args.is_empty() {
        #[cfg(feature = "tui")]
        pick(agent, &models, &providers, &mut picks, &mut notices);
        return;
    }
    let Some((provider, spec)) = providers.find(&args.args) else {
        commands.trigger(SetModel {
            entity: agent,
            model: args.args,
        });
        return;
    };
    match agents.get(agent) {
        Ok(false) => {}
        Ok(true) => {
            notices.write(Notice::turn_running(agent));
            return;
        }
        Err(_) => return,
    }
    let reference = provider.reference(spec);
    match provider.connect(spec) {
        Ok(connection) => {
            commands
                .entity(agent)
                .remove::<ModelChoice>()
                .insert((CustomModel(reference.clone()), connection));
            let model = format!("Model: {} ({reference}).", spec.display_name);
            notices.write(Notice::info(agent, model));
        }
        Err(why) => {
            notices.write(Notice::error(agent, why));
        }
    }
}

/// Connects a restored agent's configured model again.
fn reconnect_restored(
    restored: On<Restored>,
    agents: Query<&CustomModel>,
    providers: Res<CustomProviders>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = restored.entity;
    if let Ok(CustomModel(reference)) = agents.get(agent) {
        reconnect(agent, reference, &providers, &mut commands, &mut notices);
    }
}

fn reconnect(
    agent: Entity,
    reference: &str,
    providers: &CustomProviders,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let connected = match providers.find(reference) {
        Some((provider, spec)) => provider.connect(spec),
        None => Err(format!("{reference} is no longer in {FILE}.")),
    };
    match connected {
        Ok(connection) => {
            commands.entity(agent).insert(connection);
        }
        Err(why) => {
            commands.entity(agent).remove::<Connection>();
            notices.write(Notice::error(agent, why));
        }
    }
}

/// A catalog model chosen after a configured one replaces it.
fn leave_custom(
    chosen: On<Insert<ModelChoice>>,
    settled: Res<Settled>,
    agents: Query<(), With<CustomModel>>,
    mut commands: Commands,
) {
    if settled.0 && agents.contains(chosen.entity) {
        commands.entity(chosen.entity).remove::<CustomModel>();
    }
}

/// On the first frame, after startup: an agent restored on a configured
/// model that was given a catalog model while starting (the remembered
/// default a new agent gets) goes back to its configured one.
fn settle(
    mut settled: ResMut<Settled>,
    agents: Query<(Entity, &CustomModel), With<ModelChoice>>,
    providers: Res<CustomProviders>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if settled.0 {
        return;
    }
    settled.0 = true;
    for (agent, CustomModel(reference)) in &agents {
        commands.entity(agent).remove::<ModelChoice>();
        reconnect(agent, reference, &providers, &mut commands, &mut notices);
    }
}

/// What in `providers.toml` could not be used, as notices.
fn report_problems(providers: Res<CustomProviders>, mut notices: MessageWriter<Notice>) {
    for problem in &providers.problems {
        notices.write(Notice::error(None, problem.clone()));
    }
}

/// The configured models that call tools, each marked "no key" when its
/// variable is unset, then the catalog's models that can be reached, as
/// rig-models' picker lists them.
#[cfg(feature = "tui")]
fn pick(
    agent: Entity,
    models: &Models,
    providers: &CustomProviders,
    picks: &mut MessageWriter<rig_tui::PickRequest>,
    notices: &mut MessageWriter<Notice>,
) {
    use rig_tui::{PickItem, PickRequest};
    let item = |reference: String, name: &str, note: &str| PickItem {
        label: format!("{reference}  {name}{note}"),
        command: format!("model {reference}"),
    };
    let mut items: Vec<PickItem> = providers
        .providers
        .iter()
        .flat_map(|provider| {
            let note = match provider.has_key() {
                Some(false) => "  (no key)",
                _ => "",
            };
            provider
                .models
                .iter()
                .filter(|spec| spec.tools)
                .map(move |spec| item(provider.reference(spec), &spec.display_name, note))
        })
        .collect();
    items.extend(models.0.reachable().into_iter().map(|spec| {
        let note = match models.0.plan(spec) {
            Some(plan) => format!("  ({plan} plan)"),
            None if spec.provider.requires_credential() => String::new(),
            None => "  (no key needed)".to_owned(),
        };
        item(spec.reference(), &spec.display_name, &note)
    }));
    if items.is_empty() {
        notices.write(Notice::error(
            agent,
            format!("No model can be reached: set a provider's key, or add one to {FILE}."),
        ));
        return;
    }
    picks.write(PickRequest {
        agent,
        title: "Model".to_owned(),
        items,
        selected: 0,
    });
}

#[cfg(test)]
mod tests;
