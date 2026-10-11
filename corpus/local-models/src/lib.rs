//! Local models (spec `58-local-models.md`): models served by an Ollama
//! daemon or a llama.cpp server on this machine, found at startup and on
//! `/models refresh` by probing the configured endpoints in the
//! background, and used like any catalog model.
//!
//! - Each model found is put in the catalog under its server's vendor
//!   (`ollama/<name>`, `llamacpp/<name>`), free and with the context window
//!   and tool support the server reports, so `/model`, the picker, usage
//!   and cost treat it as any other. `/model local/<name>` names it too.
//! - The catalog connects a local model to its server's default address,
//!   so an agent on one gets this plugin's connection instead: the model on
//!   the configured endpoint, behind a layer that fails each request at
//!   once with "Ollama not running at …" while the model is unavailable.
//! - A model the server stops listing, or whose server is down, is marked
//!   unavailable; an endpoint that is down is skipped with one debug line.
//! - A model without tool support is offered none (the kernel's rule for a
//!   catalog entry without tools), and the system prompt says so, unless
//!   the turn is routed to another connection, as the text tool-call plugin
//!   (spec 59) does for such models.
//! - Choosing a local Ollama model sends a small warm-up request, so the
//!   first turn does not wait for the model to load.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy_log::{debug, warn};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::catalog::{Catalog, Connector, ModelSpec, Pricing};
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::http_client::{DynHttpClient, HttpClientExt, LazyBody, Method, Request, Response};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectOptions, ProviderId};
use rig_core::serve::adapters::ModelAdapter;
use rig_core::serve::{Decision, ErasedHandler, Intercept, Verdict};
use rig_ecs::effects::Handler;
use rig_harness::prelude::*;
use serde_json::{Value, json};

/// The prefix `/model` takes for a local model, beside its catalog
/// reference.
pub const LOCAL: &str = "local/";

/// What the system prompt says to a model that cannot call tools.
pub const NO_TOOLS: &str = "This model cannot call tools, so none are offered: answer in \
                            text, and ask the user to run what needs running.";

/// The kind of local server at an endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Server {
    /// An Ollama daemon: `/api/tags`, `/api/show`, and its OpenAI-compatible
    /// `/v1`.
    Ollama,
    /// A llama.cpp `llama-server`: `/v1/models`, `/props` and `/v1`.
    LlamaCpp,
}

impl Server {
    /// The catalog vendor its models are filed under.
    fn vendor(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LlamaCpp => "llamacpp",
        }
    }

    /// Its name for people.
    fn label(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::LlamaCpp => "llama.cpp server",
        }
    }
}

/// A server to probe: its kind and its address, without `/v1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// The kind of server.
    pub server: Server,
    /// Its address, such as `http://127.0.0.1:11434`.
    pub url: String,
}

impl Endpoint {
    /// An Ollama daemon at `url`.
    pub fn ollama(url: impl Into<String>) -> Self {
        Self {
            server: Server::Ollama,
            url: url.into().trim_end_matches('/').to_owned(),
        }
    }

    /// A llama.cpp server at `url`.
    pub fn llama_cpp(url: impl Into<String>) -> Self {
        Self {
            server: Server::LlamaCpp,
            url: url.into().trim_end_matches('/').to_owned(),
        }
    }

    /// What a request to one of its models fails with while it is
    /// unavailable.
    fn down(&self) -> String {
        format!("{} not running at {}", self.server.label(), self.url)
    }
}

/// Where to look for local models, and whether choosing one warms it up.
/// Insert it before adding the plugin to change it.
#[derive(Resource, Clone, Debug)]
pub struct LocalModelsConfig {
    /// The servers probed, in order: a model two serve is the first's.
    pub endpoints: Vec<Endpoint>,
    /// Whether choosing an Ollama model sends a warm-up request.
    pub warm_up: bool,
    /// How long Ollama keeps a warmed-up model loaded.
    pub keep_alive: String,
}

impl Default for LocalModelsConfig {
    fn default() -> Self {
        Self {
            endpoints: vec![
                Endpoint::ollama("http://127.0.0.1:11434"),
                Endpoint::llama_cpp("http://127.0.0.1:8080"),
            ],
            warm_up: true,
            keep_alive: "10m".to_owned(),
        }
    }
}

/// A local model found on a server.
#[derive(Clone)]
pub struct LocalModel {
    /// The server's name for it.
    pub name: String,
    /// Where it is served.
    pub endpoint: Endpoint,
    /// The model on that server. Its spec is this plugin's own copy of the
    /// catalog entry, which tells it apart from the connection the kernel
    /// makes to the server's default address.
    pub connection: Connection,
    /// Whether the server listed it at the last probe; read by the layer
    /// on `connection`.
    up: Arc<AtomicBool>,
}

impl LocalModel {
    /// Whether the server listed it at the last probe.
    pub fn available(&self) -> bool {
        self.up.load(Ordering::Acquire)
    }

    /// Its catalog reference, such as `ollama/qwen-coder`.
    pub fn reference(&self) -> String {
        self.connection.spec.reference()
    }
}

/// Every local model found, by its server's name for it.
#[derive(Resource, Default)]
pub struct LocalModels(BTreeMap<String, LocalModel>);

impl LocalModels {
    /// The model a server calls `name`.
    pub fn get(&self, name: &str) -> Option<&LocalModel> {
        self.0.get(name)
    }

    /// Every model found, by name.
    pub fn iter(&self) -> impl Iterator<Item = &LocalModel> {
        self.0.values()
    }

    /// The model of the catalog reference `reference`.
    fn by_reference(&self, reference: &str) -> Option<&LocalModel> {
        self.0.values().find(|model| model.reference() == reference)
    }
}

/// Finds the models of local servers and connects agents to them.
#[derive(Default)]
pub struct LocalModelsPlugin;

impl Plugin for LocalModelsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LocalModelsConfig>()
            .init_resource::<LocalModels>()
            .add_command(
                "models",
                "List the local models found, or look again with /models refresh",
                models_command,
            )
            .add_systems(Startup, probe_at_startup)
            .add_observer(found)
            .add_observer(use_local_connection)
            .add_observer(local_alias)
            .add_observer(say_no_tools);
    }
}

/// What one server listed.
struct Found {
    name: String,
    context: Option<u32>,
    tools: bool,
}

/// Each endpoint probed, with its models or why it did not answer.
type Probed = Vec<(Endpoint, Result<Vec<Found>, String>)>;

/// On a probe's entity while it runs.
#[derive(Component)]
struct Probe;

/// Model listings and warm-ups wait on the network.
fn io_pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

fn probe_at_startup(config: Res<LocalModelsConfig>, wake: Res<Wake>, mut commands: Commands) {
    start_probe(&mut commands, &config, &wake);
}

/// Probes every endpoint, in the background; [`found`] takes the result.
fn start_probe(commands: &mut Commands, config: &LocalModelsConfig, wake: &Wake) {
    let endpoints = config.endpoints.clone();
    let work = async move {
        let http = rig_reqwest::shared();
        let mut probed = Probed::new();
        for endpoint in endpoints {
            let found = match endpoint.server {
                Server::Ollama => probe_ollama(&http, &endpoint.url).await,
                Server::LlamaCpp => probe_llama_cpp(&http, &endpoint.url).await,
            };
            probed.push((endpoint, found));
        }
        probed
    };
    commands.spawn((
        Name::new("local model probe"),
        Probe,
        Running::spawn(io_pool(), wake, work),
    ));
}

/// The models an Ollama daemon has pulled, each with what `/api/show` says
/// of it: its context length and whether it calls tools (assumed when the
/// daemon is too old to say).
async fn probe_ollama(http: &DynHttpClient, url: &str) -> Result<Vec<Found>, String> {
    let tags = fetch(http, Method::GET, &format!("{url}/api/tags"), None).await?;
    let names = tags
        .get("models")
        .and_then(Value::as_array)
        .ok_or("no model list in /api/tags")?
        .iter()
        .filter_map(|model| model.get("name").and_then(Value::as_str))
        .map(str::to_owned);
    let mut found = Vec::new();
    for name in names {
        let shown = fetch(
            http,
            Method::POST,
            &format!("{url}/api/show"),
            Some(json!({ "model": name })),
        )
        .await
        .unwrap_or(Value::Null);
        let context = shown
            .get("model_info")
            .and_then(Value::as_object)
            .and_then(|info| {
                info.iter()
                    .find(|(key, _)| key.ends_with(".context_length"))
                    .and_then(|(_, length)| length.as_u64())
            })
            .and_then(|length| u32::try_from(length).ok());
        let tools = shown
            .get("capabilities")
            .and_then(Value::as_array)
            .is_none_or(|capabilities| capabilities.iter().any(|it| it == "tools"));
        found.push(Found {
            name,
            context,
            tools,
        });
    }
    Ok(found)
}

/// The model a llama.cpp server serves, with the context it was started
/// with and whether its chat template takes tools (assumed when it does
/// not say).
async fn probe_llama_cpp(http: &DynHttpClient, url: &str) -> Result<Vec<Found>, String> {
    let listed = fetch(http, Method::GET, &format!("{url}/v1/models"), None).await?;
    let props = fetch(http, Method::GET, &format!("{url}/props"), None)
        .await
        .unwrap_or(Value::Null);
    let context = props
        .pointer("/default_generation_settings/n_ctx")
        .and_then(Value::as_u64)
        .and_then(|length| u32::try_from(length).ok());
    let tools = props
        .pointer("/chat_template_caps/supports_tools")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let models = listed
        .get("data")
        .and_then(Value::as_array)
        .ok_or("no model list in /v1/models")?;
    Ok(models
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(|name| Found {
            name: name.to_owned(),
            context,
            tools,
        })
        .collect())
}

/// The JSON `url` answers with, sent `body` when given.
async fn fetch(
    http: &DynHttpClient,
    method: Method,
    url: &str,
    body: Option<Value>,
) -> Result<Value, String> {
    let bytes = match &body {
        Some(body) => serde_json::to_vec(body).map_err(|error| error.to_string())?,
        None => Vec::new(),
    };
    let request = Request::builder()
        .method(method)
        .uri(url)
        .header("content-type", "application/json")
        .body(bytes)
        .map_err(|error| error.to_string())?;
    let response: Response<LazyBody<Vec<u8>>> = http
        .send(request)
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("{url} answered {}", response.status()));
    }
    let body = response
        .into_body()
        .await
        .map_err(|error| error.to_string())?;
    serde_json::from_slice(&body).map_err(|error| error.to_string())
}

/// Takes a finished probe: adds what is new to the catalog, marks each
/// model available or not, and connects the agents on a local model.
fn found(
    done: On<Add<Done<Probed>>>,
    mut probes: Query<&mut Done<Probed>>,
    mut local: ResMut<LocalModels>,
    mut models: ResMut<Models>,
    agents: Query<(Entity, &ModelChoice, Option<&Connection>), With<Agent>>,
    mut commands: Commands,
) {
    let Ok(mut probe) = probes.get_mut(done.entity) else {
        return;
    };
    let probed = std::mem::take(&mut probe.0);
    commands.entity(done.entity).despawn();
    let mut catalog = models.0.catalog().clone();
    let mut added = false;
    for (endpoint, result) in probed {
        let listed = match result {
            Ok(listed) => listed,
            Err(why) => {
                debug!("no local models at {}: {why}", endpoint.url);
                Vec::new()
            }
        };
        for model in local.0.values().filter(|model| model.endpoint == endpoint) {
            let up = listed.iter().any(|found| found.name == model.name);
            model.up.store(up, Ordering::Release);
        }
        for found in listed {
            match local.0.get(&found.name) {
                // Known: the same server again, or a model an earlier
                // server serves.
                Some(known) if known.endpoint == endpoint => {
                    let spec = &known.connection.spec;
                    if spec.context_window == found.context && spec.tools == found.tools {
                        continue;
                    }
                }
                Some(known) if known.available() => continue,
                _ => {}
            }
            match add(&mut catalog, &endpoint, &found) {
                Ok(model) => {
                    local.0.insert(found.name, model);
                    added = true;
                }
                Err(why) => warn!("cannot use the local model {}: {why}", found.name),
            }
        }
    }
    if added {
        // `Connector` has no way to change its catalog, so this drops the
        // sign-ins other plugins added to it.
        *models = Models(Connector::new(catalog));
    }
    for (agent, choice, connection) in &agents {
        let Some(model) = local.by_reference(&choice.0) else {
            continue;
        };
        if connection.is_none_or(|connection| !ours(connection, model)) {
            commands.entity(agent).insert(model.connection.clone());
        }
    }
}

/// `found` in `catalog`, free, and the model on `endpoint`, available.
fn add(catalog: &mut Catalog, endpoint: &Endpoint, found: &Found) -> Result<LocalModel, String> {
    let vendor = endpoint.server.vendor();
    let provider = ProviderId::catalog(vendor).ok_or(format!("no vendor {vendor}"))?;
    let mut spec = ModelSpec::new(provider, &found.name)
        .with_display_name(format!("{} (local)", found.name))
        .with_tools(found.tools)
        .with_pricing(Pricing::new(0.0, 0.0));
    if let Some(context) = found.context {
        spec = spec.with_context_window(context);
    }
    catalog.insert(spec.clone());
    // A copy of its own, not the catalog's: see `LocalModel::connection`.
    let spec = Arc::new(spec);
    let options = ConnectOptions::new().base_url(format!("{}/v1", endpoint.url));
    let model = catalog
        .connect_with(&*spec, options)
        .map_err(|error| error.to_string())?;
    let up = Arc::new(AtomicBool::new(true));
    let available = Available {
        up: up.clone(),
        down: endpoint.down(),
    };
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(spec.reference(), model))
        .layered(available);
    Ok(LocalModel {
        name: found.name.clone(),
        endpoint: endpoint.clone(),
        connection: Connection {
            spec,
            handler: Handler(handler),
        },
        up,
    })
}

/// Whether `connection` is this plugin's connection to `model`.
fn ours(connection: &Connection, model: &LocalModel) -> bool {
    Arc::ptr_eq(&connection.spec, &model.connection.spec)
}

/// Fails each request at once, and for good, while the model is
/// unavailable.
struct Available {
    up: Arc<AtomicBool>,
    down: String,
}

impl Intercept for Available {
    fn name(&self) -> String {
        "local-models".to_owned()
    }

    async fn before(&self, _id: EffectId, _kind: &EffectKind) -> Decision {
        if self.up.load(Ordering::Acquire) {
            return Decision::Proceed;
        }
        let report = ErrorReport::new(ErrorKind::HandlerUnavailable, self.down.clone());
        Decision::Deny(report.with_retryable(false))
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// An agent connected to a local model by the kernel, which reaches the
/// server's default address, gets this plugin's connection instead; once
/// it has it, the model is warmed up.
fn use_local_connection(
    inserted: On<Insert<Connection>>,
    agents: Query<&Connection, With<Agent>>,
    local: Res<LocalModels>,
    config: Res<LocalModelsConfig>,
    mut commands: Commands,
) {
    let Ok(connection) = agents.get(inserted.entity) else {
        return;
    };
    let Some(model) = local.by_reference(&connection.spec.reference()) else {
        return;
    };
    if !ours(connection, model) {
        commands
            .entity(inserted.entity)
            .insert(model.connection.clone());
        return;
    }
    if config.warm_up && model.available() && model.endpoint.server == Server::Ollama {
        let url = format!("{}/api/generate", model.endpoint.url);
        let body = json!({ "model": model.name, "keep_alive": config.keep_alive });
        io_pool()
            .spawn(async move {
                let http = rig_reqwest::shared();
                if let Err(why) = fetch(&http, Method::POST, &url, Some(body)).await {
                    debug!("the warm-up at {url} failed: {why}");
                }
            })
            .detach();
    }
}

/// `/model local/<name>` chooses the local model `name`.
fn local_alias(mut deliver: On<Deliver>, local: Res<LocalModels>) {
    let Some(line) = deliver.command() else {
        return;
    };
    let Some(asked) = line.strip_prefix("model") else {
        return;
    };
    if !asked.starts_with(char::is_whitespace) {
        return;
    }
    let Some(model) = asked
        .trim()
        .strip_prefix(LOCAL)
        .and_then(|name| local.get(name))
    else {
        return;
    };
    deliver.text = format!("/model {}", model.reference());
}

/// Tells a local model without tools that it has none, unless its turn is
/// routed to another connection, such as one that turns tool calls into
/// text.
fn say_no_tools(
    mut prepare: On<PrepareRequest>,
    routed: Query<(), (With<TurnOf>, With<Connection>)>,
    agents: Query<&Connection>,
    local: Res<LocalModels>,
) {
    if routed.contains(prepare.entity) {
        return;
    }
    let Ok(connection) = agents.get(prepare.agent) else {
        return;
    };
    if connection.spec.tools || local.by_reference(&connection.spec.reference()).is_none() {
        return;
    }
    prepare.tools.clear();
    prepare.preamble.push_str("\n\n");
    prepare.preamble.push_str(NO_TOOLS);
}

/// `/models` lists the local models; `/models refresh` looks again.
fn models_command(
    In(args): In<CommandArgs>,
    local: Res<LocalModels>,
    config: Res<LocalModelsConfig>,
    probes: Query<(), With<Probe>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.as_str() {
        "refresh" => {
            if probes.is_empty() {
                start_probe(&mut commands, &config, &wake);
            }
            notices.write(Notice::info(args.agent, "Looking for local models."));
        }
        "" => {
            let lines: Vec<String> = local.iter().map(describe).collect();
            let text = match lines.is_empty() {
                true => "No local models found: /models refresh looks again.".to_owned(),
                false => lines.join("\n"),
            };
            notices.write(Notice::info(args.agent, text));
        }
        _ => {
            notices.write(Notice::error(
                args.agent,
                "/models takes nothing, or `refresh`.",
            ));
        }
    }
}

/// One line about `model`.
fn describe(model: &LocalModel) -> String {
    let spec = &model.connection.spec;
    let window = spec.context_window.map_or_else(
        || "window unknown".to_owned(),
        |tokens| format!("{tokens} tokens"),
    );
    let tools = if spec.tools { "tools" } else { "no tools" };
    let state = match model.available() {
        true => "available".to_owned(),
        false => format!("unavailable: {}", model.endpoint.down()),
    };
    format!(
        "{LOCAL}{} ({}): {window}, {tools}, {state}",
        model.name,
        model.reference()
    )
}

#[cfg(test)]
mod tests;
