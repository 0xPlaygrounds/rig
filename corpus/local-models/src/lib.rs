//! Local models (spec `58-local-models.md`): models served by an Ollama
//! daemon or a llama.cpp server on this machine, found at startup and on
//! `/models refresh` by probing the `[local_models]` endpoints in the
//! background, and used like any catalog model.
//!
//! - The plugin declares the vendor `local` and puts each model found in
//!   the catalog as `local/<name>`, free and with the context window and
//!   tool support its server reports, and provides the vendor: its models
//!   connect to their server's OpenAI-compatible `/v1`. So `/model`, the
//!   picker, `--model`, the saved choice, usage and cost take them like any
//!   other model.
//! - A model the server stops listing, or whose server is down, is marked
//!   unavailable (and so named in the picker); a model call to it is
//!   refused at the gate with "Ollama not running at …". An endpoint that
//!   is down is skipped with one debug line.
//! - A model without tool support is offered none (the kernel's rule for a
//!   catalog entry without tools), and its agent's system prompt says so,
//!   in words that leave room for a plugin that describes tools as text
//!   (spec 59).
//! - An agent connected to an Ollama model sends it a small warm-up
//!   request, so the first turn does not wait for the model to load.

use std::collections::BTreeMap;

use bevy_log::{debug, warn};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Catalog, ModelSpec, Pricing};
use rig_harness::rig_core::http_client::{
    DynHttpClient, HttpClientExt, LazyBody, Method, Request, Response,
};
use rig_harness::rig_core::operation::Completion;
use rig_harness::rig_core::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ModelAdapter;
use serde::Deserialize;
use serde_json::{Value, json};

/// The vendor local models are filed under: `local/<name>`.
pub const VENDOR: &str = "local";

/// The tag of the system prompt's section for a model without tools.
const TAG: &str = "local_model";

/// What the system prompt says to a model that cannot call tools.
pub const NO_TOOLS: &str = "This model has no native tool calling, so no tools are offered to \
                            it as functions. Unless another section describes tools and how to \
                            call them, answer in text and ask the user to run what needs running.";

/// The kind of local server at an endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Server {
    /// An Ollama daemon: `/api/tags`, `/api/show`, and its OpenAI-compatible
    /// `/v1`.
    Ollama,
    /// A llama.cpp `llama-server` (or LM Studio): `/v1/models`, `/props`
    /// and `/v1`.
    LlamaCpp,
}

impl Server {
    /// The vendor whose dialect its `/v1` speaks.
    fn dialect(self) -> &'static str {
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
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    /// The kind of server.
    pub server: Server,
    /// Its address, such as `http://127.0.0.1:11434`.
    pub url: String,
}

impl Endpoint {
    fn new(server: Server, url: &str) -> Self {
        Self {
            server,
            url: url.to_owned(),
        }
    }

    /// Its address without a trailing `/`.
    fn base(&self) -> &str {
        self.url.trim_end_matches('/')
    }

    /// Why a model call to one of its models is refused while the model
    /// is unavailable.
    fn down(&self) -> String {
        format!("{} not running at {}", self.server.label(), self.base())
    }
}

/// The `[local_models]` table: where to look for local models, and whether
/// choosing one warms it up.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalModelsConfig {
    /// The servers probed, in order: a model two serve is the first's.
    pub endpoints: Vec<Endpoint>,
    /// Whether an agent connected to an Ollama model warms it up.
    pub warm_up: bool,
    /// How long Ollama keeps a warmed-up model loaded.
    pub keep_alive: String,
}

impl Default for LocalModelsConfig {
    fn default() -> Self {
        Self {
            endpoints: vec![
                Endpoint::new(Server::Ollama, "http://127.0.0.1:11434"),
                Endpoint::new(Server::LlamaCpp, "http://127.0.0.1:8080"),
            ],
            warm_up: true,
            keep_alive: "10m".to_owned(),
        }
    }
}

/// A local model found on a server.
#[derive(Clone, Debug)]
pub struct LocalModel {
    /// The server's name for it.
    pub name: String,
    /// Where it is served.
    pub endpoint: Endpoint,
    /// Its context window, when the server says.
    pub context: Option<u32>,
    /// Whether it calls tools natively.
    pub tools: bool,
    /// Whether its server listed it at the last probe.
    pub available: bool,
}

impl LocalModel {
    /// Its catalog entry.
    fn spec(&self, vendor: ProviderId) -> ModelSpec {
        let state = if self.available { "" } else { ", unavailable" };
        let spec = ModelSpec::new(vendor, &self.name)
            .with_display_name(format!("{} (local{state})", self.name))
            .with_tools(self.tools)
            .with_pricing(Pricing::new(0.0, 0.0));
        match self.context {
            Some(tokens) => spec.with_context_window(tokens),
            None => spec,
        }
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

    /// The local model `spec` is the catalog entry of.
    fn of(&self, spec: &ModelSpec) -> Option<&LocalModel> {
        (spec.provider.vendor() == VENDOR)
            .then(|| self.0.get(&spec.id))
            .flatten()
    }
}

/// Finds the models of local servers and serves them as `local/<name>`.
#[derive(Default)]
pub struct LocalModelsPlugin;

impl Plugin for LocalModelsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<LocalModelsConfig>("local_models")
            .init_resource::<LocalModels>()
            .add_command(
                "models",
                "List the local models found, or look again with /models refresh",
                models_command,
            )
            .add_systems(Startup, probe_at_startup)
            .add_systems(AgentPass, refuse_unavailable.in_set(Stage::Gate))
            .add_observer(found)
            .add_observer(connected);
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
                Server::Ollama => probe_ollama(&http, endpoint.base()).await,
                Server::LlamaCpp => probe_llama_cpp(&http, endpoint.base()).await,
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

/// Takes a finished probe: marks each model available or not, lists the
/// models in the catalog, points the `local` vendor at their servers, and
/// connects the agents whose saved choice named a model not found yet.
fn found(
    done: On<Add<Done<Probed>>>,
    mut probes: Query<&mut Done<Probed>>,
    mut local: ResMut<LocalModels>,
    mut models: ResMut<Models>,
    agents: Query<(Entity, &ModelChoice), (With<Agent>, Without<Connection>)>,
    mut commands: Commands,
) {
    let Ok(mut probe) = probes.get_mut(done.entity) else {
        return;
    };
    let probed = std::mem::take(&mut probe.0);
    commands.entity(done.entity).despawn();
    for (endpoint, result) in probed {
        let listed = result.unwrap_or_else(|why| {
            debug!("no local models at {}: {why}", endpoint.base());
            Vec::new()
        });
        for model in local.0.values_mut().filter(|it| it.endpoint == endpoint) {
            model.available = listed.iter().any(|found| found.name == model.name);
        }
        for found in listed {
            let earlier = local.0.get(&found.name);
            if earlier.is_some_and(|it| it.endpoint != endpoint && it.available) {
                continue;
            }
            let model = LocalModel {
                name: found.name.clone(),
                endpoint: endpoint.clone(),
                context: found.context,
                tools: found.tools,
                available: true,
            };
            local.0.insert(found.name, model);
        }
    }
    let vendor = match ProviderId::declare(VENDOR) {
        Ok(vendor) => vendor,
        Err(why) => return warn!("cannot declare the `{VENDOR}` vendor: {why}"),
    };
    let catalog = models.0.catalog_mut();
    for model in local.0.values() {
        catalog.insert(model.spec(vendor));
    }
    let servers: BTreeMap<String, Endpoint> = local
        .0
        .iter()
        .map(|(name, model)| (name.clone(), model.endpoint.clone()))
        .collect();
    models
        .0
        .provide(vendor, move |spec: &ModelSpec, catalog: &Catalog| {
            serve(&servers, spec, catalog)
        });
    // A session restored on a local model could not connect it before
    // this probe found it: choosing it again connects it now.
    for (agent, choice) in &agents {
        let name = choice
            .0
            .strip_prefix(VENDOR)
            .and_then(|it| it.strip_prefix('/'));
        if name.is_some_and(|name| local.0.contains_key(name)) {
            commands.entity(agent).insert(choice.clone());
        }
    }
}

/// The local model `spec` on its server's OpenAI-compatible `/v1`.
fn serve(
    servers: &BTreeMap<String, Endpoint>,
    spec: &ModelSpec,
    catalog: &Catalog,
) -> Result<ErasedHandler, ConnectError> {
    let endpoint = servers.get(&spec.id).ok_or(ConnectError::CatalogOnly {
        vendor: VENDOR.to_owned(),
        served_by: "a local server, once one lists it",
    })?;
    let served = format!("{}/{}", endpoint.server.dialect(), spec.id);
    let options = ConnectOptions::new().base_url(format!("{}/v1", endpoint.base()));
    let model = catalog.connect_with(served.as_str(), options)?;
    let adapter = ModelAdapter::<Completion>::new(spec.reference(), model);
    Ok(ErasedHandler::new(adapter))
}

/// Refuses a model call to a local model that is unavailable, before it is
/// sent.
fn refuse_unavailable(
    calls: Query<(Entity, &ModelRequest, Option<&ForAgent>, Option<&BuildFrom>), Undecided>,
    agents: Query<&Connection, With<Agent>>,
    local: Res<LocalModels>,
    mut commands: Commands,
) {
    for (call, request, for_agent, from) in &calls {
        let agent = from.map(|it| it.0).or(for_agent.map(|it| it.0));
        let own = agent.and_then(|agent| agents.get(agent).ok());
        let Some(connection) = request.connection.as_ref().or(own) else {
            continue;
        };
        if let Some(model) = local.of(&connection.spec)
            && !model.available
        {
            let refused = Outcome::refused(model.endpoint.down());
            commands.entity(call).insert_if_new(refused);
        }
    }
}

/// An agent connected to a local model without tools is told so in its
/// system prompt, and one connected to an Ollama model warms it up.
fn connected(
    inserted: On<Insert<Connection>>,
    agents: Query<&Connection, With<Agent>>,
    sections: Query<(Entity, &PromptSection, &SectionOf)>,
    local: Res<LocalModels>,
    config: Res<LocalModelsConfig>,
    mut commands: Commands,
) {
    let agent = inserted.entity;
    let Ok(connection) = agents.get(agent) else {
        return;
    };
    let model = local.of(&connection.spec);
    let told = sections
        .iter()
        .find(|(_, section, of)| of.0 == agent && section.tag == TAG);
    match (model.is_some_and(|it| !it.tools), told) {
        (true, None) => {
            let section = PromptSection::new(PromptSection::ORDER_ROLE, TAG, NO_TOOLS);
            commands.spawn((section, SectionOf(agent)));
        }
        (false, Some((section, ..))) => commands.entity(section).despawn(),
        _ => {}
    }
    let Some(model) = model else {
        return;
    };
    if config.warm_up && model.available && model.endpoint.server == Server::Ollama {
        let url = format!("{}/api/generate", model.endpoint.base());
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
    match args.args.trim() {
        "refresh" => {
            if probes.is_empty() {
                start_probe(&mut commands, &config, &wake);
            }
            notices.write(Notice::info(args.agent, "Looking for local models."));
        }
        "" => {
            let lines: Vec<String> = local.0.values().map(describe).collect();
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
    let window = model.context.map_or_else(
        || "window unknown".to_owned(),
        |tokens| format!("{tokens} tokens"),
    );
    let tools = if model.tools { "tools" } else { "no tools" };
    let state = match model.available {
        true => "available".to_owned(),
        false => format!("unavailable: {}", model.endpoint.down()),
    };
    format!("{VENDOR}/{}: {window}, {tools}, {state}", model.name)
}

#[cfg(test)]
mod tests;
