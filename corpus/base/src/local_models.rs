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
//! - Each model call gets an outermost adapter, which sees the model the
//!   call finally goes to as every other adapter left it. When that is a
//!   local model, the adapter:
//!   - refuses the call with "Ollama not running at …" while the model is
//!     unavailable: its server stopped listing it, or is down. The vendor's
//!     provider refuses to connect it with the same words, so the picker
//!     leaves it out and choosing it fails. An endpoint that is down is
//!     skipped with one debug line. A saved choice of a local model
//!     connects once a probe finds it (the kernel reconnects when the
//!     catalog changes);
//!   - when the model takes no tools, adds a line saying so to the system
//!     prompt. The kernel already offers it none; a plugin that gives it
//!     tools as text (spec 59) says it takes tools, and the line is left
//!     out.
//! - An agent connected to an Ollama model sends it a small warm-up
//!   request, so the first turn does not wait for the model to load.
//! - Probes, warm-ups and model calls go through the connector's HTTP
//!   transport, so its wrappers see them all.

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_log::{debug, warn};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Catalog, ModelSpec, Pricing, Transport};
use rig_harness::rig_core::completion::Message;
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::http_client::{
    DynHttpClient, HttpClientExt, LazyBody, Method, Request, Response,
};
use rig_harness::rig_core::operation::Completion;
use rig_harness::rig_core::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use rig_harness::rig_core::providers::{llamacpp, ollama};
use rig_harness::rig_core::serve::adapters::ModelAdapter;
use rig_harness::rig_core::serve::{AfterServe, BeforeServe, ErasedHandler, Intercept};
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;
use serde_json::{Value, json};

/// The vendor local models are filed under: `local/<name>`.
pub const VENDOR: &str = "local";

/// The adapter of calls to local models, and the tag of what it adds to the
/// system prompt.
const TAG: &str = "local_model";

/// What the system prompt says to a model that cannot call tools.
pub const NO_TOOLS: &str = "This model cannot call tools, so none are offered. Answer in text, \
                            and ask the user to run what needs running.";

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
            Self::Ollama => ollama::PROVIDER_NAME,
            Self::LlamaCpp => llamacpp::PROVIDER_NAME,
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
        let spec = ModelSpec::new(vendor, &self.name)
            .with_tools(self.tools)
            .with_pricing(Pricing::new(0.0, 0.0));
        match self.context {
            Some(tokens) => spec.with_context_window(tokens),
            None => spec,
        }
    }
}

/// Every local model found, by its server's name for it. Each probe makes
/// a new map, which the vendor's provider and each call's adapter keep.
#[derive(Resource, Clone, Default)]
pub struct LocalModels(Arc<BTreeMap<String, LocalModel>>);

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

/// The plugin's stage system, for others to order against: in Prepare, it
/// pushes the outermost adapter of calls to local models, in
/// [`PrepareSystems::Edit`], before compaction measures the adapted call.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalModelSystems;

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
            .configure_sets(
                AgentPass,
                LocalModelSystems.in_set(PrepareSystems::AdaptOutermost),
            )
            .add_systems(AgentPass, adapt.in_set(LocalModelSystems))
            .add_observer(found)
            .add_observer(warm_up);
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

/// The shared HTTP client, through the connector's wrappers.
fn client(models: &Models) -> DynHttpClient {
    models.0.transport().wrap(rig_reqwest::shared())
}

fn probe_at_startup(
    config: Res<LocalModelsConfig>,
    models: Res<Models>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    start_probe(&mut commands, &config, &models, &wake);
}

/// Probes every endpoint, in the background; [`found`] takes the result.
fn start_probe(commands: &mut Commands, config: &LocalModelsConfig, models: &Models, wake: &Wake) {
    let endpoints = config.endpoints.clone();
    let http = client(models);
    let work = async move {
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
/// models in the catalog and points the `local` vendor at their servers.
fn found(
    done: On<Add<Done<Probed>>>,
    mut probes: Query<&mut Done<Probed>>,
    mut local: ResMut<LocalModels>,
    mut models: ResMut<Models>,
    mut commands: Commands,
) {
    let Ok(mut probe) = probes.get_mut(done.entity) else {
        return;
    };
    let probed = std::mem::take(&mut probe.0);
    commands.entity(done.entity).despawn();
    let found_now = Arc::make_mut(&mut local.0);
    for (endpoint, result) in probed {
        let listed = result.unwrap_or_else(|why| {
            debug!("no local models at {}: {why}", endpoint.base());
            Vec::new()
        });
        for model in found_now.values_mut().filter(|it| it.endpoint == endpoint) {
            model.available = listed.iter().any(|found| found.name == model.name);
        }
        for found in listed {
            let earlier = found_now.get(&found.name);
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
            found_now.insert(found.name, model);
        }
    }
    let vendor = match ProviderId::declare(VENDOR) {
        Ok(vendor) => vendor,
        Err(why) => return warn!("cannot declare the `{VENDOR}` vendor: {why}"),
    };
    let catalog = models.0.catalog_mut();
    for model in found_now.values() {
        catalog.insert(model.spec(vendor));
    }
    let servers = local.clone();
    models.0.provide(
        vendor,
        move |spec: &ModelSpec, catalog: &Catalog, transport: &Transport| {
            serve(&servers, spec, catalog, transport)
        },
    );
}

/// The local model `spec` on its server's OpenAI-compatible `/v1`, through
/// `transport`, or why it cannot be reached.
fn serve(
    servers: &LocalModels,
    spec: &ModelSpec,
    catalog: &Catalog,
    transport: &Transport,
) -> Result<ErasedHandler, ConnectError> {
    let model = servers
        .of(spec)
        .ok_or_else(|| ConnectError::Provider(format!("no local server lists {}", spec.id)))?;
    let endpoint = &model.endpoint;
    if !model.available {
        return Err(ConnectError::Provider(endpoint.down()));
    }
    let served = format!("{}/{}", endpoint.server.dialect(), spec.id);
    let options = ConnectOptions::new()
        .base_url(format!("{}/v1", endpoint.base()))
        .http(transport.wrap(rig_reqwest::shared()));
    let model = catalog.connect_with(served.as_str(), options)?;
    let adapter = ModelAdapter::<Completion>::new(spec.reference(), model);
    Ok(ErasedHandler::new(adapter))
}

/// Serves each model call that goes to a local model, outermost: it sees
/// the model the call finally goes to, whoever routed or adapted it, with
/// the local models as they are now.
fn adapt(mut calls: Query<&mut Adapted, Added<Admitted>>, local: Res<LocalModels>) {
    for mut adapted in &mut calls {
        adapted.0 = local.adapt(adapted.0.clone());
    }
}

impl LocalModels {
    /// `connection`, when it is a local model, refusing calls while the
    /// model is unavailable, and saying so in the system prompt when it
    /// takes no tools.
    fn adapt(&self, mut connection: Connection) -> Connection {
        let Some(model) = self.of(&connection.spec) else {
            return connection;
        };
        let layer = match (model.available, connection.spec.tools) {
            (false, _) => Local::Down(model.endpoint.down()),
            (true, false) => Local::NoTools,
            (true, true) => return connection,
        };
        connection.handler = Handler(connection.handler.0.layered(layer));
        connection
    }
}

/// What a call to a local model needs on its way.
enum Local {
    /// Refused: its server is down or no longer lists it, for this reason.
    Down(String),
    /// [`NO_TOOLS`] added to its system prompt.
    NoTools,
}

impl Intercept for Local {
    fn name(&self) -> String {
        TAG.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> BeforeServe {
        if let Self::Down(why) = self {
            return BeforeServe::Deny(ErrorReport::new(ErrorKind::Http, why).with_retryable(false));
        }
        let EffectKind::Completion { request, stream } = kind else {
            return BeforeServe::Proceed;
        };
        let mut request = request.clone();
        let Some(Message::System { content }) = std::sync::Arc::make_mut(&mut request)
            .chat_history
            .first_mut()
            .map(std::sync::Arc::make_mut)
        else {
            return BeforeServe::Proceed;
        };
        content.push_str(&format!("\n\n<{TAG}>\n{NO_TOOLS}\n</{TAG}>"));
        BeforeServe::Patch(EffectKind::Completion {
            request,
            stream: *stream,
        })
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// An agent connected to an Ollama model warms it up, so its first turn
/// does not wait for the model to load.
fn warm_up(
    inserted: On<Insert<Connection>>,
    agents: Query<&Connection, With<Agent>>,
    local: Res<LocalModels>,
    config: Res<LocalModelsConfig>,
    models: Res<Models>,
) {
    let Some(model) = agents
        .get(inserted.entity)
        .ok()
        .and_then(|connection| local.of(&connection.spec))
    else {
        return;
    };
    if config.warm_up && model.available && model.endpoint.server == Server::Ollama {
        let url = format!("{}/api/generate", model.endpoint.base());
        let body = json!({ "model": model.name, "keep_alive": config.keep_alive });
        let http = client(&models);
        io_pool()
            .spawn(async move {
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
    models: Res<Models>,
    probes: Query<(), With<Probe>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.trim() {
        "refresh" => {
            if probes.is_empty() {
                start_probe(&mut commands, &config, &models, &wake);
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
