//! Plugin corpus: 16. Custom provider registration (spec `16-custom-provider.md`).
//!
//! Providers listed in `RIG_HOME/providers.toml` (a company gateway, say)
//! become catalog vendors: each is declared, its models are added to the
//! catalog in [`Models`] as `<name>/<model>`, and the vendor is reached
//! through this plugin's [`Provide`], never the environment's keys for a
//! built-in vendor. So `--model`, `/model`, its picker and completions
//! (with "no key: set …" for one whose `key_env` is unset), the remembered
//! default, subagents and session restore take them like any built-in
//! model. Each is reached on the wire of its `api`, at its `base_url`, with
//! its extra `headers` and its key from `key_env` or `key_command` (run
//! again once the token it printed expired), and priced and windowed by its
//! own entry. `/reload` restarts the process, so it re-reads the file.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use http::{HeaderMap, HeaderName, HeaderValue};
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{
    Catalog, ModelFacts, ModelSpec, Pricing, Provide, ReasoningSupport, Transport,
};
use rig_harness::rig_core::http_client::{self, Exchange, HttpMiddleware};
use rig_harness::rig_core::providers::anthropic::wire::{AnthropicConfig, compatible};
use rig_harness::rig_core::providers::openai::wire::{Auth, Dialect, OpenAIConfig, Route};
use rig_harness::rig_core::providers::registry::{ConnectError, ProviderConfig, ProviderId};
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::wasm_compat::WasmBoxedFuture;
use serde::Deserialize;

/// The file under `RIG_HOME`.
pub const FILE: &str = "providers.toml";

/// How long a token a `key_command` printed is used when it says nothing
/// of its expiry.
const TOKEN_LIFETIME: Duration = Duration::from_secs(300);

/// Adds the providers of `RIG_HOME/providers.toml` (or of a
/// [`CustomProviders`] inserted before it) to the catalog.
#[derive(Default)]
pub struct CustomProviderPlugin;

impl Plugin for CustomProviderPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<CustomProviders>() {
            let path = RigHome::of(app.world()).root().join(FILE);
            app.insert_resource(CustomProviders::load(&path));
        }
        app.add_systems(Startup, report_problems);
    }

    /// Registers the providers once every plugin's `build` ran, so the
    /// process launcher and tokio runtime they run `key_command` with exist.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let processes = world.get_resource_or_init::<Processes>().clone();
        let runtime = world
            .get_resource::<TokioRuntime>()
            .cloned()
            .or_else(|| rig_http::runtime::handle().ok().map(TokioRuntime));
        let providers = world
            .get_resource::<CustomProviders>()
            .map(|providers| providers.providers.clone())
            .unwrap_or_default();
        let mut models = world.get_resource_or_init::<Models>();
        for provider in providers {
            for spec in &provider.models {
                models.0.catalog_mut().insert(spec.clone());
            }
            let gateway = Gateway {
                provider: Arc::clone(&provider),
                processes: processes.clone(),
                runtime: runtime.clone(),
            };
            models.0.provide(provider.id, gateway);
        }
    }
}

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
    /// The vendor its models are filed under.
    pub id: ProviderId,
    /// Its wire, at its URL, without a key: the key goes on each request.
    config: ProviderConfig,
    api: Api,
    key: KeySource,
    headers: HeaderMap,
    /// Its models, as catalog specs of its vendor.
    pub models: Vec<ModelSpec>,
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
}

impl Provider {
    fn new(entry: ProviderEntry) -> Result<Self, String> {
        let id = ProviderId::declare(&entry.name).map_err(|why| why.to_string())?;
        if !id.is_declared() {
            return Err("the name is a built-in provider's".to_owned());
        }
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
        let config = match entry.api {
            Api::OpenaiChat | Api::OpenaiResponses => {
                let route = match entry.api {
                    Api::OpenaiResponses => Route::Responses,
                    _ => Route::Chat,
                };
                let mut config = OpenAIConfig::with_key(&Dialect::gateway(id.vendor(), "", ""), "");
                // The key is put on each request: the wire sends none.
                config.auth = Auth::OptionalBearer;
                ProviderConfig::OpenAi(config.with_base_url(&entry.base_url).with_route(route))
            }
            Api::Anthropic => {
                let dialect = compatible(id.vendor(), "", "", None);
                ProviderConfig::Anthropic(
                    AnthropicConfig::with_key(&dialect, "").with_base_url(&entry.base_url),
                )
            }
        };
        let models = entry
            .models
            .into_iter()
            .map(|model| model_spec(id, model))
            .collect();
        Ok(Self {
            id,
            config,
            api: entry.api,
            key,
            headers,
            models,
        })
    }

    /// The token a `key_command` printed, if it has not expired.
    fn cached(&self) -> Option<String> {
        let KeySource::Command { cached, .. } = &self.key else {
            return None;
        };
        let cached = cached.lock().unwrap_or_else(PoisonError::into_inner);
        cached
            .as_ref()
            .filter(|(_, until)| Instant::now() < *until)
            .map(|(token, _)| token.clone())
    }

    /// The header the key goes in.
    fn auth(&self, key: &str) -> Result<(HeaderName, HeaderValue), String> {
        let (name, value) = match self.api {
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

/// A configured model as a catalog spec of the vendor `id`.
fn model_spec(id: ProviderId, model: ModelEntry) -> ModelSpec {
    let mut spec = ModelSpec::new(id, &model.id)
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
    spec
}

/// How a configured provider's models are reached, and the HTTP middleware
/// that puts its headers and current key on each of its requests.
#[derive(Clone)]
struct Gateway {
    provider: Arc<Provider>,
    /// How `key_command` starts.
    processes: Processes,
    /// Where `key_command` runs, off the request's future.
    runtime: Option<TokioRuntime>,
}

impl Provide for Gateway {
    /// A model of the provider, priced and windowed by its spec. A missing
    /// `key_env` refuses it, so `/model` marks it and choosing it says what
    /// to set.
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
    ) -> Result<ErasedHandler, ConnectError> {
        if let KeySource::Env(name) = &self.provider.key
            && env_key(name).is_none()
        {
            return Err(ConnectError::MissingKey {
                vendor: self.provider.id.vendor().to_owned(),
                tried: vec![name.clone().into()],
            });
        }
        let http = transport.wrap(rig_reqwest::shared().with_middleware(self.clone()));
        let facts = ModelFacts::new(spec.clone()).with_catalog(catalog);
        Ok(self
            .provider
            .config
            .completion_handler(&spec.reference(), &spec.id, http, facts))
    }
}

impl Gateway {
    /// The key: the variable's value, or the command's token, run again
    /// once the last one expired. A failed command says its stderr.
    async fn key(&self) -> Result<String, String> {
        let command = match &self.provider.key {
            KeySource::Env(name) => {
                return env_key(name).ok_or_else(|| format!("no key: set `{name}`"));
            }
            KeySource::Command { command, .. } => command,
        };
        if let Some(token) = self.provider.cached() {
            return Ok(token);
        }
        let mut sh = self.processes.launcher().command("sh");
        sh.args(["-c", command]);
        let run = {
            let command = command.clone();
            move || run_key_command(sh, &command)
        };
        let (token, lifetime) = match &self.runtime {
            Some(runtime) => runtime
                .0
                .spawn_blocking(run)
                .await
                .map_err(|why| format!("`{command}` did not finish: {why}"))??,
            None => run()?,
        };
        if let KeySource::Command { cached, .. } = &self.provider.key {
            *cached.lock().unwrap_or_else(PoisonError::into_inner) =
                Some((token.clone(), Instant::now() + lifetime));
        }
        Ok(token)
    }
}

impl HttpMiddleware for Gateway {
    fn before_request_headers<'a>(
        &'a self,
        _exchange: Exchange<'a>,
        headers: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, http_client::Result<()>> {
        Box::pin(async move {
            let failed = |why: String| http_client::Error::permanent(why);
            let key = self.key().await.map_err(failed)?;
            let (name, value) = self.provider.auth(&key).map_err(failed)?;
            for (name, value) in &self.provider.headers {
                headers.insert(name.clone(), value.clone());
            }
            headers.insert(name, value);
            Ok(())
        })
    }
}

/// The non-empty value of the variable `name`.
fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|key| !key.is_empty())
}

/// Runs `sh`, a `key_command` started as every child process starts: its
/// output is the token, or JSON with `token` and `expires_in` seconds.
fn run_key_command(mut sh: Command, command: &str) -> Result<(String, Duration), String> {
    let output = sh
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

/// What in `providers.toml` could not be used, as notices.
fn report_problems(providers: Res<CustomProviders>, mut notices: MessageWriter<Notice>) {
    for problem in &providers.problems {
        notices.write(Notice::error(None, problem.clone()));
    }
}

#[cfg(test)]
mod tests;
