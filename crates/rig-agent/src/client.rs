//! Classic runtime construction extensions for portable completion clients.

use crate::agent::AgentBuilder;
use crate::provider::ProviderConfig;
use rig_core::providers::descriptor::ApiKeyLocation;

/// Surrender this client's connection details as plain provider configuration.
///
/// The classic `Client<Ext, H>` bakes its credential into the default header
/// map it attaches to every request, so the produced config carries
/// [`ApiKeyLocation::None`] and forwards those headers verbatim as
/// `extra_headers` — the authorization header travels with them. Providers
/// whose credential is *not* a baked header (ChatGPT and Copilot resolve
/// OAuth tokens per request) produce a config without a usable credential;
/// construct their `functions::Config` directly for those flows.
///
/// # Configs may carry credentials
///
/// A produced config is connection data, and connection data includes
/// secrets: `extra_headers` copied from a classic client can include an
/// `authorization` / `x-api-key` header, and some impls carry the key as
/// [`ApiKeyLocation::Inline`]. `Debug` on `ApiKeyLocation` redacts the
/// inline key, but serde stays faithful (resuming a serialized config
/// requires the real value) — treat serialized configs, and any `{:?}`
/// dump that includes `extra_headers`, as secrets.
///
/// # Custom HTTP transports are not carried
///
/// Only plain connection data crosses this bridge. A classic client's
/// custom HTTP transport — proxy, TLS settings, middleware — is NOT part
/// of the produced config. To keep it, pair the config with
/// `AgentBuilder::runtime(Arc<Runtime>)` where the [`crate::provider::Runtime`]
/// was built via `Runtime::with_http` over the same transport.
pub trait ToProviderConfig {
    /// This client's connection details as a [`ProviderConfig`] targeting `model`.
    fn provider_config(&self, model: &str) -> ProviderConfig;
}

/// Convert a client's default header map into `(name, value)` config pairs,
/// dropping `skip` entries (headers the provider's `functions` path derives
/// from dedicated config fields) and non-UTF8 values (with a warning).
fn header_pairs(headers: &http::HeaderMap, skip: &[&str]) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|(name, _)| !skip.contains(&name.as_str()))
        .filter_map(|(name, value)| match value.to_str() {
            Ok(value) => Some((name.as_str().to_string(), value.to_string())),
            Err(_) => {
                tracing::warn!(
                    header = %name,
                    "skipping non-UTF8 header value while converting a client to provider config"
                );
                None
            }
        })
        .collect()
}

/// Collect every UTF-8 value of a (lowercase) header, comma-joined —
/// `HeaderMap::get` would silently return only the first appended value.
fn header_values_joined(headers: &http::HeaderMap, name: &str) -> Option<String> {
    let values: Vec<&str> = headers
        .get_all(name)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    if values.is_empty() {
        None
    } else {
        Some(values.join(","))
    }
}

/// The uniform mapping shared by every provider whose `functions::Config` is
/// exactly `{base_url, api_key, model, extra_headers}`.
macro_rules! impl_to_provider_config_uniform {
    ($(($variant:ident, $module:ident),)*) => {$(
        impl<H> ToProviderConfig for rig_core::providers::$module::Client<H> {
            fn provider_config(&self, model: &str) -> ProviderConfig {
                let mut cfg = rig_core::providers::$module::functions::Config::new(model);
                cfg.base_url = self.base_url().to_string();
                cfg.api_key = ApiKeyLocation::None;
                cfg.extra_headers = header_pairs(self.headers(), &[]);
                ProviderConfig::$variant(cfg)
            }
        }
    )*};
}

impl_to_provider_config_uniform! {
    (Cohere, cohere),
    (DeepSeek, deepseek),
    (Doubleword, doubleword),
    (Groq, groq),
    (Hyperbolic, hyperbolic),
    (Minimax, minimax),
    (Mira, mira),
    (Mistral, mistral),
    (Moonshot, moonshot),
    (Ollama, ollama),
    (OpenRouter, openrouter),
    (Perplexity, perplexity),
    (Together, together),
    (Xai, xai),
    (XiaomiMimo, xiaomimimo),
    (Zai, zai),
}

// ChatGPT's classic client resolves its OAuth/access-token credential per
// request and carries codex-specific state (`originator` / `user-agent` /
// default instructions) in its extension rather than its header map —
// transfer all of it into the functions config. The credential is resolved
// through the non-interactive cached path; flows that would need a token
// refresh or a device-code prompt fall back to the config's
// `CHATGPT_ACCESS_TOKEN` environment default (see `functions::Config::new`).
//
// The transferred token is a point-in-time snapshot: the config does NOT
// refresh it mid-session the way the classic authenticator would. Long
// sessions should rebuild the config from the client when the token nears
// expiry, or rely on the environment-credential fallback instead.
impl<H> ToProviderConfig for rig_core::providers::chatgpt::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        let mut cfg = rig_core::providers::chatgpt::functions::Config::new(model);
        let ext = self.ext();
        cfg.base_url = self.base_url().to_string();
        cfg.default_instructions = ext.default_instructions().map(str::to_string);
        cfg.originator = ext.originator().to_string();
        cfg.user_agent = ext.user_agent().to_string();
        if let Some(context) = ext.authenticator().cached_auth_context() {
            cfg.api_key = ApiKeyLocation::Inline(context.access_token);
            cfg.account_id = context.account_id;
        }
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::ChatGpt(cfg)
    }
}

// Copilot's classic client also resolves its credential per request (API-key,
// GitHub access-token exchange, or OAuth) and may derive its base URL from
// the resolved token's endpoints. Transfer the non-interactive cached
// credential inline; flows that would need a token exchange or device-code
// prompt fall back to the config's `GITHUB_COPILOT_API_KEY` environment
// default (see `functions::Config::new`).
//
// As with ChatGPT, the transferred credential is a point-in-time snapshot:
// Copilot exchange tokens expire and the config does NOT refresh them
// mid-session. Long sessions should rebuild the config from the client, or
// rely on the environment-credential fallback instead.
impl<H> ToProviderConfig for rig_core::providers::copilot::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        let mut cfg = rig_core::providers::copilot::functions::Config::new(model);
        cfg.base_url = self.base_url().to_string();
        if let Some(context) = self.ext().authenticator().cached_auth_context() {
            cfg.api_key = ApiKeyLocation::Inline(context.api_key);
            // The classic runtime prefers the endpoint the token exchange
            // reported over the default base URL (`runtime_base_url`).
            if let Some(api_base) = context.api_base
                && cfg.base_url == rig_core::providers::copilot::functions::DEFAULT_BASE_URL
            {
                cfg.base_url = api_base;
            }
        }
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::Copilot(cfg)
    }
}

// Gemini's classic client authenticates via the `key=` query parameter
// (`build_uri`), not a baked header — the functions path emits that query
// only when the config carries a resolvable key, so transfer it as an
// inline credential instead of `ApiKeyLocation::None`.
impl<H> ToProviderConfig for rig_core::providers::gemini::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        let mut cfg = rig_core::providers::gemini::functions::Config::new(model);
        cfg.base_url = self.base_url().to_string();
        cfg.api_key = ApiKeyLocation::Inline(self.ext().api_key().to_string());
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::Gemini(cfg)
    }
}

// Llamafile's classic ext inserts `/v1` between the client base URL and the
// chat-completions path (`build_uri`), while the `functions` face expects the
// `/v1` inside `Config::base_url` — append it here so both faces hit the
// same endpoint.
impl<H> ToProviderConfig for rig_core::providers::llamafile::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        let mut cfg = rig_core::providers::llamafile::functions::Config::new(model);
        cfg.base_url = format!("{}/v1", self.base_url().trim_end_matches('/'));
        cfg.api_key = ApiKeyLocation::None;
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::Llamafile(cfg)
    }
}

// The canonical OpenAI client speaks the Responses API, so it maps onto the
// responses `functions` face — including its system-instructions placement
// knob, which OpenAI-compatible backends (mistral.rs, OpenRouter) flip via
// `with_system_instructions_as_messages()` and which must survive the bridge
// for the request bytes to stay faithful.
impl<H> ToProviderConfig for rig_core::providers::openai::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        use rig_core::providers::openai::responses_api::ResponsesProviderExt as _;

        let mut cfg = rig_core::providers::openai::responses_api::functions::Config::new(model);
        cfg.base_url = self.base_url().to_string();
        cfg.api_key = ApiKeyLocation::None;
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        cfg.system_instructions_placement = self.ext().system_instructions_placement();
        ProviderConfig::OpenAiResponses(cfg)
    }
}

// The chat-completions flavored OpenAI client (`client.completions_api()`)
// rides the same chat-completions `functions` path as the canonical variant.
impl<H> ToProviderConfig for rig_core::providers::openai::CompletionsClient<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        let mut cfg = rig_core::providers::openai::functions::Config::new(model);
        cfg.base_url = self.base_url().to_string();
        cfg.api_key = ApiKeyLocation::None;
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::OpenAi(cfg)
    }
}

/// Build an anthropic `functions::Config` from a classic client's connection
/// details — shared by the anthropic client and every anthropic-flavored
/// alias client (`minimax`/`moonshot`/`xiaomimimo`/`zai::AnthropicClient`),
/// all of which drive `anthropic::completion::GenericCompletionModel`.
///
/// `default_max_tokens` overrides the config's `max_tokens` fallback: the
/// alias providers classically defaulted to `Some(4096)` for every model
/// (their `AnthropicCompatibleProvider::default_max_tokens`), while the
/// canonical anthropic client passes `None` to keep the model-derived
/// default `Config::new` resolves (which only matches `claude-*` models).
fn anthropic_config(
    base_url: &str,
    headers: &http::HeaderMap,
    model: &str,
    default_max_tokens: Option<u64>,
) -> rig_core::providers::anthropic::functions::Config {
    // `Config::new` also resolves the per-model `default_max_tokens`
    // fallback, mirroring what the classic model type computes.
    let mut cfg = rig_core::providers::anthropic::functions::Config::new(model);
    if let Some(max_tokens) = default_max_tokens {
        cfg.default_max_tokens = Some(max_tokens);
    }
    cfg.base_url = base_url.to_string();
    cfg.api_key = ApiKeyLocation::None;
    // The classic client bakes `anthropic-version` / `anthropic-beta` into
    // its header map, but the functions path emits them from dedicated
    // config fields — transfer the values (all of them: the builder appends
    // one `anthropic-beta` header per beta) instead of duplicating headers.
    if let Some(version) = header_values_joined(headers, "anthropic-version") {
        cfg.anthropic_version = version;
    }
    if let Some(betas) = header_values_joined(headers, "anthropic-beta") {
        cfg.anthropic_betas = betas
            .split(',')
            .map(|beta| beta.trim().to_string())
            .filter(|beta| !beta.is_empty())
            .collect();
    }
    cfg.extra_headers = header_pairs(headers, &["anthropic-version", "anthropic-beta"]);
    cfg
}

impl<H> ToProviderConfig for rig_core::providers::anthropic::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        ProviderConfig::Anthropic(anthropic_config(
            self.base_url(),
            self.headers(),
            model,
            None,
        ))
    }
}

/// The anthropic mapping for each anthropic-flavored alias client: they ride
/// the anthropic completion path, so their configs are anthropic configs
/// pointed at the alias provider's base URL.
macro_rules! impl_to_provider_config_anthropic_alias {
    ($($module:ident :: $ext:ident,)*) => {$(
        impl<H> ToProviderConfig
            for rig_core::client::Client<rig_core::providers::$module::$ext, H>
        {
            fn provider_config(&self, model: &str) -> ProviderConfig {
                ProviderConfig::Anthropic(anthropic_config(
                    self.base_url(),
                    self.headers(),
                    model,
                    // Alias providers classically defaulted `max_tokens`
                    // to 4096 for every model.
                    Some(4096),
                ))
            }
        }
    )*};
}

impl_to_provider_config_anthropic_alias! {
    minimax::MiniMaxAnthropicExt,
    moonshot::MoonshotAnthropicExt,
    xiaomimimo::XiaomiMimoAnthropicExt,
    zai::ZAiAnthropicExt,
}

impl<H> ToProviderConfig for rig_core::providers::azure::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        // Azure routes through `{endpoint}/openai/deployments/{model}` with an
        // `api-version` query parameter; both live in the client extension,
        // not the (empty) base URL.
        let mut cfg =
            rig_core::providers::azure::functions::Config::new(self.ext().endpoint(), model);
        cfg.api_version = self.ext().api_version().to_string();
        cfg.api_key = ApiKeyLocation::None;
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::Azure(cfg)
    }
}

impl<H> ToProviderConfig for rig_core::providers::huggingface::Client<H> {
    fn provider_config(&self, model: &str) -> ProviderConfig {
        // The sub-provider only affects completions through model-identifier
        // qualification (Fireworks); apply it up front so the plain config
        // stays faithful to the classic request bytes.
        let model = self.ext().subprovider().model_identifier(model);
        let mut cfg = rig_core::providers::huggingface::functions::Config::new(model);
        cfg.base_url = self.base_url().to_string();
        cfg.api_key = ApiKeyLocation::None;
        cfg.extra_headers = header_pairs(self.headers(), &[]);
        ProviderConfig::HuggingFace(cfg)
    }
}

/// Classic-runtime construction sugar layered on any provider client that can
/// surrender its connection details as plain configuration.
///
/// Blanket-implemented for every [`ToProviderConfig`] type (each bundled
/// in-core provider `Client`), so `openai.agent(model)` keeps working: it
/// captures the client's base URL and default headers into a
/// [`ProviderConfig`] and hands it to the builder. `use rig::prelude::*;`
/// brings this trait into scope.
///
/// The former `extractor::<T>(model)` sugar is **gone** along with
/// `Extractor<T>`/`ExtractorBuilder<T>`. Structured extraction is now the
/// free-function surface in [`crate::extract`], which takes the same plain
/// configuration directly:
///
/// ```rust,no_run
/// # use rig_agent::{agent::AgentConfig, client::{AgentClientExt, ToProviderConfig},
/// #     extract::{ExtractOptions, extract_with_options}, provider::Runtime};
/// # use std::sync::Arc;
/// # #[derive(serde::Deserialize, schemars::JsonSchema)] struct Person { name: String }
/// # async fn run(client: rig_core::providers::openai::Client) -> Result<(), Box<dyn std::error::Error>> {
/// let person: Person = extract_with_options(
///     AgentConfig::new(),
///     client.provider_config("gpt-4o"),
///     Arc::new(Runtime::new()),
///     "John Doe is a 30 year old doctor.",
///     ExtractOptions::classic_extractor(),
/// )
/// .await?
/// .value;
/// # Ok(())
/// # }
/// ```
pub trait AgentClientExt: ToProviderConfig {
    /// Construct a classic agent builder for `model`.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(&model.into()))
    }
}

impl<C: ToProviderConfig> AgentClientExt for C {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_alias_config_defaults_max_tokens_to_4096() {
        let client = rig_core::providers::minimax::AnthropicClient::new("test-key")
            .expect("client should build");
        let ProviderConfig::Anthropic(cfg) = client.provider_config("MiniMax-M2") else {
            panic!("minimax anthropic alias must bridge to an anthropic config");
        };
        // Classic alias models defaulted `max_tokens` to 4096; without this
        // a bridged alias agent without `.max_tokens(...)` errors out.
        assert_eq!(cfg.default_max_tokens, Some(4096));
    }

    #[test]
    fn canonical_anthropic_config_keeps_model_derived_max_tokens_default() {
        let client =
            rig_core::providers::anthropic::Client::new("test-key").expect("client should build");
        let ProviderConfig::Anthropic(cfg) = client.provider_config("claude-sonnet-4-5") else {
            panic!("anthropic client must bridge to an anthropic config");
        };
        assert_eq!(cfg.default_max_tokens, Some(64_000));
    }

    #[test]
    fn anthropic_config_transfers_every_appended_beta_header() {
        let mut headers = http::HeaderMap::new();
        headers.append(
            "anthropic-beta",
            http::HeaderValue::from_static("token-efficient-tools-2025-02-19"),
        );
        headers.append(
            "anthropic-beta",
            http::HeaderValue::from_static("output-128k-2025-02-19"),
        );
        let cfg = anthropic_config("https://api.anthropic.com", &headers, "claude-3", None);
        assert_eq!(
            cfg.anthropic_betas,
            vec![
                "token-efficient-tools-2025-02-19".to_string(),
                "output-128k-2025-02-19".to_string(),
            ]
        );
        // The beta headers were transferred to config fields, so none may
        // survive as extra headers (which would duplicate them on the wire).
        assert!(
            cfg.extra_headers
                .iter()
                .all(|(name, _)| name != "anthropic-beta"),
            "beta headers leaked into extra_headers: {:?}",
            cfg.extra_headers
        );
    }
}
