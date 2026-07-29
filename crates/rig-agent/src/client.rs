//! Classic runtime construction extensions for portable completion clients.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::provider::ProviderConfig;
use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};
use rig_core::providers::descriptor::ApiKeyLocation;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Surrender this client's connection details as plain provider configuration.
///
/// The classic `Client<Ext, H>` bakes its credential into the default header
/// map it attaches to every request, so the produced config carries
/// [`ApiKeyLocation::None`] and forwards those headers verbatim as
/// `extra_headers` — the authorization header travels with them. Providers
/// whose credential is *not* a baked header (ChatGPT and Copilot resolve
/// OAuth tokens per request) produce a config without a usable credential;
/// construct their `functions::Config` directly for those flows.
pub trait ToProviderConfig {
    /// This client's connection details as a [`ProviderConfig`] targeting `model`.
    fn provider_config(&self, model: &str) -> ProviderConfig;
}

/// Convert a client's default header map into `(name, value)` config pairs,
/// dropping `skip` entries (headers the provider's `functions` path derives
/// from dedicated config fields) and non-UTF8 values (with a debug log).
fn header_pairs(headers: &http::HeaderMap, skip: &[&str]) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|(name, _)| !skip.contains(&name.as_str()))
        .filter_map(|(name, value)| match value.to_str() {
            Ok(value) => Some((name.as_str().to_string(), value.to_string())),
            Err(_) => {
                tracing::debug!(
                    header = %name,
                    "skipping non-UTF8 header value while converting a client to provider config"
                );
                None
            }
        })
        .collect()
}

/// Look up a single UTF-8 header value by (lowercase) name.
fn header_value(headers: &http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
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
fn anthropic_config(
    base_url: &str,
    headers: &http::HeaderMap,
    model: &str,
) -> rig_core::providers::anthropic::functions::Config {
    // `Config::new` also resolves the per-model `default_max_tokens`
    // fallback, mirroring what the classic model type computes.
    let mut cfg = rig_core::providers::anthropic::functions::Config::new(model);
    cfg.base_url = base_url.to_string();
    cfg.api_key = ApiKeyLocation::None;
    // The classic client bakes `anthropic-version` / `anthropic-beta` into
    // its header map, but the functions path emits them from dedicated
    // config fields — transfer the values instead of duplicating headers.
    if let Some(version) = header_value(headers, "anthropic-version") {
        cfg.anthropic_version = version;
    }
    if let Some(betas) = header_value(headers, "anthropic-beta") {
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
        ProviderConfig::Anthropic(anthropic_config(self.base_url(), self.headers(), model))
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
/// in-core provider `Client`), so `openai.agent(model)` and
/// `openai.extractor::<T>(model)` keep working: they capture the client's base
/// URL and default headers into a [`ProviderConfig`] and hand it to the
/// builder. `use rig::prelude::*;` brings this trait into scope.
pub trait AgentClientExt: ToProviderConfig {
    /// Construct a classic agent builder for `model`.
    fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(&model.into()))
    }

    /// Construct a classic typed extractor builder for `model`.
    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: JsonSchema
            + for<'de> Deserialize<'de>
            + Serialize
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        ExtractorBuilder::new(self.provider_config(&model.into()))
    }
}

impl<C: ToProviderConfig> AgentClientExt for C {}
