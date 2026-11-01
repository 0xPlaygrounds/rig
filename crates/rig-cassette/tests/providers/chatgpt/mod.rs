use rig_test_support::cassette_models::OpenAiModels;
mod support;

mod cassette {
    // The model the fixtures were recorded with.
    const GPT_5_4: &str = "gpt-5.4";

    mod codex_behaviors;
    mod codex_sessions;
    mod codex_tool_args;
    mod codex_tool_choice;
    mod raw_capture_matrix;
    mod raw_completion_parity_matrix;
    mod raw_stream_capture_matrix;
    mod streaming_tools;
}

mod agent;
mod auth;
mod completion;
mod extractor;
mod extractor_usage;
mod multi_extract;
mod permission_control;
mod reasoning_roundtrip;
mod reasoning_tool_roundtrip;
mod request_hook;
mod streaming;
mod streaming_tools;

use rig::http_client::DynHttpClient;
use rig::providers::chatgpt;
use rig::providers::openai::OpenAIConfig;
use serde::Deserialize;
use std::path::PathBuf;

const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 60;
// The model the fixtures were recorded with. ChatGPT accounts no longer
// get it, so a new recording names a model the account lists.
pub(crate) const LIVE_MODEL: &str = "gpt-5.3-codex";

#[derive(Debug, Deserialize)]
struct CachedAuthRecord {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_at: Option<i64>,
}

/// The live ChatGPT provider configuration, credential already exchanged.
///
/// `OpenAI` holds a resolved token, so the exchange — which is not a
/// wire — runs first, through the authenticator on `http`: the same
/// transport the completion then speaks over. The OAuth cache wins when there is a usable one, exactly as
/// the deleted builder's default did; otherwise the variables the dialect
/// names describe the provider outright.
async fn live_provider(http: &DynHttpClient) -> OpenAIConfig {
    if !has_usable_oauth_cache() && std::env::var_os("CHATGPT_ACCESS_TOKEN").is_some() {
        return OpenAIConfig::from_env_with(&chatgpt::DIALECT)
            .expect("the ChatGPT environment should describe a provider");
    }

    let store: std::sync::Arc<dyn chatgpt::auth::TokenStore> = match default_auth_dir() {
        Some(dir) => std::sync::Arc::new(chatgpt::auth::FileTokenStore::new(dir)),
        None => std::sync::Arc::new(chatgpt::auth::MemoryTokenStore::default()),
    };
    let authenticator = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        store,
        chatgpt::auth::DeviceCodeHandler::default(),
        true,
    );
    let context = authenticator
        .auth_context(http)
        .await
        .expect("ChatGPT OAuth should resolve an access token");
    let mut provider = OpenAIConfig::with_key(&chatgpt::DIALECT, context.api_key);
    if let Some(account_id) = context.account_id {
        provider = provider.with_account_id(account_id);
    }
    if let Ok(base_url) =
        std::env::var("CHATGPT_API_BASE").or_else(|_| std::env::var("OPENAI_CHATGPT_API_BASE"))
    {
        provider = provider.with_base_url(base_url);
    }
    if let Ok(instructions) = std::env::var("CHATGPT_DEFAULT_INSTRUCTIONS")
        && !instructions.trim().is_empty()
    {
        provider = provider.with_instructions(instructions);
    }

    provider
}

pub(crate) async fn live_client() -> OpenAiModels {
    let http = rig_test_support::cassettes::local_http();
    OpenAiModels::new(live_provider(&http).await, http)
}

fn has_usable_oauth_cache() -> bool {
    let Some(path) = default_auth_dir()
        .map(|dir| chatgpt::auth::FileTokenStore::new(dir).path(chatgpt::PROVIDER_NAME))
    else {
        return false;
    };

    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };

    let Ok(record) = serde_json::from_slice::<CachedAuthRecord>(&bytes) else {
        return false;
    };

    record.refresh_token.is_some() || has_unexpired_access_token(&record)
}

fn has_unexpired_access_token(record: &CachedAuthRecord) -> bool {
    if record.access_token.is_none() {
        return false;
    }

    match record.expires_at {
        Some(expires_at) => current_unix_timestamp() + TOKEN_EXPIRY_SKEW_SECONDS < expires_at,
        None => false,
    }
}

fn current_unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after the unix epoch")
        .as_secs() as i64
}

fn default_auth_dir() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("chatgpt"))
}

fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA").map(PathBuf::from)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }
}
