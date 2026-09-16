mod support;

mod cassette {
    mod codex_behaviors;
    mod codex_sessions;
    mod codex_tool_args;
    mod codex_tool_choice;
    mod http_errors;
    mod noninteractive_oauth;
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

use rig::driver::{Bind as _, Bound};
use rig::http_client::BoxedHttpClient;
use rig::providers::chatgpt;
use rig::providers::openai::OpenAI;
use rig::rig_reqwest::client::bundled;
use serde::Deserialize;
use std::path::PathBuf;

const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 60;
pub(crate) const LIVE_MODEL: &str = chatgpt::GPT_5_3_CODEX;

#[derive(Debug, Deserialize)]
struct CachedAuthRecord {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_at: Option<i64>,
}

/// The live ChatGPT provider configuration, credential already exchanged.
///
/// `OpenAI` holds a resolved token, so the exchange — which is not a
/// wire — runs first, on `http`: the same transport the completion then
/// speaks over. The OAuth cache wins when there is a usable one, exactly as
/// the deleted builder's default did; otherwise the variables the dialect
/// names describe the provider outright.
async fn live_provider(http: &BoxedHttpClient) -> OpenAI {
    if !has_usable_oauth_cache() && std::env::var_os("CHATGPT_ACCESS_TOKEN").is_some() {
        return OpenAI::from_env_with(&chatgpt::DIALECT)
            .expect("the ChatGPT environment should describe a provider");
    }

    let context = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        default_auth_file(),
        chatgpt::auth::DeviceCodeHandler::default(),
        true,
    )
    .auth_context(http)
    .await
    .expect("ChatGPT OAuth should resolve an access token");

    let mut provider = OpenAI::with_key(&chatgpt::DIALECT, context.access_token);
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

pub(crate) async fn live_client() -> Bound<OpenAI> {
    let http = bundled().expect("the bundled transport should build");
    live_provider(&http).await.bind(http)
}

fn has_usable_oauth_cache() -> bool {
    let Some(path) = default_auth_file() else {
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

fn default_auth_file() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("chatgpt").join("auth.json"))
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
