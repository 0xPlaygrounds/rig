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

use rig_core::providers::chatgpt::{self, ChatGPTAuth};
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

pub(crate) fn live_builder() -> chatgpt::ClientBuilder {
    let mut builder = chatgpt::Client::builder();

    if let Ok(base_url) =
        std::env::var("CHATGPT_API_BASE").or_else(|_| std::env::var("OPENAI_CHATGPT_API_BASE"))
    {
        builder = builder.base_url(base_url);
    }

    if has_usable_oauth_cache() {
        builder.oauth()
    } else if let Ok(access_token) = std::env::var("CHATGPT_ACCESS_TOKEN") {
        let account_id = std::env::var("CHATGPT_ACCOUNT_ID").ok();
        builder.api_key(ChatGPTAuth::AccessToken {
            access_token,
            account_id,
        })
    } else {
        builder.oauth()
    }
}

pub(crate) fn live_client() -> chatgpt::Client {
    live_builder().build().expect("ChatGPT client should build")
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
