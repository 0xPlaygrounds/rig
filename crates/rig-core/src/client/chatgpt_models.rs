//! The model slugs a signed-in ChatGPT account may use, in backend order.
//!
//! ```no_run
//! use rig_core::http_client::DynHttpClient;
//! use rig_core::client::chatgpt_models::account_models;
//! use rig_core::providers::chatgpt::auth::AuthContext;
//!
//! # async fn example(context: &AuthContext, http: &DynHttpClient)
//! # -> Result<(), rig_core::error::ProviderError> {
//! let slugs = account_models(context, http).await?;
//! # let _ = slugs;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use serde::Deserialize;

use crate::error::ProviderError;
use crate::http_client::{DynHttpClient, HttpClientExt};
use crate::providers::chatgpt::auth::AuthContext;
use crate::providers::chatgpt::signed_in_config;

/// The Codex release whose request contract the ChatGPT dialect follows.
/// The backend lists only the models a client this new may use.
pub const CLIENT_VERSION: &str = "0.161.0";
/// The largest model-list response body read, in bytes.
pub const LIST_LIMIT: usize = 8 << 20;
/// How long the complete model-list request may take.
pub const LIST_TIMEOUT: Duration = Duration::from_secs(10);

/// Fetch the model slugs offered to `context`'s account, in backend order.
/// An empty list is valid. Sends one request through `http`, with the
/// ChatGPT dialect's headers and endpoint overrides.
/// Returns an error for transport failures, a timeout, a non-success status,
/// a body larger than [`LIST_LIMIT`], invalid JSON, more than 1024 models,
/// or an empty slug.
pub async fn account_models(
    context: &AuthContext,
    http: &DynHttpClient,
) -> Result<Vec<String>, ProviderError> {
    account_models_with_limits(context, http, LIST_LIMIT, LIST_TIMEOUT).await
}

pub(crate) async fn account_models_with_limits(
    context: &AuthContext,
    http: &DynHttpClient,
    limit: usize,
    timeout: Duration,
) -> Result<Vec<String>, ProviderError> {
    rig_http::wasm_compat::timeout(timeout, async {
        let config = signed_in_config(context).map_err(ProviderError::request)?;
        let uri = config.uri(&format!("/models?client_version={CLIENT_VERSION}"), None);
        let request = config
            .headers_in_session(http::Request::get(uri), None)
            .body(Bytes::new())?;
        let response = http
            .send_streaming(request)
            .await
            .map_err(|error| match error {
                crate::http_client::Error::InvalidStatusCodeWithDetails {
                    status,
                    mut body,
                    headers,
                } => {
                    if body.len() > limit {
                        body.truncate(body.floor_char_boundary(limit));
                        body.push_str(&format!(
                            "\nChatGPT model list is larger than the {limit}-byte limit"
                        ));
                    }
                    ProviderError::from_http_response(status, body)
                        .with_response_headers(Some(headers))
                }
                other => other.into(),
            })?;
        let (parts, mut stream) = response.into_parts();
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if chunk.len() > limit - body.len() {
                let why = format!("ChatGPT model list is larger than the {limit}-byte limit");
                if !parts.status.is_success() {
                    body.extend(chunk.iter().take(limit - body.len()));
                    return Err(ProviderError::from_http_response(
                        parts.status,
                        format!("{}\n{why}", String::from_utf8_lossy(&body)),
                    )
                    .with_response_headers(Some(parts.headers)));
                }
                return Err(ProviderError::Response(why));
            }
            body.extend_from_slice(&chunk);
        }
        if !parts.status.is_success() {
            return Err(ProviderError::from_http_response(
                parts.status,
                String::from_utf8_lossy(&body),
            )
            .with_response_headers(Some(parts.headers)));
        }
        let list: ModelList = serde_json::from_slice(&body)?;
        if list.models.len() > 1024 {
            return Err(ProviderError::Response(
                "ChatGPT model list contains more than 1024 entries".to_owned(),
            ));
        }
        if list.models.iter().any(|model| model.slug.is_empty()) {
            return Err(ProviderError::Response(
                "ChatGPT model list contains an empty slug".to_owned(),
            ));
        }
        Ok(list.models.into_iter().map(|model| model.slug).collect())
    })
    .await
    .map_err(|_| {
        ProviderError::from(crate::http_client::Error::transient(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("ChatGPT model list request timed out after {timeout:?}"),
        )))
    })?
}

#[derive(Deserialize)]
struct ModelList {
    models: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    slug: String,
}

#[cfg(test)]
mod tests;
