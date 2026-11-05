//! Gemini provider configuration and endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::gemini;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = gemini::Gemini::from_env()?;
//!
//! let embeddings = provider.embedding(gemini::EMBEDDING_001, None);
//! # Ok(())
//! # }
//! ```
//!
//! Pair a wire with a transport in a [`Model`](crate::driver::Model) to call it.

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod cached_content;
pub mod caching;
pub mod completion;
pub mod embedding;
pub mod extension;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod interactions_api;
pub mod model_listing;
mod options;
pub mod streaming;
pub mod transcription;

pub use crate::client::gemini::Gemini;
pub use crate::client::gemini_caching::Caching;
#[cfg(feature = "audio")]
pub use audio_generation::{GEMINI_3_8_FLASH_LITE_TTS, GEMINI_3_8_FLASH_TTS};
pub use cached_content::{CacheExpiry, CachedContent, CachedContents, NewCachedContent};
pub use caching::{AutoCache, CacheBook, CacheEvent, CacheReport, Clock, CreatedCache, Lease};
pub use completion::ThoughtReplay;
pub use embedding::{EMBEDDING_001, EMBEDDING_004};
#[cfg(feature = "image")]
pub use image_generation::{
    GEMINI_2_5_FLASH_IMAGE, GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE,
    GEMINI_3_PRO_IMAGE, GEMINI_NANO_BANANA_2_1,
};
pub use interactions_api::{DEEP_RESEARCH_MAX_PREVIEW_04_2026, DEEP_RESEARCH_PREVIEW_04_2026};
pub use model_listing::*;

use crate::client::env::{self, EnvError};
use crate::wire::Secret;

/// Stable descriptor name for both Gemini surfaces, as records and
/// telemetry spell it.
pub use completion::PROVIDER_NAME;

/// Where both Gemini surfaces live.
pub const BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// The environment variable holding the API key.
pub const API_KEY_ENV: &str = "GEMINI_API_KEY";

/// Where a Gemini request carries its API key.
///
/// The REST documentation puts the key in the query string. The official SDKs
/// send the `x-goog-api-key` header instead, which also keeps the credential
/// out of URLs that proxies and logs may retain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Auth {
    /// As the query string's `key` parameter.
    #[default]
    Query,
    /// As the `x-goog-api-key` request header.
    Header,
}

impl Auth {
    /// Whether this placement is the default one, serialized as absence.
    fn is_query(&self) -> bool {
        matches!(self, Self::Query)
    }
}

/// The settings of Gemini's GenerateContent and Interactions APIs:
/// serializable, and the key is never serialized. [`connect`](Self::connect)
/// puts it on a transport as a [`Gemini`] client.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiConfig {
    /// The API key, redacted in `Debug` and in serialized form.
    pub api_key: Secret,
    /// The API root every wire builds its URI from.
    pub base_url: String,
    /// Where a request carries `api_key`; the query string unless set.
    #[serde(default, skip_serializing_if = "Auth::is_query")]
    pub auth: Auth,
}

impl GeminiConfig {
    /// The provider configured with `api_key` and the public base URL.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_owned(),
            auth: Auth::default(),
        }
    }

    /// Load `GEMINI_API_KEY`, returning an error if it is missing or invalid.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Send to a different API root (a proxy, a regional endpoint).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Carry the key in `auth`'s place instead of the query string.
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Build a URI whose credential placement follows `auth`. The returned URI
    /// may contain credentials and must not be logged.
    pub(crate) fn uri_with(&self, path: &str, auth: Auth) -> String {
        let trimmed = path.trim_start_matches('/');
        let base = self.base_url.trim_end_matches('/');
        match auth {
            Auth::Query => {
                let separator = if trimmed.contains('?') { "&" } else { "?" };
                format!("{base}/{trimmed}{separator}key={}", self.api_key.expose())
            }
            Auth::Header => format!("{base}/{trimmed}"),
        }
    }

    /// Build a GenerateContent URI with the configured credential placement.
    /// The returned URI may contain credentials and must not be logged.
    pub(crate) fn uri(&self, path: &str) -> String {
        self.uri_with(path, self.auth)
    }

    /// Put the key on `request` where `auth` says it goes. A query-auth
    /// request already carries it in the URI.
    pub(crate) fn authenticate_with<B>(
        &self,
        mut request: http::Request<B>,
        auth: Auth,
    ) -> Result<http::Request<B>, crate::error::EncodeError> {
        if auth == Auth::Header {
            let value = http::HeaderValue::from_str(self.api_key.expose())
                .map_err(crate::error::EncodeError::request)?;
            request
                .headers_mut()
                .insert(http::HeaderName::from_static(Self::API_KEY_HEADER), value);
        }
        Ok(request)
    }

    /// Put the key on `request` where the configured [`Auth`] says it goes.
    pub(crate) fn authenticate<B>(
        &self,
        request: http::Request<B>,
    ) -> Result<http::Request<B>, crate::error::EncodeError> {
        self.authenticate_with(request, self.auth)
    }

    /// The URI for an Interactions-family `path`. No key: that family
    /// authenticates by the `x-goog-api-key` header instead.
    pub(crate) fn interactions_uri(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/{}", path.trim_start_matches('/'))
    }

    /// The header that authenticates a header-auth request.
    pub(crate) const API_KEY_HEADER: &'static str = "x-goog-api-key";

    /// The header that authenticates an Interactions-family request.
    pub(crate) const INTERACTIONS_KEY_HEADER: &'static str = Self::API_KEY_HEADER;

    /// The `generateContent` / `streamGenerateContent` completion wire.
    pub(crate) fn completion(&self, model: impl Into<String>) -> completion::GenerateContent {
        completion::GenerateContent::new(self.clone(), model)
    }

    /// The Interactions API completion wire.
    pub(crate) fn interactions(&self, model: impl Into<String>) -> interactions_api::Interactions {
        interactions_api::Interactions::new(self.clone(), model)
    }
}

/// The size of a generated image. Each model takes some of them; the
/// API refuses the others.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ImageSize {
    /// `512`: 512 by 512 pixels at 1:1.
    #[serde(rename = "512")]
    FiveTwelve,
    /// `1K`: 1024 by 1024 pixels at 1:1, the API's default.
    #[serde(rename = "1K")]
    OneK,
    /// `2K`: 2048 by 2048 pixels at 1:1.
    #[serde(rename = "2K")]
    TwoK,
    /// `4K`: 4096 by 4096 pixels at 1:1.
    #[serde(rename = "4K")]
    FourK,
}

#[cfg(test)]
mod tests;
