use crate::http_client::HttpClientExt;
use crate::providers::internal::auth::{
    Shell, load_record, request, save_record, send_json, token_expired,
};
use http::Method;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

#[cfg(not(target_family = "wasm"))]
pub use crate::providers::internal::auth::FileTokenStore;
pub use crate::providers::internal::auth::{
    DeviceCodeHandler, DeviceCodePrompt, MemoryTokenStore, StoreLock, TokenStore,
};

#[cfg(not(target_family = "wasm"))]
mod native;

/// The key the record is kept under in the store.
const STORE_KEY: &str = super::PROVIDER_NAME;

const GITHUB_API_KEY_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// Derive the Copilot REST base URL from a chat token's `proxy-ep=` segment.
///
/// The endpoint is parsed from a credential string, not from explicit caller
/// configuration. For that reason, token-derived routing is limited to GitHub
/// Copilot service hosts and HTTPS. Callers that need a custom non-GitHub host
/// can still opt in explicitly with [`OpenAIConfig::with_base_url`](crate::providers::openai::OpenAIConfig::with_base_url).
pub(crate) fn base_url_from_token(token: &str) -> Option<String> {
    let proxy_ep = token
        .split(';')
        .find_map(|part| part.trim().strip_prefix("proxy-ep="))?
        .trim();

    normalize_copilot_proxy_endpoint(proxy_ep)
}

fn normalize_copilot_proxy_endpoint(proxy_ep: &str) -> Option<String> {
    if proxy_ep.is_empty() {
        return None;
    }

    let candidate = if proxy_ep.starts_with("http://") || proxy_ep.starts_with("https://") {
        proxy_ep.to_string()
    } else {
        format!("https://{proxy_ep}")
    };

    let mut url = url::Url::parse(&candidate).ok()?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return None;
    }

    let host = url.host_str()?.to_ascii_lowercase();
    if !is_allowed_token_derived_copilot_host(&host) {
        return None;
    }

    let api_host = host
        .strip_prefix("proxy.")
        .map(|suffix| format!("api.{suffix}"))
        .unwrap_or(host);
    url.set_host(Some(&api_host)).ok()?;

    Some(url.to_string().trim_end_matches('/').to_string())
}

fn is_allowed_token_derived_copilot_host(host: &str) -> bool {
    host == "githubcopilot.com" || host.ends_with(".githubcopilot.com")
}

#[derive(Clone)]
pub enum AuthSource {
    ApiKey(String),
    GitHubAccessToken(String),
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::GitHubAccessToken(_) => f.write_str("GitHubAccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

/// Signs in to GitHub, exchanges the GitHub token for a Copilot session, and
/// keeps both in its [`TokenStore`]. Clones share the store and the refresh
/// lock.
#[derive(Clone)]
pub struct Authenticator(Shell<AuthSource>);

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub use crate::providers::internal::auth::{AuthContext, AuthError};

impl Authenticator {
    /// An authenticator over `source`, keeping the GitHub token and the
    /// session it exchanges for in `store` under
    /// [`PROVIDER_NAME`](super::PROVIDER_NAME), `copilot`;
    /// `device_code_handler` shows a device code when the device flow is
    /// allowed.
    pub fn new(
        source: AuthSource,
        store: impl TokenStore + 'static,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self(Shell::new(
            source,
            store,
            device_code_handler,
            allow_device_flow,
        ))
    }

    /// Resolve the API key and optional API base, refreshing through `http` as needed.
    /// Return store, transport, or authorization errors. OAuth device login is
    /// unsupported on WASM; the GitHub token exchange remains available.
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.0.source {
            AuthSource::ApiKey(api_key) => Ok(AuthContext {
                api_key: api_key.clone().into(),
                account_id: None,
                api_base: None,
            }),
            AuthSource::GitHubAccessToken(access_token) => {
                self.auth_context_with_github_access_token(http, access_token)
                    .await
            }
            #[cfg(not(target_family = "wasm"))]
            AuthSource::OAuth => self.auth_context_oauth(http).await,
            #[cfg(target_family = "wasm")]
            AuthSource::OAuth => Err(AuthError::Message(
                "GitHub Copilot OAuth is not supported on wasm targets".into(),
            )),
        }
    }

    async fn auth_context_with_github_access_token<H>(
        &self,
        http: &H,
        access_token: &str,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        if let Some(context) = self.0.remembered.live(0) {
            return Ok(context);
        }
        let _held = self.0.hold(STORE_KEY).await?;
        let record: ApiKeyRecord = load_record(&*self.0.store, STORE_KEY).await?;
        if record.can_reuse_for_bootstrap_token(access_token) {
            return Ok(self.remember(record));
        }

        // A GitHub token an OAuth sign-in kept stays for that sign-in.
        let session = refresh_api_key(http, access_token)
            .await?
            .bound_to(access_token, record.github_token);
        save_record(&*self.0.store, STORE_KEY, &session).await?;
        Ok(self.remember(session))
    }

    /// `session`'s context, remembered until the session expires.
    fn remember(&self, session: ApiKeyRecord) -> AuthContext {
        let context = AuthContext {
            api_base: session.api_base(),
            api_key: session.token.unwrap_or_default().into(),
            account_id: None,
        };
        self.0.remembered.keep(&context, session.expires_at);
        context
    }
}

/// Copilot API key exchanged for a GitHub access token, and the record kept
/// under the provider's name.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ApiKeyRecord {
    /// The GitHub token an OAuth sign-in got, which the exchange takes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    github_token: Option<String>,
    token: Option<String>,
    expires_at: Option<i64>,
    endpoints: Option<ApiKeyEndpoints>,
    bootstrap_token_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ApiKeyEndpoints {
    api: Option<String>,
}

impl ApiKeyRecord {
    fn api_base(&self) -> Option<String> {
        self.endpoints
            .as_ref()
            .and_then(|endpoints| endpoints.api.as_ref())
            .cloned()
    }

    fn can_reuse_for_bootstrap_token(&self, bootstrap_token: &str) -> bool {
        self.has_live_api_key() && self.matches_bootstrap_token(bootstrap_token)
    }

    /// This session, exchanged for `bootstrap_token`, kept with
    /// `github_token`.
    fn bound_to(mut self, bootstrap_token: &str, github_token: Option<String>) -> Self {
        self.bootstrap_token_fingerprint = Some(bootstrap_token_fingerprint(bootstrap_token));
        self.github_token = github_token;
        self
    }

    fn has_live_api_key(&self) -> bool {
        self.token
            .as_ref()
            .is_some_and(|token| !token.trim().is_empty())
            && !token_expired(self.expires_at, 0)
    }

    fn matches_bootstrap_token(&self, bootstrap_token: &str) -> bool {
        self.bootstrap_token_fingerprint.as_deref()
            == Some(bootstrap_token_fingerprint(bootstrap_token).as_str())
    }
}

fn bootstrap_token_fingerprint(bootstrap_token: &str) -> String {
    let mut hasher = DefaultHasher::new();
    bootstrap_token.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Exchange a GitHub access token for a Copilot API key.
/// Return transport errors, or an error when the response has no non-blank token.
async fn refresh_api_key<H>(http: &H, access_token: &str) -> Result<ApiKeyRecord, AuthError>
where
    H: HttpClientExt,
{
    let response: ApiKeyRecord = send_json(
        http,
        request(Method::GET, GITHUB_API_KEY_URL)
            .header(http::header::ACCEPT, "application/json")
            .header("editor-version", super::EDITOR_VERSION)
            .header("editor-plugin-version", super::EDITOR_PLUGIN_VERSION)
            .header("user-agent", super::USER_AGENT)
            .header(http::header::AUTHORIZATION, format!("token {access_token}"))
            .body(bytes::Bytes::new()),
    )
    .await?;

    if response
        .token
        .as_ref()
        .is_none_or(|token| token.trim().is_empty())
    {
        return Err(AuthError::Message(
            "GitHub Copilot API key response did not include a token".into(),
        ));
    }

    Ok(response)
}
