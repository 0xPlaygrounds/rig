use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
    wasm_compat::WasmCompatSend,
};
use std::path::{Path, PathBuf};

/// Original API-key xAI provider extension.
///
/// This remains zero-sized, `Copy`, and `Default` for backwards compatibility.
#[derive(Debug, Default, Clone, Copy)]
pub struct XAiExt;

/// Original API-key xAI provider builder.
///
/// This remains zero-sized, `Copy`, and `Default` for backwards compatibility.
#[derive(Debug, Default, Clone, Copy)]
pub struct XAiExtBuilder;

/// xAI OAuth provider extension backed by a refreshable token cache.
#[derive(Debug, Clone)]
pub struct XAiOAuthExt {
    auth: super::auth::Authenticator,
}

/// Builder state for the xAI OAuth provider extension.
#[derive(Debug, Clone)]
pub struct XAiOAuthExtBuilder {
    auth_file: Option<PathBuf>,
}

type XAiApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<XAiExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XAiExtBuilder, XAiApiKey, H>;
/// xAI client that authorizes requests from a SuperGrok OAuth token cache.
pub type OAuthClient<H = reqwest::Client> = client::Client<XAiOAuthExt, H>;

const XAI_BASE_URL: &str = "https://api.x.ai";

impl Default for XAiOAuthExtBuilder {
    fn default() -> Self {
        Self {
            auth_file: default_auth_file(),
        }
    }
}

impl Provider for XAiExt {
    type Builder = XAiExtBuilder;

    const VERIFY_PATH: &'static str = "/v1/api-key";
}

impl Provider for XAiOAuthExt {
    type Builder = XAiOAuthExtBuilder;

    // xAI's API-key verification endpoint does not accept subscription OAuth.
    // Generic VerifyClient is therefore not a supported OAuth operation.
    const VERIFY_PATH: &'static str = "/v1/api-key";
}

impl<H> Capabilities<H> for XAiExt {
    type Completion = Capable<super::completion::CompletionModel<H, Self>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::image_generation::ImageGenerationModel<H, Self>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H, Self>>;
    type Rerank = Nothing;
}

impl<H> Capabilities<H> for XAiOAuthExt {
    type Completion = Capable<super::completion::CompletionModel<H, Self>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::image_generation::ImageGenerationModel<H, Self>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H, Self>>;
    type Rerank = Nothing;
}

impl DebugExt for XAiExt {}
impl DebugExt for XAiOAuthExt {}

impl ProviderBuilder for XAiExtBuilder {
    type Extension<H>
        = XAiExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = XAiApiKey;

    const BASE_URL: &'static str = XAI_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(XAiExt)
    }
}

impl ProviderBuilder for XAiOAuthExtBuilder {
    type Extension<H>
        = XAiOAuthExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = XAI_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        validate_oauth_api_base(builder.get_base_url())?;
        Ok(XAiOAuthExt {
            auth: super::auth::Authenticator::new(builder.ext().auth_file.clone()),
        })
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new API-key xAI client from `XAI_API_KEY`.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("XAI_API_KEY")?;
        let mut builder = Self::builder();
        if let Some(base_url) = crate::client::optional_env_var("XAI_API_BASE")? {
            builder = builder.base_url(base_url);
        }
        builder.api_key(api_key).build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

impl ProviderClient for OAuthClient {
    type Input = PathBuf;
    type Error = crate::client::ProviderClientError;

    /// Create an OAuth xAI client using the LiteLLM-compatible default cache.
    fn from_env() -> Result<Self, Self::Error> {
        Client::builder().oauth().build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Client::builder()
            .oauth()
            .auth_file(input)
            .build()
            .map_err(Into::into)
    }
}

impl<H> client::ClientBuilder<XAiExtBuilder, crate::markers::Missing, H> {
    /// Switch this builder to SuperGrok OAuth token-cache authentication.
    ///
    /// OAuth requests are restricted to the canonical `https://api.x.ai`
    /// origin. `XAI_API_BASE` remains an API-key-only override.
    pub fn oauth(self) -> client::ClientBuilder<XAiOAuthExtBuilder, Nothing, H> {
        self.over_ext(|_| XAiOAuthExtBuilder::default())
            .api_key(Nothing)
    }
}

impl<H> client::ClientBuilder<XAiOAuthExtBuilder, crate::markers::Missing, H> {
    /// Complete an [`OAuthClient`] builder with OAuth authentication.
    pub fn oauth(self) -> client::ClientBuilder<XAiOAuthExtBuilder, Nothing, H> {
        self.api_key(Nothing)
    }
}

impl<H> client::ClientBuilder<XAiOAuthExtBuilder, Nothing, H> {
    /// Set the directory containing xAI OAuth `auth.json`.
    pub fn token_dir(self, path: impl AsRef<Path>) -> Self {
        self.auth_file(path.as_ref().join("auth.json"))
    }

    /// Set the xAI OAuth `auth.json` file.
    pub fn auth_file(self, path: impl AsRef<Path>) -> Self {
        let auth_file = path.as_ref().to_path_buf();
        self.over_ext(|mut ext| {
            ext.auth_file = Some(auth_file);
            ext
        })
    }
}

/// Request-authorization behavior shared by xAI API-key and OAuth clients.
#[doc(hidden)]
pub trait XAiRequestAuth: Provider {
    fn authorize_request<B>(
        &self,
        req: http::Request<B>,
    ) -> impl Future<Output = http_client::Result<http::Request<B>>> + WasmCompatSend
    where
        B: WasmCompatSend;
}

impl XAiRequestAuth for XAiExt {
    async fn authorize_request<B>(
        &self,
        req: http::Request<B>,
    ) -> http_client::Result<http::Request<B>>
    where
        B: WasmCompatSend,
    {
        Ok(req)
    }
}

impl XAiRequestAuth for XAiOAuthExt {
    async fn authorize_request<B>(
        &self,
        mut req: http::Request<B>,
    ) -> http_client::Result<http::Request<B>>
    where
        B: WasmCompatSend,
    {
        match self.auth.access_token().await {
            Ok(Some(access_token)) => {
                http_client::bearer_auth_header(req.headers_mut(), &access_token)?;
                Ok(req)
            }
            Ok(None) => Ok(req),
            Err(err) => Err(super::auth::into_http_error(err)),
        }
    }
}

fn validate_oauth_api_base(base_url: &str) -> http_client::Result<()> {
    let parsed = url::Url::parse(base_url).map_err(|error| {
        http_client::Error::Instance(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid xAI OAuth API base: {error}"),
        )))
    })?;
    let canonical = parsed.scheme() == "https"
        && parsed.host_str() == Some("api.x.ai")
        && parsed.port().is_none()
        && matches!(parsed.path(), "" | "/")
        && parsed.query().is_none()
        && parsed.fragment().is_none();
    if !canonical {
        return Err(http_client::Error::Instance(Box::new(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "xAI OAuth bearer tokens may only be sent to https://api.x.ai",
        ))));
    }
    Ok(())
}

fn default_auth_file() -> Option<PathBuf> {
    resolve_auth_file(
        std::env::var_os("XAI_OAUTH_AUTH_FILE").map(PathBuf::from),
        std::env::var_os("XAI_OAUTH_TOKEN_DIR").map(PathBuf::from),
        home_dir(),
    )
}

fn resolve_auth_file(
    auth_file: Option<PathBuf>,
    token_dir: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    let token_dir =
        token_dir.or_else(|| home.map(|home| home.join(".config/litellm/xai_oauth")))?;
    match auth_file {
        Some(path) if path.is_absolute() => Some(path),
        Some(path) => Some(token_dir.join(path)),
        None => Some(token_dir.join("auth.json")),
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProviderClient;
    use http::header::AUTHORIZATION;

    #[test]
    fn api_key_public_types_keep_copy_default_contract() {
        fn assert_copy_default<T: Copy + Default>() {}
        assert_copy_default::<XAiExt>();
        assert_copy_default::<XAiExtBuilder>();
        let _client_from_builder = Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn oauth_builder_accepts_auth_file() {
        let _client = Client::builder()
            .oauth()
            .auth_file("/tmp/rig-xai-oauth/auth.json")
            .build()
            .expect("OAuth client should build");
        let _explicit_oauth_client = OAuthClient::builder()
            .oauth()
            .build()
            .expect("OAuthClient::builder() should build");
    }

    #[test]
    fn oauth_rejects_custom_api_origins() {
        assert!(
            Client::builder()
                .base_url("https://example.com")
                .oauth()
                .build()
                .is_err()
        );
        assert!(
            Client::builder()
                .base_url("https://api.x.ai.evil.test")
                .oauth()
                .build()
                .is_err()
        );
    }

    #[test]
    fn provider_client_from_val_still_accepts_api_key_string() {
        let _client = Client::from_val("dummy-key".to_string())
            .expect("ProviderClient::from_val should keep accepting API key strings");
    }

    #[test]
    fn default_cache_path_matches_litellm_and_env_precedence() {
        let temp = assert_fs::TempDir::new().unwrap();
        assert_eq!(
            resolve_auth_file(None, None, Some(temp.path().to_path_buf())).unwrap(),
            temp.path().join(".config/litellm/xai_oauth/auth.json")
        );

        assert_eq!(
            resolve_auth_file(
                Some(PathBuf::from("custom.json")),
                Some(temp.path().join("tokens")),
                None,
            )
            .unwrap(),
            temp.path().join("tokens/custom.json")
        );
        assert_eq!(
            resolve_auth_file(
                Some(temp.path().join("absolute.json")),
                Some(PathBuf::from("ignored")),
                None,
            )
            .unwrap(),
            temp.path().join("absolute.json")
        );
    }

    #[tokio::test]
    async fn oauth_authorization_reads_a_litellm_cache() {
        let temp = assert_fs::TempDir::new().unwrap();
        let auth_file = temp.path().join("auth.json");
        std::fs::write(
            &auth_file,
            format!(
                r#"{{"access_token":"oauth-access","refresh_token":"oauth-refresh","expires_at":{}}}"#,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    + 3600.5
            ),
        )
        .unwrap();
        let client = Client::builder()
            .oauth()
            .auth_file(auth_file)
            .build()
            .unwrap();
        let request = http::Request::get("https://api.x.ai/v1/responses")
            .body(crate::http_client::NoBody)
            .unwrap();

        let request = client.ext().authorize_request(request).await.unwrap();

        assert_eq!(
            request.headers().get(AUTHORIZATION).unwrap(),
            "Bearer oauth-access"
        );
    }

    #[tokio::test]
    async fn generic_verify_is_explicitly_not_oauth_aware() {
        use crate::client::{VerifyClient, VerifyError};
        use crate::test_utils::RecordingHttpClient;

        let temp = assert_fs::TempDir::new().unwrap();
        let auth_file = temp.path().join("auth.json");
        std::fs::write(
            &auth_file,
            format!(
                r#"{{"access_token":"oauth-access","expires_at":{}}}"#,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    + 3600.0
            ),
        )
        .unwrap();
        let http = RecordingHttpClient::with_error_response(
            http::StatusCode::UNAUTHORIZED,
            "api-key endpoint rejects OAuth",
        );
        let client = Client::builder()
            .oauth()
            .auth_file(auth_file)
            .http_client(http.clone())
            .build()
            .unwrap();

        assert!(matches!(
            client.verify().await,
            Err(VerifyError::InvalidAuthentication)
        ));
        assert!(
            http.requests()[0]
                .headers
                .get(http::header::AUTHORIZATION)
                .is_none()
        );
    }
}
