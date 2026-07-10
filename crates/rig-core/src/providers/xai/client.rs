use crate::{
    client::{
        self, ApiKey, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider,
        ProviderBuilder, ProviderClient,
    },
    http_client,
};
use std::fmt::Debug;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub enum XAiAuth {
    ApiKey(String),
    OAuth,
}

impl ApiKey for XAiAuth {
    fn into_header(self) -> Option<http_client::Result<(http::HeaderName, http::HeaderValue)>> {
        match self {
            Self::ApiKey(api_key) => BearerAuth::from(api_key).into_header(),
            Self::OAuth => None,
        }
    }
}

impl<S> From<S> for XAiAuth
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::ApiKey(value.into())
    }
}

impl Debug for XAiAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct XAiExt {
    auth: super::auth::Authenticator,
}

#[derive(Debug, Clone)]
pub struct XAiExtBuilder {
    auth_file: Option<PathBuf>,
}

pub type Client<H = reqwest::Client> = client::Client<XAiExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XAiExtBuilder, XAiAuth, H>;

const XAI_BASE_URL: &str = "https://api.x.ai";

impl Default for XAiExtBuilder {
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

impl<H> Capabilities<H> for XAiExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::image_generation::ImageGenerationModel<H>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H>>;
    type Rerank = Nothing;
}

impl DebugExt for XAiExt {}

impl ProviderBuilder for XAiExtBuilder {
    type Extension<H>
        = XAiExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = XAiAuth;

    const BASE_URL: &'static str = XAI_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        let source = match builder.get_api_key() {
            XAiAuth::ApiKey(_) => super::auth::AuthSource::ApiKey,
            XAiAuth::OAuth => super::auth::AuthSource::OAuth,
        };
        Ok(XAiExt {
            auth: super::auth::Authenticator::new(source, builder.ext().auth_file.clone()),
        })
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new xAI client from `XAI_API_KEY`, or an OAuth token cache when
    /// no API key is set.
    fn from_env() -> Result<Self, Self::Error> {
        let mut builder = Self::builder();
        if let Some(base_url) = crate::client::optional_env_var("XAI_API_BASE")? {
            builder = builder.base_url(base_url);
        }

        if let Some(api_key) = crate::client::optional_env_var("XAI_API_KEY")? {
            builder.api_key(api_key).build().map_err(Into::into)
        } else {
            builder.oauth().build().map_err(Into::into)
        }
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

impl<H> client::ClientBuilder<XAiExtBuilder, crate::markers::Missing, H> {
    pub fn oauth(self) -> client::ClientBuilder<XAiExtBuilder, XAiAuth, H> {
        self.api_key(XAiAuth::OAuth)
    }
}

impl<H> ClientBuilder<H> {
    /// Set the directory containing xAI OAuth `auth.json`.
    pub fn token_dir(self, path: impl AsRef<Path>) -> Self {
        let auth_file = path.as_ref().join("auth.json");
        self.auth_file(auth_file)
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

impl XAiExt {
    pub(crate) async fn authorize_request<B>(
        &self,
        mut req: http::Request<B>,
    ) -> http_client::Result<http::Request<B>> {
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

fn default_auth_file() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XAI_OAUTH_AUTH_FILE").map(PathBuf::from) {
        return Some(path);
    }
    if let Some(dir) = std::env::var_os("XAI_OAUTH_TOKEN_DIR").map(PathBuf::from) {
        return Some(dir.join("auth.json"));
    }
    config_dir().map(|dir| dir.join("xai_oauth").join("auth.json"))
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

#[cfg(test)]
mod tests {
    use http::header::AUTHORIZATION;

    #[test]
    fn test_client_initialization() {
        let _client_from_builder = crate::providers::xai::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn oauth_builder_accepts_auth_file() {
        let _client = crate::providers::xai::Client::builder()
            .oauth()
            .auth_file("/tmp/rig-xai-oauth/auth.json")
            .build()
            .expect("OAuth client should build");
    }

    #[test]
    fn provider_client_from_val_still_accepts_api_key_string() {
        use crate::client::ProviderClient;

        let _client = crate::providers::xai::Client::from_val("dummy-key".to_string())
            .expect("ProviderClient::from_val should keep accepting API key strings");
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
        let client = crate::providers::xai::Client::builder()
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
}
