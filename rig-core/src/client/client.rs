use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue};
use std::fmt::{Debug, DebugStruct};

#[cfg(feature = "image")]
use crate::client::ImageGenerationClient;
use crate::{
    client::{
        AsAudioGeneration, AsCompletion, AsEmbeddings, AsImageGeneration, AsTranscription,
        CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient, VerifyClient,
        VerifyError,
    },
    http_client::{self, Builder, HttpClientExt, LazyBody, Request, Response},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub struct Nothing;

pub struct ApiKey(String);

impl From<&str> for ApiKey {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

pub trait IntoHeader: Sized {
    fn make_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        None
    }
}

impl IntoHeader for ApiKey {
    fn make_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        let header = HeaderValue::from_str(&self.0)
            .map(|val| (HeaderName::from_static("AUTHORIZATION"), val))
            .map_err(|e| http_client::Error::from(http::Error::from(e)));

        Some(header)
    }
}

// So that i.e Ollama can ignore auth
impl IntoHeader for Nothing {}

#[derive(Clone)]
pub struct Client<Ext = Nothing, H = reqwest::Client> {
    base_url: &'static str,
    headers: HeaderMap,
    http_client: H,
    ext: Ext,
}

pub trait DebugExt {
    fn with_fields<'a, 'b>(&'a self, f: &'b mut DebugStruct) -> &'b mut DebugStruct
    where
        'a: 'b;
}

impl<Ext, H> std::fmt::Debug for Client<Ext, H>
where
    Ext: DebugExt,
    H: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ext
            .with_fields(
                f.debug_struct("Client")
                    .field("base_url", &self.base_url)
                    .field("headers", &self.headers)
                    .field("http_client", &self.http_client),
            )
            .finish()
    }
}

pub trait Provider {
    type ApiKey: IntoHeader + From<String>;
    type Builder;

    const VERIFY_PATH: &'static str;

    fn build(builder: Self::Builder) -> Self;

    fn with_custom(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req)
    }
}

pub trait ProviderBuilder: Sized {
    const BASE_URL: &'static str;

    fn finish<Key, H>(&self, headers: &mut HeaderMap) -> http_client::Result<()>;
}

impl<Ext, ExtBuilder, ApiKey, H> Client<Ext, H>
where
    ApiKey: From<String> + IntoHeader,
    ExtBuilder: Default + ProviderBuilder,
    Ext: Provider<ApiKey = ApiKey, Builder = ExtBuilder> + Default,
    H: Default,
{
    pub fn new(api_key: &str) -> http_client::Result<Self> {
        Self::builder().api_key::<ApiKey>(api_key).build()
    }
}

impl<Ext, ExtBuilder, H> Client<Ext, H>
where
    H: Default,
    Ext: Default + Provider<Builder = ExtBuilder>,
    ExtBuilder: Default + ProviderBuilder,
{
    pub fn builder() -> ClientBuilder<ExtBuilder, NeedsApiKey, H> {
        ClientBuilder::default()
    }
}

impl<Ext, H> Client<Ext, H>
where
    H: HttpClientExt,
    Ext: Provider,
{
    fn build_uri(&self, path: &str) -> String {
        self.base_url.to_string() + "/" + path.trim_start_matches('/')
    }

    pub fn post(&self, path: &'static str) -> http_client::Result<Builder> {
        self.ext.with_custom(Request::post(self.build_uri(path)))
    }

    pub fn get(&self, path: &'static str) -> http_client::Result<Builder> {
        self.ext.with_custom(Request::get(self.build_uri(path)))
    }

    pub async fn send<T, U>(&self, req: Request<T>) -> http_client::Result<Response<LazyBody<U>>>
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.http_client.send(req).await
    }

    pub async fn send_streaming<U, R>(
        &self,
        req: Request<U>,
    ) -> Result<http_client::StreamingResponse, http_client::Error>
    where
        U: Into<Bytes>,
    {
        self.http_client.send_streaming(req).await
    }
}

/*
impl<Ext, H> EmbeddingsClient for Client<Ext, H>
where
    H: Default + Debug + Clone + WasmCompatSend + WasmCompatSync,
    Ext: EmbeddingsClient + Provider + DebugExt,
    Ext::Builder: Default,
    Self: ProviderClient,
{
    type EmbeddingModel = Ext::EmbeddingModel;

    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        self.ext.embedding_model(model)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        self.ext.embedding_model_with_ndims(model, ndims)
    }
}

impl<Ext, H> TranscriptionClient for Client<Ext, H>
where
    H: Default + Debug + Clone + WasmCompatSend + WasmCompatSync,
    Ext: TranscriptionClient + Provider + DebugExt,
    Ext::Builder: Default,
    Self: ProviderClient,
{
    type TranscriptionModel = Ext::TranscriptionModel;

    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel {
        self.ext.transcription_model(model)
    }
}

#[cfg(feature = "image")]
impl<Ext, H> ImageGenerationClient for Client<Ext, H>
where
    H: Default + Debug + Clone + WasmCompatSend + WasmCompatSync,
    Ext: ImageGenerationClient + Provider + DebugExt,
    Ext::Builder: Default,
    Self: ProviderClient,
{
    type ImageGenerationModel = <Ext as ImageGenerationClient>::ImageGenerationModel;

    fn image_generation_model(
        &self,
        model: &str,
    ) -> <Self as ImageGenerationClient>::ImageGenerationModel {
        self.ext.image_generation_model(model)
    }
}

impl<Ext, H> CompletionClient for Client<Ext, H>
where
    H: Default + Debug + Clone + WasmCompatSend + WasmCompatSync,
    Ext: CompletionClient + Provider + DebugExt,
    Ext::Builder: Default,
    Self: ProviderClient,
{
    type CompletionModel = Ext::CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        self.ext.completion_model(model)
    }

    fn agent(&self, model: &str) -> crate::agent::AgentBuilder<Self::CompletionModel> {
        self.ext.agent(model)
    }

    fn extractor<T>(
        &self,
        model: &str,
    ) -> crate::extractor::ExtractorBuilder<Self::CompletionModel, T>
    where
        T: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + serde::Serialize + Send + Sync,
    {
        self.ext.extractor(model)
    }
}

#[cfg(feature = "audio")]
impl<Ext, H> AudioGenerationClient for Client<Ext, H>
where
    H: Default + Debug + Clone + WasmCompatSend + WasmCompatSync,
    Ext: CompletionClient + Provider + DebugExt,
    Ext::Builder: Default,
{
    type AudioGenerationModel = Ext::AudioGenerationModel;

    fn audio_generation_model(&self, model: &str) -> Self::AudioGenerationModel {
        self.ext.audio_generation_model(model)
    }
}

impl<Ext, Builder, H> ProviderClient for Client<Ext, H>
where
    H: std::fmt::Debug + Default + Clone + WasmCompatSend + WasmCompatSync,
    Ext: Provider<Builder = Builder> + DebugExt + Clone + WasmCompatSend + WasmCompatSync,
    Builder: Default,
    Self: AsEmbeddings + AsImageGeneration + AsAudioGeneration + AsCompletion + AsTranscription,
{
    // FIXME: Realistically, the users API key could contain some invalid characters which could
    // cause this to fail i.e. from incorrectly reading secrets
    // We may want this trait to return results?
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

        Client::builder()
            .api_key(&api_key)
            .build()
            .expect("Default client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };

        Client::builder()
            .api_key(&api_key)
            .build()
            .expect("Default client should build")
    }
}
*/

impl<Ext, H> VerifyClient for Client<Ext, H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + std::fmt::Debug + Default + Clone,
    Ext: DebugExt + Provider + WasmCompatSend + WasmCompatSync + Default + Clone,
    Ext::Builder: Default,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        use http::StatusCode;

        let req = self
            .get(Ext::VERIFY_PATH)?
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            StatusCode::OK => Ok(()),
            StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(VerifyError::InvalidAuthentication)
            }
            StatusCode::INTERNAL_SERVER_ERROR => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            status if status.as_u16() == 529 => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                let status = response.status();

                if status.is_success() {
                    Ok(())
                } else {
                    let text: String = String::from_utf8_lossy(&response.into_body().await?).into();
                    Err(VerifyError::HttpError(http_client::Error::Instance(
                        format!("Failed with '{status}': {text}").into(),
                    )))
                }
            }
        }
    }
}

pub struct NeedsApiKey;

// ApiKey is generic because Anthropic uses custom auth header, local models like Ollama use none
pub struct ClientBuilder<Ext, ApiKey = NeedsApiKey, H = reqwest::Client> {
    base_url: &'static str,
    api_key: ApiKey,
    headers: HeaderMap,
    http_client: H,
    ext: Ext,
}

impl<'a, ExtBuilder, H> Default for ClientBuilder<ExtBuilder, NeedsApiKey, H>
where
    H: Default,
    ExtBuilder: ProviderBuilder + Default,
{
    fn default() -> Self {
        Self {
            api_key: NeedsApiKey,
            headers: Default::default(),
            base_url: ExtBuilder::BASE_URL,
            http_client: Default::default(),
            ext: Default::default(),
        }
    }
}

impl<'a, Ext, H> ClientBuilder<Ext, NeedsApiKey, H> {
    pub fn api_key<ApiKey>(self, api_key: &'a str) -> ClientBuilder<Ext, ApiKey, H>
    where
        ApiKey: From<String>,
    {
        ClientBuilder {
            api_key: ApiKey::from(api_key.into()),
            base_url: self.base_url,
            headers: self.headers,
            http_client: self.http_client,
            ext: self.ext,
        }
    }
}

impl<'a, Ext, ApiKey, H> ClientBuilder<Ext, ApiKey, H> {
    /// Map over the ext field
    pub(crate) fn over_ext<F>(self, f: F) -> Self
    where
        F: Fn(Ext) -> Ext,
    {
        Self {
            ext: f(self.ext),
            ..self
        }
    }

    pub fn base_url(self, base_url: &'static str) -> Self {
        Self { base_url, ..self }
    }

    pub fn http_client<U>(self, http_client: U) -> ClientBuilder<Ext, ApiKey, U> {
        ClientBuilder {
            http_client,
            base_url: self.base_url,
            api_key: self.api_key,
            headers: self.headers,
            ext: self.ext,
        }
    }

    pub(crate) fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    pub(crate) fn ext(&self) -> &Ext {
        &self.ext
    }

    pub(crate) fn ext_mut(&mut self) -> &mut Ext {
        &mut self.ext
    }
}

impl<HasApiKey, ExtBuilder, H> ClientBuilder<ExtBuilder, HasApiKey, H>
where
    HasApiKey: IntoHeader + From<String>,
    ExtBuilder: ProviderBuilder,
{
    pub fn build<Ext>(mut self) -> http_client::Result<Client<Ext, H>>
    where
        Ext: Provider<Builder = ExtBuilder, ApiKey = HasApiKey>,
    {
        self.ext.finish::<HasApiKey, H>(self.headers_mut());

        let ClientBuilder {
            http_client,
            base_url,
            mut headers,
            ext,
            ..
        } = self;

        if let Some((k, v)) = self.api_key.make_header().transpose()? {
            headers.insert(k, v);
        }

        Ok(Client {
            http_client,
            base_url,
            headers,
            ext: Ext::build(ext),
        })
    }
}
