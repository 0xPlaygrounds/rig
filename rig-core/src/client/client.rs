use std::fmt::DebugStruct;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue};

use crate::{
    client::{EmbeddingsClient, ProviderClient, VerifyClient, VerifyError},
    http_client::{self, Builder, HttpClientExt, LazyBody, Request, Response},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub struct Nothing;

#[derive(Clone)]
pub struct Client<'a, Ext = Nothing, H = reqwest::Client> {
    base_url: &'a str,
    headers: HeaderMap,
    http_client: H,
    ext: Ext,
}

pub struct ApiKey<'a>(&'a str);

impl<'a> From<&'a str> for ApiKey<'a> {
    fn from(value: &'a str) -> Self {
        Self(value)
    }
}

pub trait IntoHeader {
    fn make_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>>;
}

impl<'a> IntoHeader for ApiKey<'a> {
    fn make_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(self.0)
                .map(|val| (HeaderName::from_static("AUTHORIZATION"), val))
                .map_err(|e| http_client::Error::from(http::Error::from(e))),
        )
    }
}

// So that i.e Ollama can ignore auth
impl IntoHeader for Nothing {
    fn make_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        None
    }
}

pub trait ClientSpecific<'a> {
    type ApiKey: IntoHeader + From<&'a str>;

    const BASE_URL: &'static str;
    const VERIFY_PATH: &'static str;

    fn with_custom(&self, req: Builder) -> http_client::Result<Builder> {
        Ok(req)
    }
}

pub trait ClientExtBuilder<'a> {
    type Extension: ClientSpecific<'a>;

    fn customize(&self, headers: HeaderMap) -> http_client::Result<HeaderMap>;
    fn build(self) -> Self::Extension;
}

pub trait DebugExt {
    fn with_fields<'a, 'b>(&'a self, f: &'b mut DebugStruct) -> &'b mut DebugStruct
    where
        'a: 'b;
}

// #[derive(Clone)]
// pub struct Client<'a, Ext = Nothing, H = reqwest::Client> {
//     base_url: &'a str,
//     headers: HeaderMap,
//     http_client: H,
//     ext: Ext,
// }

impl<'a, Ext, H> std::fmt::Debug for Client<'a, Ext, H>
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

impl<'a, Ext, H> Client<'a, Ext, H>
where
    Ext: ClientSpecific<'a> + Default,
    H: Default,
{
    pub fn new(api_key: &str) -> http_client::Result<Self> {
        ClientBuilder::default().api_key(api_key).build()
    }
}

impl<'a, Ext, H> Client<'a, Ext, H>
where
    H: HttpClientExt,
    Ext: ClientSpecific<'a>,
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

    pub async fn send_streaming<U>(
        &self,
        req: Request<U>,
    ) -> Result<http_client::StreamingResponse, http_client::Error>
    where
        U: Into<Bytes>,
    {
        self.http_client.send_streaming(req).await
    }
}

impl<'a, Ext, H, Key> ProviderClient for Client<'a, Ext, H>
where
    Ext: ClientSpecific<'a, ApiKey = Key> + Default + Clone,
    Key: From<&'a str>,
    H: Default + Clone,
{
    // FIXME: Realistically, the users API key could contain some invalid characters which could
    // cause this to fail i.e. from incorrectly reading secrets
    // We may want this trait to return results?
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

        Client::new(&api_key).expect("Default client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };

        Client::new(&api_key).expect("Default client should build")
    }
}

impl<'a, Ext, H> Client<'a, Ext, H>
where
    H: Default,
    Ext: Default + ClientSpecific<'a>,
{
    pub fn builder() -> ClientBuilder<NeedsApiKey, Ext, H> {
        ClientBuilder::default()
    }
}

impl<'a, Ext, H> VerifyClient for Client<'a, Ext, H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + Default + Clone,
    Ext: ClientSpecific<'a> + WasmCompatSend + WasmCompatSync + Default + Clone,
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
    pub(crate) ext: Ext,
}

impl<'a, Ext, H> Default for ClientBuilder<Ext, NeedsApiKey, H>
where
    H: Default,
    Ext: Default + ClientSpecific<'a>,
{
    fn default() -> Self {
        Self {
            api_key: NeedsApiKey,
            headers: Default::default(),
            base_url: Ext::BASE_URL,
            http_client: Default::default(),
            ext: Default::default(),
        }
    }
}

impl<'a, Ext, ApiKey, H> ClientBuilder<Ext, ApiKey, H>
where
    ApiKey: From<&'a str>,
{
    /// Map over the ext field
    pub(crate) fn with_client_specific<F>(self, f: F) -> Self
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

    pub fn api_key(self, api_key: &'a str) -> ClientBuilder<Ext, ApiKey, H> {
        ClientBuilder {
            api_key: ApiKey::from(api_key),
            base_url: self.base_url,
            headers: self.headers,
            http_client: self.http_client,
            ext: self.ext,
        }
    }

    pub fn http_client<U>(self, http_client: U) -> ClientBuilder<ApiKey, Ext, U> {
        ClientBuilder {
            http_client,
            base_url: self.base_url,
            api_key: self.api_key,
            headers: self.headers,
            ext: self.ext,
        }
    }
}

impl<'a, HasApiKey, Ext, H> ClientBuilder<Ext, HasApiKey, H>
where
    HasApiKey: IntoHeader + From<&'a str>,
{
    pub fn build<ClientExt>(self) -> http_client::Result<Client<'a, ClientExt, H>>
    where
        ClientExt: ClientSpecific<'a, ApiKey = HasApiKey>,
        Ext: ClientExtBuilder<'a, Extension = ClientExt>,
    {
        let ClientBuilder {
            http_client,
            base_url,
            mut headers,
            ext,
            ..
        } = self;

        let ext = ext.build();

        if let Some((k, v)) = self.api_key.make_header().transpose()? {
            headers.insert(k, v);
        }

        Ok(Client {
            http_client,
            base_url,
            headers,
            ext,
        })
    }
}
