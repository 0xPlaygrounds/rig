//! Typesafe's unary evaluation operation, using Rig's shared HTTP driver.

use rig_core::client::{EnvError, env};
use rig_core::driver::Model;
use rig_core::error::EncodeError;
use rig_core::http_client::{DynHttpClient, HttpClientExt};
use rig_core::operation::Whole;
use rig_core::wire::{
    Body, Call, Descriptor, Encoded, Framing, Free, Json, Mode, Operation, Secret, Wire,
};
use serde::{Deserialize, Serialize};

use crate::types::{Request, Response};

/// A batch of independent questions about one state.
pub struct Evaluation;

impl Operation for Evaluation {
    type Request = Request;
    // No events: the reply's end is the response.
    type Event = std::convert::Infallible;
    type End = Response;
    type Response = Response;
    type Fold = Whole<Self>;
    type Emit = Free;

    /// The transport request id and the reply document reach the response.
    fn fold(_request: &Request, _call: &mut Call<'_>) -> Whole<Self> {
        Whole::<Self>::stamping(|response, reply| {
            response
                .provider_request_id
                .clone_from(&reply.provider_request_id);
            response.raw.clone_from(&reply.raw);
        })
    }
}

/// The settings of Jev's `POST /v1/systemone` endpoint, which are also its
/// wire. [`connect`](Self::connect) puts them on a transport as a [`Jev`]
/// client.
///
/// Credentials are redacted when this configuration is printed or serialized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JevConfig {
    token: Secret,
    model: String,
    endpoint: String,
}

impl JevConfig {
    /// Use `jev-latest` at the Typesafe API.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into().into(),
            model: "jev-latest".into(),
            endpoint: "https://api.typesafe.ai/v1/systemone".into(),
        }
    }

    /// Read the credential from `JEV_TOKEN` without loading a secrets file.
    pub fn from_env() -> Result<Self, EnvError> {
        let token = env::required("JEV_TOKEN")?;
        if token.trim().is_empty() {
            return Err(EnvError::Invalid {
                name: "JEV_TOKEN",
                detail: "credential is empty".into(),
            });
        }
        Ok(Self::new(token))
    }

    /// Select the provider model identifier.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the complete endpoint URL, including `/v1/systemone`.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// A client that sends these settings' requests through `http`.
    pub fn connect(self, http: impl HttpClientExt + 'static) -> Jev {
        Jev {
            config: self,
            http: DynHttpClient::new(http),
        }
    }

    /// A client that sends through the shared reqwest client, built once
    /// per process.
    #[cfg(feature = "reqwest")]
    pub fn client(self) -> Jev {
        self.connect(rig_reqwest::shared())
    }
}

/// Jev: its [`JevConfig`] on a transport. The model it builds sends through
/// that transport.
#[derive(Debug, Clone)]
pub struct Jev {
    config: JevConfig,
    http: DynHttpClient,
}

impl Jev {
    /// `jev-latest` at the Typesafe API with `token`, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    pub fn new(token: impl Into<String>) -> Self {
        JevConfig::new(token).client()
    }

    /// The credential from `JEV_TOKEN`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(JevConfig::from_env()?.client())
    }

    /// The same settings, sending through `http`.
    pub fn with_http(self, http: impl HttpClientExt + 'static) -> Self {
        Self {
            http: DynHttpClient::new(http),
            ..self
        }
    }

    /// The settings this client sends with.
    pub fn config(&self) -> &JevConfig {
        &self.config
    }

    /// The evaluation model: the configured Jev model on this client's
    /// transport.
    pub fn evaluation(&self) -> Model<JevConfig> {
        Model::new(self.config.clone(), self.http.clone())
    }
}

impl Wire for JevConfig {
    type Op = Evaluation;
    type Payload = Encoded;
    type Frame = rig_core::wire::WireFrame;
    // One JSON document in, the reply's end out, the document kept as `raw`.
    type Decoder<'id> = Json;
    type Reassembler = rig_core::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("typesafeai").model(self.model.as_str())
    }
    fn encode(&self, request: Request, _mode: Mode) -> Result<Encoded, EncodeError> {
        if self.token.is_empty() || self.model.trim().is_empty() {
            return Err(EncodeError::request(
                "credential and model must be nonempty",
            ));
        }
        #[derive(Serialize)]
        struct Payload<'a> {
            model: &'a str,
            #[serde(flatten)]
            request: Request,
        }
        let body = Payload {
            model: &self.model,
            request,
        };
        let mut authorization =
            http::HeaderValue::from_str(&format!("Bearer {}", self.token.expose()))
                .map_err(EncodeError::request)?;
        authorization.set_sensitive(true);
        let request = http::Request::post(&self.endpoint)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::AUTHORIZATION, authorization)
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(Some("x-typesafe-request-id"))
            .with_route(Some("/v1/systemone")))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        Json
    }
}
