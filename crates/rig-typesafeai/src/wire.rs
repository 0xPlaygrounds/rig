//! Typesafe's unary evaluation operation, using Rig's shared HTTP driver.

use rig_core::client::{EnvError, env};
use rig_core::driver::Model;
use rig_core::error::EncodeError;
use rig_core::http_client::{DynHttpClient, HttpClientExt};
use rig_core::operation::{Events, Take};
use rig_core::wire::{
    Body, Decoder, Encoded, Framing, Mode, Operation, Output, Reply, Secret, Sink, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use crate::types::{Request, Response};

/// A batch of independent questions about one state.
pub struct Evaluation;

impl Operation for Evaluation {
    type Request = Request;
    type Event = Response;
    type Response = Response;
    type Capabilities = ();
    type Output = Events<Self>;
    type Fold = Take<Self>;
    type Telemetry = ();
    const NAME: &'static str = "evaluation";

    fn is_terminal(_event: &Response) -> bool {
        true
    }
    fn fold<W: Wire<Op = Self>>(_request: &Request, _wire: &W, _mode: Mode) -> Take<Self> {
        Take::default()
    }
    fn telemetry(_mode: Mode) {}
    fn stamp_reply(response: &mut Response, reply: &Reply) {
        response
            .provider_request_id
            .clone_from(&reply.provider_request_id);
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
    type Decoder = JevDecoder;

    fn name(&self) -> &str {
        "typesafeai"
    }
    fn id(&self) -> Option<&str> {
        Some(&self.model)
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

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        JevDecoder
    }
}

/// Decoder for the endpoint's single JSON response.
pub struct JevDecoder;

impl Decoder<Evaluation> for JevDecoder {
    type Event = Response;

    fn classify(&self, frame: WireFrame) -> WireEvent<Response> {
        rig_core::providers::internal::wire::classify_marker_keyed_frame(
            &frame.as_str(),
            &["answers", "model"],
        )
    }

    fn interpret(&mut self, response: Response, out: &mut Output<Evaluation>) {
        out.push(Ok(response));
    }
}
