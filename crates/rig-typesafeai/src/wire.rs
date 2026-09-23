//! Typesafe's unary evaluation operation, using Rig's shared HTTP driver.

use rig_core::client::{EnvError, env};
use rig_core::error::EncodeError;
use rig_core::operation::{One, Take};
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
    type Output = One<Self>;
    type Fold = Take<Self>;
    type Telemetry = ();
    const NAME: &'static str = "evaluation";

    fn is_terminal(_event: &Response) -> bool {
        true
    }
    fn telemetry(_streaming: bool) {}
    fn stamp_request_id(event: &mut Response, request_id: &Option<String>) {
        event.provider_request_id.clone_from(request_id);
    }
    fn stamp_reply(response: &mut Response, reply: Reply) {
        response.provider_request_id = reply.provider_request_id;
    }
}

/// Configuration for Jev's `POST /v1/systemone` endpoint.
///
/// Credentials are redacted when this configuration is printed or serialized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Jev {
    token: Secret,
    model: String,
    endpoint: String,
}

impl Jev {
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
}

impl Wire for Jev {
    type Op = Evaluation;
    type Decoder = JevDecoder;

    fn name(&self) -> &str {
        "typesafeai"
    }
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }
    fn route(&self) -> Option<&str> {
        Some("/v1/systemone")
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
        let mut encoded = Encoded::new(request, Framing::Whole);
        encoded.request_id_header = Some("x-typesafe-request-id");
        Ok(encoded)
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
