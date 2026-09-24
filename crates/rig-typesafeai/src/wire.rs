//! Typesafe's unary evaluation operation, driven by a Rig [`Model`](rig_core::driver::Model).

use rig_core::client::{EnvError, env};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::One;
use rig_core::wire::{
    Body, Decoder, Encoded, Fold, Framing, Mode, Operation, Output, Reply, Secret, Sink, Wire,
    WireEvent, WireFrame,
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
    type Fold = Answered;
    const NAME: &'static str = "evaluation";

    fn is_terminal(_event: &Response) -> bool {
        true
    }
}

/// The evaluation's one answer, carrying the reply's transport request id.
#[derive(Default)]
pub struct Answered(Option<Response>);

impl Fold<Evaluation> for Answered {
    fn absorb(&mut self, response: Response) -> Result<(), ProviderError> {
        if self.0.is_none() {
            self.0 = Some(response);
        }
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<Response, ProviderError> {
        let mut response = self.0.ok_or_else(|| {
            ProviderError::Response("evaluation reply carried no payload".to_owned())
        })?;
        response.provider_request_id = reply.provider_request_id;
        Ok(response)
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
    type Payload = rig_core::wire::Encoded;
    type Frame = rig_core::wire::WireFrame;
    type Decoder = JevDecoder;

    fn name(&self) -> &str {
        "typesafeai"
    }
    fn model(&self) -> Option<&str> {
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
            .with_route("/v1/systemone"))
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
