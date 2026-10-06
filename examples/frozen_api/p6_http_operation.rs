use std::convert::Infallible;

use rig::driver::Model;
use rig::error::EncodeError;
use rig::operation::Whole;
use rig::wire::{
    Body, Call, Descriptor, Encoded, Framing, Free, Json, Mode, Operation, Secret, Wire, WireFrame,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// A batch of questions about one state, answered in one JSON document.
pub struct Evaluation;

#[derive(Debug, Clone, Serialize)]
pub struct EvalRequest {
    pub state: serde_json::Value,
    pub questions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalResponse {
    pub answers: Vec<serde_json::Value>,
    pub model: String,
    #[serde(skip)]
    pub provider_request_id: Option<String>,
    /// The whole reply document, including fields this type does not model.
    #[serde(skip)]
    pub raw: serde_json::Value,
}

impl Operation for Evaluation {
    type Request = EvalRequest;
    // No events: the reply's end *is* the response, so "no payload" is not a
    // state this operation can reach.
    type Event = Infallible;
    type End = EvalResponse;
    type Response = EvalResponse;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &EvalRequest, _call: &mut Call<'_>) -> Whole<Self> {
        Whole::<Self>::stamping(|response, reply| {
            response
                .provider_request_id
                .clone_from(&reply.provider_request_id);
            response.raw.clone_from(&reply.raw);
        })
    }
}

/// Typesafe's endpoint: plain data, credentials redacted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Jev {
    token: Secret,
    endpoint: String,
}

impl Jev {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into().into(),
            endpoint: "https://api.typesafe.ai/v1/systemone".to_owned(),
        }
    }
}

impl Wire for Jev {
    type Op = Evaluation;
    type Payload = Encoded;
    type Frame = WireFrame;
    // One JSON document in, the reply's end out, the document kept as `raw`.
    type Decoder<'id> = Json;
    type Reassembler = rig::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("typesafeai")
    }

    fn encode(&self, request: EvalRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let mut authorization =
            http::HeaderValue::from_str(&format!("Bearer {}", self.token.expose()))
                .map_err(EncodeError::request)?;
        authorization.set_sensitive(true);
        let request = http::Request::post(&self.endpoint)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::AUTHORIZATION, authorization)
            .body(Body::Bytes(serde_json::to_vec(&request)?))?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(Some("x-typesafe-request-id")))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        Json
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new(Jev::new(std::env::var("JEV_TOKEN")?), rig_reqwest::shared());
    let verdict = model
        .call(EvalRequest {
            state: json!({"door": "open"}),
            questions: vec!["Is the door open?".to_owned()],
        })
        .await?;
    println!("{} answered {:?}", verdict.model, verdict.answers);
    println!(
        "request {:?}, raw {}",
        verdict.provider_request_id, verdict.raw
    );
    Ok(())
}
