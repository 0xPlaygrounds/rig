//! OpenAI's Batch API, through which a Chat Completions or Responses model
//! submits [batch completions](crate::completion::batch): the requests are
//! uploaded as a JSON Lines file, a batch is created over it, and its
//! output and error files hold the results.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::completion::batch::BatchItem;
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.responses(openai::GPT_5_2);
//! let item = BatchItem::new("hello", CompletionRequest::new("Say hello"));
//! let batch = model.submit_batch(vec![item]).await?;
//! # let _ = batch;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::responses_api::wire::Responses;
use super::wire::{Chat, OpenAIConfig, OpenAiWire};
use crate::ProviderResponseError;
use crate::completion::batch::{
    BatchCounts, BatchOutcome, BatchPrice, BatchState, BatchStatus, Batching, EncodedItem,
    ItemReply, json_body, path_id,
};
use crate::driver::{Model, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::http_client::MultipartForm;
use crate::http_client::multipart::Part;
use crate::operation::BatchCall;
use crate::providers::internal::wire::{classify_json_lines, classify_or, classify_tagged_frame};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

/// The endpoints a batch line may address, by the path suffix of the live
/// request that becomes the line.
const ENDPOINTS: [(&str, &str); 2] = [
    ("/chat/completions", "/v1/chat/completions"),
    ("/responses", "/v1/responses"),
];

/// One Batch API call.
#[derive(Debug)]
pub enum BatchRequest {
    /// `POST /files` with `purpose=batch`: the JSON Lines input file.
    Upload(Vec<u8>),
    /// `POST /batches` over an uploaded input file, for `endpoint`.
    Create {
        /// The uploaded file's id.
        input_file_id: String,
        /// The endpoint every line addresses, `/v1/responses` or
        /// `/v1/chat/completions`.
        endpoint: String,
    },
    /// `GET /batches/<id>`.
    Get(String),
    /// `GET /files/<id>/content`: an output or error file.
    FileContent(String),
    /// `POST /batches/<id>/cancel`.
    Cancel(String),
}

/// A Batch API reply.
#[derive(Debug)]
pub enum BatchReply {
    /// A batch or a file object.
    Object(Object, Value),
    /// The lines of an output or error file.
    Lines(Vec<OutputLine>),
}

impl BatchReply {
    fn batch(self) -> Result<(BatchObject, Value), ProviderError> {
        match self {
            Self::Object(Object::Batch(batch), raw) => Ok((batch, raw)),
            _ => Err(mismatch("a batch")),
        }
    }
}

fn mismatch(wanted: &str) -> ProviderError {
    ProviderError::Response(format!("OpenAI's Batch API did not answer with {wanted}"))
}

/// An object the Batch API answers with, by its `object` tag.
#[derive(Debug, Deserialize)]
#[serde(tag = "object", rename_all = "snake_case")]
pub enum Object {
    /// A batch.
    Batch(BatchObject),
    /// An uploaded file.
    File {
        /// The file's id.
        id: String,
    },
}

/// A batch, as OpenAI reports it.
#[derive(Debug, Deserialize)]
pub struct BatchObject {
    /// The batch id, `batch_…`.
    pub id: String,
    /// Where the batch is in its life.
    pub status: Status,
    /// The file holding a line per request that succeeded.
    #[serde(default)]
    pub output_file_id: Option<String>,
    /// The file holding a line per request that failed.
    #[serde(default)]
    pub error_file_id: Option<String>,
    /// The requests in each state.
    #[serde(default)]
    pub request_counts: Option<RequestCounts>,
}

/// Where an OpenAI batch is in its life.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The input file is being validated.
    Validating,
    /// Validation failed.
    Failed,
    /// Running.
    InProgress,
    /// Writing the results.
    Finalizing,
    /// Every request has a result.
    Completed,
    /// Expired before every request ran.
    Expired,
    /// Being cancelled.
    Cancelling,
    /// Cancelled.
    Cancelled,
}

/// A batch's requests in each state.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RequestCounts {
    /// Every request.
    pub total: u64,
    /// Requests answered.
    pub completed: u64,
    /// Requests that failed.
    pub failed: u64,
}

impl BatchObject {
    fn state(&self, raw: Value) -> BatchState {
        let status = match self.status {
            Status::Validating => BatchStatus::Pending,
            Status::InProgress | Status::Finalizing => BatchStatus::Running,
            Status::Cancelling => BatchStatus::Cancelling,
            Status::Completed => BatchStatus::Completed,
            Status::Cancelled => BatchStatus::Cancelled,
            Status::Expired => BatchStatus::Expired,
            Status::Failed => BatchStatus::Failed,
        };
        let counts = self.request_counts.unwrap_or_default();
        BatchState {
            id: self.id.clone(),
            status,
            counts: BatchCounts {
                pending: counts
                    .total
                    .saturating_sub(counts.completed + counts.failed),
                succeeded: counts.completed,
                failed: counts.failed,
            },
            raw,
        }
    }
}

/// One line of an output or error file.
#[derive(Debug, Deserialize)]
pub struct OutputLine {
    /// The request's id in the batch.
    pub custom_id: String,
    /// The endpoint's reply, when the request was sent.
    #[serde(default)]
    pub response: Option<LineResponse>,
    /// Why the request has no reply, `{code, message}`.
    #[serde(default)]
    pub error: Option<Value>,
}

/// The endpoint's reply to one request.
#[derive(Debug, Deserialize)]
pub struct LineResponse {
    /// The HTTP status the endpoint answered with.
    pub status_code: u16,
    /// The endpoint's request id.
    #[serde(default)]
    pub request_id: Option<String>,
    /// The reply body a live call would have received.
    pub body: Value,
}

impl From<OutputLine> for BatchOutcome<ItemReply> {
    fn from(line: OutputLine) -> Self {
        if let Some(error) = line.error {
            return match error.get("code").and_then(Value::as_str) {
                Some("batch_expired") => Self::Expired,
                Some("batch_cancelled") => Self::Canceled,
                _ => Self::Failed(ProviderError::ProviderResponse(
                    ProviderResponseError::from_body(json!({ "error": error }).to_string()),
                )),
            };
        }
        let Some(response) = line.response else {
            return Self::Failed(ProviderError::Response(
                "the batch line has neither a response nor an error".to_owned(),
            ));
        };
        match http::StatusCode::from_u16(response.status_code) {
            Ok(status) if status.is_success() => Self::Succeeded(ItemReply {
                body: response.body,
                request_id: response.request_id,
            }),
            status => {
                let body = response.body.to_string();
                let reply = match status {
                    Ok(status) => ProviderResponseError::new(status, body),
                    Err(_) => ProviderResponseError::from_body(body),
                };
                Self::Failed(ProviderError::ProviderResponse(
                    reply.with_provider_request_id(response.request_id),
                ))
            }
        }
    }
}

/// The Batch API wire of an OpenAI configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batches {
    /// The provider this wire speaks to.
    pub provider: OpenAIConfig,
}

impl Batches {
    fn request(&self, method: http::Method, path: &str) -> http::request::Builder {
        self.provider.headers(
            http::Request::builder()
                .method(method)
                .uri(self.provider.uri(path, None)),
        )
    }
}

impl Wire for Batches {
    type Op = BatchCall<BatchRequest, BatchReply>;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = BatchesDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, request: BatchRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        use http::Method;
        let request = match request {
            BatchRequest::Upload(lines) => {
                let form = MultipartForm::new()
                    .text("purpose", "batch")
                    .part(Part::bytes("file", lines).filename("batch.jsonl"));
                self.request(Method::POST, "/files")
                    .body(Body::Multipart(form))?
            }
            BatchRequest::Create {
                input_file_id,
                endpoint,
            } => {
                let body = json!({
                    "input_file_id": input_file_id,
                    "endpoint": endpoint,
                    "completion_window": "24h",
                });
                self.request(Method::POST, "/batches")
                    .body(Body::Bytes(serde_json::to_vec(&body)?))?
            }
            BatchRequest::Get(id) => self
                .request(Method::GET, &format!("/batches/{}", path_id(&id)?))
                .body(Body::empty())?,
            BatchRequest::FileContent(id) => self
                .request(Method::GET, &format!("/files/{}/content", path_id(&id)?))
                .body(Body::empty())?,
            BatchRequest::Cancel(id) => self
                .request(Method::POST, &format!("/batches/{}/cancel", path_id(&id)?))
                .body(Body::empty())?,
        };
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        BatchesDecoder
    }
}

/// Decodes a batch, a file object, or an output or error file.
pub struct BatchesDecoder;

impl<'id> Decoder<'id, BatchCall<BatchRequest, BatchReply>> for BatchesDecoder {
    type Event = BatchReply;

    /// An output file is JSON Lines, which an object never is: a
    /// pretty-printed object's lines do not parse, and a compact one is not
    /// an output line.
    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_or(
            &frame.as_str(),
            |data| classify_json_lines(data).map(BatchReply::Lines),
            |data| {
                classify_tagged_frame::<Object>(data, "object", |tag| {
                    matches!(tag, "batch" | "file")
                })
                .map(|object| {
                    BatchReply::Object(object, serde_json::from_str(data).unwrap_or_default())
                })
            },
        )
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, BatchCall<BatchRequest, BatchReply>>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }

    /// An empty output or error file has no lines.
    fn eof(
        &mut self,
        out: Out<'id, BatchCall<BatchRequest, BatchReply>>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(BatchReply::Lines(Vec::new())))
    }
}

/// The batch wire for `provider`, which only OpenAI itself serves.
fn batches(provider: &OpenAIConfig) -> Result<Batches, ProviderError> {
    if provider.dialect.name != super::PROVIDER_NAME {
        return Err(ProviderError::request(format!(
            "{} has no OpenAI Batch API",
            provider.dialect.name
        )));
    }
    Ok(Batches {
        provider: provider.clone(),
    })
}

/// The input file for `items` and the endpoint its lines address. One
/// batch addresses one endpoint and one model.
fn input_file(items: Vec<EncodedItem>) -> Result<(Vec<u8>, String), ProviderError> {
    let mut lines = Vec::new();
    let mut batch: Option<(String, Option<String>)> = None;
    for EncodedItem { id, request } in items {
        let path = request.uri().path();
        let Some((_, endpoint)) = ENDPOINTS.iter().find(|(suffix, _)| path.ends_with(suffix))
        else {
            return Err(ProviderError::request(format!(
                "`{path}` is not an endpoint OpenAI's Batch API serves"
            )));
        };
        let body = json_body(&request)?;
        let model = body.get("model").and_then(Value::as_str).map(str::to_owned);
        match &batch {
            None => batch = Some(((*endpoint).to_owned(), model)),
            Some(first) if *first == ((*endpoint).to_owned(), model) => {}
            Some(_) => {
                return Err(ProviderError::request(
                    "an OpenAI batch sends every request to one endpoint and one model",
                ));
            }
        }
        let line = json!({ "custom_id": id, "method": "POST", "url": endpoint, "body": body });
        serde_json::to_writer(&mut lines, &line)?;
        lines.push(b'\n');
    }
    let endpoint = batch.map(|(endpoint, _)| endpoint).unwrap_or_default();
    Ok((lines, endpoint))
}

async fn submit<T: Transport<Batches>>(
    batches: &Model<Batches, T>,
    items: Vec<EncodedItem>,
) -> Result<String, ProviderError> {
    let (lines, endpoint) = input_file(items)?;
    let input_file_id = match batches.call(BatchRequest::Upload(lines)).await? {
        BatchReply::Object(Object::File { id }, _) => id,
        _ => return Err(mismatch("a file")),
    };
    let request = BatchRequest::Create {
        input_file_id,
        endpoint,
    };
    let (batch, _) = batches.call(request).await?.batch()?;
    Ok(batch.id)
}

async fn status<T: Transport<Batches>>(
    batches: &Model<Batches, T>,
    id: &str,
) -> Result<BatchState, ProviderError> {
    let (batch, raw) = batches
        .call(BatchRequest::Get(id.to_owned()))
        .await?
        .batch()?;
    Ok(batch.state(raw))
}

/// The lines of the batch's output and error files.
async fn results<T: Transport<Batches>>(
    batches: &Model<Batches, T>,
    id: &str,
) -> Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError> {
    let (batch, raw) = batches
        .call(BatchRequest::Get(id.to_owned()))
        .await?
        .batch()?;
    let state = batch.state(raw);
    if !state.status.is_finished() {
        return Err(ProviderError::Response(format!(
            "OpenAI batch `{id}` is {:?}: its results are not ready",
            batch.status
        )));
    }
    let mut outcomes = Vec::new();
    for file in [batch.output_file_id, batch.error_file_id]
        .into_iter()
        .flatten()
    {
        match batches.call(BatchRequest::FileContent(file)).await? {
            BatchReply::Lines(lines) => outcomes.extend(
                lines
                    .into_iter()
                    .map(|line| (line.custom_id.clone(), line.into())),
            ),
            BatchReply::Object(..) => return Err(mismatch("a results file")),
        }
    }
    Ok(outcomes)
}

async fn cancel<T: Transport<Batches>>(
    batches: &Model<Batches, T>,
    id: &str,
) -> Result<BatchState, ProviderError> {
    let (batch, raw) = batches
        .call(BatchRequest::Cancel(id.to_owned()))
        .await?
        .batch()?;
    Ok(batch.state(raw))
}

/// The configuration an OpenAI completion wire sends with.
trait Configured {
    fn config(&self) -> &OpenAIConfig;
}

impl Configured for Chat {
    fn config(&self) -> &OpenAIConfig {
        &self.provider
    }
}

impl Configured for Responses {
    fn config(&self) -> &OpenAIConfig {
        &self.provider
    }
}

impl Configured for OpenAiWire {
    fn config(&self) -> &OpenAIConfig {
        match self {
            Self::Chat(wire) => &wire.provider,
            Self::Responses(wire) => &wire.provider,
        }
    }
}

/// Implement [`Batching`] for an OpenAI completion wire.
macro_rules! openai_batching {
    ($wire:ty) => {
        impl Batching for $wire {
            type Batches = Batches;

            const PRICE: BatchPrice = BatchPrice::HALF;

            fn batches(&self) -> Result<Batches, ProviderError> {
                batches(self.config())
            }

            fn submit<T: Transport<Batches>>(
                batches: &Model<Batches, T>,
                items: Vec<EncodedItem>,
            ) -> impl Future<Output = Result<String, ProviderError>> + WasmCompatSend {
                submit(batches, items)
            }

            fn status<T: Transport<Batches>>(
                batches: &Model<Batches, T>,
                id: &str,
            ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
                status(batches, id)
            }

            fn results<T: Transport<Batches>>(
                batches: &Model<Batches, T>,
                id: &str,
            ) -> impl Future<Output = Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError>>
            + WasmCompatSend {
                results(batches, id)
            }

            fn cancel<T: Transport<Batches>>(
                batches: &Model<Batches, T>,
                id: &str,
            ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
                cancel(batches, id)
            }
        }
    };
}

openai_batching!(Chat);
openai_batching!(Responses);
openai_batching!(OpenAiWire);

#[cfg(test)]
mod tests;
