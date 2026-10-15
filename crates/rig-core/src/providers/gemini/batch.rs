//! Gemini's Batch API, through which a [`GenerateContent`] model submits
//! [batch completions](crate::completion::batch): `batchGenerateContent`
//! with the requests inline, read back as a long-running operation whose
//! response holds each request's reply in order.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::completion::batch::BatchItem;
//! use rig_core::providers::gemini::{self, Gemini};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.completion(gemini::completion::GEMINI_2_5_FLASH);
//! let item = BatchItem::new("hello", CompletionRequest::new("Say hello"));
//! let batch = model.submit_batch(vec![item]).await?;
//! # let _ = batch;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::GeminiConfig;
use super::completion::GenerateContent;
use crate::ProviderResponseError;
use crate::completion::batch::{
    BatchCounts, BatchOutcome, BatchState, BatchStatus, Batching, EncodedItem, json_body, path_id,
};
use crate::driver::{Model, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::operation::BatchCall;
use crate::providers::internal::wire::{
    classify_marker_keyed_frame, classify_or, classify_untyped_line,
};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

/// The prefix of a batch's resource name.
const NAME_PREFIX: &str = "batches/";

/// One Batch API call.
#[derive(Debug)]
pub enum BatchRequest {
    /// `POST /v1beta/models/<model>:batchGenerateContent` with `requests`
    /// inline, each `{request, metadata: {key}}`.
    Create {
        /// The model every request addresses.
        model: String,
        /// The batch's requests.
        requests: Vec<Value>,
    },
    /// `GET /v1beta/batches/<id>`.
    Get(String),
    /// `POST /v1beta/batches/<id>:cancel`.
    Cancel(String),
}

/// A Batch API reply.
#[derive(Debug)]
pub enum BatchReply {
    /// The batch's long-running operation, and the document it came in.
    Operation(BatchOperation, Value),
    /// An empty success, as a cancellation answers.
    Acknowledged,
}

impl BatchReply {
    fn operation(self) -> Result<(BatchOperation, Value), ProviderError> {
        match self {
            Self::Operation(operation, raw) => Ok((operation, raw)),
            Self::Acknowledged => Err(ProviderError::Response(
                "Gemini's Batch API answered with nothing, not a batch".to_owned(),
            )),
        }
    }
}

/// A batch's long-running operation. Its metadata is the batch, and its
/// response, once done, the inline replies.
#[derive(Debug, Deserialize)]
pub struct BatchOperation {
    /// The batch's resource name, `batches/…`.
    pub name: String,
    /// The batch: its `state`, `batchStats` and `output`.
    #[serde(default)]
    pub metadata: Value,
    /// Whether the batch has finished.
    #[serde(default)]
    pub done: bool,
    /// Why the batch failed, when it did.
    #[serde(default)]
    pub error: Option<Value>,
    /// The batch's output, once done.
    #[serde(default)]
    pub response: Option<Value>,
}

impl BatchOperation {
    fn state(&self, raw: Value) -> Result<BatchState, ProviderError> {
        let state = self.metadata.str("state").unwrap_or_default();
        let state = ["BATCH_STATE_", "JOB_STATE_"]
            .iter()
            .find_map(|prefix| state.strip_prefix(prefix))
            .unwrap_or(state);
        let status = match state {
            "PENDING" => BatchStatus::Pending,
            "RUNNING" => BatchStatus::Running,
            "SUCCEEDED" => BatchStatus::Completed,
            "FAILED" => BatchStatus::Failed,
            "CANCELLED" => BatchStatus::Cancelled,
            "EXPIRED" => BatchStatus::Expired,
            "" | "UNSPECIFIED" if self.error.is_some() => BatchStatus::Failed,
            "" | "UNSPECIFIED" if self.done => BatchStatus::Completed,
            "" | "UNSPECIFIED" => BatchStatus::Pending,
            other => {
                return Err(ProviderError::Response(format!(
                    "Gemini reported batch `{}` in an unknown state `{other}`",
                    self.name
                )));
            }
        };
        // Counts are int64, which Gemini writes as strings and omits at zero.
        let stats = self.metadata.get("batchStats").unwrap_or(&Value::Null);
        let total = stats.u64("requestCount").unwrap_or_default();
        let succeeded = stats.u64("successfulRequestCount").unwrap_or_default();
        let failed = stats.u64("failedRequestCount").unwrap_or_default();
        let pending = stats
            .u64("pendingRequestCount")
            .unwrap_or_else(|| total.saturating_sub(succeeded + failed));
        Ok(BatchState {
            id: self.name.clone(),
            status,
            counts: BatchCounts {
                pending,
                succeeded,
                failed,
            },
            raw,
        })
    }

    /// The inline replies, from the operation's response or the batch's
    /// output.
    fn inlined(&self) -> &[Value] {
        let path = "/inlinedResponses/inlinedResponses";
        self.response
            .as_ref()
            .and_then(|response| response.at(path))
            .or_else(|| self.metadata.at(&format!("/output{path}")))
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice)
    }
}

/// The outcome of one inline reply: `{metadata: {key}, response | error}`.
fn outcome(reply: &Value) -> Result<(String, BatchOutcome<Value>), ProviderError> {
    let Some(key) = reply.at("/metadata/key").and_then(Value::as_str) else {
        return Err(ProviderError::Response(
            "a Gemini batch reply names no request key".to_owned(),
        ));
    };
    let outcome = match (reply.get("response"), reply.get("error")) {
        (_, Some(error)) => BatchOutcome::Failed(ProviderError::ProviderResponse(
            ProviderResponseError::from_body(json!({ "error": error }).to_string()),
        )),
        (Some(response), None) => BatchOutcome::Succeeded(response.clone()),
        (None, None) => BatchOutcome::Failed(ProviderError::Response(format!(
            "the Gemini batch reply for `{key}` has neither a response nor an error"
        ))),
    };
    Ok((key.to_owned(), outcome))
}

/// The Batch API wire of a Gemini configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batches {
    /// The provider this wire speaks to.
    pub provider: GeminiConfig,
}

/// `/v1beta/batches/<id>` for `name`, a `batches/<id>` name or a bare id.
fn batch_path(name: &str) -> Result<String, EncodeError> {
    let id = name.strip_prefix(NAME_PREFIX).unwrap_or(name);
    Ok(format!("/v1beta/{NAME_PREFIX}{}", path_id(id)?))
}

impl Wire for Batches {
    type Op = BatchCall<BatchRequest, BatchReply>;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = BatchesDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    fn encode(&self, request: BatchRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = match request {
            BatchRequest::Create { model, requests } => {
                let body = json!({
                    "batch": {
                        "displayName": "rig",
                        "inputConfig": { "requests": { "requests": requests } },
                    }
                });
                let path = format!("/v1beta/models/{model}:batchGenerateContent");
                http::Request::post(self.provider.uri(&path))
                    .body(Body::Bytes(serde_json::to_vec(&body)?))?
            }
            BatchRequest::Get(name) => {
                http::Request::get(self.provider.uri(&batch_path(&name)?)).body(Body::empty())?
            }
            BatchRequest::Cancel(name) => {
                let path = format!("{}:cancel", batch_path(&name)?);
                http::Request::post(self.provider.uri(&path)).body(Body::empty())?
            }
        };
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        BatchesDecoder
    }
}

/// Decodes a batch operation or an empty acknowledgement.
pub struct BatchesDecoder;

impl<'id> Decoder<'id, BatchCall<BatchRequest, BatchReply>> for BatchesDecoder {
    type Event = BatchReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_or(&frame.as_str(), as_acknowledgement, |data| {
            classify_marker_keyed_frame::<BatchOperation>(data, &["name"]).map(|operation| {
                BatchReply::Operation(operation, serde_json::from_str(data).unwrap_or_default())
            })
        })
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, BatchCall<BatchRequest, BatchReply>>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }

    /// A reply with no body is an acknowledgement: the status answered.
    fn eof(
        &mut self,
        out: Out<'id, BatchCall<BatchRequest, BatchReply>>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(BatchReply::Acknowledged))
    }
}

/// An empty object, or a body that is not one.
fn as_acknowledgement(data: &str) -> WireEvent<BatchReply> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Acknowledgement {}

    classify_untyped_line::<Acknowledgement>(data.as_bytes()).map(|_| BatchReply::Acknowledged)
}

/// The create call for `items`, which must all address one model.
fn create_request(items: Vec<EncodedItem>) -> Result<BatchRequest, ProviderError> {
    let mut model: Option<String> = None;
    let mut requests = Vec::with_capacity(items.len());
    for EncodedItem { id, request } in items {
        let path = request.uri().path();
        let Some(addressed) = path
            .split_once("/models/")
            .and_then(|(_, rest)| rest.strip_suffix(":generateContent"))
        else {
            return Err(ProviderError::request(format!(
                "`{path}` is not a generateContent request"
            )));
        };
        match &model {
            None => model = Some(addressed.to_owned()),
            Some(model) if model == addressed => {}
            Some(_) => {
                return Err(ProviderError::request(
                    "a Gemini batch sends every request to one model",
                ));
            }
        }
        requests.push(json!({ "request": json_body(&request)?, "metadata": { "key": id } }));
    }
    Ok(BatchRequest::Create {
        model: model.unwrap_or_default(),
        requests,
    })
}

impl Batching for GenerateContent {
    type Batches = Batches;

    const PRICE: f64 = 0.5;

    fn batches(&self) -> Result<Batches, ProviderError> {
        Ok(Batches {
            provider: self.provider.clone(),
        })
    }

    fn submit<T: Transport<Batches>>(
        batches: &Model<Batches, T>,
        items: Vec<EncodedItem>,
    ) -> impl Future<Output = Result<String, ProviderError>> + WasmCompatSend {
        let request = create_request(items);
        async move {
            let (operation, _) = batches.call(request?).await?.operation()?;
            Ok(operation.name)
        }
    }

    fn status<T: Transport<Batches>>(
        batches: &Model<Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
        let reply = batches.call(BatchRequest::Get(id.to_owned()));
        async move {
            let (operation, raw) = reply.await?.operation()?;
            operation.state(raw)
        }
    }

    fn results<T: Transport<Batches>>(
        batches: &Model<Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<Vec<(String, BatchOutcome<Value>)>, ProviderError>> + WasmCompatSend
    {
        let reply = batches.call(BatchRequest::Get(id.to_owned()));
        let id = id.to_owned();
        async move {
            let (operation, raw) = reply.await?.operation()?;
            let state = operation.state(raw)?;
            if !state.status.is_finished() {
                return Err(ProviderError::Response(format!(
                    "Gemini batch `{id}` is {:?}: its results are not ready",
                    state.status
                )));
            }
            operation.inlined().iter().map(outcome).collect()
        }
    }

    /// Gemini answers a cancellation with nothing, so the batch is read
    /// back after it.
    fn cancel<T: Transport<Batches>>(
        batches: &Model<Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
        let cancel = batches.call(BatchRequest::Cancel(id.to_owned()));
        let get = batches.call(BatchRequest::Get(id.to_owned()));
        async move {
            cancel.await?;
            let (operation, raw) = get.await?.operation()?;
            operation.state(raw)
        }
    }
}

#[cfg(test)]
mod tests;
