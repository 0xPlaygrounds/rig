//! Anthropic's Message Batches API, through which a
//! [`Messages`](super::Messages) model submits
//! [batch completions](crate::completion::batch): `POST
//! /v1/messages/batches`, read back by id, its results a JSON Lines file.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::completion::batch::BatchItem;
//! use rig_core::providers::anthropic::{self, Anthropic};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Anthropic::from_env()?.completion(anthropic::CLAUDE_HAIKU_4_5);
//! let item = BatchItem::new("hello", CompletionRequest::new("Say hello"));
//! let batch = model.submit_batch(vec![item]).await?;
//! # let _ = batch;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::wire::AnthropicConfig;
use crate::ProviderResponseError;
use crate::completion::batch::{
    BatchCounts, BatchOutcome, BatchState, BatchStatus, EncodedItem, ItemReply, json_body, path_id,
};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::operation::BatchCall;
use crate::providers::internal::wire::{
    classify_json_lines, classify_marker_keyed_frame, classify_or,
};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

/// The Message Batches collection path.
const BATCHES_PATH: &str = "/v1/messages/batches";

/// The header naming a request's beta features.
const BETA_HEADER: &str = "anthropic-beta";

/// One Message Batches call.
#[derive(Debug)]
pub enum MessageBatchRequest {
    /// `POST /v1/messages/batches` with `requests`, each `{custom_id,
    /// params}`, under the `anthropic-beta` flags `betas`, which apply to
    /// the whole batch.
    Create {
        /// The batch's requests.
        requests: Vec<Value>,
        /// The beta flags the requests need.
        betas: Vec<String>,
    },
    /// `GET /v1/messages/batches/<id>`.
    Get(String),
    /// `GET /v1/messages/batches/<id>/results`.
    Results(String),
    /// `POST /v1/messages/batches/<id>/cancel`.
    Cancel(String),
}

/// A Message Batches reply.
#[derive(Debug)]
pub enum MessageBatchReply {
    /// A message batch object, and the document it came in.
    Batch(MessageBatch, Value),
    /// The lines of a results file, in the order Anthropic wrote them.
    Results(Vec<ResultLine>),
}

impl MessageBatchReply {
    pub(crate) fn batch(self) -> Result<BatchState, ProviderError> {
        match self {
            Self::Batch(batch, raw) => Ok(batch.state(raw)),
            Self::Results(_) => Err(ProviderError::Response(
                "Anthropic answered with a results file, not a message batch".to_owned(),
            )),
        }
    }
}

/// A message batch, as Anthropic reports it.
#[derive(Debug, Deserialize)]
pub struct MessageBatch {
    /// The batch id, `msgbatch_…`.
    pub id: String,
    /// `in_progress`, `canceling` or `ended`.
    pub processing_status: ProcessingStatus,
    /// The requests in each state.
    pub request_counts: RequestCounts,
    /// When cancellation was asked for, if it was.
    #[serde(default)]
    pub cancel_initiated_at: Option<String>,
}

/// Where a message batch is in its life.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingStatus {
    /// Running.
    InProgress,
    /// Being cancelled.
    Canceling,
    /// Every request has a result.
    Ended,
}

/// A message batch's requests in each state. Every count but `processing`
/// stays zero until the batch ends.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RequestCounts {
    /// Requests still running.
    pub processing: u64,
    /// Requests answered with a message.
    pub succeeded: u64,
    /// Requests that failed.
    pub errored: u64,
    /// Requests cancelled before they ran.
    pub canceled: u64,
    /// Requests that expired before they ran.
    pub expired: u64,
}

impl MessageBatch {
    fn state(self, raw: Value) -> BatchState {
        let counts = self.request_counts;
        let status = match self.processing_status {
            ProcessingStatus::InProgress => BatchStatus::Running,
            ProcessingStatus::Canceling => BatchStatus::Cancelling,
            ProcessingStatus::Ended if self.cancel_initiated_at.is_some() => BatchStatus::Cancelled,
            ProcessingStatus::Ended if counts.expired > 0 => BatchStatus::Expired,
            ProcessingStatus::Ended => BatchStatus::Completed,
        };
        BatchState {
            id: self.id,
            status,
            counts: BatchCounts {
                pending: counts.processing,
                succeeded: counts.succeeded,
                failed: counts.errored + counts.canceled + counts.expired,
            },
            raw,
        }
    }
}

/// One line of a results file.
#[derive(Debug, Deserialize)]
pub struct ResultLine {
    /// The request's id in the batch.
    pub custom_id: String,
    /// What became of it.
    pub result: ItemResult,
}

/// What became of one request.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ItemResult {
    /// The message a live call would have answered with.
    Succeeded {
        /// The Messages response.
        message: Value,
    },
    /// Anthropic's error document for the request.
    Errored {
        /// `{"type": "error", "error": {...}, "request_id": ...}`.
        error: Value,
    },
    /// The batch was cancelled before the request ran.
    Canceled,
    /// The batch expired before the request ran.
    Expired,
}

impl From<ItemResult> for BatchOutcome<ItemReply> {
    fn from(result: ItemResult) -> Self {
        match result {
            ItemResult::Succeeded { message } => Self::Succeeded(ItemReply::new(message)),
            ItemResult::Errored { error } => {
                let request_id = error.str("request_id").map(str::to_owned);
                let reply = ProviderResponseError::from_body(error.to_string())
                    .with_provider_request_id(request_id);
                Self::Failed(ProviderError::ProviderResponse(reply))
            }
            ItemResult::Canceled => Self::Canceled,
            ItemResult::Expired => Self::Expired,
        }
    }
}

/// The Message Batches wire of a Messages-format provider.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessageBatches {
    /// The provider this wire speaks to.
    pub provider: AnthropicConfig,
}

impl Wire for MessageBatches {
    type Op = BatchCall<MessageBatchRequest, MessageBatchReply>;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = MessageBatchesDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, request: MessageBatchRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let base = &self.provider.base_url;
        let request = match request {
            MessageBatchRequest::Create { requests, betas } => {
                let betas: Vec<&str> = betas.iter().map(String::as_str).collect();
                let body = serde_json::to_vec(&json!({ "requests": requests }))?;
                self.provider
                    .headers_with(http::Request::post(format!("{base}{BATCHES_PATH}")), &betas)
                    .body(Body::Bytes(body))?
            }
            MessageBatchRequest::Get(id) => self
                .provider
                .headers(http::Request::get(format!(
                    "{base}{BATCHES_PATH}/{}",
                    path_id(&id)?
                )))
                .body(Body::empty())?,
            MessageBatchRequest::Results(id) => self
                .provider
                .headers(http::Request::get(format!(
                    "{base}{BATCHES_PATH}/{}/results",
                    path_id(&id)?
                )))
                .body(Body::empty())?,
            MessageBatchRequest::Cancel(id) => self
                .provider
                .headers(http::Request::post(format!(
                    "{base}{BATCHES_PATH}/{}/cancel",
                    path_id(&id)?
                )))
                .body(Body::empty())?,
        };
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        MessageBatchesDecoder
    }
}

/// Decodes a message batch or a results file.
pub struct MessageBatchesDecoder;

impl<'id> Decoder<'id, BatchCall<MessageBatchRequest, MessageBatchReply>>
    for MessageBatchesDecoder
{
    type Event = MessageBatchReply;

    /// A results file is JSON Lines, which a batch object never is: a
    /// pretty-printed object's lines do not parse, and a compact one is not
    /// a result line.
    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_or(
            &frame.as_str(),
            |data| classify_json_lines(data).map(MessageBatchReply::Results),
            |data| {
                classify_marker_keyed_frame::<MessageBatch>(data, &["processing_status"]).map(
                    |batch| {
                        MessageBatchReply::Batch(
                            batch,
                            serde_json::from_str(data).unwrap_or_default(),
                        )
                    },
                )
            },
        )
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, BatchCall<MessageBatchRequest, MessageBatchReply>>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }

    /// An empty results file has no lines.
    fn eof(
        &mut self,
        out: Out<'id, BatchCall<MessageBatchRequest, MessageBatchReply>>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(MessageBatchReply::Results(Vec::new())))
    }
}

/// The create call for `items`: each item's Messages body as its `params`,
/// under every beta flag an item's request carried.
pub(crate) fn create_request(
    items: Vec<EncodedItem>,
) -> Result<MessageBatchRequest, ProviderError> {
    let mut requests = Vec::with_capacity(items.len());
    let mut betas: Vec<String> = Vec::new();
    for EncodedItem { id, request } in items {
        let flags = request
            .headers()
            .get_all(BETA_HEADER)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .flat_map(|value| value.split(','))
            .map(str::trim);
        for flag in flags {
            if !flag.is_empty() && !betas.iter().any(|known| known == flag) {
                betas.push(flag.to_owned());
            }
        }
        requests.push(json!({ "custom_id": id, "params": json_body(&request)? }));
    }
    Ok(MessageBatchRequest::Create { requests, betas })
}

#[cfg(test)]
mod tests;
