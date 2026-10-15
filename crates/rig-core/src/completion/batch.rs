//! Batch completions: many requests submitted together through a provider's
//! batch API, answered asynchronously within a day at a lower price. Each
//! request is prepared and encoded exactly as a live unary call to the same
//! [`Model`] would send it, and each reply is decoded as that call's reply
//! would be.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::completion::batch::{BatchItem, BatchOutcome};
//! use rig_core::providers::anthropic::{self, Anthropic};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Anthropic::from_env()?.completion(anthropic::CLAUDE_HAIKU_4_5);
//! let batch = model
//!     .submit_batch(vec![
//!         BatchItem::new("capital", CompletionRequest::new("Capital of France?")),
//!         BatchItem::new("sky", CompletionRequest::new("Color of the sky?")),
//!     ])
//!     .await?;
//! if model.batch_status(&batch.id).await?.status.is_finished() {
//!     for (id, outcome) in model.batch_results(&batch).await? {
//!         if let BatchOutcome::Succeeded(response) = outcome {
//!             println!("{id}: {:?}", response.choice);
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, HashSet};
use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::driver::{Model, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::operation::Completion;
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Body, Encoded, Mode, Operation, Wire, WireFrame, decode_reply};

/// The longest item id: the strictest provider's limit, so an id is valid
/// for every provider.
const MAX_ID_LEN: usize = 64;

/// One request of a batch, under the caller's id.
#[derive(Clone, Debug)]
pub struct BatchItem {
    /// The caller's id for the request, which its result is keyed by: 1 to
    /// 64 ASCII letters, digits, `_` or `-`, unique within its batch.
    pub id: String,
    /// The request, sent as a live unary call would send it.
    pub request: CompletionRequest,
}

impl BatchItem {
    /// `request` under `id`.
    pub fn new(id: impl Into<String>, request: impl Into<CompletionRequest>) -> Self {
        Self {
            id: id.into(),
            request: request.into(),
        }
    }
}

/// A submitted batch: the provider's id for it, and each request as it was
/// sent, keyed by item id, which its reply is decoded against. It
/// serializes, so another process can fetch the results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    /// The provider's id for the batch.
    pub id: String,
    /// Each request as prepared for the model and sent, by item id.
    pub requests: BTreeMap<String, CompletionRequest>,
}

/// Where a batch is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStatus {
    /// Accepted, waiting to run or being validated.
    Pending,
    /// Running.
    Running,
    /// Being cancelled. Requests already running may still finish.
    Cancelling,
    /// Every request has a result.
    Completed,
    /// Cancelled. The requests that ran have results; the others are
    /// [`BatchOutcome::Canceled`], or absent where the provider reports
    /// nothing for them.
    Cancelled,
    /// Expired before every request ran. The requests that ran have
    /// results; the others are [`BatchOutcome::Expired`], or absent where
    /// the provider reports nothing for them.
    Expired,
    /// The batch failed as a whole, such as on input the provider rejected,
    /// and has no results.
    Failed,
}

impl BatchStatus {
    /// Whether the batch has stopped changing: its results, if any, are
    /// ready.
    pub fn is_finished(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Expired | Self::Failed
        )
    }
}

/// How many of a batch's requests have finished, as the provider counts
/// them. A provider that counts only once the batch ends reports every
/// request as pending until then.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchCounts {
    /// Requests not yet finished.
    pub pending: u64,
    /// Requests answered with a reply.
    pub succeeded: u64,
    /// Requests that failed, were cancelled or expired.
    pub failed: u64,
}

/// A batch's progress, as the provider reports it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchState {
    /// The provider's id for the batch.
    pub id: String,
    /// Where the batch is in its life.
    pub status: BatchStatus,
    /// How many of its requests have finished.
    pub counts: BatchCounts,
    /// The provider's batch object, as it sent it.
    pub raw: Value,
}

/// What became of one request of a batch.
#[derive(Debug)]
pub enum BatchOutcome<R = CompletionResponse> {
    /// The model's reply.
    Succeeded(R),
    /// The provider's error for the request, or a reply that did not decode.
    Failed(ProviderError),
    /// The batch was cancelled before the request ran.
    Canceled,
    /// The batch expired before the request ran.
    Expired,
}

/// What batched tokens cost, as a share of their price in a request sent
/// alone, by kind of token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchPrice {
    /// Uncached input tokens.
    pub input: f64,
    /// Output tokens.
    pub output: f64,
    /// Input tokens read from a cache.
    pub cache_read: f64,
    /// Input tokens written to a cache.
    pub cache_write: f64,
}

impl BatchPrice {
    /// Every kind of token at half price.
    pub const HALF: Self = Self {
        input: 0.5,
        output: 0.5,
        cache_read: 0.5,
        cache_write: 0.5,
    };
}

/// A provider's reply to one request of a batch, before it is decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemReply {
    /// The response document a live unary call would receive.
    pub body: Value,
    /// The provider's id for the request, when it reports one.
    pub request_id: Option<String>,
}

impl ItemReply {
    /// `body`, with no request id.
    pub fn new(body: Value) -> Self {
        Self {
            body,
            request_id: None,
        }
    }
}

/// One item of a batch, encoded as a live unary call would send it: what a
/// [`Batching`] wire builds its batch from.
#[derive(Debug)]
pub struct EncodedItem {
    /// The caller's id for the item.
    pub id: String,
    /// The request a live unary call would send.
    pub request: http::Request<Body>,
}

/// A completion wire whose provider has a batch API. The provider-neutral
/// calls are the batch methods of a [`Model`] over the wire; an
/// implementation translates them to its provider's batch endpoints.
pub trait Batching: Wire<Op = Completion, Payload = Encoded, Frame = WireFrame> {
    /// The wire of the provider's batch API.
    type Batches: Wire<Payload = Encoded, Frame = WireFrame>;

    /// What a batched request's tokens cost, as a share of the same
    /// request's sent alone. Batch replies are priced by it.
    const PRICE: BatchPrice;

    /// The batch wire for this wire's provider.
    ///
    /// # Errors
    ///
    /// When the provider this wire is configured for has no batch API.
    fn batches(&self) -> Result<Self::Batches, ProviderError>;

    /// Create a batch of `items` and answer its id.
    fn submit<T: Transport<Self::Batches>>(
        batches: &Model<Self::Batches, T>,
        items: Vec<EncodedItem>,
    ) -> impl Future<Output = Result<String, ProviderError>> + WasmCompatSend;

    /// The batch `id`'s progress.
    fn status<T: Transport<Self::Batches>>(
        batches: &Model<Self::Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend;

    /// Each request's outcome in the finished batch `id`, by item id, its
    /// reply not yet decoded.
    fn results<T: Transport<Self::Batches>>(
        batches: &Model<Self::Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError>>
    + WasmCompatSend;

    /// Ask the provider to cancel the batch `id`, and answer its progress.
    fn cancel<T: Transport<Self::Batches>>(
        batches: &Model<Self::Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend;
}

impl<W, T> Model<W, T>
where
    W: Batching,
    T: Transport<W::Batches>,
{
    /// Submit `items` as one batch. Each request is prepared and encoded as
    /// [`Model::call`] would send it, and the answer keeps each as sent.
    ///
    /// # Errors
    ///
    /// When there are no items, an id is invalid or used twice, a request
    /// cannot be encoded, the provider has no batch API, or the provider
    /// rejects the batch.
    pub async fn submit_batch(&self, items: Vec<BatchItem>) -> Result<Batch, ProviderError> {
        let batches = self.batches()?;
        let (requests, encoded) = self.encode_batch(items)?;
        let id = W::submit(&batches, encoded).await?;
        Ok(Batch { id, requests })
    }

    /// The progress of the batch `id`.
    ///
    /// # Errors
    ///
    /// When the provider has no batch API or the request fails.
    pub async fn batch_status(&self, id: &str) -> Result<BatchState, ProviderError> {
        W::status(&self.batches()?, id).await
    }

    /// Each request's outcome in the finished `batch`, by item id. A reply
    /// is decoded against its request as sent, as a live call's would be,
    /// and priced at the batch price. A request the provider reports
    /// nothing for is absent.
    ///
    /// # Errors
    ///
    /// When the batch has not finished, the provider has no batch API, or
    /// the request fails.
    pub async fn batch_results(
        &self,
        batch: &Batch,
    ) -> Result<BTreeMap<String, BatchOutcome>, ProviderError> {
        let outcomes = W::results(&self.batches()?, &batch.id).await?;
        Ok(outcomes
            .into_iter()
            .map(|(id, outcome)| {
                let outcome = self.decode_outcome(batch.requests.get(&id), &id, outcome);
                (id, outcome)
            })
            .collect())
    }

    /// Ask the provider to cancel the batch `id`, and answer its progress.
    /// Requests already running may still finish.
    ///
    /// # Errors
    ///
    /// When the provider has no batch API or the request fails.
    pub async fn cancel_batch(&self, id: &str) -> Result<BatchState, ProviderError> {
        W::cancel(&self.batches()?, id).await
    }

    /// This model's batch wire on its transport.
    fn batches(&self) -> Result<Model<W::Batches, T>, ProviderError> {
        Ok(Model::new(self.wire.batches()?, self.transport.clone()))
    }

    /// Each item prepared and encoded as a live unary call would send it.
    fn encode_batch(
        &self,
        items: Vec<BatchItem>,
    ) -> Result<(BTreeMap<String, CompletionRequest>, Vec<EncodedItem>), ProviderError> {
        check_ids(&items)?;
        let describe = self.wire.describe();
        let mut requests = BTreeMap::new();
        let mut encoded = Vec::with_capacity(items.len());
        for BatchItem { id, request } in items {
            let request = <Completion as Operation>::prepare(request, &describe)?;
            let payload = self.wire.encode(request.clone(), Mode::Unary)?;
            encoded.push(EncodedItem {
                id: id.clone(),
                request: payload.request,
            });
            requests.insert(id, request);
        }
        Ok((requests, encoded))
    }

    /// `outcome` with its reply decoded against `request` and priced at the
    /// batch price.
    fn decode_outcome(
        &self,
        request: Option<&CompletionRequest>,
        id: &str,
        outcome: BatchOutcome<ItemReply>,
    ) -> BatchOutcome {
        match outcome {
            BatchOutcome::Succeeded(reply) => {
                let Some(request) = request else {
                    return BatchOutcome::Failed(ProviderError::Response(format!(
                        "the batch answered item `{id}`, which it was not sent"
                    )));
                };
                let frame = WireFrame::Text(reply.body.to_string());
                match decode_reply(&self.wire, request, Mode::Unary, [frame]) {
                    Ok(mut response) => {
                        response.provider_request_id = reply.request_id;
                        response.usage.cost = response.usage.cost.map(|cost| cost.at(W::PRICE));
                        BatchOutcome::Succeeded(response)
                    }
                    Err(error) => BatchOutcome::Failed(error),
                }
            }
            BatchOutcome::Failed(error) => BatchOutcome::Failed(error),
            BatchOutcome::Canceled => BatchOutcome::Canceled,
            BatchOutcome::Expired => BatchOutcome::Expired,
        }
    }
}

/// Refuse an empty batch, and an item id that is invalid or used twice.
fn check_ids(items: &[BatchItem]) -> Result<(), ProviderError> {
    if items.is_empty() {
        return Err(ProviderError::request("a batch needs at least one item"));
    }
    let mut seen = HashSet::new();
    for BatchItem { id, .. } in items {
        if !is_id(id) || id.len() > MAX_ID_LEN {
            return Err(ProviderError::request(format!(
                "batch item id `{id}` must be 1 to {MAX_ID_LEN} ASCII letters, digits, `_` or `-`"
            )));
        }
        if !seen.insert(id) {
            return Err(ProviderError::request(format!(
                "batch item id `{id}` is used twice"
            )));
        }
    }
    Ok(())
}

/// Whether `id` is non-empty ASCII letters, digits, `_` and `-`.
fn is_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
}

/// The provider's batch id `id`, checked before it is spliced into a
/// request path, where a `/`, `?` or `#` would retarget the request.
pub(crate) fn path_id(id: &str) -> Result<&str, EncodeError> {
    if is_id(id) {
        Ok(id)
    } else {
        Err(EncodeError::request(format!(
            "`{id}` is not a batch id: expected ASCII letters, digits, `_` and `-`"
        )))
    }
}

/// A reply document decoded as `T`, kept beside the JSON it came in, so a
/// classifier's one decode yields both.
pub(crate) struct Document<T>(pub(crate) T, pub(crate) Value);

impl<'de, T: serde::de::DeserializeOwned> Deserialize<'de> for Document<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let typed = T::deserialize(&value).map_err(serde::de::Error::custom)?;
        Ok(Self(typed, value))
    }
}

/// The JSON body of an item's request.
pub(crate) fn json_body(request: &http::Request<Body>) -> Result<Value, ProviderError> {
    match request.body() {
        Body::Bytes(bytes) => Ok(serde_json::from_slice(bytes)?),
        Body::Multipart(_) => Err(ProviderError::request(
            "a multipart request cannot go in a batch",
        )),
    }
}

#[cfg(test)]
mod tests;
