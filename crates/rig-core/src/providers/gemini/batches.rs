//! Batch mode: many GenerateContent requests run asynchronously at half the
//! price. Each request is rendered exactly as the model would send it.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::gemini::{self, Gemini};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::from_env()?;
//! let model = gemini.completion(gemini::GEMINI_3_8_FLASH);
//! let job = gemini
//!     .batches()
//!     .create(&model.wire, "nightly", [CompletionRequest::new("Summarize A")])
//!     .await?;
//! println!("{:?}", job.name);
//! # Ok(())
//! # }
//! ```

use serde::Deserialize;

use super::api::{self, Recognized};
use super::completion::GenerateContent;
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::operation::BatchJobs;
use crate::providers::internal::wire::{classify_or, classify_untyped_line};
use crate::providers::internal::with_query_pairs;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

const BATCHES_PATH: &str = "/v1beta/batches";

/// One batch verb.
#[derive(Debug)]
pub enum BatchRequest {
    /// Start a batch on `model`; answers with its operation.
    Create {
        /// The model id.
        model: String,
        /// The batch: its name and inlined requests.
        batch: Box<api::GenerateContentBatch>,
    },
    /// Read a batch by name, `batches/<id>` or `<id>`.
    Get(String),
    /// One page of the listing, after `page_token` when continuing.
    List {
        /// The previous page's cursor.
        page_token: Option<String>,
    },
    /// Cancel a running batch.
    Cancel(String),
    /// Delete a batch.
    Delete(String),
}

/// A batch operation, a page of them, or an empty acknowledgement.
#[derive(Clone, Debug, Default)]
pub enum BatchReply {
    /// `create` and `get`: the batch's long-running operation.
    Operation(Box<api::Operation>),
    /// `list`: one page.
    Page(api::ListOperationsResponse),
    /// `cancel` and `delete`.
    #[default]
    Acknowledged,
}

impl BatchReply {
    /// The operation a create or read returned.
    pub fn operation(self) -> Result<api::Operation, ProviderError> {
        match self {
            Self::Operation(operation) => Ok(*operation),
            _ => Err(ProviderError::Response(
                "the batch reply carried no operation".to_owned(),
            )),
        }
    }

    /// The cursor of the next page, when there is one.
    pub fn next_page_token(&self) -> Option<String> {
        match self {
            Self::Page(page) => page
                .next_page_token
                .clone()
                .filter(|token| !token.is_empty()),
            _ => None,
        }
    }

    /// The operations of one listing page.
    pub fn entries(self) -> Result<Vec<api::Operation>, ProviderError> {
        match self {
            Self::Page(page) => Ok(page.operations),
            Self::Acknowledged => Ok(Vec::new()),
            Self::Operation(_) => Err(ProviderError::Response(
                "the batch reply carried an operation, not a listing page".to_owned(),
            )),
        }
    }
}

/// The batch wire.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Batches {
    /// The key and the API root.
    pub provider: super::GeminiConfig,
}

impl super::GeminiConfig {
    /// The batch wire.
    pub(crate) fn batches(&self) -> Batches {
        Batches {
            provider: self.clone(),
        }
    }
}

/// The batch that runs `requests` as `model` would send them, keyed by
/// position in `metadata.key`.
pub fn batch(
    model: &GenerateContent,
    display_name: impl Into<String>,
    requests: impl IntoIterator<Item = CompletionRequest>,
) -> Result<api::GenerateContentBatch, EncodeError> {
    let requests = requests
        .into_iter()
        .enumerate()
        .map(|(index, request)| {
            Ok(api::InlinedRequest {
                request: Some(model.to_api(request)?),
                metadata: Some(serde_json::Map::from_iter([(
                    "key".to_owned(),
                    serde_json::Value::String(index.to_string()),
                )])),
                ..Default::default()
            })
        })
        .collect::<Result<Vec<_>, EncodeError>>()?;
    Ok(api::GenerateContentBatch {
        display_name: Some(display_name.into()),
        model: Some(format!("models/{}", model.model)),
        input_config: Some(api::InputConfig {
            requests: Some(api::InlinedRequests {
                requests,
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    })
}

/// `/v1beta/batches/<id>` from a name, refusing anything that could retarget
/// the path.
fn batch_path(name: &str) -> Result<String, EncodeError> {
    let id = name.strip_prefix("batches/").unwrap_or(name);
    let is_id_char = |ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_');
    if id.is_empty() || !id.chars().all(is_id_char) {
        return Err(EncodeError::request(format!(
            "`{name}` is not a batch name `batches/<id>`"
        )));
    }
    Ok(format!("{BATCHES_PATH}/{id}"))
}

impl Wire for Batches {
    type Op = BatchJobs;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = BatchesDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    fn encode(&self, request: BatchRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = match request {
            BatchRequest::Create { model, batch } => {
                let body = api::BatchGenerateContentRequest {
                    batch: Some(*batch),
                    ..Default::default()
                };
                http::Request::post(
                    self.provider
                        .uri(&format!("/v1beta/models/{model}:batchGenerateContent")),
                )
                .header("Content-Type", "application/json")
                .body(Body::Bytes(serde_json::to_vec(&body)?))?
            }
            BatchRequest::Get(name) => {
                http::Request::get(self.provider.uri(&batch_path(&name)?)).body(Body::empty())?
            }
            BatchRequest::List { page_token } => {
                let pairs: Vec<(&str, &str)> = page_token
                    .as_deref()
                    .map(|token| ("pageToken", token))
                    .into_iter()
                    .collect();
                http::Request::get(self.provider.uri(&with_query_pairs(BATCHES_PATH, &pairs)))
                    .body(Body::empty())?
            }
            BatchRequest::Cancel(name) => {
                let path = format!("{}:cancel", batch_path(&name)?);
                http::Request::post(self.provider.uri(&path)).body(Body::empty())?
            }
            BatchRequest::Delete(name) => {
                http::Request::delete(self.provider.uri(&batch_path(&name)?)).body(Body::empty())?
            }
        };
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        BatchesDecoder
    }
}

/// Decodes one batch reply.
pub struct BatchesDecoder;

impl<'id> Decoder<'id, BatchJobs> for BatchesDecoder {
    type Event = BatchReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<BatchReply> {
        let body = frame.as_str();
        classify_or(&body, as_page, |data| {
            classify_or(data, as_acknowledgement, as_operation)
        })
    }

    fn decode(
        &mut self,
        reply: BatchReply,
        out: Out<'id, BatchJobs>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }

    /// A reply with no body at all is an acknowledgement.
    fn eof(&mut self, out: Out<'id, BatchJobs>) -> Result<Flow, ProviderError> {
        Ok(out.end(BatchReply::Acknowledged))
    }
}

fn as_page(data: &str) -> WireEvent<BatchReply> {
    classify_untyped_line::<Recognized<api::ListOperationsResponse>>(data.as_bytes())
        .map(|Recognized(page)| BatchReply::Page(page))
}

fn as_acknowledgement(data: &str) -> WireEvent<BatchReply> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Acknowledgement {}
    classify_untyped_line::<Acknowledgement>(data.as_bytes()).map(|_| BatchReply::Acknowledged)
}

fn as_operation(data: &str) -> WireEvent<BatchReply> {
    classify_untyped_line::<Recognized<api::Operation>>(data.as_bytes())
        .map(|Recognized(operation)| BatchReply::Operation(Box::new(operation)))
}

#[cfg(test)]
mod tests;
