//! The envelope a model operation answers with: its output and the metadata
//! the provider reported beside it.
//!
//! Each fact in [`ResponseMeta`] has one owner. The driver that sent the
//! request records the provider, the transport request id and the raw reply;
//! the decoder that read the reply reports the model, the response id and the
//! usage ([`Reported`]).
//!
//! ```
//! use rig_core::completion::Usage;
//! use rig_core::id::ProviderName;
//! use rig_core::response::{Response, ResponseMeta};
//!
//! let response = Response {
//!     output: "hello".to_owned(),
//!     meta: ResponseMeta::new(ProviderName::new("openai")?),
//! };
//! assert_eq!(response.meta.usage, Usage::default());
//! # Ok::<(), rig_core::id::EmptyId>(())
//! ```

use serde::{Deserialize, Serialize};

use crate::completion::Usage;
use crate::error::ProviderError;
use crate::id::{ModelName, ProviderName, RequestId, ResponseId};

/// What the provider reported about one response, beside its output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMeta {
    /// The provider that answered, as its wire names itself (`"openai"`).
    pub provider: ProviderName,
    /// The model the provider says answered, which can differ from the one
    /// requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelName>,
    /// The provider's id for the response. Never replayed as a message id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<ResponseId>,
    /// The transport request id from the reply's headers or SDK metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<RequestId>,
    /// The tokens the provider reported; a counter it did not report is
    /// `None`.
    #[serde(default)]
    pub usage: Usage,
    /// The provider's reply document: the whole reply for a buffered call
    /// (an array of the page documents when a call sent several requests),
    /// `null` when the reply was not JSON.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl ResponseMeta {
    /// Metadata for a response from `provider`, with nothing else reported.
    pub fn new(provider: ProviderName) -> Self {
        Self {
            provider,
            model: None,
            response_id: None,
            provider_request_id: None,
            usage: Usage::default(),
            raw: serde_json::Value::Null,
        }
    }
}

/// An operation's output with the metadata the provider reported beside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response<T> {
    /// What the operation produced.
    pub output: T,
    /// What the provider reported about it.
    pub meta: ResponseMeta,
}

/// What a decoder read off one reply: the output and the facts the reply
/// states about it. The driver completes it into a [`Response`].
#[derive(Debug, Clone, PartialEq)]
pub struct Reported<T> {
    /// What the reply carried.
    pub output: T,
    /// The model the reply names.
    pub model: Option<ModelName>,
    /// The response id the reply names.
    pub response_id: Option<ResponseId>,
    /// The tokens the reply reports.
    pub usage: Usage,
}

impl<T> Reported<T> {
    /// A reply that reports its output and nothing else.
    pub fn new(output: T) -> Self {
        Self {
            output,
            model: None,
            response_id: None,
            usage: Usage::default(),
        }
    }
}

/// A provider payload that reads as one operation's reply.
pub trait Normalize<T> {
    /// What the payload reports, or the reason it does not answer the
    /// request.
    fn normalize(self) -> Result<Reported<T>, ProviderError>;
}

#[cfg(test)]
mod tests;
