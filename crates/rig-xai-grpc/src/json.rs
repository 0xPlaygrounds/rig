//! The proto3 JSON of xAI's messages, under the proto's own field names:
//! snake_case keys, enum values by name, bytes in base64 and 64-bit integers
//! as numbers. A reply's `raw` document is its message in this form, and
//! requests are read from it, driven by the descriptors `build.rs` compiles.
//!
//! ```
//! use rig_xai_grpc::proto::GetChatCompletionResponse;
//!
//! let response = GetChatCompletionResponse {
//!     id: "resp-1".to_owned(),
//!     ..GetChatCompletionResponse::default()
//! };
//! let json = rig_xai_grpc::json::to_json(&response)?;
//! assert_eq!(json["id"], "resp-1");
//! assert_eq!(rig_xai_grpc::json::from_json::<GetChatCompletionResponse>(json)?, response);
//! # Ok::<(), rig_xai_grpc::json::TranscodeError>(())
//! ```

use std::sync::LazyLock;

use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, SerializeOptions};
use serde_json::Value;

use crate::proto;

static DESCRIPTORS: LazyLock<Result<DescriptorPool, prost_reflect::DescriptorError>> =
    LazyLock::new(|| {
        DescriptorPool::decode(
            include_bytes!(concat!(env!("OUT_DIR"), "/xai_descriptor.bin")).as_ref(),
        )
    });

/// Why a message could not be transcoded.
#[derive(Debug, thiserror::Error)]
pub enum TranscodeError {
    /// The compiled descriptors do not load or do not name the message.
    #[error("the xAI proto descriptors do not describe `{0}`")]
    Descriptor(&'static str),
    /// The message bytes do not match its descriptor.
    #[error("xAI protobuf message does not decode: {0}")]
    Decode(#[from] prost::DecodeError),
    /// The JSON is not the message's proto3 JSON.
    #[error("JSON is not an xAI protobuf message: {0}")]
    Json(#[from] serde_json::Error),
}

/// A generated message and its full protobuf name.
pub trait Proto: prost::Message + Default {
    /// The message's full name in the `xai_api` package.
    const NAME: &'static str;
}

impl Proto for proto::GetCompletionsRequest {
    const NAME: &'static str = "xai_api.GetCompletionsRequest";
}

impl Proto for proto::GetChatCompletionResponse {
    const NAME: &'static str = "xai_api.GetChatCompletionResponse";
}

impl Proto for proto::GetChatCompletionChunk {
    const NAME: &'static str = "xai_api.GetChatCompletionChunk";
}

fn descriptor<M: Proto>() -> Result<MessageDescriptor, TranscodeError> {
    DESCRIPTORS
        .as_ref()
        .ok()
        .and_then(|pool| pool.get_message_by_name(M::NAME))
        .ok_or(TranscodeError::Descriptor(M::NAME))
}

/// `message` as proto3 JSON under the proto's field names.
///
/// # Errors
///
/// When the descriptors do not describe the message.
pub fn to_json<M: Proto>(message: &M) -> Result<Value, TranscodeError> {
    let mut dynamic = DynamicMessage::new(descriptor::<M>()?);
    dynamic.transcode_from(message)?;
    let options = SerializeOptions::new()
        .use_proto_field_name(true)
        .stringify_64_bit_integers(false);
    Ok(dynamic.serialize_with_options(serde_json::value::Serializer, &options)?)
}

/// The message `json` describes, under either field spelling proto3 JSON
/// allows. A field the proto does not declare is an error rather than
/// dropped.
///
/// # Errors
///
/// When `json` is not the message's proto3 JSON.
pub fn from_json<M: Proto>(json: Value) -> Result<M, TranscodeError> {
    let dynamic = DynamicMessage::deserialize(descriptor::<M>()?, json)?;
    Ok(dynamic.transcode_to()?)
}
