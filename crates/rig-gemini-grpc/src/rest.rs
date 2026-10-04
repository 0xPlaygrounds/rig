//! The canonical proto3 JSON of the Gemini messages, which is the REST
//! JSON: camelCase field names, enum values by name, bytes in base64, and
//! `google.protobuf.Struct` as a JSON object. Replies are read as this JSON
//! by the REST wire's decoder, and request contents built as REST JSON by
//! the shared Gemini encoder are read back into protobuf messages. One
//! generic mapping, driven by the descriptors `build.rs` compiles, covers
//! every field the proto declares.
//!
//! ```
//! use rig_gemini_grpc::proto::{Content, Part, part::Data};
//!
//! let content = Content {
//!     parts: vec![Part {
//!         data: Some(Data::Text("hi".to_owned())),
//!         ..Part::default()
//!     }],
//!     role: "user".to_owned(),
//! };
//! let json = rig_gemini_grpc::rest::to_rest(&content)?;
//! assert_eq!(json, serde_json::json!({"parts": [{"text": "hi"}], "role": "user"}));
//! assert_eq!(rig_gemini_grpc::rest::from_rest::<Content>(json)?, content);
//! # Ok::<(), rig_gemini_grpc::rest::TranscodeError>(())
//! ```

use std::sync::LazyLock;

use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor};
use serde_json::Value;

use crate::proto;

static DESCRIPTORS: LazyLock<Result<DescriptorPool, prost_reflect::DescriptorError>> =
    LazyLock::new(|| {
        DescriptorPool::decode(
            include_bytes!(concat!(env!("OUT_DIR"), "/gemini_descriptor.bin")).as_ref(),
        )
    });

/// Why a message could not be transcoded.
#[derive(Debug, thiserror::Error)]
pub enum TranscodeError {
    /// The compiled descriptors do not load or do not name the message.
    #[error("the Gemini proto descriptors do not describe `{0}`")]
    Descriptor(&'static str),
    /// The message bytes do not match its descriptor.
    #[error("Gemini protobuf message does not decode: {0}")]
    Decode(#[from] prost::DecodeError),
    /// The JSON is not the message's proto3 JSON.
    #[error("JSON is not a Gemini protobuf message: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<TranscodeError> for rig_core::error::ProviderError {
    fn from(error: TranscodeError) -> Self {
        Self::Response(error.to_string())
    }
}

impl From<TranscodeError> for rig_core::error::EncodeError {
    fn from(error: TranscodeError) -> Self {
        Self::request(error.to_string())
    }
}

/// A generated message and its full protobuf name.
pub trait Rest: prost::Message + Default {
    /// The message's full name in the `google.ai.generativelanguage.v1beta`
    /// package.
    const NAME: &'static str;
}

macro_rules! rest_messages {
    ($($message:ident),+ $(,)?) => {
        $(impl Rest for proto::$message {
            const NAME: &'static str =
                concat!("google.ai.generativelanguage.v1beta.", stringify!($message));
        })+
    };
}

rest_messages!(Content, GenerateContentRequest, GenerateContentResponse);

fn descriptor<M: Rest>() -> Result<MessageDescriptor, TranscodeError> {
    DESCRIPTORS
        .as_ref()
        .ok()
        .and_then(|pool| pool.get_message_by_name(M::NAME))
        .ok_or(TranscodeError::Descriptor(M::NAME))
}

/// `message` as its REST JSON. A value of an enum the proto does not list
/// is spelled as its number, as proto3 JSON spells it. A whole-number
/// double is written as an integer, as Google's JSON printer writes it, so
/// a call's `args` read the same on every Gemini wire.
///
/// # Errors
///
/// When the descriptors do not describe the message.
pub fn to_rest<M: Rest>(message: &M) -> Result<Value, TranscodeError> {
    let mut dynamic = DynamicMessage::new(descriptor::<M>()?);
    dynamic.transcode_from(message)?;
    let mut json = serde_json::to_value(&dynamic)?;
    integral_numbers(&mut json);
    Ok(json)
}

/// The largest magnitude a double holds every integer up to.
const EXACT: f64 = 9_007_199_254_740_992.0;

fn integral_numbers(value: &mut Value) {
    match value {
        Value::Number(number) => {
            if let Some(float) = number.as_f64().filter(|_| number.is_f64())
                && float.fract() == 0.0
                && float.abs() <= EXACT
            {
                *number = (float as i64).into();
            }
        }
        Value::Array(values) => values.iter_mut().for_each(integral_numbers),
        Value::Object(fields) => fields.values_mut().for_each(integral_numbers),
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
}

/// The message `json`, its REST JSON, describes. Bytes are read as
/// standard or URL-safe base64, so Google's placeholder thought signature
/// becomes the bytes it spells. A field the proto does not declare is an
/// error rather than dropped.
///
/// # Errors
///
/// When `json` is not the message's proto3 JSON.
pub fn from_rest<M: Rest>(json: Value) -> Result<M, TranscodeError> {
    let dynamic = DynamicMessage::deserialize(descriptor::<M>()?, json)?;
    Ok(dynamic.transcode_to()?)
}
