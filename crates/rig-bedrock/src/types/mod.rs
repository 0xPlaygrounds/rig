//! Bedrock request conversions, Converse stop-reason mapping, and the
//! image-generation wire types. A Converse response's `raw` is the JSON
//! Bedrock sent, so it has no mirror type here.
//!
//! ```
//! use rig_bedrock::types::text_to_image::TextToImageResponse;
//!
//! let response: TextToImageResponse = serde_json::from_str(r#"{"images":[]}"#)?;
//! assert!(response.error.is_none());
//! # Ok::<(), serde_json::Error>(())
//! ```

pub mod assistant_content;

pub(crate) mod block;

pub(crate) mod completion_request;
pub(crate) mod document;
pub(crate) mod errors;
pub(crate) mod image;
pub(crate) mod json;
pub(crate) mod message;
/// Bedrock's text-to-image request and response wire types; the image
/// generation reply decodes from [`TextToImageResponse`](text_to_image::TextToImageResponse).
pub mod text_to_image;
pub(crate) mod tool;
pub(crate) mod user_content;
