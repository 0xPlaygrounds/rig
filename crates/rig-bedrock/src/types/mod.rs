//! Bedrock response types and request conversions.
//! Raw completion and image-generation methods return these provider-native types.
//!
//! ```
//! use rig_bedrock::types::text_to_image::TextToImageResponse;
//!
//! let response: TextToImageResponse = serde_json::from_str(r#"{"images":[]}"#)?;
//! assert!(response.error.is_none());
//! # Ok::<(), serde_json::Error>(())
//! ```

pub mod assistant_content;
pub mod converse_output;

pub(crate) mod completion_request;
pub(crate) mod document;
pub(crate) mod errors;
pub(crate) mod image;
pub(crate) mod json;
pub(crate) mod media_types;
pub(crate) mod message;
/// Bedrock's text-to-image request and response wire types; the image
/// generation reply decodes from [`TextToImageResponse`](text_to_image::TextToImageResponse).
pub mod text_to_image;
pub(crate) mod tool;
pub(crate) mod user_content;
