//! Bedrock's error conversions and the image-generation wire types.
//!
//! ```
//! use rig_bedrock::types::text_to_image::TextToImageResponse;
//!
//! let response: TextToImageResponse = serde_json::from_str(r#"{"images":[]}"#)?;
//! assert!(response.error.is_none());
//! # Ok::<(), serde_json::Error>(())
//! ```

pub(crate) mod errors;
/// Bedrock's text-to-image request and response wire types; the image
/// generation reply decodes from [`TextToImageResponse`](text_to_image::TextToImageResponse).
pub mod text_to_image;
