//! Image model identifiers for OpenAI's `/images/generations` endpoint.
//!
//! ```
//! use rig_core::providers::openai::{OpenAI, image_generation::GPT_IMAGE_2};
//! let wire = OpenAI::new("key").images(GPT_IMAGE_2);
//! ```

pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";
pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";
