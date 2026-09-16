//! Image model identifiers for OpenAI's `/images/generations` endpoint.
//!
//! The endpoint itself is [`Images`](super::wire::Images); what is left here
//! is the model vocabulary the wire is pointed at.

pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";
pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";
