//! xAI's image-generation model identifiers.
//!
//! The request runs on the shared OpenAI image wire, whose
//! [`xai::DIALECT`](crate::providers::xai::DIALECT) dialect carries the
//! `/v1/images/generations` path and xAI's body — no `size`, an explicit
//! `response_format: "b64_json"`, and an `aspect_ratio`.

pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";
