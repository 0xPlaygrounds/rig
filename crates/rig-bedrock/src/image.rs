//! Bedrock image-generation model identifiers.
//!
//! The generation call itself is the free function
//! [`crate::functions::generate_image`]; this module is just the model-id
//! constants it is called with.

/// `amazon.titan-image-generator-v1`
pub const AMAZON_TITAN_IMAGE_GENERATOR_V1: &str = "amazon.titan-image-generator-v1";
/// `amazon.titan-image-generator-v2:0`
pub const AMAZON_TITAN_IMAGE_GENERATOR_V2_0: &str = "amazon.titan-image-generator-v2:0";
/// `amazon.nova-canvas-v1:0`
pub const AMAZON_NOVA_CANVAS: &str = "amazon.nova-canvas-v1:0";
