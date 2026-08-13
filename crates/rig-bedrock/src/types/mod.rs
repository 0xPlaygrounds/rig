//! Bedrock wire types.
//!
//! `converse_output` and `assistant_content` are public because
//! [`CompletionModel::raw_completion`](crate::completion::CompletionModel::raw_completion)
//! returns `assistant_content::AwsConverseOutput`, which wraps
//! `converse_output::InternalConverseOutput`: an escape hatch whose type a
//! caller cannot name is only half an escape hatch. The remaining modules are
//! request-side conversions with no public return type.

pub mod assistant_content;
pub mod converse_output;

pub(crate) mod completion_request;
pub(crate) mod document;
pub(crate) mod errors;
pub(crate) mod image;
pub(crate) mod json;
pub(crate) mod media_types;
pub(crate) mod message;
pub(crate) mod text_to_image;
pub(crate) mod tool;
pub(crate) mod user_content;
