//! Integration tests for llama.cpp extractor usage tracking.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub(super) struct Person {
    #[schemars(required)]
    pub(super) name: Option<String>,
    #[schemars(required)]
    pub(super) age: Option<u8>,
    #[schemars(required)]
    pub(super) profession: Option<String>,
}

pub(super) const EXTRACTOR_PREAMBLE: &str = "\
Extract every field explicitly stated in the input text.
Do not omit keys when the value is present in the text.
Return the exact stated values through the submit tool.";
