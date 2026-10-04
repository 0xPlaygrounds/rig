//! Copilot integration tests for extractor usage tracking.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub(super) struct Person {
    pub(super) name: Option<String>,
    pub(super) age: Option<u8>,
    pub(super) profession: Option<String>,
}
