//! The extraction types the OpenAI `ecs_extractor_usage` cells read.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub(super) struct Person {
    pub(super) name: Option<String>,
    pub(super) age: Option<u8>,
    pub(super) profession: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub(super) struct Address {
    pub(super) street: Option<String>,
    pub(super) city: Option<String>,
    pub(super) state: Option<String>,
    pub(super) zip_code: Option<String>,
}
