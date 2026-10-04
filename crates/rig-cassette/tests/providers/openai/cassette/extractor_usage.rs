//! Integration tests for extractor usage tracking.
//!
//! These tests verify that:
//! - `extract()` yields the extracted value as `TypedPromptResponse::output`
//! - the same `TypedPromptResponse` carries the run's usage
//! - Usage accumulates across retry attempts
//! - Both plain and `.history(..)` extractions work

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
