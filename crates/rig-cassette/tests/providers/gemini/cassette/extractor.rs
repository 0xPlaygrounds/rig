//! The `Person` extraction type the Gemini `ecs_extractor` cells read.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub(super) struct Person {
    pub(super) first_name: Option<String>,
    pub(super) last_name: Option<String>,
    pub(super) job: Option<String>,
}
