use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub struct Person {
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub job: Option<String>,
}
