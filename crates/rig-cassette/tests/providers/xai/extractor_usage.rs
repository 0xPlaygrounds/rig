//! The `Person` extraction type and profession check the xAI
//! `ecs_extractor_usage` cells share.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub(super) struct Person {
    pub(super) name: Option<String>,
    pub(super) age: Option<u8>,
    pub(super) profession: Option<String>,
}

pub(super) fn assert_compatible_professions(left: Option<&str>, right: &str) -> Result<()> {
    let left = left
        .ok_or_else(|| anyhow::anyhow!("profession should be present"))?
        .trim()
        .to_ascii_lowercase();
    let right = right.trim().to_ascii_lowercase();

    anyhow::ensure!(
        left == right || left.contains(&right) || right.contains(&left),
        "expected compatible professions, got {left:?} and {right:?}"
    );
    Ok(())
}
