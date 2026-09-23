//! Explicit runtime-defined evaluation queries.
use crate::{Query, types::Question};
use rig_core::error::ProviderError;
use std::collections::BTreeMap;

/// Runtime-defined questions with runtime-keyed answers. Choose this explicitly
/// when the application cannot know the question set at compile time.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(transparent)]
pub struct DynamicQuery {
    definitions: BTreeMap<String, Question>,
}
impl DynamicQuery {
    /// Validate runtime question definitions before accepting them.
    pub fn new(definitions: BTreeMap<String, Question>) -> Result<Self, ProviderError> {
        if definitions.is_empty() {
            return Err(ProviderError::Request(
                "at least one question is required".into(),
            ));
        }
        for (id, question) in &definitions {
            crate::questions::validate_definition(id, question)?;
        }
        Ok(Self { definitions })
    }
}
impl Query for DynamicQuery {
    type Output = BTreeMap<String, crate::types::Answer>;
    type Response = BTreeMap<String, crate::types::Answer>;
    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        if !self.definitions.keys().eq(response.keys()) {
            return Err(ProviderError::Response(
                "response question IDs differ from request".into(),
            ));
        }
        self.definitions
            .iter()
            .map(|(id, question)| {
                let answer = response
                    .get(id)
                    .ok_or_else(|| ProviderError::Response(format!("missing answer: {id}")))?;
                crate::questions::validate(question, answer)?;
                Ok((id.clone(), answer.clone()))
            })
            .collect()
    }
}
