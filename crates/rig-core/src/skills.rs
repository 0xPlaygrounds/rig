//! Provider-independent, progressively disclosed agent skills.
//!
//! Catalog listings contain only names/descriptions. Full instructions and
//! assets are loaded on demand through [`SkillCatalog`], allowing hosts to own
//! filesystem discovery, trust, and provenance policy.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Lightweight skill advertisement suitable for model context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillSummary {
    /// Stable catalog name.
    pub name: String,
    /// Short progressive-disclosure description.
    pub description: String,
    /// Host-defined provenance (package, project, provider, and so on).
    pub provenance: Option<String>,
}

/// Asset bundled with a loaded skill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillAsset {
    /// Relative logical path.
    pub path: String,
    /// Raw asset bytes.
    pub content: Vec<u8>,
    /// Optional media type.
    pub media_type: Option<String>,
}

/// Fully loaded skill instructions and assets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    /// Lightweight catalog metadata.
    pub summary: SkillSummary,
    /// Full instructions loaded only after selection.
    pub instructions: String,
    /// Optional supporting assets.
    pub assets: Vec<SkillAsset>,
    /// Optional allowlist of tool names the host may enforce.
    pub allowed_tools: Option<Vec<String>>,
}

/// Skill catalog failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SkillError {
    /// No skill exists under the requested name.
    #[error("skill `{0}` was not found")]
    NotFound(String),
    /// Catalog backend failure.
    #[error("skill catalog backend error: {0}")]
    Backend(String),
}

/// Provider-independent progressive-disclosure skill catalog.
pub trait SkillCatalog: WasmCompatSend + WasmCompatSync {
    /// List lightweight advertisements without loading full instructions/assets.
    fn list(&self) -> WasmBoxedFuture<'_, Result<Vec<SkillSummary>, SkillError>>;

    /// Load one full skill by stable name.
    fn load<'a>(&'a self, name: &'a str) -> WasmBoxedFuture<'a, Result<Skill, SkillError>>;
}

impl<C> SkillCatalog for Arc<C>
where
    C: SkillCatalog + ?Sized,
{
    fn list(&self) -> WasmBoxedFuture<'_, Result<Vec<SkillSummary>, SkillError>> {
        (**self).list()
    }

    fn load<'a>(&'a self, name: &'a str) -> WasmBoxedFuture<'a, Result<Skill, SkillError>> {
        (**self).load(name)
    }
}

/// Mutable in-process catalog useful for hosts and tests.
#[derive(Clone, Default)]
pub struct InMemorySkillCatalog {
    skills: Arc<RwLock<HashMap<String, Skill>>>,
}

impl InMemorySkillCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a skill by its summary name.
    pub fn insert(&self, skill: Skill) -> Option<Skill> {
        self.skills
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(skill.summary.name.clone(), skill)
    }

    /// Remove a skill.
    pub fn remove(&self, name: &str) -> Option<Skill> {
        self.skills
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name)
    }
}

impl SkillCatalog for InMemorySkillCatalog {
    fn list(&self) -> WasmBoxedFuture<'_, Result<Vec<SkillSummary>, SkillError>> {
        Box::pin(async move {
            let mut summaries = self
                .skills
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .values()
                .map(|skill| skill.summary.clone())
                .collect::<Vec<_>>();
            summaries.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(summaries)
        })
    }

    fn load<'a>(&'a self, name: &'a str) -> WasmBoxedFuture<'a, Result<Skill, SkillError>> {
        Box::pin(async move {
            self.skills
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .get(name)
                .cloned()
                .ok_or_else(|| SkillError::NotFound(name.to_owned()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[allow(clippy::panic_in_result_fn)]
    async fn lists_summaries_before_loading_instructions() -> Result<(), SkillError> {
        let catalog = InMemorySkillCatalog::new();
        catalog.insert(Skill {
            summary: SkillSummary {
                name: "rust".into(),
                description: "Rust help".into(),
                provenance: Some("project".into()),
            },
            instructions: "Full private instructions".into(),
            assets: Vec::new(),
            allowed_tools: Some(vec!["cargo".into()]),
        });
        let summaries = catalog.list().await?;
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].description, "Rust help");
        assert_eq!(
            catalog.load("rust").await?.instructions,
            "Full private instructions"
        );
        Ok(())
    }
}
