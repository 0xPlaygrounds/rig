//! Provider-independent progressive-disclosure skills.
//!
//! Core catalogs descriptors and lazily loads trusted bundles. Filesystem
//! discovery and project-trust decisions belong in hosts or companion crates.

use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Lightweight skill advertisement suitable for initial model context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillDescriptor {
    /// Stable catalog name.
    pub name: String,
    /// Short progressive-disclosure description.
    pub description: String,
    /// Source identifier retained across loading/import.
    pub provenance: String,
    /// Tools this skill may use. Empty means no skill-specific grant; host
    /// policy still decides effective access.
    pub allowed_tools: Vec<String>,
}

/// Named asset loaded only after a skill is activated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillAsset {
    /// Bundle-relative logical name.
    pub name: String,
    /// Media type when known.
    pub media_type: Option<String>,
    /// Asset bytes.
    pub data: Vec<u8>,
}

/// Full trusted skill content loaded on demand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillBundle {
    /// Advertised descriptor and provenance.
    pub descriptor: SkillDescriptor,
    /// Full instructions injected only on activation.
    pub instructions: String,
    /// Optional references/assets.
    pub assets: Vec<SkillAsset>,
}

/// Skill catalog failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SkillError {
    /// Requested skill was not present.
    #[error("skill `{0}` not found")]
    NotFound(String),
    /// Catalog backend failed.
    #[error("skill catalog error: {0}")]
    Backend(String),
}

/// Lazy provider-independent skill catalog.
pub trait SkillCatalog: WasmCompatSend + WasmCompatSync {
    /// Enumerate lightweight descriptors without loading full instructions.
    fn list<'a>(&'a self) -> WasmBoxedFuture<'a, Result<Vec<SkillDescriptor>, SkillError>>;

    /// Load a trusted skill bundle by descriptor name.
    fn load<'a>(&'a self, name: &'a str) -> WasmBoxedFuture<'a, Result<SkillBundle, SkillError>>;
}

/// Immutable in-memory catalog useful for hosts and tests.
#[derive(Clone, Default)]
pub struct InMemorySkillCatalog {
    bundles: Arc<HashMap<String, SkillBundle>>,
}

impl InMemorySkillCatalog {
    /// Build a catalog, rejecting duplicate names.
    pub fn new(bundles: impl IntoIterator<Item = SkillBundle>) -> Result<Self, SkillError> {
        let mut by_name = HashMap::new();
        for bundle in bundles {
            let name = bundle.descriptor.name.clone();
            if by_name.insert(name.clone(), bundle).is_some() {
                return Err(SkillError::Backend(format!("duplicate skill `{name}`")));
            }
        }
        Ok(Self {
            bundles: Arc::new(by_name),
        })
    }
}

impl SkillCatalog for InMemorySkillCatalog {
    fn list<'a>(&'a self) -> WasmBoxedFuture<'a, Result<Vec<SkillDescriptor>, SkillError>> {
        Box::pin(async move {
            let mut descriptors = self
                .bundles
                .values()
                .map(|bundle| bundle.descriptor.clone())
                .collect::<Vec<_>>();
            descriptors.sort_by(|left, right| left.name.cmp(&right.name));
            Ok(descriptors)
        })
    }

    fn load<'a>(&'a self, name: &'a str) -> WasmBoxedFuture<'a, Result<SkillBundle, SkillError>> {
        Box::pin(async move {
            self.bundles
                .get(name)
                .cloned()
                .ok_or_else(|| SkillError::NotFound(name.to_owned()))
        })
    }
}
