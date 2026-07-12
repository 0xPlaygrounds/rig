//! Provider-independent progressive-disclosure skills.

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

/// Skill provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillProvenance {
    BuiltIn,
    Filesystem(PathBuf),
    ProviderHosted(String),
}

/// Skill metadata and lazily disclosed instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub provenance: SkillProvenance,
    pub assets: Vec<PathBuf>,
    pub allowed_tools: Vec<String>,
}

/// Provider-independent skill lookup.
pub trait SkillCatalog: WasmCompatSend + WasmCompatSync {
    /// List names/descriptions without disclosing instructions.
    fn list(&self) -> Vec<(String, String)>;
    /// Activate and disclose one complete skill.
    fn activate(&self, name: &str) -> Option<Skill>;
}

/// Mutable in-memory skill catalog.
#[derive(Clone, Default)]
pub struct InMemorySkillCatalog(Arc<RwLock<HashMap<String, Skill>>>);
impl InMemorySkillCatalog {
    pub fn insert(&self, skill: Skill) {
        self.0
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(skill.name.clone(), skill);
    }
}
impl SkillCatalog for InMemorySkillCatalog {
    fn list(&self) -> Vec<(String, String)> {
        let mut values = self
            .0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.name.clone(), s.description.clone()))
            .collect::<Vec<_>>();
        values.sort();
        values
    }
    fn activate(&self, name: &str) -> Option<Skill> {
        self.0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }
}

/// Trusted filesystem discovery for `SKILL.md` directories.
pub struct FilesystemSkillCatalog;
impl FilesystemSkillCatalog {
    /// Discover direct child `SKILL.md` files under a canonical trusted root.
    pub fn discover(
        root: impl AsRef<Path>,
        trusted_root: impl AsRef<Path>,
    ) -> std::io::Result<InMemorySkillCatalog> {
        let trusted = trusted_root.as_ref().canonicalize()?;
        let root = root.as_ref().canonicalize()?;
        if !root.starts_with(&trusted) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "skill root escapes trusted project",
            ));
        }
        let catalog = InMemorySkillCatalog::default();
        for entry in std::fs::read_dir(&root)? {
            let directory = entry?.path();
            let path = directory.join("SKILL.md");
            if !path.is_file() {
                continue;
            }
            let canonical = path.canonicalize()?;
            if !canonical.starts_with(&trusted) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "skill path escapes trusted project",
                ));
            }
            let contents = std::fs::read_to_string(&canonical)?;
            let mut lines = contents.lines();
            let name = lines
                .next()
                .unwrap_or("skill")
                .trim_start_matches('#')
                .trim()
                .to_string();
            let description = lines.next().unwrap_or("").trim().to_string();
            let assets = std::fs::read_dir(&directory)?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|asset| asset.file_name().is_some_and(|name| name != "SKILL.md"))
                .collect();
            catalog.insert(Skill {
                name,
                description,
                instructions: contents,
                provenance: SkillProvenance::Filesystem(canonical),
                assets,
                allowed_tools: Vec::new(),
            });
        }
        Ok(catalog)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disclosure_hides_instructions_until_activation() {
        let catalog = InMemorySkillCatalog::default();
        catalog.insert(Skill {
            name: "review".into(),
            description: "review code".into(),
            instructions: "secret detail".into(),
            provenance: SkillProvenance::BuiltIn,
            assets: vec![],
            allowed_tools: vec!["read".into()],
        });
        assert_eq!(
            catalog.list(),
            vec![("review".into(), "review code".into())]
        );
        assert_eq!(
            catalog.activate("review").unwrap().instructions,
            "secret detail"
        );
    }

    #[test]
    fn filesystem_discovery_respects_trusted_root() {
        let root = std::env::temp_dir().join(format!("rig-skills-{}", crate::id::generate()));
        let skill = root.join("review");
        std::fs::create_dir_all(&skill).unwrap();
        std::fs::write(
            skill.join("SKILL.md"),
            "# review\nReview code\nDetailed instructions",
        )
        .unwrap();
        let catalog = FilesystemSkillCatalog::discover(&root, &root).unwrap();
        assert_eq!(
            catalog.list(),
            vec![("review".into(), "Review code".into())]
        );
        assert!(FilesystemSkillCatalog::discover(&root, root.join("review")).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
