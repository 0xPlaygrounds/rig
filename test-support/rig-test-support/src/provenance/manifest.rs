//! Shared declaration schema for provider cassette provenance.
//!
//! Compiled by `rig-test-support` and included by `xtask` via `#[path]`.
//! Serde rejects unknown/missing fields and wrong types; validation below
//! rejects empty declarations and duplicate identities.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub(crate) const MANIFEST_PATH: &str = "crates/rig-cassette/fixtures/scenarios.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Category {
    Live,
    Derived,
    Scripted,
}

impl Category {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Derived => "derived",
            Self::Scripted => "scripted",
        }
    }

    pub(crate) fn parse(text: &str) -> Option<Self> {
        match text {
            "live" => Some(Self::Live),
            "derived" => Some(Self::Derived),
            "scripted" => Some(Self::Scripted),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) providers: Vec<ProviderScenarios>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProviderScenarios {
    pub(crate) provider: String,
    /// Relative to `crates/rig-cassette/`.
    pub(crate) source_dir: String,
    /// Functions whose first argument names a scenario.
    pub(crate) wrappers: Vec<String>,
    pub(crate) live: Vec<String>,
    #[serde(default)]
    pub(crate) derived: Vec<DerivedScenario>,
    #[serde(default)]
    pub(crate) scripted: Vec<ScriptedFamilyDecl>,
    /// Live by intent; no fixture yet because its producer is ignored.
    #[serde(default)]
    pub(crate) unrecorded: Vec<UnrecordedScenario>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnrecordedScenario {
    pub(crate) scenario: String,
    pub(crate) reason: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DerivedScenario {
    pub(crate) scenario: String,
    /// Fully qualified `provider/scenario` ids.
    pub(crate) sources: Vec<String>,
    pub(crate) reason: String,
    pub(crate) rebuild: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScriptedFamilyDecl {
    pub(crate) family: String,
    /// Relative to `crates/rig-cassette/`, like `source_dir`.
    pub(crate) module: String,
    pub(crate) sources: Vec<String>,
    pub(crate) reason: String,
    pub(crate) cases: Vec<String>,
}

impl Manifest {
    pub(crate) fn parse(json: &str) -> Result<Self, String> {
        let manifest: Self =
            serde_json::from_str(json).map_err(|error| format!("invalid manifest: {error}"))?;
        unique(
            manifest.providers.iter().map(|p| p.provider.as_str()),
            "provider",
        )?;
        for provider in &manifest.providers {
            provider
                .validate()
                .map_err(|error| format!("{}: {error}", provider.provider))?;
        }
        Ok(manifest)
    }

    pub(crate) fn load(workspace_root: &Path) -> Result<Self, String> {
        let path = Self::path(workspace_root);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        Self::parse(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub(crate) fn path(workspace_root: &Path) -> PathBuf {
        workspace_root.join(MANIFEST_PATH)
    }

    pub(crate) fn provider(&self, provider: &str) -> Option<&ProviderScenarios> {
        self.providers.iter().find(|p| p.provider == provider)
    }
}

impl ProviderScenarios {
    fn validate(&self) -> Result<(), String> {
        text(&self.provider, "provider", 1)?;
        text(&self.source_dir, "source_dir", 1)?;
        strings(&self.wrappers, "wrappers", true)?;
        strings(&self.live, "live", false)?;
        for entry in &self.derived {
            text(&entry.scenario, "derived.scenario", 1)?;
            strings(&entry.sources, "derived.sources", true)?;
            text(&entry.reason, "derived.reason", 3)?;
            text(&entry.rebuild, "derived.rebuild", 3)?;
        }
        for entry in &self.unrecorded {
            text(&entry.scenario, "unrecorded.scenario", 1)?;
            text(&entry.reason, "unrecorded.reason", 3)?;
        }
        for entry in &self.scripted {
            text(&entry.family, "scripted.family", 1)?;
            text(&entry.module, "scripted.module", 1)?;
            strings(&entry.sources, "scripted.sources", false)?;
            strings(&entry.cases, "scripted.cases", true)?;
            text(&entry.reason, "scripted.reason", 3)?;
        }
        unique(
            self.live_scenarios()
                .into_iter()
                .chain(self.derived.iter().map(|d| d.scenario.as_str())),
            "scenario",
        )?;
        unique(
            self.scripted.iter().map(|s| s.family.as_str()),
            "scripted family",
        )
    }

    /// Every scenario that must have a committed fixture.
    pub(crate) fn fixture_scenarios(&self) -> Vec<&str> {
        self.live
            .iter()
            .map(String::as_str)
            .chain(self.derived.iter().map(|d| d.scenario.as_str()))
            .collect()
    }

    /// Scenarios a live recording is supposed to produce, committed or not.
    pub(crate) fn live_scenarios(&self) -> Vec<&str> {
        self.live
            .iter()
            .map(String::as_str)
            .chain(self.unrecorded.iter().map(|u| u.scenario.as_str()))
            .collect()
    }
}

fn text(value: &str, field: &str, minimum: usize) -> Result<(), String> {
    if value.trim().len() < minimum {
        return Err(format!(
            "{field} must contain at least {minimum} non-whitespace byte(s)"
        ));
    }
    Ok(())
}

fn strings(values: &[String], field: &str, required: bool) -> Result<(), String> {
    if required && values.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    for (index, value) in values.iter().enumerate() {
        text(value, &format!("{field}[{index}]"), 1)?;
    }
    Ok(())
}

fn unique<'a>(values: impl IntoIterator<Item = &'a str>, kind: &str) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(format!("{kind} {value:?} is declared twice"));
        }
    }
    Ok(())
}
