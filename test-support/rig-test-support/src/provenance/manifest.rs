//! Parser for `crates/rig-cassette/fixtures/scenarios.json`, the authoritative
//! declaration of where every committed provider cassette came from.
//!
//! Two crates compile this file: `rig-test-support` declares it as a module of
//! `provenance`, and `xtask` includes it by `#[path]` the way `tests/core/mod.rs`
//! includes `xtask/src/verify/checks.rs`. That dual life is the reason for the
//! constraints here: std plus `serde_json` only, no derive, no `crate::` paths,
//! and `#[allow(dead_code)]` where one includer uses less of the surface than
//! the other.
//!
//! Parsing is strict on purpose. The manifest is an ownership ledger, so an
//! unknown key, a missing key, an empty justification or a scenario declared
//! twice is a rejected manifest rather than a silently ignored field: a typo in
//! `"derived"` must not quietly turn a hand-edited fixture into a live one.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Repository-relative location of the declaration file.
#[allow(dead_code)]
pub(crate) const MANIFEST_PATH: &str = "crates/rig-cassette/fixtures/scenarios.json";

/// How a committed cassette fixture came to exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub(crate) enum Category {
    /// Recorded against the real provider.
    Live,
    /// Hand-derived from one or more live recordings.
    Derived,
    /// Synthesised at runtime by a scripted transport; no fixture on disk.
    Scripted,
}

impl Category {
    /// The lowercase spelling used in the manifest and on the CLI.
    #[allow(dead_code)]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Derived => "derived",
            Self::Scripted => "scripted",
        }
    }

    /// Parse a CLI or manifest spelling.
    #[allow(dead_code)]
    pub(crate) fn parse(text: &str) -> Option<Self> {
        match text {
            "live" => Some(Self::Live),
            "derived" => Some(Self::Derived),
            "scripted" => Some(Self::Scripted),
            _ => None,
        }
    }
}

/// The whole declaration file.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct Manifest {
    /// One entry per provider suite, in declaration order.
    pub(crate) providers: Vec<ProviderScenarios>,
}

/// Every scenario owned by one provider suite.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct ProviderScenarios {
    /// Provider directory name under `fixtures/cassettes/`.
    pub(crate) provider: String,
    /// Test sources that reference this provider's cassettes, relative to
    /// `crates/rig-cassette/`.
    pub(crate) source_dir: String,
    /// Cassette wrapper function names whose first argument names a scenario.
    pub(crate) wrappers: Vec<String>,
    /// Scenarios recorded against the real provider.
    pub(crate) live: Vec<String>,
    /// Scenarios hand-derived from live recordings.
    pub(crate) derived: Vec<DerivedScenario>,
    /// Families synthesised by scripted transports.
    pub(crate) scripted: Vec<ScriptedFamilyDecl>,
    /// Scenarios that are live by intent but have no committed fixture yet,
    /// because the test that would record them is `#[ignore]`d.
    pub(crate) unrecorded: Vec<UnrecordedScenario>,
}

/// A scenario a live capture is meant to produce, not yet captured. It is
/// `Live` for recording purposes and has no file: the producing test is
/// `#[ignore]`d, so neither the fixture walk nor AST discovery sees it.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct UnrecordedScenario {
    /// Scenario id relative to the provider.
    pub(crate) scenario: String,
    /// Why no recording exists yet.
    pub(crate) reason: String,
}

/// A fixture that exists on disk but was never served by the provider.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct DerivedScenario {
    /// Scenario id relative to the provider.
    pub(crate) scenario: String,
    /// Fully qualified `provider/scenario` ids this was derived from.
    pub(crate) sources: Vec<String>,
    /// Why the derivation exists.
    pub(crate) reason: String,
    /// How to reproduce the derivation from its sources.
    pub(crate) rebuild: String,
}

/// A scripted family: no fixture on disk, behaviour injected in test code.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct ScriptedFamilyDecl {
    /// Family name as spelled in the test module.
    pub(crate) family: String,
    /// Test source implementing the family, relative to `crates/rig-cassette/`
    /// like `source_dir`.
    pub(crate) module: String,
    /// Fully qualified `provider/scenario` ids the script starts from.
    pub(crate) sources: Vec<String>,
    /// What behaviour the script injects.
    pub(crate) reason: String,
    /// Case names the family generates.
    pub(crate) cases: Vec<String>,
}

const PROVIDER_KEYS: &[&str] = &[
    "provider",
    "source_dir",
    "wrappers",
    "live",
    "derived",
    "scripted",
    "unrecorded",
];
const DERIVED_KEYS: &[&str] = &["scenario", "sources", "reason", "rebuild"];
const SCRIPTED_KEYS: &[&str] = &["family", "module", "sources", "reason", "cases"];
const UNRECORDED_KEYS: &[&str] = &["scenario", "reason"];

impl Manifest {
    /// Parse and validate the manifest text.
    #[allow(dead_code)]
    pub(crate) fn parse(json: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(json).map_err(|error| format!("invalid JSON: {error}"))?;
        let root = object(&root, "manifest")?;
        reject_unknown(root, &["providers"], "manifest")?;
        let providers = array(root, "providers", "manifest")?;

        let mut parsed = Vec::with_capacity(providers.len());
        let mut seen_providers = BTreeSet::new();
        for (index, entry) in providers.iter().enumerate() {
            let where_ = format!("providers[{index}]");
            let provider = ProviderScenarios::parse(entry, &where_)?;
            if !seen_providers.insert(provider.provider.clone()) {
                return Err(format!(
                    "{where_}: provider {:?} is declared twice",
                    provider.provider
                ));
            }
            parsed.push(provider);
        }

        Ok(Self { providers: parsed })
    }

    /// Read and parse `crates/rig-cassette/fixtures/scenarios.json`.
    #[allow(dead_code)]
    pub(crate) fn load(workspace_root: &Path) -> Result<Self, String> {
        let path = Self::path(workspace_root);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        Self::parse(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    /// Absolute path of the declaration file under `workspace_root`.
    #[allow(dead_code)]
    pub(crate) fn path(workspace_root: &Path) -> PathBuf {
        workspace_root.join(MANIFEST_PATH)
    }

    /// The declaration for one provider, if it is declared at all.
    #[allow(dead_code)]
    pub(crate) fn provider(&self, provider: &str) -> Option<&ProviderScenarios> {
        self.providers.iter().find(|p| p.provider == provider)
    }
}

impl ProviderScenarios {
    fn parse(value: &Value, where_: &str) -> Result<Self, String> {
        let object = object(value, where_)?;
        reject_unknown(object, PROVIDER_KEYS, where_)?;

        let provider = non_empty_string(object, "provider", where_)?;
        let where_ = format!("{where_} ({provider})");
        let source_dir = non_empty_string(object, "source_dir", &where_)?;
        let wrappers = string_array(object, "wrappers", &where_)?;
        if wrappers.is_empty() {
            return Err(format!("{where_}: wrappers must not be empty"));
        }
        let live = string_array(object, "live", &where_)?;

        // Only `provider`, `source_dir`, `wrappers` and `live` are required.
        // A suite with nothing hand-derived, nothing scripted and nothing
        // awaiting its first capture says so by omission: three empty arrays
        // per provider would be noise in a file whose whole job is to be read.
        let mut derived = Vec::new();
        for (index, entry) in optional_array(object, "derived", &where_)?
            .iter()
            .enumerate()
        {
            derived.push(DerivedScenario::parse(
                entry,
                &format!("{where_}.derived[{index}]"),
            )?);
        }

        let mut scripted = Vec::new();
        for (index, entry) in optional_array(object, "scripted", &where_)?
            .iter()
            .enumerate()
        {
            scripted.push(ScriptedFamilyDecl::parse(
                entry,
                &format!("{where_}.scripted[{index}]"),
            )?);
        }

        let mut unrecorded = Vec::new();
        for (index, entry) in optional_array(object, "unrecorded", &where_)?
            .iter()
            .enumerate()
        {
            unrecorded.push(UnrecordedScenario::parse(
                entry,
                &format!("{where_}.unrecorded[{index}]"),
            )?);
        }

        // A scenario id names exactly one cassette, so the same id cannot be
        // both recorded and hand-derived, nor listed twice in either, nor be
        // simultaneously committed and awaiting its first recording.
        let mut seen = BTreeSet::new();
        for scenario in live
            .iter()
            .chain(derived.iter().map(|d| &d.scenario))
            .chain(unrecorded.iter().map(|u| &u.scenario))
        {
            if !seen.insert(scenario.as_str()) {
                return Err(format!("{where_}: scenario {scenario:?} is declared twice"));
            }
        }
        let mut families = BTreeSet::new();
        for family in &scripted {
            if !families.insert(family.family.as_str()) {
                return Err(format!(
                    "{where_}: scripted family {:?} is declared twice",
                    family.family
                ));
            }
        }

        Ok(Self {
            provider,
            source_dir,
            wrappers,
            live,
            derived,
            scripted,
            unrecorded,
        })
    }

    /// Every scenario that must have a committed fixture: live and derived
    /// alike. `unrecorded` is deliberately absent — it is the set that has no
    /// file yet.
    #[allow(dead_code)]
    pub(crate) fn fixture_scenarios(&self) -> Vec<&str> {
        self.live
            .iter()
            .map(String::as_str)
            .chain(self.derived.iter().map(|d| d.scenario.as_str()))
            .collect()
    }

    /// Scenarios a live recording is supposed to produce, committed or not.
    #[allow(dead_code)]
    pub(crate) fn live_scenarios(&self) -> Vec<&str> {
        self.live
            .iter()
            .map(String::as_str)
            .chain(self.unrecorded.iter().map(|u| u.scenario.as_str()))
            .collect()
    }
}

impl DerivedScenario {
    fn parse(value: &Value, where_: &str) -> Result<Self, String> {
        let object = object(value, where_)?;
        reject_unknown(object, DERIVED_KEYS, where_)?;
        let scenario = non_empty_string(object, "scenario", where_)?;
        let where_ = format!("{where_} ({scenario})");
        let sources = string_array(object, "sources", &where_)?;
        if sources.is_empty() {
            return Err(format!("{where_}: sources must name at least one scenario"));
        }
        Ok(Self {
            scenario,
            sources,
            reason: justification(object, "reason", &where_)?,
            rebuild: justification(object, "rebuild", &where_)?,
        })
    }
}

impl UnrecordedScenario {
    fn parse(value: &Value, where_: &str) -> Result<Self, String> {
        let object = object(value, where_)?;
        reject_unknown(object, UNRECORDED_KEYS, where_)?;
        let scenario = non_empty_string(object, "scenario", where_)?;
        let where_ = format!("{where_} ({scenario})");
        Ok(Self {
            scenario,
            reason: justification(object, "reason", &where_)?,
        })
    }
}

impl ScriptedFamilyDecl {
    fn parse(value: &Value, where_: &str) -> Result<Self, String> {
        let object = object(value, where_)?;
        reject_unknown(object, SCRIPTED_KEYS, where_)?;
        let family = non_empty_string(object, "family", where_)?;
        let where_ = format!("{where_} ({family})");
        let module = non_empty_string(object, "module", &where_)?;
        let sources = string_array(object, "sources", &where_)?;
        let cases = string_array(object, "cases", &where_)?;
        if cases.is_empty() {
            return Err(format!("{where_}: cases must not be empty"));
        }
        Ok(Self {
            family,
            module,
            sources,
            reason: justification(object, "reason", &where_)?,
            cases,
        })
    }
}

type Object = serde_json::Map<String, Value>;

fn object<'a>(value: &'a Value, where_: &str) -> Result<&'a Object, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{where_}: expected an object"))
}

fn reject_unknown(object: &Object, allowed: &[&str], where_: &str) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!(
                "{where_}: unknown key {key:?} (allowed: {})",
                allowed.join(", ")
            ));
        }
    }
    Ok(())
}

fn member<'a>(object: &'a Object, key: &str, where_: &str) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("{where_}: missing key {key:?}"))
}

fn array<'a>(object: &'a Object, key: &str, where_: &str) -> Result<&'a Vec<Value>, String> {
    member(object, key, where_)?
        .as_array()
        .ok_or_else(|| format!("{where_}: {key:?} must be an array"))
}

/// An array whose absence means "none of these", distinguished from a present
/// value of the wrong shape, which is still a mistake worth reporting.
fn optional_array<'a>(object: &'a Object, key: &str, where_: &str) -> Result<&'a [Value], String> {
    match object.get(key) {
        None => Ok(&[]),
        Some(value) => value
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| format!("{where_}: {key:?} must be an array")),
    }
}

fn string_array(object: &Object, key: &str, where_: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for (index, entry) in array(object, key, where_)?.iter().enumerate() {
        let text = entry
            .as_str()
            .ok_or_else(|| format!("{where_}: {key}[{index}] must be a string"))?;
        if text.trim().is_empty() {
            return Err(format!("{where_}: {key}[{index}] must not be empty"));
        }
        out.push(text.to_owned());
    }
    Ok(out)
}

fn non_empty_string(object: &Object, key: &str, where_: &str) -> Result<String, String> {
    let text = member(object, key, where_)?
        .as_str()
        .ok_or_else(|| format!("{where_}: {key:?} must be a string"))?;
    if text.trim().is_empty() {
        return Err(format!("{where_}: {key:?} must not be empty"));
    }
    Ok(text.to_owned())
}

/// A justification whose whole point is being read by a human: whitespace is
/// not an explanation, so it is rejected rather than stored.
fn justification(object: &Object, key: &str, where_: &str) -> Result<String, String> {
    let text = non_empty_string(object, key, where_)?;
    if text.trim().len() < 3 {
        return Err(format!("{where_}: {key:?} must explain itself"));
    }
    Ok(text)
}
