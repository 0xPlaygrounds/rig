use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use crate::{Capture, Inventory, Scenario, Test};

/// A malformed declaration or unsafe recording selection.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error::Invalid(message()))
    }
}

fn canonical(id: &str) -> bool {
    id.contains('/')
        && id.split('/').all(|s| {
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        })
}

fn scenarios(tests: &[Test]) -> Result<BTreeMap<&str, &Scenario>, Error> {
    let mut out = BTreeMap::new();
    let mut names = BTreeSet::new();
    for test in tests {
        require(!test.name.is_empty() && names.insert(&test.name), || {
            format!("duplicate or empty test identity: {}", test.name)
        })?;
        let mut local = BTreeSet::new();
        for scenario in &test.scenarios {
            require(canonical(&scenario.id), || {
                format!("noncanonical scenario: {}", scenario.id)
            })?;
            require(local.insert(&scenario.id), || {
                format!("{} declares {} twice", test.name, scenario.id)
            })?;
            if let Some(previous) = out.insert(scenario.id.as_str(), scenario) {
                // Several assertions may replay the same fixture. Its policy must agree.
                require(previous == scenario, || {
                    format!("conflicting declarations for {}", scenario.id)
                })?;
            }
            if let Some(reason) = &scenario.missing {
                require(
                    !reason.trim().is_empty() && scenario.capture == Capture::Allowed,
                    || format!("{}: invalid missing-capture allowance", scenario.id),
                )?;
            }
            if let Capture::Forbidden(reason) = &scenario.capture {
                require(!reason.trim().is_empty(), || {
                    format!("{}: forbidden capture needs a reason", scenario.id)
                })?;
            }
            if let Some(derived) = &scenario.derivation {
                require(
                    matches!(scenario.capture, Capture::Forbidden(_))
                        && !derived.sources.is_empty()
                        && !derived.reason.trim().is_empty()
                        && !derived.rebuild.trim().is_empty(),
                    || format!("{}: incomplete or recordable derivation", scenario.id),
                )?;
            }
        }
    }
    Ok(out)
}

/// Validate compiled declarations against the provider fixture directories.
/// `providers` is the exact set of binaries whose inventory was collected.
pub fn validate(
    inventory: &Inventory,
    root: &Path,
    providers: &BTreeSet<String>,
) -> Result<(), Error> {
    require(!providers.is_empty() && !inventory.tests.is_empty(), || {
        "empty cassette inventory".into()
    })?;
    let declarations = scenarios(&inventory.tests)?;
    for (id, scenario) in &declarations {
        let provider = id.split('/').next().unwrap_or_default();
        require(providers.contains(provider), || {
            format!("{id}: outside inventoried providers")
        })?;
        let exists = root.join(format!("{id}.yaml")).is_file();
        require(exists || scenario.missing.is_some(), || {
            format!("{id}: declared fixture is missing")
        })?;
        // First captures remain recordable, but the allowance must be removed before commit.
        require(!exists || scenario.missing.is_none(), || {
            format!("{id}: captured fixture still has a missing-capture allowance")
        })?;
        if let Some(derived) = &scenario.derivation {
            let mut sources = BTreeSet::new();
            for source in &derived.sources {
                require(
                    source != id
                        && sources.insert(source)
                        && declarations.contains_key(source.as_str()),
                    || format!("{id}: duplicate, missing or self-referencing source {source}"),
                )?;
            }
        }
    }
    fn visit<'a>(
        id: &'a str,
        entries: &BTreeMap<&'a str, &'a Scenario>,
        active: &mut BTreeSet<&'a str>,
        done: &mut BTreeSet<&'a str>,
    ) -> Result<(), Error> {
        if done.contains(id) {
            return Ok(());
        }
        require(active.insert(id), || {
            format!("derived-source cycle at {id}")
        })?;
        if let Some(derived) = entries.get(id).and_then(|s| s.derivation.as_ref()) {
            for source in &derived.sources {
                visit(source, entries, active, done)?;
            }
        }
        active.remove(id);
        done.insert(id);
        Ok(())
    }
    let mut done = BTreeSet::new();
    for id in declarations.keys() {
        visit(id, &declarations, &mut BTreeSet::new(), &mut done)?;
    }
    let mut families = BTreeSet::new();
    for family in &inventory.families {
        require(
            !family.name.is_empty() && families.insert(&family.name),
            || format!("duplicate or empty scripted family {}", family.name),
        )?;
        let mut sources = BTreeSet::new();
        for source in &family.sources {
            require(
                sources.insert(source) && declarations.contains_key(source.as_str()),
                || {
                    format!(
                        "{}: duplicate or undeclared scripted source {source}",
                        family.name
                    )
                },
            )?;
        }
    }
    fn walk(root: &Path, dir: &Path, entries: &BTreeMap<&str, &Scenario>) -> Result<(), Error> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, entries)?;
            } else if path.extension().is_some_and(|ext| ext == "yaml") {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|e| Error::Invalid(e.to_string()))?;
                let id = relative
                    .with_extension("")
                    .to_string_lossy()
                    .replace('\\', "/");
                require(entries.contains_key(id.as_str()), || {
                    format!("orphan fixture {id}")
                })?;
            }
        }
        Ok(())
    }
    for provider in providers {
        walk(root, &root.join(provider), &declarations)?;
    }
    Ok(())
}

/// Check one session before upstream, credential or write operations.
pub fn authorize(
    tests: &[Test],
    root: &Path,
    id: &str,
    scope: Option<&BTreeSet<String>>,
) -> Result<(), Error> {
    let entries = scenarios(tests)?;
    let entry = entries
        .get(id)
        .ok_or_else(|| Error::Invalid(format!("{id}: unknown cassette scenario")))?;
    if let Capture::Forbidden(reason) = &entry.capture {
        let rebuild = entry.derivation.as_ref().map_or("", |d| d.rebuild.as_str());
        return Err(Error::Invalid(format!(
            "{id}: live capture forbidden: {reason}; rebuild: {rebuild}"
        )));
    }
    require(
        entry.missing.is_some() || root.join(format!("{id}.yaml")).is_file(),
        || format!("{id}: missing capture without first-capture allowance"),
    )?;
    require(scope.is_none_or(|scope| scope.contains(id)), || {
        format!("{id}: outside recording plan")
    })
}

/// One exact libtest invocation and its complete authorized cassette scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recording {
    pub test: String,
    pub provider: String,
    pub ignored: bool,
    pub scenarios: BTreeSet<String>,
}

/// Select whole tests, never silently widening a requested scenario scope.
/// Ignored tests require an explicit scenario selection or `include_ignored`.
pub fn plan(
    inventory: &Inventory,
    root: &Path,
    selected: Option<&str>,
    include_ignored: bool,
) -> Result<Vec<Recording>, Error> {
    let declarations = scenarios(&inventory.tests)?;
    if let Some(id) = selected {
        authorize(&inventory.tests, root, id, None)?;
    }
    let mut out = Vec::new();
    let mut covered = BTreeSet::new();
    for test in &inventory.tests {
        if test.scenarios.is_empty() || (test.ignored && selected.is_none() && !include_ignored) {
            continue;
        }
        if selected.is_some_and(|id| !test.scenarios.iter().any(|s| s.id == id)) {
            continue;
        }
        let ids: BTreeSet<_> = test.scenarios.iter().map(|s| s.id.clone()).collect();
        if selected.is_some() && ids.len() != 1 {
            return Err(Error::Invalid(format!(
                "{} opens multiple scenarios; select the complete provider plan instead",
                test.name
            )));
        }
        if test
            .scenarios
            .iter()
            .any(|s| matches!(s.capture, Capture::Forbidden(_)))
        {
            continue;
        }
        for id in &ids {
            authorize(&inventory.tests, root, id, Some(&ids))?;
        }
        if ids.iter().all(|id| covered.contains(id)) {
            continue;
        }
        let providers: BTreeSet<_> = ids.iter().filter_map(|id| id.split('/').next()).collect();
        require(providers.len() == 1, || {
            format!("{} spans providers", test.name)
        })?;
        let provider = providers.into_iter().next().unwrap_or_default().to_owned();
        covered.extend(ids.iter().cloned());
        out.push(Recording {
            test: test.name.clone(),
            provider,
            ignored: test.ignored,
            scenarios: ids,
        });
    }
    require(!out.is_empty(), || {
        "selection contains no executable live captures".into()
    })?;
    if selected.is_none() {
        for (id, scenario) in declarations {
            if scenario.capture == Capture::Allowed && !covered.contains(id) {
                let producers: Vec<_> = inventory
                    .tests
                    .iter()
                    .filter(|t| t.scenarios.iter().any(|s| s.id == id))
                    .collect();
                require(
                    !include_ignored && producers.iter().all(|t| t.ignored),
                    || format!("{id}: no safe complete producing test"),
                )?;
            }
        }
    }
    Ok(out)
}
