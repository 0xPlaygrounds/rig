//! Validate maintained scenario correspondences against current files and,
//! optionally, a fresh nextest listing. This does not certify execution/parity.
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};
#[cfg(test)]
mod tests;
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
type Result<T> = std::result::Result<T, Error>;
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
fn text<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("missing string {k}")))
}
fn file(root: &Path, name: &str) -> Result<()> {
    if Path::new(name)
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe catalog path: {name}")));
    }
    let target = root
        .join(name)
        .canonicalize()
        .map_err(|e| invalid(format!("missing catalog file {name}: {e}")))?;
    if !target.starts_with(root.canonicalize()?) || !target.is_file() {
        return Err(invalid(format!(
            "catalog path escapes root or is not a file: {name}"
        )));
    }
    Ok(())
}
fn references(root: &Path, v: &Value) -> Result<()> {
    match v {
        Value::Object(m) => {
            for (key, value) in m {
                if matches!(
                    key.as_str(),
                    "path" | "source" | "family_contract" | "contract"
                ) {
                    let path = value
                        .as_str()
                        .ok_or_else(|| invalid(format!("{key} must be a file path string")))?;
                    file(root, path)?;
                }
                if key == "external_helper_source" {
                    file(root, text(value, "path")?)?;
                }
                if key.contains("sha256")
                    || matches!(
                        key.as_str(),
                        "scoped_parity"
                            | "candidate_execution_evidence"
                            | "classification_review"
                            | "review_state"
                            | "baseline_execution_result"
                    )
                {
                    return Err(invalid(format!("obsolete historical-proof field: {key}")));
                }
                references(root, value)?;
            }
        }
        Value::Array(a) => {
            for value in a {
                references(root, value)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn compiled(list: &Value) -> Result<BTreeSet<String>> {
    let suites = list["rust-suites"]
        .as_object()
        .ok_or_else(|| invalid("expected full nextest JSON listing"))?;
    let mut tests = BTreeSet::new();
    for suite in suites.values().filter(|s| s["package-name"] == "rig") {
        let binary = text(suite, "binary-name")?;
        for (name, test) in suite["testcases"]
            .as_object()
            .ok_or_else(|| invalid("missing testcases"))?
        {
            if test.pointer("/filter-match/status").and_then(Value::as_str) == Some("mismatch")
                && !(test.pointer("/filter-match/reason").and_then(Value::as_str)
                    == Some("ignored")
                    && test["ignored"] == true)
            {
                return Err(invalid("use an unfiltered nextest listing"));
            }
            tests.insert(format!("rig::{binary}::{name}"));
        }
    }
    if tests.is_empty() {
        return Err(invalid("listing contains no rig tests"));
    }
    Ok(tests)
}
fn validate(
    root: &Path,
    catalog: &Value,
    listing: Option<&Value>,
) -> Result<(usize, usize, usize)> {
    if catalog["schema"] != 1 {
        return Err(invalid("unsupported scenario catalog schema"));
    }
    let rows = catalog["scenarios"]
        .as_array()
        .ok_or_else(|| invalid("missing scenarios"))?;
    if rows.is_empty() {
        return Err(invalid("empty scenario catalog"));
    }
    let compiled = listing.map(compiled).transpose()?;
    let mut originals = BTreeSet::new();
    let mut natives = BTreeSet::new();
    let mut unlisted_unmapped = 0;
    for row in rows {
        let id = text(row, "id")?;
        if !originals.insert(id) {
            return Err(invalid(format!("duplicate original scenario: {id}")));
        }
        if !matches!(
            text(row, "classification")?,
            "agent" | "shared_provider" | "infrastructure" | "unclassified"
        ) {
            return Err(invalid(format!("invalid classification: {id}")));
        }
        file(root, text(row, "source")?)?;
        let configuration = text(row, "configuration")?;
        if !catalog
            .get("configurations")
            .and_then(|c| c.get(configuration))
            .is_some_and(Value::is_object)
        {
            return Err(invalid(format!("unknown configuration: {id}")));
        }
        references(root, row)?;
        if let Some(tests) = &compiled
            && !tests.contains(id)
        {
            if row.get("ecs").is_some_and(|v| !v.is_null()) {
                return Err(invalid(format!(
                    "original scenario is not compiled: {id}; regenerate the full rig listing with --features bedrock"
                )));
            }
            unlisted_unmapped += 1;
        }
        if let Some(ecs) = row.get("ecs").filter(|v| !v.is_null()) {
            file(root, text(ecs, "source")?)?;
            let native = format!("rig::{}::{}", text(ecs, "binary")?, text(ecs, "test")?);
            if !natives.insert(native.clone()) {
                return Err(invalid(format!("duplicate native mapping: {native}")));
            }
            if compiled
                .as_ref()
                .is_some_and(|tests| !tests.contains(&native))
            {
                return Err(invalid(format!(
                    "native scenario is not compiled: {native}"
                )));
            }
        }
    }
    if let Some(shared) = catalog.get("shared_provider_correspondences") {
        let shared = shared
            .as_array()
            .ok_or_else(|| invalid("shared-provider correspondences must be an array"))?;
        let mut seen = BTreeSet::new();
        for pair in shared {
            let original = text(pair, "original")?;
            let counterpart = text(pair, "counterpart")?;
            if !seen.insert((original, counterpart))
                || !rows
                    .iter()
                    .any(|row| row["id"] == original && row["classification"] == "shared_provider")
            {
                return Err(invalid(
                    "duplicate or incorrectly classified shared-provider correspondence",
                ));
            }
            if compiled
                .as_ref()
                .is_some_and(|tests| !tests.contains(original) || !tests.contains(counterpart))
            {
                return Err(invalid(format!(
                    "shared-provider correspondence is not compiled: {original} -> {counterpart}"
                )));
            }
        }
    }
    Ok((originals.len(), natives.len(), unlisted_unmapped))
}
pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<()> {
    let listing = match args.as_slice() {
        [] => None,
        [path] => Some(serde_json::from_slice(&fs::read(path)?)?),
        _ => {
            return Err(invalid(
                "usage: cargo xtask check-ecs-scenarios [full-nextest-list.json]",
            ));
        }
    };
    let catalog = serde_json::from_slice(&fs::read(root.join("tests/ecs_parity/scenarios.json"))?)?;
    let (originals, natives, unlisted) = validate(root, &catalog, listing.as_ref())?;
    if listing.is_some() {
        println!(
            "{unlisted} unmapped source scenarios are outside this compiled listing; no execution claim"
        );
    }
    println!(
        "{originals} original scenarios, {natives} native mappings: current {} checked; execution and exhaustive parity are not certified",
        if listing.is_some() {
            "files and compiled registrations"
        } else {
            "file references"
        }
    );
    Ok(())
}
