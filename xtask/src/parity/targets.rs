//! Cargo target discovery independent of the caller's compiled-test selection.
//!
//! This is a target-registration guard, not feature, scenario, or execution
//! coverage. Conditional source tests still need source/listing reconciliation.

use super::*;
use serde_json::json;
use std::collections::BTreeSet;

#[cfg(test)]
mod tests;

const METADATA: &[&str] = &[
    "cargo",
    "metadata",
    "--locked",
    "--offline",
    "--no-deps",
    "--format-version",
    "1",
];

fn metadata(root: &Path) -> Result<(Value, String)> {
    let raw = capture(root, METADATA)?;
    let value = serde_json::from_str(&raw)?;
    Ok((value, raw))
}

fn workspace_packages(metadata: &Value) -> Result<Vec<&Value>> {
    let members = strings(metadata, "workspace_members")?;
    let mut packages = BTreeMap::new();
    for package in array(metadata, "packages")? {
        let id = text(package, "id")?;
        if members.iter().any(|member| member == id) && packages.insert(id, package).is_some() {
            return Err(format!("duplicate workspace package {id}").into());
        }
    }
    if packages.is_empty() || packages.len() != members.len() {
        return Err("metadata does not resolve every workspace member exactly once".into());
    }
    let mut packages: Vec<_> = packages.into_values().collect();
    packages.sort_by_key(|package| text(package, "name").unwrap_or_default());
    Ok(packages)
}

fn relative(root: &Path, path: &str) -> Result<String> {
    Ok(Path::new(path)
        .canonicalize()?
        .strip_prefix(root.canonicalize()?)?
        .to_str()
        .ok_or("non-UTF8 workspace path")?
        .replace('\\', "/"))
}

fn test_kind(target: &Value) -> Result<Option<String>> {
    let enabled = field(target, "test")
        .as_bool()
        .ok_or("missing target test flag")?;
    let kinds = strings(target, "kind")?;
    if !enabled {
        return Ok(None);
    }
    // Cargo includes examples and benches when their test flag is explicitly
    // enabled. Doctests and build scripts do not become nextest test binaries.
    let kind = kinds.iter().find_map(|kind| match kind.as_str() {
        "lib" | "rlib" | "dylib" | "cdylib" | "staticlib" | "proc-macro" => Some("lib"),
        "test" => Some("test"),
        "bin" => Some("bin"),
        "example" => Some("example"),
        "bench" => Some("bench"),
        _ => None,
    });
    Ok(kind.map(str::to_owned))
}

fn inventory(root: &Path, metadata: &Value) -> Result<Value> {
    let defaults = strings(metadata, "workspace_default_members")?;
    let mut packages = Vec::new();
    for package in workspace_packages(metadata)? {
        let manifest = relative(root, text(package, "manifest_path")?)?;
        let mut targets = Vec::new();
        for target in array(package, "targets")? {
            let source = relative(root, text(target, "src_path")?)?;
            targets.push(json!({
                "name": text(target, "name")?, "kind": strings(target, "kind")?,
                "source": source, "source_sha256": hash(&fs::read(root.join(&source))?),
                "test": field(target, "test"), "doctest": field(target, "doctest"),
                "required_features": target.get("required-features").cloned().unwrap_or(json!([])),
                "nextest_kind": test_kind(target)?,
                "execution_result": null,
            }));
        }
        targets.sort_by_key(|target| (target["name"].to_string(), target["kind"].to_string()));
        packages.push(json!({
            "name": text(package, "name")?, "manifest": manifest,
            "manifest_sha256": hash(&fs::read(root.join(&manifest))?),
            "default_member": defaults.iter().any(|id| id == text(package, "id").unwrap_or_default()),
            "declared_features": field(package, "features"), "targets": targets,
        }));
    }
    Ok(json!({
        "schema": 1, "evidence_kind": "workspace_target_discovery_not_coverage",
        "scope": "all Cargo workspace members and targets, including non-default members and examples",
        "limitations": [
            "Feature declarations and target requirements are inventory inputs, not an executed feature matrix.",
            "Target roots are hashed; this inventory does not trace cfg gates, helper obligations, or nested modules.",
            "An empty compiled target can hide conditionally excluded source tests; reconcile those separately.",
            "Doctests, targets with test=false, custom harnesses and platform-specific execution require separate runner evidence."
        ],
        "packages": packages,
    }))
}

fn reconcile(metadata: &Value, package_name: &str, listing: &Value) -> Result<Value> {
    let packages = workspace_packages(metadata)?;
    let matching: Vec<_> = packages
        .into_iter()
        .filter(|package| field(package, "name") == package_name)
        .collect();
    let [package] = matching.as_slice() else {
        return Err(format!("expected exactly one workspace package {package_name}").into());
    };
    let mut expected = BTreeSet::new();
    for target in array(package, "targets")? {
        if let Some(kind) = test_kind(target)? {
            let key = (text(target, "name")?.to_owned(), kind);
            if !expected.insert(key) {
                return Err("duplicate Cargo test target".into());
            }
        }
    }
    if expected.is_empty() {
        return Err("package has no default nextest test targets; audit its other runners".into());
    }
    let suites = field(listing, "rust-suites")
        .as_object()
        .ok_or("missing rust-suites")?;
    let mut actual = BTreeSet::new();
    let mut cases = 0;
    let mut ignored = 0;
    for suite in suites.values() {
        if text(suite, "package-name")? != package_name {
            return Err("listing contains another package; use a package-specific listing".into());
        }
        if field(suite, "package-id") != field(package, "id") {
            return Err("listing package identity differs from Cargo metadata".into());
        }
        if field(suite, "status") != "listed" {
            return Err("test target was skipped instead of listed".into());
        }
        let key = (
            text(suite, "binary-name")?.to_owned(),
            text(suite, "kind")?.to_owned(),
        );
        if !actual.insert(key) {
            return Err("duplicate listed test target".into());
        }
        let tests = field(suite, "testcases")
            .as_object()
            .ok_or("missing testcases")?;
        for test in tests.values() {
            let is_ignored = field(test, "ignored")
                .as_bool()
                .ok_or("missing ignored state")?;
            let filter = field(test, "filter-match");
            if field(filter, "status") != "matches"
                && !(is_ignored
                    && field(filter, "status") == "mismatch"
                    && field(filter, "reason") == "ignored")
            {
                return Err(
                    "listing filtered a scenario; use an unfiltered package listing".into(),
                );
            }
            ignored += usize::from(is_ignored);
        }
        cases += tests.len();
    }
    if field(listing, "test-count").as_u64() != Some(cases as u64) {
        return Err("listing test-count differs from enumerated testcases".into());
    }
    let missing: Vec<_> = expected.difference(&actual).collect();
    let unexpected: Vec<_> = actual.difference(&expected).collect();
    Ok(json!({
        "schema": 1, "evidence_kind": "target_registration_only",
        "package": package_name, "expected_targets": expected.len(), "listed_targets": actual.len(),
        "missing_targets": missing, "unexpected_targets": unexpected,
        "complete_target_registration": missing.is_empty() && unexpected.is_empty(),
        "compiled_testcases": cases, "ignored_testcases": ignored,
        "execution_result": null, "parity_verdict": null,
        "limitations": "Checks every default nextest target, including targets with required-features. Enable those features before claiming full target registration. Does not establish source, feature, fixture, assertion or execution coverage, or authenticate a supplied listing's original command/source contents."
    }))
}

pub(crate) fn run(args: Vec<String>) -> Result<()> {
    let (mode, root, output, check) = match args.as_slice() {
        [mode, root, output] if mode == "inventory" => (mode, root, output, None),
        [mode, root, package, listing, output] if mode == "check" =>
            (mode, root, output, Some((package, listing))),
        _ => return Err("usage: parity-targets inventory <root> <output> | check <root> <package> <listing> <output>".into()),
    };
    let root = Path::new(root).canonicalize()?;
    let output = Path::new(output);
    let evidence = output
        .parent()
        .ok_or("output needs a parent directory")?
        .join("evidence/target-inventory");
    let (metadata, raw) = metadata(&root)?;
    let mut result = match check {
        Some((package, listing)) => {
            let raw = fs::read(listing)?;
            let mut result = reconcile(&metadata, package, &serde_json::from_slice(&raw)?)?;
            put(&mut result, &["listing_sha256"], json!(hash(&raw)))?;
            put(
                &mut result,
                &["listing_artifact"],
                json!(format!(
                    "evidence/target-inventory/{}",
                    store_raw(&evidence, &raw, "json")?
                )),
            )?;
            result
        }
        None => inventory(&root, &metadata)?,
    };
    put(&mut result, &["metadata_command"], json!(METADATA))?;
    put(
        &mut result,
        &["metadata_sha256"],
        json!(hash(raw.as_bytes())),
    )?;
    put(
        &mut result,
        &["source_revision"],
        json!(capture(&root, &["git", "rev-parse", "HEAD"])?.trim()),
    )?;
    put(
        &mut result,
        &["source_status"],
        json!(
            capture(
                &root,
                &["git", "status", "--porcelain", "--untracked-files=all"]
            )?
            .trim()
        ),
    )?;
    put(
        &mut result,
        &["toolchain"],
        json!(capture(&root, &["rustc", "-Vv"])?.trim()),
    )?;
    put(
        &mut result,
        &["metadata_artifact"],
        json!(format!(
            "evidence/target-inventory/{}",
            store_raw(&evidence, raw.as_bytes(), "json")?
        )),
    )?;
    write(output, &result)?;
    if mode == "check" && field(&result, "complete_target_registration") != true {
        return Err(format!(
            "incomplete package target listing; see {}",
            output.display()
        )
        .into());
    }
    Ok(())
}
