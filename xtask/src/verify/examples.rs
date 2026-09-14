//! Build each example in its own Cargo invocation: batching packages can
//! unify provider features and conceal a missing declaration.
use super::*;
use std::{collections::BTreeSet, fs, path::PathBuf};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Example {
    manifest: PathBuf,
    flag: String,
    name: String,
    features: Vec<String>,
}

fn collect(package: &Value, standalone: bool, examples: &mut BTreeSet<Example>) -> Result<()> {
    let manifest = PathBuf::from(
        package["manifest_path"]
            .as_str()
            .ok_or_else(|| invalid("example package has no manifest"))?,
    );
    for target in package["targets"]
        .as_array()
        .ok_or_else(|| invalid("example package has no targets"))?
    {
        let kinds = target["kind"]
            .as_array()
            .ok_or_else(|| invalid("example has no kind"))?;
        let flag = if kinds.iter().any(|k| k == "example") {
            "--example"
        } else if standalone && kinds.iter().any(|k| k == "bin") {
            "--bin"
        } else if standalone
            && kinds
                .iter()
                .any(|k| k == "lib" || k == "cdylib" || k == "rlib")
        {
            "--lib"
        } else {
            continue;
        };
        let features = target["required-features"]
            .as_array()
            .map(|fs| {
                fs.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        examples.insert(Example {
            manifest: manifest.clone(),
            flag: flag.into(),
            name: target["name"]
                .as_str()
                .ok_or_else(|| invalid("example has no name"))?
                .into(),
            features,
        });
    }
    Ok(())
}

fn discover(root: &Path) -> Result<BTreeSet<Example>> {
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)?;
    let packages = metadata
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("missing packages"))?;
    let mut manifests = BTreeSet::new();
    let mut examples = BTreeSet::new();
    for package in packages {
        let manifest = PathBuf::from(
            package["manifest_path"]
                .as_str()
                .ok_or_else(|| invalid("missing manifest"))?,
        );
        collect(
            package,
            manifest.starts_with(root.join("examples")),
            &mut examples,
        )?;
        manifests.insert(manifest);
    }
    // Excluded standalone workspaces (currently Discord) keep their own
    // dependency graph and lockfile; exclusion is not a reason to skip them.
    for entry in fs::read_dir(root.join("examples"))? {
        let manifest = entry?.path().join("Cargo.toml");
        if !manifest.is_file() || manifests.contains(&manifest) {
            continue;
        }
        let path = manifest
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 example path"))?;
        // These workspaces deliberately do not share the repository lock.
        // Resolve a missing private lock explicitly before the locked build;
        // a warm developer checkout must not be a prerequisite for CI.
        let lock = manifest.with_file_name("Cargo.lock");
        if !lock.is_file() {
            output(
                root,
                "cargo",
                &["generate-lockfile", "--manifest-path", path],
            )?;
        }
        let metadata: Value = serde_json::from_str(&output(
            root,
            "cargo",
            &[
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                path,
            ],
        )?)?;
        let package = metadata
            .get("packages")
            .and_then(Value::as_array)
            .and_then(|ps| ps.iter().find(|p| p["manifest_path"] == path))
            .ok_or_else(|| invalid(format!("{path}: excluded example missing from metadata")))?;
        collect(package, true, &mut examples)?;
    }
    if examples.is_empty() {
        return Err(invalid("no examples discovered"));
    }
    Ok(examples)
}

pub(super) fn run(root: &Path, shard: usize) -> Result<()> {
    const SHARDS: usize = 4;
    if shard >= SHARDS {
        return Err(invalid("invalid example shard"));
    }
    let examples = discover(root)?;
    println!(
        "Discovered {} independent example targets; shard {shard}/{SHARDS}",
        examples.len()
    );
    let mut failures = Vec::new();
    let mut count = 0;
    for (index, example) in examples.into_iter().enumerate() {
        if index % SHARDS != shard {
            continue;
        }
        count += 1;
        println!(
            "Example: {} {} ({})",
            example.flag,
            example.name,
            example.manifest.display()
        );
        let mut command = Command::new("cargo");
        command
            .current_dir(root)
            .args(["build", "--locked", "--manifest-path"])
            .arg(&example.manifest)
            .arg(&example.flag)
            .env("RIG_PROVIDER_TEST_MODE", "replay");
        if example.flag != "--lib" {
            command.arg(&example.name);
        }
        if !example.features.is_empty() {
            command.args(["--features", &example.features.join(",")]);
        }
        if !command.status()?.success() {
            failures.push(format!("{}: {}", example.manifest.display(), example.name));
        }
    }
    if count == 0 {
        return Err(invalid("example shard selected no targets"));
    }
    if !failures.is_empty() {
        return Err(invalid(format!(
            "example build failures: {}",
            failures.join("; ")
        )));
    }
    println!("PASS: {count} examples linked independently");
    Ok(())
}

#[cfg(test)]
mod tests;
