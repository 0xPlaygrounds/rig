//! `check-packaging`: what the workspace publishes, and what it claims to need.
//!
//! Checks properties no build catches: that the facade tarball ships only its
//! own source and registry documents rather than the workspace root, that every
//! declared dependency is actually used, that the facade feature guard lists
//! every facade feature, and that docs.rs documents every facade feature not
//! excluded here with a reason.
//!
//! Manifest data comes from `cargo metadata` and the file list from
//! `cargo package --list`, which reports what Cargo will do without building a
//! tarball. A full `cargo package` would resolve sibling path dependencies
//! against the registry, which fails on a release PR for versions that are not
//! published yet. Sizes are therefore the uncompressed sum of listed files,
//! which bounds the compressed size from above.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

/// Reads a field, treating a missing one as null, since indexing would panic.
fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}

/// Everything the published `rig` tarball may contain besides `src/` and the
/// images its README displays.
///
/// Stated independently of the manifest's `[package].include` so the guard does
/// not read the value it checks. Cargo synthesizes `Cargo.toml.orig`,
/// `Cargo.lock`, and `.cargo_vcs_info.json`, which the manifest does not list.
const FACADE_ALLOWED_FILES: &[&str] = &[
    ".cargo_vcs_info.json",
    "CHANGELOG.md",
    "Cargo.lock",
    "Cargo.toml",
    "Cargo.toml.orig",
    "LICENSE",
    "MIGRATING.md",
    "README.md",
];

/// Facade features the docs.rs build leaves out, each with the reason. Every
/// other feature except `default` must be in `[package.metadata.docs.rs]
/// features`.
const DOCS_RS_EXCLUDED: &[(&str, &str)] = &[(
    "surrealdb",
    "surrealdb's `diskann` dependency does not compile on the nightly docs.rs uses",
)];

/// crates.io's documented hard cap, on the *compressed* tarball. Not the gate:
/// the thing the gate exists to keep the workspace away from.
const CRATES_IO_CAP: u64 = 10 * 1024 * 1024;

/// The gate, on uncompressed bytes, for every publishable crate. `rig-core` is
/// the largest at roughly 4.3 MiB uncompressed and 0.9 MiB compressed, so this
/// leaves it room to grow while still failing loudly on a structural
/// regression - the facade's pre-allowlist package was 30.2 MiB uncompressed.
/// Raising it is fine, as a reviewed diff, which is the point.
const SIZE_CEILING: u64 = 8 * 1024 * 1024;

/// Dependencies a crate really needs but never names in a checked-in file,
/// with the reason. Code a build script writes into `OUT_DIR` is not on disk
/// to scan, so the one crate with a build script needs its generated code's
/// dependency spelled out here.
const GENERATED_CODE_DEPENDENCIES: &[(&str, &str, &str)] = &[
    (
        "rig-gemini-grpc",
        "tonic-prost",
        "named by the tonic service code build.rs generates into OUT_DIR",
    ),
    (
        "rig-gemini-grpc",
        "prost-types",
        "names the google.protobuf well-known types the generated messages hold",
    ),
    (
        "rig-inspect",
        "bevy_ecs",
        "named by the code `#[derive(Component)]` generates",
    ),
    (
        "rig-inspect",
        "bevy_reflect",
        "named by the code `#[derive(Reflect)]` generates",
    ),
];

pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let metadata = metadata(workspace, &["--no-deps"])?;
    let packages = packages(&metadata)?;

    let mut failures = Vec::new();
    failures.extend(facade_ships_only_its_source(workspace)?);
    failures.extend(packages_stay_under_the_ceiling(workspace, &packages)?);
    failures.extend(dependencies_are_used(&packages)?);
    failures.extend(facade_guard_covers_every_feature(workspace, &packages)?);
    failures.extend(docs_rs_documents_every_feature(&metadata, &packages)?);

    if failures.is_empty() {
        println!(
            "ok: the facade publishes only its allowlist, {} published crates are under \
             {SIZE_CEILING} B uncompressed and name only dependencies their sources use, \
             the facade guard covers every root feature, and docs.rs documents every \
             feature not excluded",
            packages.len()
        );
        return Ok(());
    }
    Err(format!(
        "check-packaging found {} problem(s):\n{}",
        failures.len(),
        failures.join("\n")
    ))
}

/// 1. The published facade contains nothing but its source and manifest.
fn facade_ships_only_its_source(workspace: &Path) -> Result<Vec<String>, String> {
    let files = package_listing(workspace, "rig")?;
    // Liveness. Everything below is an absence assertion, and an absence
    // assertion over an empty listing passes precisely because it learned
    // nothing.
    if !files.iter().any(|file| file == "src/lib.rs") {
        return Err(
            "no usable package listing for `rig` (src/lib.rs absent), so nothing \
                    asserted about its contents would be meaningful"
                .to_string(),
        );
    }
    // A registry renders the README out of the tarball, so an image it points
    // at earns its place; an image nothing points at does not.
    let readme = read(&workspace.join("README.md"))?;
    let mut stowaways: Vec<&str> = files
        .iter()
        .map(String::as_str)
        .filter(|file| {
            !(file.starts_with("src/")
                || FACADE_ALLOWED_FILES.contains(file)
                || (file.starts_with("img/") && readme.contains(file)))
        })
        .collect();
    stowaways.sort_unstable();
    if stowaways.is_empty() {
        return Ok(Vec::new());
    }
    // Dropping the allowlist entirely lists over a thousand files, which would
    // bury the error being reported.
    let shown: Vec<&str> = stowaways.iter().copied().take(20).collect();
    let more = match stowaways.len().saturating_sub(shown.len()) {
        0 => String::new(),
        rest => format!("\n      ... and {rest} more (cargo package -p rig --list)"),
    };
    Ok(vec![format!(
        "  rig: {} file(s) would be published inside the crate that are not part of it. If \
         consumers need them at build time, add them to `[package].include` near the top of \
         the root manifest AND to FACADE_ALLOWED_FILES here; otherwise they should not \
         ship:\n{}{more}",
        stowaways.len(),
        indent(&shown)
    )])
}

/// 2. Every publishable crate stays far below the registry's cap. The facade's
///    allowlist is the sharp guard; this is the backstop for the crates that
///    have no allowlist at all.
fn packages_stay_under_the_ceiling(
    workspace: &Path,
    packages: &BTreeMap<String, Package>,
) -> Result<Vec<String>, String> {
    let mut failures = Vec::new();
    for (name, package) in packages {
        // `--list` prints paths relative to the package's own directory, so
        // each crate is summed from its manifest directory; against the
        // workspace root every crate would measure the root's `src/`.
        let bytes: u64 = package_listing(workspace, name)?
            .iter()
            .filter_map(|file| std::fs::metadata(package.root.join(file)).ok())
            .map(|file| file.len())
            .sum();
        if bytes > SIZE_CEILING {
            failures.push(format!(
                "  {name}: packages to {bytes} B uncompressed, over this repository's \
                 {SIZE_CEILING} B ceiling (crates.io caps the compressed tarball at \
                 {CRATES_IO_CAP} B). Inspect it with: cargo package -p {name} --list"
            ));
        }
    }
    Ok(failures)
}

/// 3. Every non-target dependency a published crate declares is named by its
///    own sources. Target-conditional entries are exempt: a `cfg`-gated
///    dependency is sometimes present only to activate a feature on a crate
///    another dependency pulls in.
fn dependencies_are_used(packages: &BTreeMap<String, Package>) -> Result<Vec<String>, String> {
    let mut failures = Vec::new();
    for (name, package) in packages {
        let sources = package.sources()?;
        if sources.is_empty() {
            return Err(format!(
                "{name}: no source files found; refusing to pass vacuously"
            ));
        }
        let mut unused: Vec<String> = Vec::new();
        for dependency in &package.dependencies {
            let identifier = dependency.identifier();
            let generated = GENERATED_CODE_DEPENDENCIES
                .iter()
                .any(|(package, dep, _)| *package == name && *dep == dependency.name);
            if !generated && !sources.iter().any(|source| names(source, &identifier)) {
                unused.push(format!(
                    "{} (as `{identifier}`)",
                    dependency
                        .rename
                        .clone()
                        .unwrap_or_else(|| dependency.name.clone())
                ));
            }
        }
        if !unused.is_empty() {
            let refs: Vec<&str> = unused.iter().map(String::as_str).collect();
            failures.push(format!(
                "  {name}: `[dependencies]` names {} crate(s) no source under src/ or build.rs \
                 uses. Delete them, or move them to `[dev-dependencies]` if only tests and \
                 examples need them:\n{}",
                unused.len(),
                indent(&refs)
            ));
        }
    }
    Ok(failures)
}

/// 4. The facade additivity fixture enables every root feature.
fn facade_guard_covers_every_feature(
    workspace: &Path,
    packages: &BTreeMap<String, Package>,
) -> Result<Vec<String>, String> {
    let fixture = workspace.join("tests/fixtures/tool_facade/Cargo.toml");
    // The fixture is its own workspace and is `publish = false`, so it is read
    // directly rather than through the published-package map.
    let metadata = metadata(
        workspace,
        &["--no-deps", "--manifest-path", &fixture.to_string_lossy()],
    )?;
    let guard: BTreeSet<String> = field(&metadata, "packages")
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|package| field(field(package, "features"), "all_root").as_array())
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|entry| entry.strip_prefix("rig/").map(str::to_owned))
        .collect();
    if guard.is_empty() {
        return Err(format!(
            "the facade fixture at {} declares no `all_root` feature list",
            fixture.display()
        ));
    }
    let facade = packages
        .get("rig")
        .ok_or_else(|| "the workspace has no `rig` package".to_string())?;
    // `default` is what the guard's other rows already build, and
    // `facade-build-tests` only selects the guard itself.
    let exempt = ["default", "facade-build-tests"];
    let mut missing: Vec<&str> = facade
        .features
        .keys()
        .map(String::as_str)
        .filter(|feature| !exempt.contains(feature) && !guard.contains(*feature))
        .collect();
    missing.sort_unstable();
    if missing.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![format!(
        "  rig: {} facade feature(s) are outside the additivity guard; add them to `all_root` \
         in tests/fixtures/tool_facade/Cargo.toml:\n{}",
        missing.len(),
        indent(&missing)
    )])
}

/// 5. docs.rs documents every facade feature except `default` and the
///    exclusions in [`DOCS_RS_EXCLUDED`].
fn docs_rs_documents_every_feature(
    metadata: &Value,
    packages: &BTreeMap<String, Package>,
) -> Result<Vec<String>, String> {
    let facade = packages
        .get("rig")
        .ok_or_else(|| "the workspace has no `rig` package".to_string())?;
    let docs_rs = field(metadata, "packages")
        .as_array()
        .into_iter()
        .flatten()
        .find(|package| field(package, "name").as_str() == Some("rig"))
        // TOML reads `[package.metadata.docs.rs]` as a `docs` table holding `rs`.
        .map(|package| field(field(field(package, "metadata"), "docs"), "rs"))
        .ok_or_else(|| "cargo metadata has no `rig` package".to_string())?;
    let documented: BTreeSet<String> = field(docs_rs, "features")
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let all_features = field(docs_rs, "all-features").as_bool().unwrap_or(false);
    let features: BTreeSet<&str> = facade.features.keys().map(String::as_str).collect();
    let excluded: BTreeSet<&str> = DOCS_RS_EXCLUDED.iter().map(|(name, _)| *name).collect();
    Ok(docs_rs_gaps(
        &features,
        &documented,
        all_features,
        &excluded,
    ))
}

/// The mismatches between the facade's features and its docs.rs feature list.
fn docs_rs_gaps(
    features: &BTreeSet<&str>,
    documented: &BTreeSet<String>,
    all_features: bool,
    excluded: &BTreeSet<&str>,
) -> Vec<String> {
    let mut failures = Vec::new();
    if all_features {
        // `all-features` documents everything, which is right once nothing is excluded.
        if !excluded.is_empty() {
            failures.push(format!(
                "  rig: `[package.metadata.docs.rs] all-features = true` also builds the excluded \
                 feature(s) {}; list the features instead",
                excluded.iter().copied().collect::<Vec<_>>().join(", ")
            ));
        }
        return failures;
    }
    let missing: Vec<&str> = features
        .iter()
        .copied()
        .filter(|feature| {
            *feature != "default" && !excluded.contains(feature) && !documented.contains(*feature)
        })
        .collect();
    if !missing.is_empty() {
        failures.push(format!(
            "  rig: {} facade feature(s) are missing from `[package.metadata.docs.rs] features` \
             in the root manifest; add them, or exclude them in DOCS_RS_EXCLUDED with a reason:\n{}",
            missing.len(),
            indent(&missing)
        ));
    }
    let stray: Vec<&str> = documented
        .iter()
        .map(String::as_str)
        .filter(|feature| excluded.contains(feature) || !features.contains(feature))
        .collect();
    if !stray.is_empty() {
        failures.push(format!(
            "  rig: `[package.metadata.docs.rs] features` lists excluded or unknown feature(s):\n{}",
            indent(&stray)
        ));
    }
    failures
}

/// Whether `source` uses `identifier` as a crate name: a whole word, not a
/// substring of a longer path segment (`serde` must not match `serde_json`).
fn names(source: &str, identifier: &str) -> bool {
    let boundary =
        |byte: Option<&u8>| !matches!(byte, Some(b) if b.is_ascii_alphanumeric() || *b == b'_');
    let bytes = source.as_bytes();
    source.match_indices(identifier).any(|(at, _)| {
        boundary(at.checked_sub(1).and_then(|before| bytes.get(before)))
            && boundary(bytes.get(at + identifier.len()))
    })
}

fn indent(lines: &[&str]) -> String {
    lines
        .iter()
        .map(|line| format!("      {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The files `cargo package` would publish for `name`, relative to that
/// package's own directory.
///
/// `--allow-dirty` so this is runnable on a working tree, and so it keeps
/// seeing untracked files: those are exactly what the contents check is here
/// to catch. CI checks out clean, where the flag is a no-op.
fn package_listing(workspace: &Path, name: &str) -> Result<Vec<String>, String> {
    let listing = cargo(
        workspace,
        &["package", "--list", "--quiet", "--allow-dirty", "-p", name],
    )?;
    Ok(listing
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

struct Package {
    root: std::path::PathBuf,
    dependencies: Vec<Dependency>,
    features: BTreeMap<String, Vec<String>>,
}

struct Dependency {
    name: String,
    rename: Option<String>,
}

impl Dependency {
    fn identifier(&self) -> String {
        self.rename
            .clone()
            .unwrap_or_else(|| self.name.clone())
            .replace('-', "_")
    }
}

impl Package {
    /// Every Rust source Cargo compiles into the library, plus its build script.
    fn sources(&self) -> Result<Vec<String>, String> {
        let mut sources = Vec::new();
        let build = self.root.join("build.rs");
        if build.is_file() {
            sources.push(read(&build)?);
        }
        let src = self.root.join("src");
        if src.is_dir() {
            for path in crate::support::files_under(&src, Some("rs"))? {
                sources.push(read(&path)?);
            }
        }
        Ok(sources)
    }
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))
}

/// Published workspace members, keyed by package name.
fn packages(metadata: &Value) -> Result<BTreeMap<String, Package>, String> {
    let listed = field(metadata, "packages")
        .as_array()
        .ok_or_else(|| "cargo metadata reported no packages".to_string())?;
    let mut packages = BTreeMap::new();
    for package in listed {
        // `publish = []` is the manifest's own statement that a package is not
        // distributed; only what consumers can depend on is in scope here.
        if field(package, "publish")
            .as_array()
            .is_some_and(|deny| deny.is_empty())
        {
            continue;
        }
        let name = field(package, "name")
            .as_str()
            .ok_or_else(|| "a package has no name".to_string())?
            .to_owned();
        let manifest = Path::new(
            field(package, "manifest_path")
                .as_str()
                .ok_or_else(|| format!("{name} has no manifest path"))?,
        );
        let dependencies = field(package, "dependencies")
            .as_array()
            .ok_or_else(|| format!("{name} has no dependency list"))?
            .iter()
            .filter(|dependency| {
                field(dependency, "kind").is_null() && field(dependency, "target").is_null()
            })
            .map(|dependency| Dependency {
                name: field(dependency, "name")
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                rename: field(dependency, "rename").as_str().map(str::to_owned),
            })
            .collect();
        let features = field(package, "features")
            .as_object()
            .map(|features| {
                features
                    .iter()
                    .map(|(feature, enables)| {
                        let enables = enables
                            .as_array()
                            .map(|entries| {
                                entries
                                    .iter()
                                    .filter_map(|entry| entry.as_str().map(str::to_owned))
                                    .collect()
                            })
                            .unwrap_or_default();
                        (feature.clone(), enables)
                    })
                    .collect()
            })
            .unwrap_or_default();
        packages.insert(
            name,
            Package {
                root: manifest.parent().unwrap_or(manifest).to_path_buf(),
                dependencies,
                features,
            },
        );
    }
    if packages.is_empty() {
        return Err("cargo metadata reported no published packages".to_string());
    }
    Ok(packages)
}

fn metadata(workspace: &Path, arguments: &[&str]) -> Result<Value, String> {
    let mut command = vec!["metadata", "--format-version", "1"];
    command.extend_from_slice(arguments);
    serde_json::from_str(&cargo(workspace, &command)?)
        .map_err(|error| format!("could not read cargo metadata: {error}"))
}

fn cargo(workspace: &Path, arguments: &[&str]) -> Result<String, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    crate::support::output(workspace, &cargo, arguments)
}

#[cfg(test)]
mod tests;
