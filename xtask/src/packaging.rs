//! `check-packaging`: what the workspace publishes, and what it claims to need.
//!
//! Checks three properties no build catches: that the facade tarball ships only
//! its own source and registry documents rather than the workspace root, that
//! every declared dependency is actually used, and that the facade feature guard
//! lists every facade feature.
//!
//! Manifest data comes from `cargo metadata` and the file list from
//! `cargo package --list`, which reports what Cargo will do without building a
//! tarball. A full `cargo package` would resolve sibling path dependencies
//! against the registry, which fails on a release PR for versions that are not
//! published yet. Sizes are therefore the uncompressed sum of listed files,
//! which bounds the compressed size from above.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

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
const GENERATED_CODE_DEPENDENCIES: &[(&str, &str, &str)] = &[(
    "rig-gemini-grpc",
    "tonic-prost",
    "named by the tonic service code build.rs generates into OUT_DIR",
)];

pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let metadata = metadata(workspace, &["--no-deps"])?;
    let packages = packages(&metadata)?;

    let mut failures = Vec::new();
    failures.extend(facade_ships_only_its_source(workspace)?);
    failures.extend(packages_stay_under_the_ceiling(workspace, &packages)?);
    failures.extend(dependencies_are_used(&packages)?);
    failures.extend(facade_guard_covers_every_feature(workspace, &packages)?);

    if failures.is_empty() {
        println!(
            "ok: the facade publishes only its allowlist, {} published crates are under \
             {SIZE_CEILING} B uncompressed and name only dependencies their sources use, and \
             the facade guard covers every root feature",
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
        let mut stack = vec![self.root.join("src")];
        while let Some(directory) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries {
                let path = entry
                    .map_err(|error| format!("{}: {error}", directory.display()))?
                    .path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|extension| extension == "rs") {
                    sources.push(read(&path)?);
                }
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
    let output = Command::new(std::env::var("CARGO").as_deref().unwrap_or("cargo"))
        .args(arguments)
        .current_dir(workspace)
        .output()
        .map_err(|error| format!("could not run cargo {}: {error}", arguments.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "cargo {} failed:\n{}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| {
        format!(
            "cargo {} produced non-UTF-8 output: {error}",
            arguments.join(" ")
        )
    })
}

#[cfg(test)]
mod tests;
