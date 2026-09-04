//! The Bevy pin: every `bevy_*` dependency of the workspace is a crates.io
//! version — never `git =`, never `path =` — and the lock file resolves each
//! to the release rig-ecs was proved on. Moved here from the Bevy host
//! fixture's bound test when the fixture became `crates/rig-ecs`'s `bus`
//! suite; the manifest it reads is now the workspace's.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The workspace manifest pins every `bevy_*` crate to the same crates.io
/// version, and the lock file resolves each to it.
#[test]
fn bevy_is_pinned_to_one_crates_io_release() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(workspace_root().join("Cargo.toml"))?;
    let pins = bevy_pins(&manifest);
    if pins.is_empty() {
        return Err("the workspace manifest declares no bevy_* dependency".into());
    }
    for (name, pin) in &pins {
        match pin {
            BevyPin::Version(version) if version == BEVY_VERSION => {}
            BevyPin::Version(version) => {
                return Err(format!("{name} pins `{version}`, not `{BEVY_VERSION}`").into());
            }
            BevyPin::Git => return Err(format!("{name} comes from git, not crates.io").into()),
            BevyPin::Path => return Err(format!("{name} comes from a path, not crates.io").into()),
            BevyPin::Unpinned => return Err(format!("{name} declares no version").into()),
        }
    }
    let lock = std::fs::read_to_string(workspace_root().join("Cargo.lock"))?;
    let resolved = bevy_lock_versions(&lock);
    if resolved.is_empty() {
        return Err("the lock file resolves no bevy_* package".into());
    }
    for (name, version) in &resolved {
        if version != BEVY_VERSION {
            return Err(format!(
                "the lock file resolves {name} to `{version}`, not `{BEVY_VERSION}`"
            )
            .into());
        }
    }
    Ok(())
}

/// rig-ecs's manifest takes every Bevy crate from the workspace table, so
/// the pin above is the only one.
#[test]
fn rig_ecs_takes_bevy_from_the_workspace() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(workspace_root().join("crates/rig-ecs/Cargo.toml"))?;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with("bevy_") && !line.contains("workspace = true") {
            return Err(format!("rig-ecs pins Bevy itself: `{line}`").into());
        }
    }
    Ok(())
}

/// The crates.io release rig-ecs was proved on.
const BEVY_VERSION: &str = "0.19.1";

/// How a `bevy_*` dependency is declared.
#[derive(Debug, PartialEq, Eq)]
enum BevyPin {
    Version(String),
    Git,
    Path,
    Unpinned,
}

fn classify(text: &str) -> BevyPin {
    fn quoted_value(text: &str, key: &str) -> Option<String> {
        let at = text.find(key)?;
        let rest = text[at + key.len()..].trim_start();
        let rest = rest.strip_prefix('=')?.trim_start();
        let rest = rest.strip_prefix('"')?;
        rest.split('"').next().map(str::to_owned)
    }
    if quoted_value(text, "git").is_some() {
        return BevyPin::Git;
    }
    if quoted_value(text, "path").is_some() {
        return BevyPin::Path;
    }
    // `bevy_ecs = "0.19.1"` (a bare string) or `version = "0.19.1"`.
    let bare = text
        .trim()
        .strip_prefix('"')
        .and_then(|rest| rest.split('"').next())
        .map(str::to_owned);
    match quoted_value(text, "version").or(bare) {
        Some(version) => BevyPin::Version(version),
        None => BevyPin::Unpinned,
    }
}

/// Every `bevy_*` dependency in a manifest with how it is pinned, whether
/// written as an inline table (`bevy_ecs = { version = "…", … }`), a bare
/// string, or a dependency table (`[dependencies.bevy_ecs]` followed by
/// its keys).
fn bevy_pins(manifest: &str) -> Vec<(String, BevyPin)> {
    let mut pins = Vec::new();
    let mut table: Option<(String, String)> = None;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            if let Some((name, body)) = table.take() {
                pins.push((name, classify(&body)));
            }
            if let Some(name) = line
                .strip_prefix("[dependencies.")
                .and_then(|rest| rest.strip_suffix(']'))
                .filter(|name| name.starts_with("bevy_"))
            {
                table = Some((name.to_owned(), String::new()));
            }
            continue;
        }
        if let Some((_, body)) = table.as_mut() {
            body.push_str(line);
            body.push('\n');
            continue;
        }
        if line.starts_with("bevy_")
            && let Some((name, value)) = line.split_once('=')
        {
            pins.push((name.trim().to_owned(), classify(value)));
        }
    }
    if let Some((name, body)) = table.take() {
        pins.push((name, classify(&body)));
    }
    pins
}

/// Every `bevy_*` package in a lock file with its resolved version.
fn bevy_lock_versions(lock: &str) -> Vec<(String, String)> {
    let mut versions = Vec::new();
    let mut current: Option<String> = None;
    for line in lock.lines().map(str::trim) {
        if line == "[[package]]" {
            current = None;
            continue;
        }
        if let Some(name) = line.strip_prefix("name = \"") {
            let name = name.trim_end_matches('"');
            current = name.starts_with("bevy_").then(|| name.to_owned());
            continue;
        }
        if let (Some(name), Some(version)) = (current.take(), line.strip_prefix("version = \"")) {
            versions.push((name, version.trim_end_matches('"').to_owned()));
        }
    }
    versions
}

#[test]
fn bevy_pins_reads_every_manifest_form() {
    let manifest = r#"
[dependencies]
rig-core = { path = "x" }
bevy_ecs = { version = "0.19.1", default-features = false }
bevy_tasks = "0.19.1"
bevy_app = { git = "https://github.com/bevyengine/bevy", rev = "823bcc935" }

[dependencies.bevy_asset]
version = "0.19.1"
features = ["multi_threaded"]

[dependencies.bevy_reflect]
path = "../bevy/crates/bevy_reflect"

[dependencies.bevy_time]
features = ["x"]
"#;
    assert_eq!(
        bevy_pins(manifest),
        vec![
            ("bevy_ecs".to_owned(), BevyPin::Version("0.19.1".to_owned())),
            (
                "bevy_tasks".to_owned(),
                BevyPin::Version("0.19.1".to_owned())
            ),
            ("bevy_app".to_owned(), BevyPin::Git),
            (
                "bevy_asset".to_owned(),
                BevyPin::Version("0.19.1".to_owned())
            ),
            ("bevy_reflect".to_owned(), BevyPin::Path),
            ("bevy_time".to_owned(), BevyPin::Unpinned),
        ]
    );
}

#[test]
fn bevy_lock_versions_reads_packages() {
    let lock = r#"
[[package]]
name = "bevy_ecs"
version = "0.19.1"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "serde"
version = "1.0.0"

[[package]]
name = "bevy_tasks"
version = "0.19.0"
"#;
    assert_eq!(
        bevy_lock_versions(lock),
        vec![
            ("bevy_ecs".to_owned(), "0.19.1".to_owned()),
            ("bevy_tasks".to_owned(), "0.19.0".to_owned()),
        ]
    );
}
