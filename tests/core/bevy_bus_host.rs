//! The Bevy host fixture: the bound claims proven in a running Bevy world.
//!
//! `tests/fixtures/bevy_bus_host` is its own workspace (bevy stays out of the
//! main lock file) depending on rig-core, rig-bus and rig-effect-log only on the rig side and on
//! `bevy_ecs` + `bevy_tasks` only on the host side, from crates.io at one
//! release version. This test runs the fixture binary and checks that graph.

use std::{
    io::Read,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

/// How long the fixture may take to build and run: a cold build of the
/// pinned bevy crates is minutes; a proof that hangs is caught by the
/// fixture's own ten-second guard. Past this the child is killed and the
/// test fails, so a wedged build never turns into a CI job that runs until
/// the runner's own limit.
const FIXTURE_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// `cargo`, resolved through the rustup proxy on `PATH` — deliberately not
/// `env!("CARGO")`: that is the cargo of the toolchain building *this* test,
/// while the fixture is its own workspace with its own `rust-toolchain.toml`
/// pin, which only the proxy resolves (hence `RUSTUP_TOOLCHAIN` and `CARGO`
/// are removed from the environment too).
fn fixture_cargo() -> Command {
    let mut command = Command::new("cargo");
    command
        .current_dir(fixture_dir())
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO")
        // CI compiles the fixture with warnings denied; do the same here so
        // a warning in the fixture fails locally, not on the runner.
        .env("RUSTFLAGS", "-D warnings");
    command
}

/// Wait for `child` up to `timeout`; on expiry kill it and return an error
/// carrying what it printed so far.
fn wait_with_timeout(
    mut child: Child,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String, String), Box<dyn std::error::Error>> {
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let out = std::thread::spawn(move || {
        let mut buffer = String::new();
        let _ = stdout.read_to_string(&mut buffer);
        buffer
    });
    let err = std::thread::spawn(move || {
        let mut buffer = String::new();
        let _ = stderr.read_to_string(&mut buffer);
        buffer
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            let stdout = out.join().unwrap_or_default();
            let stderr = err.join().unwrap_or_default();
            return Err(format!(
                "bevy bus host fixture exceeded {timeout:?} and was killed:\n{stdout}\n{stderr}"
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let stdout = out.join().unwrap_or_default();
    let stderr = err.join().unwrap_or_default();
    Ok((status, stdout, stderr))
}

fn fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bevy_bus_host/Cargo.toml")
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bevy_bus_host")
}

fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bevy-bus-host-fixture")
}

#[test]
fn bevy_host_runs_the_thirteen_proofs() -> Result<(), Box<dyn std::error::Error>> {
    let child = fixture_cargo()
        .args(["run", "--quiet", "--locked", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target_dir())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let (status, stdout, stderr) = wait_with_timeout(child, FIXTURE_TIMEOUT)?;
    if !status.success() {
        return Err(format!("bevy bus host fixture failed:\n{stdout}\n{stderr}").into());
    }
    for proof in 3..=13 {
        if !stdout.contains(&format!("proof {proof}:")) {
            return Err(format!("fixture did not report proof {proof}:\n{stdout}").into());
        }
    }
    if !stdout.contains("bevy-bus-host: ok") {
        return Err(format!("fixture did not report success:\n{stdout}").into());
    }
    Ok(())
}

/// One vocabulary on both targets: the fixture spells the registrar as a
/// `NonSend` resource natively and on wasm, and nothing else in it forks
/// on the target either.
#[test]
fn bevy_host_fixture_has_no_target_cfg() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(fixture_dir().join("src/main.rs"))?;
    let offenders: Vec<String> = source
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("cfg("))
        .map(|(number, line)| format!("{}: {}", number + 1, line.trim()))
        .collect();
    if !offenders.is_empty() {
        return Err(format!("the fixture forks on the target:\n{}", offenders.join("\n")).into());
    }
    Ok(())
}

/// The effect corpus's log-as-script replay in a Bevy world: a replayer
/// per key on the host's bus, each record dispatched from a system in the
/// golden's order, each answer compared with the record's (Matrix O). It
/// proves the plumbing and the order, not an interpretation of the
/// program: no agent runs in the fixture.
#[test]
fn bevy_host_replays_a_golden() -> Result<(), Box<dyn std::error::Error>> {
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/rig-verify/fixtures/anthropic_tool_call_turn.effects.json");
    let child = fixture_cargo()
        .args(["run", "--quiet", "--locked", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target_dir())
        .arg("--")
        .arg("--replay")
        .arg(&golden)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let (status, stdout, stderr) = wait_with_timeout(child, FIXTURE_TIMEOUT)?;
    if !status.success() {
        return Err(format!("bevy bus host replay failed:\n{stdout}\n{stderr}").into());
    }
    if !stdout.contains("replay: 3 record(s)") || !stdout.contains("bevy-bus-host: ok") {
        return Err(format!("the fixture did not report the replay:\n{stdout}").into());
    }
    Ok(())
}

#[test]
fn bevy_host_fixture_graph_is_rig_core_and_bevy_ecs_tasks_only()
-> Result<(), Box<dyn std::error::Error>> {
    let output = fixture_cargo()
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--manifest-path",
        ])
        .arg(fixture_manifest())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let packages = metadata["packages"].as_array().cloned().unwrap_or_default();
    let names: Vec<&str> = packages
        .iter()
        .filter_map(|package| package["name"].as_str())
        .collect();
    for required in ["rig-core", "rig-bus", "bevy_ecs", "bevy_tasks"] {
        if !names.contains(&required) {
            return Err(format!("fixture graph should contain `{required}`").into());
        }
    }
    for forbidden in [
        "rig",
        "rig-agent",
        "tokio",
        "reqwest",
        "rmcp",
        "bevy",
        "bevy_app",
    ] {
        if names.contains(&forbidden) {
            return Err(format!("`{forbidden}` reached the bevy bus host fixture's graph").into());
        }
    }
    // The rig side of the graph is the vocabulary and the runtime, nothing else.
    let rig_side: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| name.starts_with("rig") && *name != "rig-bevy-bus-host-fixture")
        .collect();
    let mut rig_side = rig_side;
    rig_side.sort_unstable();
    // The replayer (`--replay`) is a handler like any other, and hosting
    // one is the fixture's purpose: rig-effect-log is the vocabulary of a
    // recorded run, itself over rig-core and rig-bus only.
    if rig_side != ["rig-bus", "rig-core", "rig-effect-log"] {
        return Err(format!(
            "rig-side dependencies must be rig-core, rig-bus and rig-effect-log only, got {rig_side:?}"
        )
        .into());
    }
    // Bevy comes from crates.io at one release version — never git, never a
    // path: every `bevy_*` dependency in the manifest (inline table or
    // `[dependencies.bevy_*]` table) requires `BEVY_VERSION`, and the lock
    // file resolves every `bevy_*` package to exactly that version.
    let manifest = std::fs::read_to_string(fixture_manifest())?;
    let pins = bevy_pins(&manifest);
    if pins.is_empty() {
        return Err("the fixture manifest declares no bevy_* dependency".into());
    }
    for (name, pin) in &pins {
        match pin {
            BevyPin::Version(version) if version == BEVY_VERSION => {}
            BevyPin::Version(version) => {
                return Err(format!("{name} requires `{version}`, not `{BEVY_VERSION}`").into());
            }
            BevyPin::Git => return Err(format!("{name} comes from git, not crates.io").into()),
            BevyPin::Path => return Err(format!("{name} comes from a path, not crates.io").into()),
            BevyPin::Unpinned => return Err(format!("{name} declares no version").into()),
        }
    }
    let lock = std::fs::read_to_string(fixture_dir().join("Cargo.lock"))?;
    let resolved = bevy_lock_versions(&lock);
    if resolved.is_empty() {
        return Err("the fixture lock file resolves no bevy_* package".into());
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

/// The crates.io release the proofs ran against.
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
