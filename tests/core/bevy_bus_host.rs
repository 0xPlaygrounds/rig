//! The Bevy host fixture: the bound claims proven in a running Bevy world.
//!
//! `tests/fixtures/bevy_bus_host` is its own workspace (bevy stays out of the
//! main lock file) depending on rig-core only on the rig side and on
//! `bevy_ecs` + `bevy_tasks` only on the host side, pinned by git revision.
//! This test runs the fixture binary and checks that graph.

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
        .env_remove("CARGO");
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
fn bevy_host_runs_the_seven_proofs() -> Result<(), Box<dyn std::error::Error>> {
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
    for proof in 3..=7 {
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
    for required in ["rig-core", "bevy_ecs", "bevy_tasks"] {
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
    // The rig side of the graph is rig-core alone.
    let rig_side: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| name.starts_with("rig") && *name != "rig-bevy-bus-host-fixture")
        .collect();
    if rig_side != ["rig-core"] {
        return Err(
            format!("rig-side dependencies must be rig-core only, got {rig_side:?}").into(),
        );
    }
    // Bevy is pinned by revision, not version: every `bevy_*` dependency
    // in the manifest — inline table or `[dependencies.bevy_*]` table —
    // carries the one rev.
    let manifest = std::fs::read_to_string(fixture_manifest())?;
    let pins = bevy_pins(&manifest);
    if pins.is_empty() {
        return Err("the fixture manifest declares no bevy_* dependency".into());
    }
    for (name, rev) in &pins {
        match rev {
            Some(rev) if rev == BEVY_REV => {}
            Some(rev) => {
                return Err(format!("{name} is pinned to `{rev}`, not `{BEVY_REV}`").into());
            }
            None => return Err(format!("{name} is not pinned by rev").into()),
        }
    }
    Ok(())
}

/// The revision the proofs ran against.
const BEVY_REV: &str = "823bcc935";

/// Every `bevy_*` dependency in a manifest with its `rev`, whether written
/// as an inline table (`bevy_ecs = { git = …, rev = "…" }`) or a dependency
/// table (`[dependencies.bevy_ecs]` followed by `rev = "…"`).
fn bevy_pins(manifest: &str) -> Vec<(String, Option<String>)> {
    fn quoted_value(text: &str, key: &str) -> Option<String> {
        let at = text.find(key)?;
        let rest = text[at + key.len()..].trim_start();
        let rest = rest.strip_prefix('=')?.trim_start();
        let rest = rest.strip_prefix('"')?;
        rest.split('"').next().map(str::to_owned)
    }
    let mut pins = Vec::new();
    let mut table: Option<(String, Option<String>)> = None;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            if let Some(pin) = table.take() {
                pins.push(pin);
            }
            if let Some(name) = line
                .strip_prefix("[dependencies.")
                .and_then(|rest| rest.strip_suffix(']'))
                .filter(|name| name.starts_with("bevy_"))
            {
                table = Some((name.to_owned(), None));
            }
            continue;
        }
        if let Some((_, rev)) = table.as_mut() {
            if line.starts_with("rev") {
                *rev = quoted_value(line, "rev");
            }
            continue;
        }
        if line.starts_with("bevy_")
            && let Some((name, value)) = line.split_once('=')
        {
            pins.push((name.trim().to_owned(), quoted_value(value, "rev")));
        }
    }
    if let Some(pin) = table.take() {
        pins.push(pin);
    }
    pins
}

#[test]
fn bevy_pins_reads_both_manifest_forms() {
    let manifest = r#"
[dependencies]
rig-core = { path = "x" }
bevy_ecs = { git = "https://github.com/bevyengine/bevy", rev = "823bcc935", default-features = false }

[dependencies.bevy_tasks]
git = "https://github.com/bevyengine/bevy"
rev = "823bcc935"
features = ["multi_threaded"]

[dependencies.bevy_app]
git = "https://github.com/bevyengine/bevy"
"#;
    assert_eq!(
        bevy_pins(manifest),
        vec![
            ("bevy_ecs".to_owned(), Some("823bcc935".to_owned())),
            ("bevy_tasks".to_owned(), Some("823bcc935".to_owned())),
            ("bevy_app".to_owned(), None),
        ]
    );
}
