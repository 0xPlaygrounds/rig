//! Recording plans consume compiled declarations, not reconstructed Rust source.

use anyhow::{Context, ensure};
use rig_cassette_inventory::{Capture, Inventory, Recording};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

const ROOT: &str = "crates/rig-cassette/fixtures/cassettes";
const MODE: &str = "RIG_PROVIDER_TEST_MODE";
use rig_cassette_inventory::RECORDING_SCOPE_ENV as SCOPE;
const PREFIX: &str = "RIG_CASSETTE_INVENTORY=";
const USAGE: &str = "cargo xtask cassettes <list|plan|record|check> [--provider P] [--scenario P/ID] [--include-ignored] [--json]";

#[derive(Debug, Default)]
struct Options {
    provider: Option<String>,
    scenario: Option<String>,
    ignored: bool,
    json: bool,
}

impl Options {
    fn parse(command: &str, args: impl IntoIterator<Item = String>) -> anyhow::Result<Self> {
        ensure!(
            ["list", "plan", "record", "check"].contains(&command),
            "{USAGE}"
        );
        let mut result = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--provider" => {
                    ensure!(result.provider.is_none(), "duplicate --provider");
                    result.provider = Some(args.next().context("--provider needs a value")?);
                }
                "--scenario" if ["plan", "record"].contains(&command) => {
                    ensure!(result.scenario.is_none(), "duplicate --scenario");
                    result.scenario = Some(args.next().context("--scenario needs a value")?);
                }
                "--include-ignored" if ["plan", "record"].contains(&command) => {
                    result.ignored = true
                }
                "--json" if command == "list" => result.json = true,
                _ => anyhow::bail!("unknown option {arg}; {USAGE}"),
            }
        }
        if let Some(id) = &result.scenario {
            let (provider, _) = id
                .split_once('/')
                .context("--scenario requires provider/scenario")?;
            ensure!(
                result.provider.as_deref().is_none_or(|p| p == provider),
                "--provider and --scenario disagree"
            );
            result.provider = Some(provider.to_owned());
        }
        if let Some(provider) = &result.provider {
            ensure!(
                !provider.is_empty()
                    && provider
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_'),
                "invalid provider {provider:?}"
            );
        }
        Ok(result)
    }
}

pub(crate) fn run(root: &Path, args: Vec<String>) -> anyhow::Result<()> {
    let mut args = args.into_iter();
    let command = args.next().context(USAGE)?;
    let options = Options::parse(&command, args)?;
    let providers = providers(root, options.provider.as_deref())?;
    let (inventory, executables) = collect(root, &providers, options.provider.is_none())?;
    let providers = executables.keys().cloned().collect::<BTreeSet<_>>();
    rig_cassette_inventory::validate(&inventory, &root.join(ROOT), &providers)?;
    if command == "check" {
        println!(
            "ok: {} compiled tests across {} providers",
            inventory.tests.len(),
            providers.len()
        );
        return Ok(());
    }
    if command == "list" {
        if options.json {
            println!("{}", serde_json::to_string_pretty(&inventory)?);
        } else {
            for test in &inventory.tests {
                for scenario in &test.scenarios {
                    println!(
                        "{} {:?} {}{}",
                        scenario.id,
                        scenario.capture,
                        test.name,
                        if test.ignored { " (ignored)" } else { "" }
                    );
                }
            }
            for family in &inventory.families {
                println!("scripted {}: {}", family.name, family.sources.join(", "));
            }
        }
        return Ok(());
    }
    let recordings = rig_cassette_inventory::plan(
        &inventory,
        &root.join(ROOT),
        options.scenario.as_deref(),
        options.ignored,
    )?;
    for test in &inventory.tests {
        for scenario in &test.scenarios {
            if let Capture::Forbidden(reason) = &scenario.capture {
                println!("EXCLUDE {}: {reason}", scenario.id);
            } else if test.ignored && options.scenario.is_none() && !options.ignored {
                println!(
                    "EXCLUDE {}: producing test {} is ignored",
                    scenario.id, test.name
                );
            }
        }
    }
    for family in &inventory.families {
        println!(
            "EXCLUDE {}: scripted family, not a live capture",
            family.name
        );
    }
    for recording in &recordings {
        println!(
            "SELECT {}\n  {}",
            recording
                .scenarios
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            display(recording)?
        );
    }
    if command == "record" {
        for recording in &recordings {
            let executable = executables
                .get(&recording.provider)
                .context("missing compiled provider executable")?;
            execute_recording(root, recording, executable)?;
            println!("RECORDED {}", recording.test);
        }
    }
    Ok(())
}

fn execute_recording(root: &Path, recording: &Recording, executable: &Path) -> anyhow::Result<()> {
    let output = Command::new(executable)
        .args(test_args(recording))
        .current_dir(root)
        .env(MODE, "record")
        .env(SCOPE, serde_json::to_string(&recording.scenarios)?)
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;
    ensure!(
        output.status.success(),
        "recording {} failed:\n{stdout}\n{}",
        recording.test,
        String::from_utf8_lossy(&output.stderr)
    );
    verify_execution(recording, &stdout)?;
    for id in &recording.scenarios {
        ensure!(
            root.join(ROOT).join(format!("{id}.yaml")).is_file(),
            "{id}: test succeeded without producing a capture"
        );
    }
    Ok(())
}

fn providers(root: &Path, selected: Option<&str>) -> anyhow::Result<BTreeSet<String>> {
    if let Some(provider) = selected {
        return Ok(BTreeSet::from([provider.to_owned()]));
    }
    let mut providers = BTreeSet::new();
    for entry in std::fs::read_dir(root.join(ROOT))? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_dir(),
            "{}: loose files or symlinks under the fixture root are not provider directories",
            entry.path().display()
        );
        let provider = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF-8 provider directory"))?;
        ensure!(
            !provider.is_empty()
                && provider
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
            "noncanonical provider directory {provider:?}"
        );
        providers.insert(provider);
    }
    Ok(providers)
}

fn compiled_binaries(
    output: &str,
    providers: &BTreeSet<String>,
    discover_all: bool,
) -> BTreeMap<String, PathBuf> {
    let mut artifacts = BTreeMap::new();
    for line in output.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(serde_json::Value::as_str) == Some("compiler-artifact")
            && value
                .pointer("/profile/test")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && let (Some(name), Some(path)) = (
                value
                    .pointer("/target/name")
                    .and_then(serde_json::Value::as_str),
                value.get("executable").and_then(serde_json::Value::as_str),
            )
            && value
                .pointer("/target/kind")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("test")))
            && (discover_all || providers.contains(name))
        {
            artifacts.insert(name.to_owned(), PathBuf::from(path));
        }
    }
    artifacts
}

fn collect(
    root: &Path,
    providers: &BTreeSet<String>,
    discover_all: bool,
) -> anyhow::Result<(Inventory, BTreeMap<String, PathBuf>)> {
    let mut command = Command::new("cargo");
    command.args([
        "test",
        "--offline",
        "--locked",
        "-p",
        "rig-cassette",
        "--all-features",
        "--no-run",
        "--message-format=json",
    ]);
    if discover_all {
        // Compiled exports also discover providers whose captures are all
        // explicitly missing; fixture directories alone cannot enumerate those.
        command.arg("--tests");
    } else {
        for provider in providers {
            command.args(["--test", provider]);
        }
    }
    let output = command.current_dir(root).env(MODE, "replay").output()?;
    ensure!(
        output.status.success(),
        "cannot compile cassette inventory:\n{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let artifacts = compiled_binaries(&String::from_utf8(output.stdout)?, providers, discover_all);
    ensure!(
        providers
            .iter()
            .all(|provider| artifacts.contains_key(provider)),
        "not all provider fixture directories have compiled test binaries"
    );
    let mut inventory = Inventory::default();
    let mut executables = BTreeMap::new();
    for (provider, executable) in artifacts {
        let invoke = |args: &[&str]| -> anyhow::Result<String> {
            let output = Command::new(&executable)
                .args(args)
                .current_dir(root)
                .env(MODE, "replay")
                .output()?;
            ensure!(
                output.status.success(),
                "{provider} inventory failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            Ok(String::from_utf8(output.stdout)?)
        };
        let listed = test_names(&invoke(&["--list"])?);
        if !listed.contains("cassette_inventory") {
            ensure!(
                !providers.contains(&provider),
                "{provider}: missing compiled cassette inventory export"
            );
            continue;
        }
        let exported = invoke(&[
            "--exact",
            "cassette_inventory",
            "--nocapture",
            "--test-threads=1",
        ])?;
        let entries = parse_inventory(&exported)?;
        let ignored = test_names(&invoke(&["--list", "--ignored"])?);
        for safety in [
            "cassette_safety::cassettes_do_not_contain_obvious_secrets",
            "cassette_safety::cassette_files_match_registered_scenarios",
        ] {
            ensure!(
                listed.contains(safety) && !ignored.contains(safety),
                "{provider}: missing executable safety test {safety}"
            );
        }
        validate_names(&entries, &listed, &ignored, &provider)?;
        inventory.tests.extend(entries.tests);
        inventory.families.extend(entries.families);
        executables.insert(provider, executable);
    }
    Ok((inventory, executables))
}

fn parse_inventory(output: &str) -> anyhow::Result<Inventory> {
    let entries: Vec<_> = output
        .lines()
        .filter_map(|line| line.split_once(PREFIX).map(|(_, json)| json))
        .collect();
    ensure!(
        entries.len() == 1,
        "expected exactly one compiled inventory export"
    );
    let json = entries.first().context("missing inventory")?;
    Ok(serde_json::from_str(json)?)
}

fn test_names(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| line.strip_suffix(": test").map(str::to_owned))
        .collect()
}

fn validate_names(
    inventory: &Inventory,
    listed: &BTreeSet<String>,
    ignored: &BTreeSet<String>,
    provider: &str,
) -> anyhow::Result<()> {
    ensure!(
        !inventory.tests.is_empty(),
        "{provider}: empty compiled inventory"
    );
    for test in &inventory.tests {
        ensure!(
            listed.contains(&test.name) && ignored.contains(&test.name) == test.ignored,
            "{}: missing exact libtest identity or mismatched ignored status",
            test.name
        );
        ensure!(
            test.scenarios
                .iter()
                .all(|s| s.id.split('/').next() == Some(provider)),
            "{}: scenario belongs to another provider binary",
            test.name
        );
    }
    Ok(())
}

fn test_args(recording: &Recording) -> Vec<String> {
    let mut args = vec![
        "--exact".into(),
        recording.test.clone(),
        "--nocapture".into(),
        "--test-threads=1".into(),
    ];
    if recording.ignored {
        args.push("--ignored".into());
    }
    args
}

fn display(recording: &Recording) -> anyhow::Result<String> {
    Ok(format!(
        "{MODE}=record {SCOPE}='{}' cargo test --offline --locked -p rig-cassette --all-features --test {} -- {}",
        serde_json::to_string(&recording.scenarios)?,
        recording.provider,
        test_args(recording).join(" ")
    ))
}

fn verify_execution(recording: &Recording, output: &str) -> anyhow::Result<()> {
    ensure!(
        output
            .lines()
            .any(|line| line.starts_with("test result: ok. 1 passed; 0 failed; 0 ignored;")),
        "{}: recording must execute exactly one passing test",
        recording.test
    );
    for id in &recording.scenarios {
        ensure!(
            output
                .lines()
                .any(|line| line.ends_with(&format!("RIG_CASSETTE_RECORDED={id}"))),
            "{id}: test did not finalize its declared cassette capture"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
