//! Independent downstream consumers prove that provider features do not leak
//! through workspace feature unification or default transport dependencies.
use super::*;
use std::{collections::BTreeSet, fs, process::Output};

fn providers(root: &Path) -> Result<Vec<String>> {
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--no-deps", "--format-version", "1", "--locked"],
    )?)?;
    let packages = metadata
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("missing packages"))?;
    let expected: BTreeSet<_> = checks::BUILTIN_PROVIDERS
        .iter()
        .map(|p| p.to_string())
        .collect();
    for name in ["rig", "rig-core"] {
        let package = packages
            .iter()
            .find(|p| p["name"] == name)
            .ok_or_else(|| invalid(format!("missing {name}")))?;
        let actual: BTreeSet<_> = package
            .pointer("/features/providers-all")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("{name}: missing providers-all")))?
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        if actual != expected {
            return Err(invalid(format!(
                "{name}: providers-all differs from verification inventory: {actual:?} vs {expected:?}"
            )));
        }
    }
    Ok(expected.into_iter().collect())
}

fn probe_command(root: &Path, target: &Path, manifest: &Path, locked: bool) -> Result<Output> {
    let mut command = Command::new("cargo");
    command
        .current_dir(root)
        .args(["check", "--message-format=json", "--manifest-path"])
        .arg(manifest)
        .arg("--target-dir")
        .arg(target.join("verify/provider-probes"))
        .env("CARGO_TERM_COLOR", "never");
    if locked {
        command.args(["--locked", "--offline"]);
    }
    Ok(command.output()?)
}

fn messages(result: &Output) -> Vec<Value> {
    String::from_utf8_lossy(&result.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn verify_dep_info(messages: &[Value], selected: &[String], all: &[String]) -> Result<()> {
    let artifact = messages
        .iter()
        .find(|m| {
            m["reason"] == "compiler-artifact"
                && m.pointer("/target/name").and_then(Value::as_str) == Some("rig_core")
        })
        .ok_or_else(|| invalid("probe emitted no rig-core artifact"))?;
    let file = artifact["filenames"]
        .as_array()
        .and_then(|files| {
            files
                .iter()
                .filter_map(Value::as_str)
                .find(|p| p.ends_with(".rmeta"))
        })
        .ok_or_else(|| invalid("rig-core artifact has no metadata file"))?;
    let file = Path::new(file);
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("lib"))
        .ok_or_else(|| invalid("unexpected rig-core artifact name"))?;
    let dep_info = fs::read_to_string(file.with_file_name(format!("{stem}.d")))?;
    for provider in all {
        let present = dep_info.contains(&format!("/providers/{provider}/"))
            || dep_info.contains(&format!("/providers/{provider}.rs"));
        if present != selected.contains(provider) {
            return Err(invalid(format!(
                "{provider}: concrete source presence {present}, selected {selected:?}"
            )));
        }
    }
    Ok(())
}

fn verify_missing_imports(messages: &[Value], disabled: &[String], crate_name: &str) -> Result<()> {
    let errors: Vec<_> = messages
        .iter()
        .filter(|m| {
            m["reason"] == "compiler-message"
                && m.pointer("/message/level").and_then(Value::as_str) == Some("error")
        })
        .collect();
    for provider in disabled {
        let expected = format!("unresolved import `{crate_name}::providers::{provider}`");
        if !errors.iter().any(|m| {
            m.pointer("/message/code/code").and_then(Value::as_str) == Some("E0432")
                && m.pointer("/message/message").and_then(Value::as_str) == Some(expected.as_str())
        }) {
            return Err(invalid(format!(
                "missing expected disabled-provider diagnostic: {expected}"
            )));
        }
    }
    if errors.len() != disabled.len() {
        return Err(invalid(
            "negative probe failed for reasons other than disabled providers",
        ));
    }
    Ok(())
}

fn consumer(
    root: &Path,
    target: &Path,
    package: &str,
    defaults: bool,
    features: &[String],
    selected: &[String],
    all: &[String],
) -> Result<()> {
    let slug = format!(
        "{package}-{}-{}",
        if defaults { "default" } else { "minimal" },
        features.join("-")
    );
    let dir = target.join("verify/provider-consumers").join(&slug);
    fs::create_dir_all(dir.join("src"))?;
    let dependency = if package == "rig" {
        root.to_path_buf()
    } else {
        root.join("crates/rig-core")
    };
    let manifest = dir.join("Cargo.toml");
    // JSON strings/arrays are valid TOML basic strings/arrays. Do not interpolate
    // filesystem paths into shell commands or assume paths contain no spaces.
    fs::write(
        &manifest,
        format!(
            "[package]\nname = \"provider-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n[dependencies]\n{package} = {{ path = {}, default-features = {defaults}, features = {} }}\n",
            serde_json::to_string(&dependency.to_string_lossy())?,
            serde_json::to_string(features)?,
        ),
    )?;
    // Seed from the repository lock: Cargo may prune unrelated workspace entries
    // and add this consumer, but retains the selected dependency versions.
    fs::copy(root.join("Cargo.lock"), dir.join("Cargo.lock"))?;
    let crate_name = package.replace('-', "_");
    let mut positive = selected
        .iter()
        .enumerate()
        .map(|(i, p)| format!("type Provider{i} = {crate_name}::providers::{p}::Client;\n"))
        .collect::<String>();
    let mut references = selected
        .iter()
        .enumerate()
        .map(|(i, _)| format!("let _ = std::any::type_name::<Provider{i}>();\n"))
        .collect::<String>();
    for provider in selected
        .iter()
        .filter(|p| ["azure", "groq", "huggingface", "openai", "venice"].contains(&p.as_str()))
    {
        // Public raw methods must return a type downstream code can name,
        // without reaching through a disabled concrete provider module.
        positive.push_str(&format!(
            "async fn transcription_{provider}(model: &{crate_name}::providers::{provider}::TranscriptionModel, request: {crate_name}::transcription::TranscriptionRequest) -> Result<{crate_name}::providers::openai_compatible::transcription::TranscriptionResponse, {crate_name}::transcription::TranscriptionError> {{ model.raw_transcription(request).await }}\n"
        ));
        references.push_str(&format!("let _ = transcription_{provider};\n"));
    }
    fs::write(
        dir.join("src/main.rs"),
        format!("{positive}\nfn main() {{ {references} }}\n"),
    )?;
    println!("Independent consumer: {slug}");
    let result = probe_command(root, target, &manifest, false)?;
    if !result.status.success() {
        return Err(invalid(format!(
            "{slug}: positive probe failed\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    verify_dep_info(&messages(&result), selected, all)?;
    let disabled: Vec<_> = all
        .iter()
        .filter(|p| !selected.contains(p))
        .cloned()
        .collect();
    if disabled.is_empty() {
        return Ok(());
    }
    let imports = disabled
        .iter()
        .map(|p| format!("use {crate_name}::providers::{p} as _;\n"))
        .collect::<String>();
    fs::write(
        dir.join("src/main.rs"),
        format!("{imports}\nfn main() {{}}\n"),
    )?;
    let result = probe_command(root, target, &manifest, true)?;
    if result.status.success() {
        return Err(invalid(format!(
            "{slug}: disabled providers were importable"
        )));
    }
    verify_missing_imports(&messages(&result), &disabled, &crate_name)?;
    println!(
        "PASS {slug}: selected sources present; {} disabled providers absent and unimportable",
        disabled.len()
    );
    Ok(())
}

fn ecs_consumer(root: &Path, target: &Path, selected: &[String], all: &[String]) -> Result<()> {
    let dir = target
        .join("verify/provider-consumers")
        .join(format!("rig-ecs-{}", selected.join("-")));
    fs::create_dir_all(dir.join("src"))?;
    let manifest = dir.join("Cargo.toml");
    fs::write(
        &manifest,
        format!(
            "[package]\nname = \"ecs-provider-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n[dependencies]\nrig-ecs = {{ path = {}, features = {} }}\n",
            serde_json::to_string(&root.join("crates/rig-ecs").to_string_lossy())?,
            serde_json::to_string(selected)?,
        ),
    )?;
    fs::copy(root.join("Cargo.lock"), dir.join("Cargo.lock"))?;
    fs::write(
        dir.join("src/main.rs"),
        format!(
            "const SELECTED: &[&str] = &{};\n{}",
            serde_json::to_string(selected)?,
            include_str!("provider_features/fixtures/disabled_bindings.rs"),
        ),
    )?;
    let result = probe_command(root, target, &manifest, false)?;
    if !result.status.success() {
        return Err(invalid(format!(
            "ECS provider probe failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    verify_dep_info(&messages(&result), selected, all)?;
    let status = Command::new("cargo")
        .current_dir(root)
        .args(["run", "--offline", "--locked", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(target.join("verify/provider-probes"))
        .status()?;
    if !status.success() {
        return Err(invalid("disabled ECS provider behavior probe failed"));
    }
    println!("PASS ECS {selected:?}: source isolation and disabled binding rejection");
    Ok(())
}

pub(super) fn run(root: &Path, target: &Path, mode: &str) -> Result<()> {
    let all = providers(root)?;
    let selected = match mode {
        "none" => Vec::new(),
        "all" => all.clone(),
        provider if all.iter().any(|p| p == provider) => vec![provider.to_owned()],
        _ => return Err(invalid(format!("unknown provider feature probe {mode}"))),
    };
    let features = if mode == "all" {
        vec!["providers-all".into()]
    } else {
        selected.clone()
    };
    for package in ["rig-core", "rig"] {
        for defaults in [false, true] {
            consumer(root, target, package, defaults, &features, &selected, &all)?;
        }
    }
    if mode == "none" {
        ecs_consumer(root, target, &[], &all)?;
        // Transport selection must not select concrete providers.
        for features in [
            vec!["websocket".into()],
            vec!["audio".into(), "image".into()],
        ] {
            consumer(root, target, "rig", false, &features, &[], &all)?;
        }
        let selected = vec!["gemini".into(), "minimax".into()];
        consumer(root, target, "rig", true, &selected, &selected, &all)?;
    }
    if mode == "openai" {
        consumer(
            root,
            target,
            "rig",
            false,
            &["openai".into(), "websocket".into()],
            &selected,
            &all,
        )?;
    }
    if ["anthropic", "deepseek", "gemini", "openai"].contains(&mode) {
        ecs_consumer(root, target, &selected, &all)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
