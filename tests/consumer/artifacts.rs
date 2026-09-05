//! Candidate artifacts and semantic comparison. Verification never writes to
//! fixture paths; promotion is a separate command after inspecting the diff.

use super::{Case, Error, Evidence, Provider, providers};
use crate::cassettes::{artifact_safety_failures, cassette_path};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub(crate) fn cassette(case: &Case) -> Result<PathBuf, Error> {
    let (provider, ..) = providers::identity(case)?;
    Ok(cassette_path(
        provider,
        &format!("ecs_consumer/{}", case.id),
    ))
}

pub(crate) fn golden(case: &Case, name: &str) -> PathBuf {
    root()
        .join("crates/rig-verify/fixtures/ecs_consumer")
        .join(format!("{}.{name}.json", case.id))
}

pub(crate) fn candidate(case: &Case) -> Result<PathBuf, Error> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| Error::Invariant("host clock precedes Unix epoch".into()))?
        .as_nanos();
    let path = root().join(".ecs-consumer/candidates").join(format!(
        "{}-{stamp}-{}",
        case.id,
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path)?;
    write(&path.join("case.json"), case)?;
    Ok(path)
}

pub(crate) fn validate_text(path: &Path, contents: &str) -> Result<(), Error> {
    let failures = artifact_safety_failures(path, contents);
    if !failures.is_empty() {
        return Err(Error::Invariant(failures.join("; ")));
    }
    for variable in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ] {
        if let Ok(secret) = std::env::var(variable)
            && !secret.is_empty()
            && contents.contains(&secret)
        {
            return Err(Error::Invariant(format!(
                "artifact contains value of {variable}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn write(path: &Path, value: &impl Serialize) -> Result<(), Error> {
    let contents = format!("{}\n", serde_json::to_string_pretty(value)?);
    validate_text(path, &contents)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let staging = path.with_extension(format!(
        "staging-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&staging, contents)?;
    std::fs::rename(staging, path)?;
    Ok(())
}

/// Names of elapsed empty passes are irrelevant to this consumer's policy.
/// Preserve every delivery and partition, renumbering only observed batches.
/// Handler registration order is normalized; descriptor contents are intact.
pub(crate) fn canonical(mut evidence: Evidence) -> Evidence {
    evidence
        .effects
        .header
        .handlers
        .sort_by(|a, b| a.key.cmp(&b.key));
    if let Some(deliveries) = &mut evidence.effects.header.deliveries {
        let mut batches = BTreeMap::new();
        for delivery in deliveries {
            let next = batches.len() as u64;
            delivery.batch = *batches.entry(delivery.batch).or_insert(next);
        }
    }
    evidence
}

pub(crate) fn save_evidence(path: &Path, evidence: &Evidence) -> Result<(), Error> {
    for (name, value) in parts(evidence)? {
        write(&path.join(format!("{name}.json")), &value)?;
    }
    Ok(())
}

pub(crate) fn parts(evidence: &Evidence) -> Result<[(&'static str, Value); 4], Error> {
    Ok([
        ("checkpoints", serde_json::to_value(&evidence.checkpoints)?),
        ("effects", serde_json::to_value(&evidence.effects)?),
        (
            "observations",
            serde_json::to_value(&evidence.observations)?,
        ),
        (
            "application",
            serde_json::json!({"files":evidence.files,"writes":evidence.writes,"result":evidence.result}),
        ),
    ])
}

#[derive(Debug, Serialize)]
pub(crate) struct Difference {
    pub pointer: String,
    pub expected: Option<Value>,
    pub actual: Option<Value>,
}

pub(crate) fn differences(expected: &Value, actual: &Value) -> Vec<Difference> {
    fn walk(
        path: &str,
        expected: Option<&Value>,
        actual: Option<&Value>,
        out: &mut Vec<Difference>,
    ) {
        if expected == actual {
            return;
        }
        match (expected, actual) {
            (Some(Value::Object(a)), Some(Value::Object(b))) => {
                let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
                for key in keys {
                    walk(
                        &format!("{path}/{}", key.replace('~', "~0").replace('/', "~1")),
                        a.get(key),
                        b.get(key),
                        out,
                    );
                }
            }
            (Some(Value::Array(a)), Some(Value::Array(b))) => {
                for index in 0..a.len().max(b.len()) {
                    walk(&format!("{path}/{index}"), a.get(index), b.get(index), out);
                }
            }
            _ => out.push(Difference {
                pointer: path.into(),
                expected: expected.cloned(),
                actual: actual.cloned(),
            }),
        }
    }
    let mut out = Vec::new();
    walk("", Some(expected), Some(actual), &mut out);
    out
}

pub(crate) fn compare(case: &Case, evidence: &Evidence) -> Result<(), Error> {
    compare_from(case, evidence, None)
}

pub(crate) fn compare_from(
    case: &Case,
    evidence: &Evidence,
    source: Option<&Path>,
) -> Result<(), Error> {
    for (name, actual) in parts(evidence)? {
        let path = source.map_or_else(
            || golden(case, name),
            |source| source.join(format!("{name}.json")),
        );
        let expected: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let diffs = differences(&expected, &actual);
        if let Some(first) = diffs.first() {
            let bundle = candidate(case)?;
            save_evidence(&bundle, evidence)?;
            write(&bundle.join("differences.json"), &diffs)?;
            return Err(Error::Invariant(format!(
                "{} {name}: {} semantic differences; first at {}; failure evidence {}; reproduce: cargo run -p rig --example ecs-consumer -- verify --case {}",
                case.id,
                diffs.len(),
                first.pointer,
                bundle.display(),
                case.id
            )));
        }
    }
    Ok(())
}

pub(crate) fn digest(path: &Path) -> Result<String, Error> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

pub(crate) fn safe_cassette(path: &Path) -> Result<String, Error> {
    let contents = std::fs::read_to_string(path)?;
    validate_text(path, &contents)?;
    let failures = crate::cassettes::cassette_safety_failures(path, &contents);
    if !failures.is_empty() {
        return Err(Error::Invariant(failures.join("; ")));
    }
    Ok(contents)
}

pub(crate) fn check_capture(
    case: &Case,
    capture: Option<&Value>,
    cassette: &Path,
) -> Result<(), Error> {
    if case.repair
        && case.provider != Provider::Synthetic
        && !capture.is_some_and(|capture| {
            capture.get("execution_succeeded") == Some(&Value::Bool(true))
                && capture.get("source") == Some(&Value::String("live-provider".into()))
        })
    {
        return Err(Error::Invariant("repository repair requires a completed genuine capture; failed or partial recordings cannot be derived or promoted".into()));
    }
    if let Some(capture) = capture.filter(|capture| !capture.is_null()) {
        let (provider, _, _, model) = providers::identity(case)?;
        if capture.get("case") != Some(&serde_json::json!(case.id))
            || capture.get("provider_model")
                != Some(&serde_json::json!({"provider":provider,"model":model}))
            || capture.get("cassette_sha256") != Some(&Value::String(digest(cassette)?))
        {
            return Err(Error::Invariant(
                "capture provenance does not match the input cassette/case/model".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn revision() -> Result<Value, Error> {
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root())
        .output()?;
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root())
        .output()?;
    if !head.status.success() || !status.status.success() {
        return Err(Error::Invariant(
            "cannot identify fixture generation revision".into(),
        ));
    }
    Ok(
        serde_json::json!({"revision":String::from_utf8_lossy(&head.stdout).trim(),"dirty":!status.stdout.is_empty()}),
    )
}

pub(crate) fn evidence_digests(path: &Path) -> Result<BTreeMap<String, String>, Error> {
    ["effects", "observations", "application", "checkpoints"]
        .into_iter()
        .map(|name| Ok((name.into(), digest(&path.join(format!("{name}.json")))?)))
        .collect()
}

pub(crate) fn validate_fixture(case: &Case) -> Result<(), Error> {
    let path = golden(case, "provenance");
    let contents = std::fs::read_to_string(&path)?;
    validate_text(&path, &contents)?;
    let manifest: Value = serde_json::from_str(&contents)?;
    if manifest.get("schema") != Some(&Value::from(1))
        || manifest.get("case") != Some(&Value::from(case.id))
    {
        return Err(Error::Invariant(format!(
            "{} has invalid fixture provenance",
            case.id
        )));
    }
    for name in ["effects", "observations", "application", "checkpoints"] {
        let path = golden(case, name);
        validate_text(&path, &std::fs::read_to_string(&path)?)?;
        if manifest
            .get("artifacts_sha256")
            .and_then(|hashes| hashes.get(name))
            != Some(&Value::String(digest(&path)?))
        {
            return Err(Error::Invariant(format!(
                "{} {name} digest differs from producer manifest",
                case.id
            )));
        }
    }
    if case.provider != Provider::Synthetic
        && manifest.get("cassette_sha256") != Some(&Value::String(digest(&cassette(case)?)?))
    {
        return Err(Error::Invariant(format!(
            "{} cassette digest differs from producer manifest",
            case.id
        )));
    }
    if case.provider != Provider::Synthetic {
        check_capture(case, manifest.get("capture"), &cassette(case)?)?;
    }
    Ok(())
}

pub(crate) fn census() -> Result<(), Error> {
    let mut expected = std::collections::BTreeSet::new();
    let mut ids = std::collections::BTreeSet::new();
    for case in super::cases() {
        if !ids.insert(case.id) {
            return Err(Error::Invariant(format!("duplicate case {}", case.id)));
        }
        validate_fixture(&case)?;
        for name in [
            "effects",
            "observations",
            "application",
            "checkpoints",
            "provenance",
        ] {
            expected.insert(golden(&case, name));
        }
    }
    let actual = std::fs::read_dir(root().join("crates/rig-verify/fixtures/ecs_consumer"))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    if actual != expected {
        return Err(Error::Invariant(
            "ECS consumer fixtures have missing or orphaned artifacts".into(),
        ));
    }
    Ok(())
}

pub(crate) fn promote(case: &Case, path: &Path, verified: &Evidence) -> Result<(), Error> {
    let provenance_path = path.join("provenance.json");
    let provenance_bytes = std::fs::read_to_string(&provenance_path)?;
    validate_text(&provenance_path, &provenance_bytes)?;
    let provenance: Value = serde_json::from_str(&provenance_bytes)?;
    if provenance.get("schema") != Some(&Value::from(1))
        || provenance.get("case") != Some(&Value::from(case.id))
    {
        return Err(Error::Invocation(
            "candidate has no successful derivation manifest".into(),
        ));
    }
    if provenance.get("artifacts_sha256") != Some(&serde_json::to_value(evidence_digests(path)?)?) {
        return Err(Error::Invariant(
            "candidate artifact digests differ from derivation".into(),
        ));
    }
    let recorded: Value = serde_json::from_slice(&std::fs::read(path.join("case.json"))?)?;
    if recorded != serde_json::to_value(case)? {
        return Err(Error::Invocation(
            "candidate case does not match registry".into(),
        ));
    }
    let mut files = Vec::new();
    let mut reports = Vec::new();
    for (name, verified) in parts(verified)? {
        let source = path.join(format!("{name}.json"));
        let contents = std::fs::read_to_string(&source)?;
        validate_text(&source, &contents)?;
        let actual: Value = serde_json::from_str(&contents)?;
        if actual != verified {
            return Err(Error::Invariant(format!(
                "candidate {name} does not match fresh producer/replay verification"
            )));
        }
        let target = golden(case, name);
        let expected: Value = if target.is_file() {
            serde_json::from_slice(&std::fs::read(&target)?)?
        } else {
            Value::Null
        };
        let report =
            serde_json::json!({"artifact":name,"differences":differences(&expected,&actual)});
        validate_text(&target, &serde_json::to_string(&report)?)?;
        reports.push(report);
        files.push((contents.into_bytes(), target));
    }
    if case.provider != Provider::Synthetic {
        let source = path.join("provider.yaml");
        check_capture(case, provenance.get("capture"), &source)?;
        let contents = std::fs::read_to_string(&source)?;
        validate_text(&source, &contents)?;
        let failures = crate::cassettes::cassette_safety_failures(&source, &contents);
        if !failures.is_empty() {
            return Err(Error::Invariant(failures.join("; ")));
        }
        if provenance.get("cassette_sha256") != Some(&Value::String(digest(&source)?)) {
            return Err(Error::Invariant(
                "candidate cassette digest differs from derivation".into(),
            ));
        }
        files.push((contents.into_bytes(), cassette(case)?));
    }
    files.push((provenance_bytes.into_bytes(), golden(case, "provenance")));
    for report in reports {
        println!("{}", serde_json::to_string(&report)?);
    }
    // Validate all candidates before mutation, retaining originals for rollback.
    let originals: Vec<_> = files
        .iter()
        .map(|(_, target)| {
            if target.exists() {
                std::fs::read(target).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<_, _>>()?;
    for (index, (contents, target)) in files.iter().enumerate() {
        let install = || -> Result<(), std::io::Error> {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let staged = target.with_extension(format!(
                "staging-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&staged, contents)?;
            std::fs::rename(staged, target)
        };
        if let Err(error) = install() {
            for ((_, restored), original) in files.iter().zip(&originals).take(index) {
                match original {
                    Some(bytes) => std::fs::write(restored, bytes)?,
                    None => std::fs::remove_file(restored)?,
                }
            }
            return Err(error.into());
        }
    }
    Ok(())
}
