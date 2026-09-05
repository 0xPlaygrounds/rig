//! Command dispatch shared with the ordinary integration-test target.

use super::{
    Case, Error, Evidence, Provider, Scripted, artifacts, cases, execute, persistence,
    providers::{self, Budget, Limits},
    replay,
};
use crate::cassettes::{CassetteMode, scrub_artifact};
use futures::FutureExt;
use serde_json::json;
use std::{panic::AssertUnwindSafe, path::PathBuf};

#[cfg(test)]
mod tests;

struct Invocation {
    command: String,
    case: Option<String>,
    matrix: Option<String>,
    candidate: Option<PathBuf>,
    cut: Option<String>,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Invocation, Error> {
    let mut args = args.into_iter();
    let mut invocation = Invocation {
        command: args.next().unwrap_or_else(|| "plan".into()),
        case: None,
        matrix: None,
        candidate: None,
        cut: None,
    };
    let mut flags = std::collections::BTreeSet::new();
    while let Some(arg) = args.next() {
        if !flags.insert(arg.clone()) {
            return Err(Error::Invocation(format!("duplicate argument {arg}")));
        }
        let value = args
            .next()
            .ok_or_else(|| Error::Invocation(format!("missing value for {arg}")))?;
        match arg.as_str() {
            "--case" => invocation.case = Some(value),
            "--matrix" => invocation.matrix = Some(value),
            "--candidate" => invocation.candidate = Some(value.into()),
            "--cut" => invocation.cut = Some(value),
            _ => return Err(Error::Invocation(format!("unknown argument {arg}"))),
        }
    }
    if invocation.case.is_some() && invocation.matrix.is_some() {
        return Err(Error::Invocation("select --case or --matrix".into()));
    }
    if invocation.candidate.is_some() && invocation.case.is_none() {
        return Err(Error::Invocation("--candidate requires --case".into()));
    }
    if invocation.candidate.is_some() && invocation.case.as_ref().is_some_and(|id| id.contains(','))
    {
        return Err(Error::Invocation(
            "one candidate belongs to exactly one case".into(),
        ));
    }
    Ok(invocation)
}

fn select(invocation: &Invocation) -> Result<Vec<Case>, Error> {
    let registry = cases();
    if let Some(ids) = &invocation.case {
        let mut selected = std::collections::BTreeSet::new();
        for id in ids.split(',') {
            if !registry.iter().any(|case| case.id == id) {
                return Err(Error::Invocation(format!("unknown case {id:?}")));
            }
            if !selected.insert(id) {
                return Err(Error::Invocation(format!("duplicate case {id}")));
            }
        }
    }
    let selected: Vec<_> = registry
        .into_iter()
        .filter(|case| {
            invocation
                .case
                .as_ref()
                .is_none_or(|ids| ids.split(',').any(|id| case.id == id))
                && invocation
                    .matrix
                    .as_ref()
                    .is_none_or(|matrix| case.matrix == matrix)
        })
        .collect();
    if selected.is_empty() {
        return Err(Error::Invocation("no cases selected".into()));
    }
    Ok(selected)
}

async fn one(
    case: &Case,
    invocation: &Invocation,
    budget: &Budget,
) -> Result<serde_json::Value, Error> {
    let input_cassette = match &invocation.candidate {
        Some(path) => path.join("provider.yaml"),
        None if case.provider != Provider::Synthetic => artifacts::cassette(case)?,
        None => PathBuf::new(),
    };
    match invocation.command.as_str() {
        "record" => {
            if case.provider == Provider::Synthetic {
                return Err(Error::Invocation(
                    "synthetic cases cannot record provider traffic; use derive".into(),
                ));
            }
            let candidate = artifacts::candidate(case)?;
            let requests_before = budget.used();
            let revision = artifacts::revision()?;
            let evidence = providers::run(
                case,
                CassetteMode::Record,
                &candidate.join("provider.yaml"),
                budget,
            )
            .await?;
            artifacts::write(
                &candidate.join("live-evidence.json"),
                &scrub_artifact(&serde_json::to_value(&evidence)?),
            )?;
            artifacts::write(
                &candidate.join("capture.json"),
                &json!({"case":case.id,"source":"live-provider",
                "provider_model":providers::identity(case.provider).ok().map(|(provider,_,_,model)|json!({"provider":provider,"model":model})),
                "cassette_sha256":artifacts::digest(&candidate.join("provider.yaml"))?,"limits":budget.limits,"requests":budget.used()-requests_before,"capture":revision}),
            )?;
            Ok(
                json!({"candidate":candidate,"next":format!("derive --case {} --candidate {}",case.id,candidate.display())}),
            )
        }
        "derive" | "verify" | "promote" => {
            let evidence = artifacts::canonical(if case.provider == Provider::Synthetic {
                execute(case, Scripted).await?
            } else {
                providers::run(case, CassetteMode::Replay, &input_cassette, budget).await?
            });
            let saved_log = serde_json::from_slice(&serde_json::to_vec(&evidence.effects)?)?;
            let replayed = artifacts::canonical(replay(case, &saved_log).await?);
            for ((name, expected), (_, actual)) in artifacts::parts(&evidence)?
                .into_iter()
                .zip(artifacts::parts(&replayed)?)
            {
                let differences = artifacts::differences(&expected, &actual);
                if let Some(first) = differences.first() {
                    let bundle = artifacts::candidate(case)?;
                    artifacts::write(&bundle.join("replay-differences.json"), &differences)?;
                    artifacts::save_evidence(&bundle, &evidence)?;
                    return Err(Error::Invariant(format!(
                        "effect replay {name} diverged at {}; evidence {}",
                        first.pointer,
                        bundle.display()
                    )));
                }
            }
            verify_resume(case, &evidence).await?;
            let mut checks: Vec<_> = persistence::check_refusals(case, &evidence)?
                .into_iter()
                .map(|name| json!({"check":name,"status":"passed"}))
                .collect();
            checks.extend(
                super::identity::check(case, &evidence)
                    .await?
                    .into_iter()
                    .map(|name| json!({"check":name,"status":"passed"})),
            );
            checks.extend(persistence::check_external_recovery(case, &evidence).await?);
            checks.extend(
                super::identity::check_replay_modes(case, &evidence)
                    .await?
                    .into_iter()
                    .map(|name| json!({"check":name,"status":"passed"})),
            );
            if invocation.command == "promote" {
                let candidate = invocation
                    .candidate
                    .as_ref()
                    .ok_or_else(|| Error::Invocation("promote requires --candidate".into()))?;
                artifacts::promote(case, candidate, &evidence)?;
                Ok(json!({"promoted":case.id,"checks":checks}))
            } else if invocation.command == "verify" {
                if invocation.candidate.is_none() {
                    artifacts::validate_fixture(case)?;
                }
                artifacts::compare_from(case, &evidence, invocation.candidate.as_deref())?;
                let mut paths = vec![
                    if case.provider == Provider::Synthetic {
                        "scripted"
                    } else {
                        "cassette"
                    },
                    "effect_replay",
                ];
                if !evidence.checkpoints.is_empty() {
                    paths.push("resume");
                }
                Ok(
                    json!({"paths":paths,"checks":checks,"records":evidence.effects.records.len(),"resume":if evidence.checkpoints.is_empty() { json!({"status":"inapplicable","reason":"case ends without an approved completed write"}) } else { json!({"status":"passed","cuts":evidence.checkpoints.iter().map(|cut|&cut.cut).collect::<Vec<_>>()}) }}),
                )
            } else {
                let candidate = artifacts::candidate(case)?;
                if case.provider != Provider::Synthetic {
                    let contents = artifacts::safe_cassette(&input_cassette)?;
                    std::fs::write(candidate.join("provider.yaml"), contents)?;
                }
                artifacts::save_evidence(&candidate, &evidence)?;
                let capture = if case.provider == Provider::Synthetic {
                    None
                } else if let Some(source) = &invocation.candidate {
                    let path = source.join("capture.json");
                    if path.is_file() {
                        Some(serde_json::from_slice::<serde_json::Value>(
                            &std::fs::read(path)?,
                        )?)
                    } else {
                        let path = source.join("provenance.json");
                        if path.is_file() {
                            serde_json::from_slice::<serde_json::Value>(&std::fs::read(path)?)?
                                .get("capture")
                                .cloned()
                        } else {
                            None
                        }
                    }
                } else {
                    let path = artifacts::golden(case, "provenance");
                    if path.is_file() {
                        serde_json::from_slice::<serde_json::Value>(&std::fs::read(path)?)?
                            .get("capture")
                            .cloned()
                    } else {
                        None
                    }
                };
                artifacts::check_capture(case, capture.as_ref(), &input_cassette)?;
                artifacts::write(
                    &candidate.join("provenance.json"),
                    &json!({"schema":1,"source":if case.provider==Provider::Synthetic {"synthetic"} else {"scrubbed-cassette-replay"},"case":case.id,
                    "cassette_sha256":if case.provider==Provider::Synthetic { None } else { Some(artifacts::digest(&input_cassette)?) },
                    "provider_model":providers::identity(case.provider).ok().map(|(provider,_,_,model)|json!({"provider":provider,"model":model})),
                    "generation":artifacts::revision()?,"capture":capture,"artifacts_sha256":artifacts::evidence_digests(&candidate)?}),
                )?;
                Ok(
                    json!({"candidate":candidate,"checks":checks,"records":evidence.effects.records.len(),"next":format!("promote --case {} --candidate {}",case.id,candidate.display())}),
                )
            }
        }
        "replay" => {
            let log = match &invocation.candidate {
                Some(path) => path.join("effects.json"),
                None => artifacts::golden(case, "effects"),
            };
            let log = serde_json::from_slice(&std::fs::read(log)?)?;
            let evidence = artifacts::canonical(replay(case, &log).await?);
            artifacts::compare_from(case, &evidence, invocation.candidate.as_deref())?;
            Ok(json!({"records":evidence.effects.records.len(),"writes":evidence.writes}))
        }
        "resume" => {
            let cut = invocation
                .cut
                .as_deref()
                .ok_or_else(|| Error::Invocation("resume requires --cut after-write".into()))?;
            let paths = |name: &str| {
                invocation.candidate.as_ref().map_or_else(
                    || artifacts::golden(case, name),
                    |path| path.join(format!("{name}.json")),
                )
            };
            let log = serde_json::from_slice(&std::fs::read(paths("effects"))?)?;
            let checkpoints: Vec<persistence::Checkpoint> =
                serde_json::from_slice(&std::fs::read(paths("checkpoints"))?)?;
            let checkpoint = checkpoints
                .iter()
                .find(|checkpoint| checkpoint.cut == cut)
                .ok_or_else(|| Error::Invocation(format!("case {} has no cut {cut}", case.id)))?;
            let evidence = artifacts::canonical(persistence::resume(case, &log, checkpoint).await?);
            let expected = artifacts::canonical(replay(case, &log).await?);
            artifacts::compare_from(case, &expected, invocation.candidate.as_deref())?;
            compare_resume(case, &expected, checkpoint, &evidence)?;
            Ok(
                json!({"cut":cut,"new_effects":evidence.effects.records.len(),"writes":evidence.writes}),
            )
        }
        _ => Err(Error::Invocation(format!(
            "unknown command {}",
            invocation.command
        ))),
    }
}

async fn verify_resume(case: &Case, evidence: &Evidence) -> Result<(), Error> {
    for cut in &evidence.checkpoints {
        let resumed =
            artifacts::canonical(persistence::resume(case, &evidence.effects, cut).await?);
        compare_resume(case, evidence, cut, &resumed)?;
    }
    Ok(())
}

fn compare_resume(
    case: &Case,
    evidence: &Evidence,
    cut: &persistence::Checkpoint,
    resumed: &Evidence,
) -> Result<(), Error> {
    let mut expected: Evidence = serde_json::from_value(serde_json::to_value(evidence)?)?;
    expected.effects = persistence::tail(&evidence.effects, cut.next_effect);
    let position = expected
        .checkpoints
        .iter()
        .position(|checkpoint| checkpoint.cut == cut.cut)
        .ok_or_else(|| Error::Invariant("resume cut absent from uninterrupted execution".into()))?;
    expected.checkpoints.drain(..position);
    let expected = artifacts::canonical(expected);
    for ((name, expected), (_, actual)) in artifacts::parts(&expected)?
        .into_iter()
        .zip(artifacts::parts(resumed)?)
    {
        let diffs = artifacts::differences(&expected, &actual);
        if let Some(first) = diffs.first() {
            let bundle = artifacts::candidate(case)?;
            artifacts::write(&bundle.join("resume-differences.json"), &diffs)?;
            artifacts::save_evidence(&bundle, evidence)?;
            return Err(Error::Invariant(format!(
                "resume {name} at {} diverges {}; evidence {}",
                cut.cut,
                first.pointer,
                bundle.display()
            )));
        }
    }
    Ok(())
}

pub(crate) async fn run(args: impl IntoIterator<Item = String>) -> Result<(), Error> {
    let invocation = parse(args)?;
    let selected = select(&invocation)?;
    let limits = Limits::default();
    let plan: Vec<_> = selected.iter().map(|case| {
        let mut fixtures: Vec<_> = ["effects", "observations", "application", "checkpoints", "provenance"].into_iter()
            .map(|name| artifacts::golden(case, name)).collect();
        if case.provider != Provider::Synthetic { fixtures.push(artifacts::cassette(case)?); }
        Ok(json!({"case":case,
            "provider_model":providers::identity(case.provider).ok().map(|(provider,_,_,model)|json!({"provider":provider,"model":model})),
            "maximum_live_requests":if case.provider == Provider::Synthetic {0} else {8},
            "required_fixtures":fixtures.iter().map(|path|json!({"path":path.strip_prefix(artifacts::root()).unwrap_or(path),"present":path.is_file()})).collect::<Vec<_>>(),
            "execution_paths":{
                "live_capture":if case.provider == Provider::Synthetic {json!({"status":"inapplicable","reason":"synthetic policy stimulus"})} else {json!({"status":"applicable"})},
                "producer":if case.provider == Provider::Synthetic {"scripted"} else {"cassette"},
                "effect_replay":"applicable",
                "resume":"cuts discovered and checked by execution; empty cuts explicitly reported"
            }
        }))
    }).collect::<Result<_, Error>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"command":invocation.command,"plan":plan,"limits":limits,"live":invocation.command=="record"})
        )?
    );
    if matches!(invocation.command.as_str(), "list" | "plan") {
        return Ok(());
    }
    if invocation.command == "record" && invocation.case.is_none() && invocation.matrix.is_none() {
        return Err(Error::Invocation(
            "live recording requires an explicit --case or --matrix".into(),
        ));
    }
    let budget = Budget::new(limits);
    let mut results = Vec::new();
    let mut failed = false;
    for case in &selected {
        let bounded = async {
            tokio::time::timeout_at(budget.deadline.into(), one(case, &invocation, &budget))
                .await
                .map_err(|_| Error::Invariant("matrix elapsed-time budget exhausted".into()))?
        };
        let result = AssertUnwindSafe(bounded).catch_unwind().await;
        let mut row = match result {
            Ok(Ok(evidence)) => json!({"case":case.id,"status":"passed","evidence":evidence}),
            Ok(Err(error)) => {
                failed = true;
                let message = scrub_artifact(&json!(error.to_string()));
                let message = if artifacts::validate_text(
                    std::path::Path::new("report"),
                    &message.to_string(),
                )
                .is_ok()
                {
                    message
                } else {
                    json!("diagnostic withheld by artifact redaction checks")
                };
                json!({"case":case.id,"status":"failed","error":message})
            }
            Err(_) => {
                failed = true;
                json!({"case":case.id,"status":"failed","error":"cassette assertion failed; rerun the selected case for its diagnostic"})
            }
        };
        if row.get("status") == Some(&json!("failed")) {
            let bundle = artifacts::candidate(case)?;
            let mut fixture_digests = serde_json::Map::new();
            for name in ["effects", "observations", "application", "checkpoints"] {
                let path = invocation.candidate.as_ref().map_or_else(
                    || artifacts::golden(case, name),
                    |path| path.join(format!("{name}.json")),
                );
                fixture_digests.insert(
                    name.into(),
                    if path.is_file() {
                        json!({"sha256":artifacts::digest(&path)?})
                    } else {
                        json!({"missing":true})
                    },
                );
            }
            let mut reproduce = vec![
                "cargo".to_owned(),
                "run".into(),
                "-p".into(),
                "rig".into(),
                "--example".into(),
                "ecs-consumer".into(),
                "--".into(),
                invocation.command.clone(),
                "--case".into(),
                case.id.into(),
            ];
            if let Some(path) = &invocation.candidate {
                reproduce.extend(["--candidate".into(), path.display().to_string()]);
            }
            if let Some(cut) = &invocation.cut {
                reproduce.extend(["--cut".into(), cut.clone()]);
            }
            artifacts::write(
                &bundle.join("failure.json"),
                &json!({"schema":1,"case":case,"result":row,"fixtures":fixture_digests,"reproduce_argv":reproduce}),
            )?;
            row.as_object_mut()
                .ok_or_else(|| Error::Invariant("invalid case report".into()))?
                .insert("failure_bundle".into(), json!(bundle));
        }
        println!("{}", serde_json::to_string(&row)?);
        results.push(row);
    }
    let passed = results
        .iter()
        .filter(|row| row.get("status") == Some(&json!("passed")))
        .count();
    let mut coverage = std::collections::BTreeMap::<
        String,
        std::collections::BTreeMap<String, serde_json::Value>,
    >::new();
    for (case, result) in selected.iter().zip(&results) {
        if let serde_json::Value::Object(axes) = serde_json::to_value(case)? {
            for (axis, value) in axes.into_iter().filter(|(axis, _)| axis != "id") {
                coverage
                    .entry(axis)
                    .or_default()
                    .entry(value.to_string())
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .ok_or_else(|| Error::Invariant("invalid coverage row".into()))?
                    .push(json!({"case":case.id,"status":result.get("status")}));
            }
        }
    }
    let report = json!({"schema":1,"command":invocation.command,"plan":plan,"cases":results,"axis_coverage":coverage,"counts":{"passed":passed,"failed":selected.len()-passed,"selected":selected.len()},"requests":budget.used(),"limits":budget.limits});
    artifacts::write(
        &artifacts::root().join(".ecs-consumer/report.json"),
        &report,
    )?;
    eprintln!(
        "ecs-consumer {}: {passed}/{} passed, {} failed; {} transport requests. Report: .ecs-consumer/report.json",
        invocation.command,
        selected.len(),
        selected.len() - passed,
        budget.used()
    );
    if failed {
        return Err(Error::Invariant(
            "required cases failed; see .ecs-consumer/report.json".into(),
        ));
    }
    Ok(())
}
