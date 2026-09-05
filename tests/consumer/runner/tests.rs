//! Runner boundaries must reject incomplete selections and altered evidence.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic_in_result_fn
)]

use super::*;

#[tokio::test]
async fn every_registered_case_verifies_its_committed_producer_and_json_replay() -> Result<(), Error>
{
    let invocation = parse(["verify".into()])?;
    // Each case has its own bounded offline request budget, as in focused CLI runs.
    for case in select(&invocation)? {
        one(&case, &invocation, &Budget::new(Limits::default()))
            .await
            .map_err(|error| Error::Invariant(format!("case {}: {error}", case.id)))?;
    }
    Ok(())
}

#[tokio::test]
async fn request_misses_and_unconsumed_interactions_fail_with_the_cassette_path()
-> Result<(), Error> {
    let case = cases()
        .into_iter()
        .find(|case| case.id == "anthropic-unary")
        .unwrap();
    let original = artifacts::cassette(&case)?;
    let digest = artifacts::digest(&original)?;
    let wire = artifacts::safe_cassette(&original)?;
    let candidate = assert_fs::TempDir::new().unwrap();
    let path = candidate.path().join("provider.yaml");
    let changed = wire.replacen("Fix the greeting typo", "A different required request", 1);
    assert_ne!(changed, wire);
    std::fs::write(&path, changed)?;
    let error = providers::run(
        &case,
        CassetteMode::Replay,
        &path,
        &Budget::new(Limits::default()),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("runtime evidence"),
        "original execution failure: {error}"
    );
    assert!(error.contains(&path.display().to_string()));
    let first = wire.split("\n---\n").next().unwrap();
    std::fs::write(
        &path,
        crate::cassettes::scrub_cassette_contents(&format!("{wire}\n---\n{first}")),
    )?;
    let error = providers::run(
        &case,
        CassetteMode::Replay,
        &path,
        &Budget::new(Limits::default()),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("finalization failed"),
        "unconsumed request: {error}"
    );
    assert!(error.contains(&path.display().to_string()));
    assert_eq!(artifacts::digest(&original)?, digest);
    Ok(())
}

#[test]
fn a_typo_cannot_silently_drop_a_required_case() {
    for ids in [
        "synthetic-approve,typo",
        "synthetic-approve,",
        "synthetic-approve,synthetic-approve",
    ] {
        let invocation = parse(["verify".into(), "--case".into(), ids.into()]).unwrap();
        assert!(select(&invocation).is_err(), "selection {ids}");
    }
    assert!(
        parse(
            [
                "record",
                "--case",
                "anthropic-unary",
                "--case",
                "openai-unary"
            ]
            .map(str::to_owned)
        )
        .is_err()
    );
}

#[tokio::test]
async fn candidate_replay_checks_observations_and_resume_checks_the_supplied_cut()
-> Result<(), Error> {
    let case = cases()
        .into_iter()
        .find(|case| case.id == "synthetic-approve")
        .unwrap();
    let evidence = artifacts::canonical(execute(&case, Scripted).await?);
    let candidate = assert_fs::TempDir::new().unwrap();
    artifacts::save_evidence(candidate.path(), &evidence)?;
    artifacts::compare_from(&case, &evidence, Some(candidate.path()))?;
    artifacts::write(&candidate.path().join("observations.json"), &json!([]))?;
    assert!(artifacts::compare_from(&case, &evidence, Some(candidate.path())).is_err());
    let invocation = parse([
        "verify".into(),
        "--case".into(),
        case.id.into(),
        "--candidate".into(),
        candidate.path().display().to_string(),
    ])?;
    assert!(
        one(&case, &invocation, &Budget::new(Limits::default()))
            .await
            .is_err(),
        "verify must check candidate sidecars"
    );
    let cut = evidence.checkpoints.first().unwrap();
    let mut changed = serde_json::to_value(cut)?;
    changed["host"]["observations"] = json!([]);
    let changed = serde_json::from_value(changed)?;
    let resumed =
        artifacts::canonical(persistence::resume(&case, &evidence.effects, &changed).await?);
    assert!(compare_resume(&case, &evidence, &changed, &resumed).is_err());
    Ok(())
}

#[tokio::test]
async fn derivation_refuses_unsafe_wire_bytes_and_stale_capture_attribution() -> Result<(), Error> {
    let case = cases()
        .into_iter()
        .find(|case| case.id == "anthropic-unary")
        .unwrap();
    let candidate = assert_fs::TempDir::new().unwrap();
    let wire = artifacts::safe_cassette(&artifacts::cassette(&case)?)?;
    std::fs::write(
        candidate.path().join("provider.yaml"),
        format!("{wire}\n# authorization: bearer controlled-test-token\n"),
    )?;
    let invocation = parse([
        "derive".into(),
        "--case".into(),
        case.id.into(),
        "--candidate".into(),
        candidate.path().display().to_string(),
    ])?;
    let budget = Budget::new(Limits::default());
    assert!(one(&case, &invocation, &budget).await.is_err());
    assert_eq!(
        budget.used(),
        0,
        "unsafe input must fail before replay begins"
    );
    std::fs::write(candidate.path().join("provider.yaml"), wire)?;
    artifacts::write(
        &candidate.path().join("capture.json"),
        &json!({"case":"a-different-case","provider_model":{"provider":"anthropic","model":"wrong-model"},"cassette_sha256":artifacts::digest(&candidate.path().join("provider.yaml"))?}),
    )?;
    let result = one(&case, &invocation, &budget).await;
    assert!(result.is_err_and(|error| error.to_string().contains("capture provenance")));
    Ok(())
}

#[tokio::test]
async fn failed_derivation_evidence_is_not_promotable() -> Result<(), Error> {
    let case = cases()
        .into_iter()
        .find(|case| case.id == "synthetic-approve")
        .unwrap();
    let evidence = artifacts::canonical(execute(&case, Scripted).await?);
    let candidate = assert_fs::TempDir::new().unwrap();
    artifacts::save_evidence(candidate.path(), &evidence)?;
    artifacts::write(&candidate.path().join("case.json"), &case)?;
    assert!(artifacts::promote(&case, candidate.path(), &evidence).is_err());
    artifacts::write(&candidate.path().join("observations.json"), &json!([]))?;
    artifacts::write(
        &candidate.path().join("provenance.json"),
        &json!({"schema":1,"case":case.id,"source":"synthetic", "artifacts_sha256":artifacts::evidence_digests(candidate.path())?}),
    )?;
    assert!(artifacts::promote(&case, candidate.path(), &evidence).is_err());
    Ok(())
}

#[tokio::test]
async fn collapsed_batches_changed_descriptors_and_lost_stream_state_are_rejected()
-> Result<(), Error> {
    let case = cases()
        .into_iter()
        .find(|case| case.id == "stream-single-events")
        .unwrap();
    let expected = artifacts::canonical(execute(&case, Scripted).await?);
    let mut collapsed = expected.effects.clone();
    for delivery in collapsed.header.deliveries.as_mut().unwrap() {
        delivery.batch = 0;
    }
    let result = replay(&case, &collapsed).await;
    assert!(
        result.is_err() || result.is_ok_and(|actual| actual.observations != expected.observations),
        "collapsing distinct policy boundaries must fail"
    );

    let mut changed = expected.effects.clone();
    let descriptor = changed
        .header
        .handlers
        .iter_mut()
        .find(|handler| handler.key.as_str() == super::super::MODEL)
        .unwrap();
    let rig_core::effect::FamilyDescriptor::Completion { model, .. } = &mut descriptor.family
    else {
        return Err(Error::Invariant("model fixture has wrong family".into()));
    };
    *model = rig_core::completion::ModelRef::new("synthetic/different-model");
    assert!(
        replay(&case, &changed).await.is_err(),
        "a semantic model change must fail the compatibility check"
    );

    let candidate = assert_fs::TempDir::new().unwrap();
    artifacts::save_evidence(candidate.path(), &expected)?;
    let mut cuts = serde_json::to_value(&expected.checkpoints)?;
    let effects = cuts[0]["scene"]["effects"]["effects"]
        .as_array_mut()
        .unwrap();
    let mut lost = 0;
    for effect in effects {
        if effect
            .get("streamed")
            .is_some_and(|stream| !stream.is_null())
        {
            effect["streamed"] = serde_json::Value::Null;
            lost += 1;
        }
    }
    assert!(lost > 0, "fault must remove actual completed stream state");
    artifacts::write(&candidate.path().join("checkpoints.json"), &cuts)?;
    let invocation = parse([
        "resume".into(),
        "--case".into(),
        case.id.into(),
        "--cut".into(),
        "after-write".into(),
        "--candidate".into(),
        candidate.path().display().to_string(),
    ])?;
    assert!(
        one(&case, &invocation, &Budget::new(Limits::default()))
            .await
            .is_err(),
        "supplied checkpoint with missing completed stream state must not pass"
    );
    Ok(())
}
