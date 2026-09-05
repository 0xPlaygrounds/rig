//! Compatibility probes over a real consumer recording, with explicit failures.

use super::{Case, Error, Evidence, MODEL, build, program, replay, tool_key};
use rig_core::{
    completion::ModelRef,
    effect::{EffectFamily, EffectKind, FamilyDescriptor},
    error::ErrorKind,
};
use rig_ecs::{
    agent::PolicyVersion,
    bus::{EffectOutcome, PendingEffect},
};

pub(super) async fn check(case: &Case, evidence: &Evidence) -> Result<Vec<&'static str>, Error> {
    if !case.identity_checks {
        return Ok(Vec::new());
    }
    let log = &evidence.effects;
    let unused = tool_key("unused_audit");
    if log
        .records
        .iter()
        .any(|record| record.key.as_str() == unused)
        || log.header.programs.len() < 2
    {
        return Err(Error::Invariant(
            "identity case requires an uncalled grant and multiple actual scopes".into(),
        ));
    }
    let mut passed = vec!["fresh-world-with-uncalled-grant-and-multiple-scopes"];
    for (name, capability) in [
        ("changed-model-refused", false),
        ("changed-capability-refused", true),
    ] {
        let mut changed = log.clone();
        let descriptor = changed
            .header
            .handlers
            .iter_mut()
            .find(|handler| handler.key.as_str() == MODEL)
            .ok_or_else(|| Error::Invariant("identity recording lost model descriptor".into()))?;
        if let FamilyDescriptor::Completion {
            model,
            capabilities,
        } = &mut descriptor.family
        {
            if capability {
                capabilities.composes_native_output_with_tools =
                    !capabilities.composes_native_output_with_tools;
            } else {
                *model = ModelRef::new("synthetic/changed-identity");
            }
        } else {
            return Err(Error::Invariant("recorded model has wrong family".into()));
        }
        if replay(case, &changed).await.is_ok() {
            return Err(Error::Invariant(format!("identity check accepted {name}")));
        }
        passed.push(name);
    }
    let mut app = build(case, Some(log))?;
    let run = program(&mut app, case)?;
    app.world_mut()
        .entity_mut(run)
        .insert(PolicyVersion("maintenance/changed-policy".into()));
    if rig_ecs::replay::check_replayable(app.world_mut(), run, log).is_ok() {
        return Err(Error::Invariant("changed policy was accepted".into()));
    }
    passed.push("changed-policy-refused");

    let mut missing = log.clone();
    missing
        .header
        .programs
        .retain(|scope, _| scope == "maintenance-audit");
    if replay(case, &missing).await.is_ok() {
        return Err(Error::Invariant(
            "missing scoped identity was accepted".into(),
        ));
    }
    passed.push("missing-scoped-identity-refused");
    let mut missing = log.clone();
    missing
        .header
        .handlers
        .retain(|handler| handler.key.as_str() != unused);
    if replay(case, &missing).await.is_ok() {
        return Err(Error::Invariant(
            "missing uncalled-grant descriptor was accepted".into(),
        ));
    }
    passed.push("missing-required-descriptor-refused");

    let mut conflicting = log.clone();
    let mut descriptor = conflicting
        .header
        .handlers
        .iter()
        .find(|handler| handler.key.as_str() == unused)
        .cloned()
        .ok_or_else(|| Error::Invariant("uncalled grant has no descriptor".into()))?;
    descriptor.family = FamilyDescriptor::Custom {
        kind: "wrong-family".into(),
    };
    conflicting.header.handlers.push(descriptor);
    if replay(case, &conflicting).await.is_ok() {
        return Err(Error::Invariant(
            "conflicting descriptors were accepted".into(),
        ));
    }
    passed.push("conflicting-descriptors-refused");
    let mut conflicting = log.clone();
    conflicting
        .header
        .programs
        .get_mut("maintenance-audit")
        .ok_or_else(|| Error::Invariant("audit scope absent".into()))?
        .required
        .insert(MODEL.into(), EffectFamily::Custom);
    if replay(case, &conflicting).await.is_ok() {
        return Err(Error::Invariant(
            "conflicting scope requirements were accepted".into(),
        ));
    }
    passed.push("conflicting-scoped-requirements-refused");

    let mut app = build(case, Some(log))?;
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(
            unused.clone(),
            EffectKind::ToolCall {
                name: "unused_audit".into(),
                args: "{}".into(),
            },
        ))
        .id();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        app.update();
        if let Some(outcome) = app.world().get::<EffectOutcome>(effect) {
            if !outcome.0.as_ref().is_err_and(|error| {
                error.kind == ErrorKind::Divergence && error.message.contains(&unused)
            }) {
                return Err(Error::Invariant(
                    "unexpected unrecorded call lacked a key-specific divergence".into(),
                ));
            }
            break;
        }
        if std::time::Instant::now() >= deadline {
            return Err(Error::Invariant(
                "unexpected call did not fail finitely".into(),
            ));
        }
        tokio::task::yield_now().await;
    }
    passed.push("unexpected-uncalled-grant-call-diverges");
    Ok(passed)
}

pub(super) async fn check_replay_modes(
    case: &Case,
    evidence: &Evidence,
) -> Result<Vec<&'static str>, Error> {
    if case.fault != super::Fault::FoldedReplay {
        return Ok(Vec::new());
    }
    let mut folded = evidence.effects.clone();
    for record in &mut folded.records {
        record.events = None;
    }
    if build(case, Some(&folded)).is_ok() {
        return Err(Error::Invariant(
            "policy replay accepted missing stream events".into(),
        ));
    }
    let mut app = super::build_with_replay(case, Some(&folded), rig_ecs::bus::Replay::default())?;
    let run = program(&mut app, case)?;
    rig_ecs::replay::check_replayable(app.world_mut(), run, &folded)?;
    let final_only = super::drive(app, run).await?;
    if final_only.files != evidence.files
        || final_only.writes != evidence.writes
        || final_only.result != evidence.result
    {
        return Err(Error::Invariant(
            "folded replay changed the final application answer".into(),
        ));
    }
    if final_only.observations == evidence.observations {
        return Err(Error::Invariant(
            "replay-mode probe did not exercise distinct partial observations".into(),
        ));
    }
    let mut absent = evidence.effects.clone();
    absent.header.deliveries = None;
    if build(case, Some(&absent)).is_ok() {
        return Err(Error::Invariant(
            "policy replay accepted absent delivery metadata".into(),
        ));
    }
    let mut malformed = evidence.effects.clone();
    malformed.header.deliveries = Some(Vec::new());
    if build(case, Some(&malformed)).is_ok() {
        return Err(Error::Invariant(
            "policy replay accepted missing outcome deliveries".into(),
        ));
    }
    Ok(vec![
        "folded-final-answer-replay",
        "folded-policy-replay-refused",
        "missing-delivery-metadata-refused",
        "malformed-delivery-metadata-refused",
    ])
}
