use super::super::{Case, Scripted, artifacts, cases, execute, persistence, replay};
use super::*;

#[test]
fn validation_before_apply_identifies_the_unapplied_proposal() -> Result<(), Error> {
    use super::super::{Host, build};
    let mut app = build(&case(Approval::Approve), None)?;
    let mut host = app.world_mut().resource_mut::<Host>();
    let state = host.repair.as_mut().unwrap();
    let initial = state.validate(Phase::Initial)?;
    state.observe_report(initial)?;
    let proposal = state.propose(
        "tests/regression.rs",
        "#[test]\nfn boundary() { assert!(page_window::page(&[1], 2, 1).is_empty()); }\n",
        "preserve contract",
    )?;
    state.observe_proposal(proposal.clone())?;
    let error = super::super::tool_answer(&mut host, "repo_validate", r#"{"phase":"regression"}"#)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("repo_apply_patch") && error.contains(&proposal.operation),
        "{error}"
    );
    assert_eq!(host.repair.as_ref().unwrap().project.writes(), 0);
    Ok(())
}

#[test]
fn premature_final_validation_does_not_request_a_duplicate_regression_edit() -> Result<(), Error> {
    use super::super::{Host, build};
    let mut app = build(&case(Approval::Approve), None)?;
    let mut host = app.world_mut().resource_mut::<Host>();
    let state = host.repair.as_mut().unwrap();
    let initial = state.validate(Phase::Initial)?;
    state.observe_report(initial)?;
    let proposal = state.propose(
        "tests/regression.rs",
        "#[test]\nfn boundary() { assert!(page_window::page(&[1], 2, 1).is_empty()); }\n",
        "preserve contract",
    )?;
    state.observe_proposal(proposal.clone())?;
    state.approve(&proposal.operation, Approval::Approve)?;
    state.apply(&proposal.operation)?;
    let error = super::super::tool_answer(&mut host, "repo_validate", r#"{"phase":"final"}"#)
        .unwrap_err()
        .to_string();
    assert!(
        !error.contains(&proposal.operation) && error.contains("src/lib.rs"),
        "{error}"
    );
    assert_eq!(host.repair.as_ref().unwrap().project.writes(), 1);
    Ok(())
}

#[test]
fn proposal_formats_scratch_before_exact_diff_and_approval() -> Result<(), Error> {
    let mut state = State::new()?;
    let initial = state.validate(Phase::Initial)?;
    state.observe_report(initial)?;
    let before = state.project.image()?;
    let raw = "#[test]\nfn boundary(){assert!(page_window::page(&[1],2,1).is_empty());}\n";
    let answer = tool_answer(
        &mut state,
        "repo_propose_patch",
        &json!({"path":"tests/regression.rs","content":raw,"justification":"prove the contract"}),
        std::time::Instant::now() + std::time::Duration::from_secs(10),
    )?;
    let proposal: Patch = serde_json::from_value(answer["proposal"].clone())?;
    assert_ne!(
        proposal.content, raw,
        "proposal must show the formatted exact diff before approval"
    );
    assert_eq!(answer["formatting"]["output"]["code"], 0);
    assert_eq!(answer["applied"], false);
    assert_eq!(state.project.image()?, before);
    assert!(state.approvals.is_empty());
    state.observe_proposal(proposal.clone())?;
    let approved = state.approve(&proposal.operation, Approval::Approve)?;
    assert_eq!(approved, proposal);
    state.apply(&proposal.operation)?;
    assert_eq!(state.project.read("tests/regression.rs")?, proposal.content);
    Ok(())
}

#[tokio::test]
async fn collect_refuses_contradictory_proposal_formatting_before_approval() -> Result<(), Error> {
    use super::super::{Host, build, bus};
    let case = case(Approval::Approve);
    let live = execute(&case, Scripted).await?;
    let record = live
        .effects
        .records
        .iter()
        .find(|record| record.key.as_str() == super::super::tool_key("repo_propose_patch"))
        .unwrap();
    for fault in [
        "exit",
        "timeout",
        "stdout",
        "command",
        "formatted",
        "path",
        "justification",
        "approval",
        "applied",
        "next-operation",
        "issued-arguments",
    ] {
        let mut app = build(&case, None)?;
        {
            let mut host = app.world_mut().resource_mut::<Host>();
            let state = host.repair.as_mut().unwrap();
            let initial = state.validate(Phase::Initial)?;
            state.observe_report(initial)?;
            host.replay = true;
        }
        let mut outcome = serde_json::to_value(&record.outcome)?;
        let value = outcome.pointer_mut("/Ok/result/value/0/value").unwrap();
        match fault {
            "exit" => value["formatting"]["output"]["code"] = json!(1),
            "timeout" => value["formatting"]["output"]["timed_out"] = json!(true),
            "stdout" => value["formatting"]["output"]["stdout"] = json!("different code"),
            "command" => value["formatting"]["command"] = json!(["another-program"]),
            "formatted" => value["formatted"] = json!(!value["formatted"].as_bool().unwrap()),
            "path" => value["proposal"]["path"] = json!("src/lib.rs"),
            "justification" => value["proposal"]["justification"] = json!("different reason"),
            "approval" => value["approval"] = json!("approved"),
            "applied" => value["applied"] = json!(true),
            "next-operation" => {
                value["next_tool"]["arguments"]["operation"] = json!("different-operation")
            }
            _ => (),
        }
        let mut kind = record.kind.clone();
        if fault == "issued-arguments"
            && let rig_core::effect::EffectKind::ToolCall { args, .. } = &mut kind
        {
            let mut parsed: Value = serde_json::from_str(args)?;
            parsed["justification"] = json!("different issued request");
            *args = parsed.to_string();
        }
        app.world_mut().spawn((
            bus::PendingEffect::new(record.key.clone(), kind),
            bus::Issued(record.id),
            bus::EffectOutcome(serde_json::from_value(outcome)?),
        ));
        super::super::observe(app.world_mut());
        let host = app.world().resource::<Host>();
        assert!(host.failure.is_some(), "{fault} must be refused");
        let state = host.repair.as_ref().unwrap();
        assert!(state.pending.is_none(), "{fault} published a proposal");
        assert!(state.approvals.is_empty());
        assert_eq!(state.project.writes(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn a_slow_validation_respects_the_remaining_capture_execution_budget() -> Result<(), Error> {
    use std::time::{Duration, Instant};
    let mut case = case(Approval::Approve);
    case.fault = super::super::Fault::RepairTimeout;
    let before = Instant::now();
    let result =
        super::super::execute_with_deadline(&case, Scripted, Some(before + Duration::from_secs(6)))
            .await;
    assert!(
        result.is_err(),
        "short execution budget must not report completed repair"
    );
    assert!(
        before.elapsed() < Duration::from_secs(8),
        "blocking validation consumed finalization reserve: {:?}",
        before.elapsed()
    );
    Ok(())
}

#[tokio::test]
async fn ecs_wrong_stale_and_timed_out_repairs_cannot_claim_success() -> Result<(), Error> {
    use super::super::Fault;
    for fault in [
        Fault::RepairInsufficient,
        Fault::RepairStaleApproval,
        Fault::RepairTimeout,
    ] {
        let mut case = case(Approval::Approve);
        case.fault = fault;
        let live = execute(&case, Scripted).await?;
        assert_eq!(live.observations.last().unwrap().data["repaired"], false);
        let replayed = replay(&case, &live.effects).await?;
        assert_eq!(live.files, replayed.files);
        assert_eq!(live.observations, replayed.observations);
        for cut in &live.checkpoints {
            let resumed = persistence::resume(&case, &live.effects, cut).await?;
            assert_eq!(live.files, resumed.files);
            assert_eq!(live.observations, resumed.observations);
        }
    }
    Ok(())
}

#[tokio::test]
async fn bad_publication_or_request_identity_is_refused_before_replay_projection()
-> Result<(), Error> {
    use super::super::{Host, build, bus};
    let case = case(Approval::Approve);
    let live = execute(&case, Scripted).await?;
    let cut = serde_json::to_value(&live.checkpoints[0])?;
    let mut snapshot: Snapshot = serde_json::from_value(cut["host"]["repair"].clone())?;
    let production = snapshot.ledger.pop().unwrap();
    snapshot.image.insert("src/lib.rs".into(), initial_source());
    snapshot.writes = 1;
    let record=live.effects.records.iter().find(|record|matches!(&record.kind,rig_core::effect::EffectKind::ToolCall{args,..} if serde_json::from_str::<Value>(args).ok().is_some_and(|args|args["operation"]==production.patch.operation))).unwrap();
    for fault in [
        "missing-publication",
        "wrong-publication",
        "wrong-issued-operation",
    ] {
        let mut app = build(&case, None)?;
        let mut state = State::restore(snapshot.clone())?;
        let before = state.project.image()?;
        state.pending = Some(production.patch.clone());
        {
            let mut host = app.world_mut().resource_mut::<Host>();
            host.repair = Some(state);
            host.replay = true;
        }
        let mut kind = record.kind.clone();
        if fault == "wrong-issued-operation"
            && let rig_core::effect::EffectKind::ToolCall { args, .. } = &mut kind
        {
            *args = json!({"operation":"wrong-issued-operation"}).to_string();
        }
        let entity = app
            .world_mut()
            .spawn((
                bus::PendingEffect::new(record.key.clone(), kind),
                bus::Issued(record.id),
                bus::EffectOutcome(record.outcome.clone()),
            ))
            .id();
        if fault != "missing-publication" {
            let mut publication = Publication {
                operation: production.patch.operation.clone(),
                applied: true,
                before: Some(production.patch.before.clone()),
                after: Some(production.after.clone()),
            };
            if fault == "wrong-publication" {
                publication.after = Some("wrong".into());
            }
            let mut context = rig_core::tool::ToolContext::new();
            context.insert_result(publication).unwrap();
            app.world_mut()
                .entity_mut(entity)
                .insert(bus::ToolOutputs(context));
        }
        super::super::observe(app.world_mut());
        let host = app.world().resource::<Host>();
        assert!(host.failure.is_some(), "{fault} must be refused");
        let state = host.repair.as_ref().unwrap();
        assert_eq!(
            state.project.image()?,
            before,
            "{fault} mutated replay before refusal"
        );
        assert_eq!(state.project.writes(), 1);
    }
    Ok(())
}

struct BatchedValidation;
impl rig_core::serve::Serve for BatchedValidation {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::serve::Serve::descriptor(&Scripted)
    }
    async fn serve(&self, kind: rig_core::effect::EffectKind, sink: rig_core::serve::OutcomeSink) {
        use rig_core::{
            completion::{CompletionResponse, Usage},
            effect::{EffectKind, Outcome},
            message::{AssistantContent, Message, UserContent},
        };
        let EffectKind::Completion { request, .. } = kind else {
            return;
        };
        let count = request
            .chat_history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(
                    content
                        .iter()
                        .filter(|content| matches!(content, UserContent::ToolResult(_)))
                        .count(),
                ),
                _ => None,
            })
            .sum::<usize>();
        let mut choice = scripted_choice(&request);
        if count == 9 {
            choice.push(AssistantContent::tool_call(
                "batched-final",
                "repo_validate",
                json!({"phase":"final"}),
            ));
        }
        if count == 11 {
            choice = vec![AssistantContent::tool_call(
                "retry-final",
                "repo_validate",
                json!({"phase":"final"}),
            )];
        }
        sink.resolve(Ok(Outcome::Completion(CompletionResponse::new(
            choice,
            Usage::new(),
            "synthetic",
        ))))
        .await;
    }
}

#[tokio::test]
async fn batched_apply_and_final_validation_can_recover_and_keep_the_required_cut()
-> Result<(), Error> {
    let case = case(Approval::Approve);
    let live = execute(&case, BatchedValidation).await?;
    assert_eq!(live.checkpoints.len(), 1);
    assert_eq!(live.writes, 2);
    let resumed = persistence::resume(&case, &live.effects, &live.checkpoints[0]).await?;
    assert_eq!(live.files, resumed.files);
    assert_eq!(live.observations, resumed.observations);
    Ok(())
}

#[tokio::test]
async fn ecs_lost_production_outcome_requires_reconciliation() -> Result<(), Error> {
    let mut case = case(Approval::Approve);
    case.fault = super::super::Fault::LostWriteOutcome;
    let live = execute(&case, Scripted).await?;
    let check = persistence::check_external_recovery(&case, &live).await?;
    assert!(check.is_some());
    Ok(())
}

fn case(approval: Approval) -> Case {
    let mut case = cases()
        .into_iter()
        .find(|case| case.id == "synthetic-approve")
        .unwrap();
    case.id = "repository-repair-module-probe";
    case.repair = true;
    case.approval = approval;
    case
}

#[tokio::test]
async fn ecs_repository_repair_replays_and_resumes_after_the_actual_production_edit()
-> Result<(), Error> {
    for (stream, batch) in [(false, 4096), (true, 1), (true, 3)] {
        let mut case = case(Approval::Approve);
        case.stream = stream;
        case.stream_batch = batch;
        let live = artifacts::canonical(execute(&case, Scripted).await?);
        assert_eq!(live.writes, 2);
        assert_ne!(live.files["src/lib.rs"], initial_source());
        let phases: Vec<_> = live
            .observations
            .iter()
            .filter(|observation| observation.boundary == "repair.validation")
            .map(|observation| (&observation.data["phase"], &observation.data["accepted"]))
            .collect();
        assert_eq!(
            phases,
            vec![
                (&json!("initial"), &json!(true)),
                (&json!("regression"), &json!(true)),
                (&json!("final"), &json!(true))
            ]
        );
        let log = serde_json::from_value(serde_json::to_value(&live.effects)?)?;
        let replayed = artifacts::canonical(replay(&case, &log).await?);
        for ((part, expected), (_, actual)) in artifacts::parts(&live)?
            .into_iter()
            .zip(artifacts::parts(&replayed)?)
        {
            assert!(
                artifacts::differences(&expected, &actual).is_empty(),
                "{part} replay: {:?}",
                artifacts::differences(&expected, &actual)
            );
        }
        assert_eq!(live.checkpoints.len(), 1);
        let cut = &live.checkpoints[0];
        assert_eq!(cut.cut, "after-write");
        let resumed = artifacts::canonical(persistence::resume(&case, &log, cut).await?);
        assert_eq!(live.files, resumed.files);
        assert_eq!(live.writes, resumed.writes);
        assert_eq!(live.observations, resumed.observations);
        assert_eq!(
            serde_json::to_value(persistence::tail(&live.effects, cut.next_effect).records)?,
            serde_json::to_value(resumed.effects.records)?
        );
    }
    Ok(())
}

#[tokio::test]
async fn ecs_production_denial_and_cancellation_preserve_the_regression_only() -> Result<(), Error>
{
    for decision in [Approval::Deny, Approval::Cancel] {
        let case = case(decision);
        let live = execute(&case, Scripted).await?;
        assert_eq!(live.writes, 1);
        assert_eq!(live.files["src/lib.rs"], initial_source());
        assert!(live.files.contains_key("tests/regression.rs"));
        assert!(live.checkpoints.is_empty());
        assert!(
            live.observations
                .iter()
                .any(|observation| observation.boundary == "repair.approval"
                    && observation.data["decision"] == serde_json::to_value(decision).unwrap())
        );
        let replayed = replay(&case, &live.effects).await?;
        assert_eq!(live.observations, replayed.observations);
        assert_eq!(live.files, replayed.files);
    }
    Ok(())
}
