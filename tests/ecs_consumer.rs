//! Independent application invariants, using the same consumer as the CLI.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unwrap_used,
    clippy::unreachable,
    reason = "shared cassette safety tests use the repository's assertion conventions"
)]

#[path = "common/cassette_safety.rs"]
mod cassette_safety;

#[path = "common/cassettes.rs"]
mod cassettes;
mod consumer;

#[test]
fn consumer_fixtures_match_the_executable_registry_and_producer_hashes()
-> Result<(), consumer::Error> {
    consumer::artifacts::census()
}

#[tokio::test]
async fn maintenance_policy_matrix_replays_actual_files_and_observations()
-> Result<(), consumer::Error> {
    for case in consumer::cases()
        .into_iter()
        .filter(|case| case.provider == consumer::Provider::Synthetic)
    {
        let live = consumer::execute(&case, consumer::Scripted).await?;
        let replay = consumer::replay(&case, &live.effects).await?;
        assert_eq!(live.observations, replay.observations, "case {}", case.id);
        assert_eq!(live.files, replay.files, "case {}", case.id);
        for checkpoint in &live.checkpoints {
            let resumed = consumer::persistence::resume(&case, &live.effects, checkpoint).await?;
            assert_eq!(
                live.observations, resumed.observations,
                "case {} resume",
                case.id
            );
            assert_eq!(live.files, resumed.files, "case {} resume", case.id);
        }
    }
    Ok(())
}

struct PanicsAfterAnswer;

impl rig::serve::Serve for PanicsAfterAnswer {
    type Family = rig::effect::family::Completion;

    fn descriptor(&self) -> rig::effect::HandlerDescriptor {
        rig::serve::Serve::descriptor(&consumer::Scripted)
    }

    #[allow(
        clippy::panic,
        reason = "deliberate handler fault, after a valid answer"
    )]
    async fn serve(&self, kind: rig::effect::EffectKind, sink: rig::serve::OutcomeSink) {
        rig::serve::Serve::serve(&consumer::Scripted, kind, sink).await;
        panic!("controlled panic after successful answer");
    }
}

#[tokio::test]
async fn maintenance_rejects_a_handler_that_panics_after_a_successful_terminal()
-> Result<(), consumer::Error> {
    let mut case = consumer::cases()
        .into_iter()
        .find(|c| c.id == "synthetic-approve")
        .ok_or_else(|| consumer::Error::Invariant("missing synthetic case".into()))?;
    case.stream = true;
    let result = consumer::execute(&case, PanicsAfterAnswer).await;
    assert!(
        result.is_err(),
        "a successful Final must not hide a producer panic"
    );
    Ok(())
}

#[tokio::test]
async fn maintenance_changes_a_real_file_and_effect_replay_projects_it_once()
-> Result<(), consumer::Error> {
    let case = consumer::cases()
        .into_iter()
        .find(|c| c.id == "synthetic-approve")
        .ok_or_else(|| consumer::Error::Invariant("missing synthetic case".into()))?;
    let live = consumer::execute(&case, consumer::Scripted).await?;
    let log = serde_json::from_str(&serde_json::to_string(&live.effects)?)?;
    let replay = consumer::replay(&case, &log).await?;
    assert_eq!(live.files, replay.files);
    assert_eq!(live.writes, 1);
    assert_eq!(replay.writes, 1);
    assert_eq!(live.result, replay.result);
    assert_eq!(live.observations, replay.observations);
    let cut = live.checkpoints.first().ok_or_else(|| {
        consumer::Error::Invariant("consumer did not capture its write cut".into())
    })?;
    let resumed = consumer::persistence::resume(&case, &log, cut).await?;
    assert_eq!(live.files, resumed.files);
    assert_eq!(live.writes, resumed.writes);
    assert_eq!(live.observations, resumed.observations);
    assert_eq!(
        serde_json::to_value(consumer::persistence::tail(&live.effects, cut.next_effect).records)?,
        serde_json::to_value(resumed.effects.records)?
    );
    Ok(())
}
