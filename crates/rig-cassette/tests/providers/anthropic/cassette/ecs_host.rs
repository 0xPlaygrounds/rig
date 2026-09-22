//! Host custom effects through native providers, bus and application policies.
use super::{
    super::support::with_anthropic_corpus_host_cassette,
    corpus_host::{ADD_PROMPT, PROMPT, note_ats},
};
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler, io_runtime},
    goldens::{NOTE_KEY, NoteTaker, families},
    support::{Adder, BASIC_PREAMBLE, TOOLS_PREAMBLE},
};
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use rig::driver::Bound;
use rig::providers::anthropic::wire::Anthropic;
use rig::{
    effect::EffectFamily, providers::anthropic::completion::CLAUDE_SONNET_4_6, serve::ServingPolicy,
};
use rig_ecs::{
    agent::{PolicyVersion, Temperature},
    bus::{EffectOutcome, Handlers, PendingEffect, Policy, RigSchedule},
    systems::{RigSet, RunCommands},
};
use std::sync::Arc;
#[path = "ecs_host/policies.rs"]
mod policies;
use policies::*;

/// The hooks a cell registers, in order.
#[derive(Clone, Copy, Debug)]
enum Hooks {
    AtStart,
    AtCompletionCall,
    AtOutcome,
    AtSettled,
    StartAndSettled,
    Twice,
    Unserved,
}

/// What a cell asks of the host.
struct Host {
    /// Register the note taker.
    notes: bool,
    /// The host's serving policy.
    serial: bool,
    /// Keep stream events.
    streamed: bool,
    /// Advertise `add` and ask for a sum.
    with_tool: bool,
}

const PLAIN: Host = Host {
    notes: true,
    serial: false,
    streamed: false,
    with_tool: false,
};

fn agent(
    model: impl rig::completion::CompletionModel + 'static,
    host: &Host,
    hooks: Hooks,
) -> EcsAgent {
    let preamble = if host.with_tool {
        TOOLS_PREAMBLE
    } else {
        BASIC_PREAMBLE
    };
    let mut ecs = EcsAgent::for_golden(model, preamble, host.streamed);
    ecs.app.world_mut().resource_mut::<Policy>().0 = ServingPolicy {
        serial_per_handler: host.serial,
        ..ServingPolicy::default()
    };
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(Temperature(Some(0.0)));
    if host.notes {
        Handlers::with(ecs.app.world_mut(), |handlers| {
            handlers.register(
                NOTE_KEY,
                RuntimeHandler {
                    inner: Arc::new(NoteTaker),
                    runtime: io_runtime(),
                },
            )
        })
        .expect("bus installed")
        .expect("fresh note key");
    }
    if host.with_tool {
        ecs.tool(Adder);
    }
    let names: &[&str] = match hooks {
        Hooks::AtStart => {
            ecs.app.add_observer(at_start);
            &["NoteAtStart"]
        }
        Hooks::AtCompletionCall => {
            ecs.app.add_systems(
                RigSchedule,
                at_completion_call
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            &["NoteAtCompletionCall"]
        }
        Hooks::AtOutcome => {
            ecs.app.add_observer(at_outcome);
            &["NoteAtOutcome"]
        }
        Hooks::AtSettled => {
            ecs.app.add_observer(at_settled);
            &["NoteAtSettled"]
        }
        Hooks::StartAndSettled => {
            ecs.app.add_observer(at_start).add_observer(at_settled);
            &["NoteAtStart", "NoteAtSettled"]
        }
        Hooks::Twice => {
            ecs.app.add_observer(twice);
            &["NoteTwice"]
        }
        Hooks::Unserved => {
            ecs.app.add_observer(unserved);
            &["NoteUnserved"]
        }
    };
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(PolicyVersion(format!("ecs-host/v1:{}", names.join(","))));
    ecs.app.configure_sets(
        RigSchedule,
        (
            RigSet::Advance.run_if(ready),
            RigSet::Assemble.run_if(ready),
            RigSet::Materialise.run_if(ready),
        ),
    );

    ecs
}
async fn run_prompt(ecs: &mut EcsAgent, host: &Host) -> String {
    let prompt = if host.with_tool { ADD_PROMPT } else { PROMPT };
    let run = ecs
        .app
        .world_mut()
        .spawn_run(ecs.agent, &[], prompt, host.streamed, Some(3));
    let output = ecs.wait_for_success(run).await;
    // Native Settled publishes before an application-owned settled note finishes.
    // Await its real acknowledgement before exposing this consumer's response.
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            ecs.app.update();
            if ecs
                .app
                .world_mut()
                .run_system_once(ready)
                .expect("note checks")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("host note acknowledgement deadline");

    output
}
async fn over_host(
    client: Bound<Anthropic>,
    host: Host,
    hooks: Hooks,
) -> rig::cassette::effect_log::EffectLog {
    let mut ecs = agent(client.completion(CLAUDE_SONNET_4_6), &host, hooks);
    let output = run_prompt(&mut ecs, &host).await;
    if host.with_tool {
        assert!(output.contains("42"), "{output}");
    }
    let log = ecs.effect_log();
    let world = ecs.app.world_mut();
    let mut effects = world.query_filtered::<Option<&EffectOutcome>, With<PendingEffect>>();
    assert!(
        effects.iter(world).all(|outcome| outcome.is_some()),
        "host work finished before teardown"
    );
    drop(ecs);
    log
}

#[tokio::test]
async fn custom_at_start_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette("corpus_host/custom_at_start", |client| async move {
            let log = over_host(client, PLAIN, Hooks::AtStart).await;
            crate::goldens::world_golden_effects("anthropic_host_custom_at_start_effect_log", &log);
            assert_eq!(
                families(&log),
                [EffectFamily::Custom, EffectFamily::Completion]
            );
            assert_eq!(note_ats(&log), ["start"]);
        })
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_at_completion_call_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette(
            "corpus_host/custom_at_completion_call",
            |client| async move {
                let log = over_host(client, PLAIN, Hooks::AtCompletionCall).await;
                crate::goldens::world_golden_effects(
                    "anthropic_host_custom_at_completion_call_effect_log",
                    &log,
                );
                assert_eq!(
                    families(&log),
                    [EffectFamily::Custom, EffectFamily::Completion]
                );
                assert_eq!(note_ats(&log), ["completion_call"]);
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_at_outcome_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette("corpus_host/custom_at_outcome", |client| async move {
            let host = Host {
                with_tool: true,
                ..PLAIN
            };
            let log = over_host(client, host, Hooks::AtOutcome).await;
            crate::goldens::world_golden_effects(
                "anthropic_host_custom_at_outcome_effect_log",
                &log,
            );
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Custom,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(note_ats(&log), ["outcome"]);
        })
        .await;
    })
    .await
}

/// A dispatch from `on_run_settled`, after the answer: the recorder is
/// still tapping the host's bus, so the record follows the completion
/// that answered the run.
#[tokio::test]
async fn custom_at_settled_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette("corpus_host/custom_at_settled", |client| async move {
            let log = over_host(client, PLAIN, Hooks::AtSettled).await;
            crate::goldens::world_golden_effects(
                "anthropic_host_custom_at_settled_effect_log",
                &log,
            );
            assert_eq!(
                families(&log),
                [EffectFamily::Completion, EffectFamily::Custom]
            );
            assert_eq!(note_ats(&log), ["settled"]);
        })
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_start_and_settled_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette(
            "corpus_host/custom_start_and_settled",
            |client| async move {
                let log = over_host(client, PLAIN, Hooks::StartAndSettled).await;
                crate::goldens::world_golden_effects(
                    "anthropic_host_custom_start_and_settled_effect_log",
                    &log,
                );
                assert_eq!(
                    families(&log),
                    [
                        EffectFamily::Custom,
                        EffectFamily::Completion,
                        EffectFamily::Custom
                    ]
                );
                assert_eq!(note_ats(&log), ["start", "settled"]);
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_twice_serial_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette(
            "corpus_host/custom_twice_serial",
            |client| async move {
                let host = Host {
                    serial: true,
                    ..PLAIN
                };
                let log = over_host(client, host, Hooks::Twice).await;
                crate::goldens::world_golden_effects(
                    "anthropic_host_custom_twice_serial_effect_log",
                    &log,
                );
                assert_eq!(
                    families(&log),
                    [
                        EffectFamily::Custom,
                        EffectFamily::Custom,
                        EffectFamily::Completion
                    ]
                );
                assert_eq!(note_ats(&log), ["first", "second"]);
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_twice_concurrent_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette(
            "corpus_host/custom_twice_concurrent",
            |client| async move {
                let log = over_host(client, PLAIN, Hooks::Twice).await;
                crate::goldens::world_golden_effects(
                    "anthropic_host_custom_twice_concurrent_effect_log",
                    &log,
                );
                assert_eq!(
                    families(&log),
                    [
                        EffectFamily::Custom,
                        EffectFamily::Custom,
                        EffectFamily::Completion
                    ]
                );
                assert_eq!(note_ats(&log), ["first", "second"]);
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_at_start_streamed_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette(
            "corpus_host/custom_at_start_streamed",
            |client| async move {
                let host = Host {
                    streamed: true,
                    ..PLAIN
                };
                let log = over_host(client, host, Hooks::AtStart).await;
                crate::goldens::world_golden_effects(
                    "anthropic_host_custom_at_start_streamed_effect_log",
                    &log,
                );
                assert_eq!(
                    families(&log),
                    [EffectFamily::Custom, EffectFamily::Completion]
                );
                assert!(log.records[1].events.is_some(), "events are kept");
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn custom_at_outcome_streamed_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette(
            "corpus_host/custom_at_outcome_streamed",
            |client| async move {
                let host = Host {
                    streamed: true,
                    with_tool: true,
                    ..PLAIN
                };
                let log = over_host(client, host, Hooks::AtOutcome).await;
                crate::goldens::world_golden_effects(
                    "anthropic_host_custom_at_outcome_streamed_effect_log",
                    &log,
                );
                assert_eq!(
                    families(&log),
                    [
                        EffectFamily::Completion,
                        EffectFamily::Tool,
                        EffectFamily::Custom,
                        EffectFamily::Completion
                    ]
                );
                assert_eq!(note_ats(&log), ["outcome"]);
            },
        )
        .await;
    })
    .await
}

/// The host registered no note taker: the hook's bind is refused, the
/// run goes on, and nothing of the hook reaches the log but its name.
#[tokio::test]
async fn custom_unserved_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_host_cassette("corpus_host/custom_unserved", |client| async move {
            let host = Host {
                notes: false,
                ..PLAIN
            };
            let log = over_host(client, host, Hooks::Unserved).await;
            crate::goldens::world_golden_effects("anthropic_host_custom_unserved_effect_log", &log);
            assert_eq!(families(&log), [EffectFamily::Completion]);
        })
        .await;
    })
    .await
}

#[path = "ecs_host/tests.rs"]
mod tests;
