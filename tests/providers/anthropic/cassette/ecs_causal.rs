//! Native nested provider completions with actual causal child effects.
use super::super::support::with_anthropic_corpus_causal_cassette;
use super::corpus_causal::{PROMPT, TOOLS_PREAMBLE};
use crate::ecs_agent::EcsAgent;
use crate::goldens::{
    Hold, Lookup, LookupArgs, NESTED_PREAMBLE, NESTING_TOOL_KEY, NEVER_KEY, NOTE_KEY, NestedChild,
    Nesting, Note, NoteAck, RELAY_KEY, RelayNote, families, parent_positions,
};
use bevy_ecs::prelude::*;
use rig::{
    effect::{EffectFamily, HandlerKey},
    prelude::*,
    providers::anthropic::completion::CLAUDE_SONNET_4_6,
    serve::{Serve, ServingPolicy},
};
use rig_ecs::{
    agent::{Grant, Order, PolicyVersion, Temperature},
    bus::{EffectOutcome, Handlers, PendingEffect, Policy},
};
// Reuse the existing native graph-producing systems, not recorded leaf handlers.
// Here their model child is served by the real provider adapter through cassettes.
#[path = "../../../../crates/rig-verify/tests/corpus/world_nesting.rs"]
#[allow(dead_code)]
mod nesting;
struct Host {
    serial: bool,
    streamed: bool,
}
async fn over_host(
    client: rig::providers::anthropic::Client,
    host: Host,
) -> rig::effect_log::EffectLog {
    let mut ecs = EcsAgent::for_golden(
        client.completion_model(CLAUDE_SONNET_4_6),
        TOOLS_PREAMBLE,
        host.streamed,
    );
    ecs.declare_bus_policy = false;
    ecs.app.world_mut().resource_mut::<Policy>().0 = ServingPolicy {
        serial_per_handler: host.serial,
        ..Default::default()
    };
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Temperature(Some(0.0)),
        PolicyVersion("ecs-causal/v1:world_nesting_completion".into()),
    ));
    let spec = Nesting {
        child: NestedChild::Completion,
        from_thread: false,
        detached: false,
    };
    // Only the neutral descriptor is shared with the original Lookup; its
    // async sink-dispatch implementation never executes in this producer.
    let descriptor = Lookup {
        nesting: spec,
        model_key: HandlerKey::from("golden/model:default"),
    }
    .descriptor();
    let tool = Handlers::with(ecs.app.world_mut(), |h| {
        h.register_open(NESTING_TOOL_KEY, descriptor.family)
    })
    .expect("bus")
    .expect("fresh lookup");
    ecs.app
        .world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(ecs.agent)));
    nesting::install(ecs.app.world_mut(), spec, "golden");
    let output = ecs
        .prompt_with_max_turns(PROMPT, host.streamed, Some(3))
        .await;
    assert!(output.contains("Paris"), "{output}");
    let log = ecs.effect_log();
    assert!(
        ecs.app
            .world_mut()
            .query::<(&PendingEffect, Option<&EffectOutcome>)>()
            .iter(ecs.app.world())
            .all(|(_, outcome)| outcome.is_some()),
        "host effects complete before teardown"
    );
    drop(ecs);
    assert_eq!(log.header.bus, None, "the policy is the host's");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion,
            EffectFamily::Completion
        ]
    );
    assert_eq!(parent_positions(&log), [None, None, Some(1), None]);
    log
}

#[tokio::test]
async fn completion_serial_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_causal_cassette("corpus_causal/completion_serial", |client| async move {
        let log = over_host(
            client,
            Host {
                serial: true,
                streamed: false,
            },
        )
        .await;
        crate::ecs_goldens::golden_effects("anthropic_causal_completion_serial", &log);
    })
    .await;
}

#[tokio::test]
async fn completion_concurrent_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_causal_cassette(
        "corpus_causal/completion_concurrent",
        |client| async move {
            let log = over_host(
                client,
                Host {
                    serial: false,
                    streamed: false,
                },
            )
            .await;
            crate::ecs_goldens::golden_effects("anthropic_causal_completion_concurrent", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn completion_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_causal_cassette(
        "corpus_causal/completion_streamed",
        |client| async move {
            let log = over_host(
                client,
                Host {
                    serial: false,
                    streamed: true,
                },
            )
            .await;
            // The run's completions are streamed with their events; the
            // tool's nested completion is unary.
            assert!(log.records[0].events.is_some());
            assert!(log.records[2].events.is_none());
            assert!(log.records[3].events.is_some());
            crate::ecs_goldens::golden_effects("anthropic_causal_completion_streamed", &log);
        },
    )
    .await;
}
