//! Native concurrent batch publication, independently inspected final history.

use super::{
    super::support::with_anthropic_cassette,
    streaming_tools::{
        OutOfOrderAlphaSignal, OutOfOrderBetaSignal, OutOfOrderSignalOrder,
        assert_cassette_groups_multiple_tool_results,
        assert_cassette_tool_results_follow_assistant_tool_use_order,
        assert_events_emit_all_tool_calls_before_results,
    },
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_observation::{install_observers, observation},
    support::{
        ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
        assert_contains_all_case_insensitive,
    },
};
use bevy_ecs::prelude::*;
use rig::{message::UserContent, prelude::*, providers::anthropic};
use rig_ecs::{
    agent::{MessageParts, Order, Parts, ToolCallSlot, ToolPolicy, Utterance},
    bus::{EffectOutcome, Policy, RigSchedule},
    systems::{RigSet, spawn_run},
};

#[derive(Resource, Default)]
struct Published(Vec<Vec<String>>);

fn result_names(parts: &Parts, slots: &[ToolCallSlot]) -> Vec<String> {
    let MessageParts::User { content } = &parts.0 else {
        return vec![];
    };
    content
        .iter()
        .filter_map(|part| match part {
            UserContent::ToolResult(result) => Some(
                slots
                    .iter()
                    .find(|slot| slot.id == result.call)
                    .expect("known tool result")
                    .name
                    .clone(),
            ),
            _ => None,
        })
        .collect()
}

type PublishedMessages<'w, 's> =
    Query<'w, 's, (&'static Order, &'static Parts), (With<Utterance>, Added<Parts>)>;

fn observe_publication(
    messages: PublishedMessages,
    tools: Query<(&ToolCallSlot, Option<&EffectOutcome>)>,
    mut published: ResMut<Published>,
) {
    let slots: Vec<_> = tools.iter().map(|(slot, _)| slot.clone()).collect();
    let mut messages: Vec<_> = messages.iter().collect();
    messages.sort_by_key(|(order, _)| order.0);
    for (_, parts) in messages {
        let names = result_names(parts, &slots);
        if !names.is_empty() {
            assert!(
                tools.iter().all(|(_, outcome)| outcome.is_some()),
                "no batch result is published before every tool outcome lands"
            );
            published.0.push(names);
        }
    }
}

async fn run(client: anthropic::Client, serial: bool) -> EcsAgent {
    let mut ecs = EcsAgent::new(
        client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6),
        TWO_TOOL_STREAM_PREAMBLE,
        1,
    );
    ecs.app
        .world_mut()
        .resource_mut::<Policy>()
        .0
        .serial_per_handler = serial;
    let order = OutOfOrderSignalOrder::default();
    ecs.tool(OutOfOrderAlphaSignal(order.clone()));
    ecs.tool(OutOfOrderBetaSignal(order));
    install_observers(&mut ecs);
    ecs.app.init_resource::<Published>().add_systems(
        RigSchedule,
        observe_publication
            .after(RigSet::Materialise)
            .before(RigSet::Settle),
    );
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        TWO_TOOL_STREAM_PROMPT,
        true,
        Some(8),
    );
    ecs.app
        .world_mut()
        .entity_mut(run)
        .insert(ToolPolicy { concurrency: 2 });
    tokio::time::timeout(std::time::Duration::from_secs(5), ecs.wait_for_success(run))
        .await
        .expect("two tools must start concurrently without deadlocking");
    ecs
}

fn history(ecs: &mut EcsAgent) -> Vec<Vec<String>> {
    let slots: Vec<_> = ecs
        .app
        .world_mut()
        .query::<&ToolCallSlot>()
        .iter(ecs.app.world())
        .cloned()
        .collect();
    let mut query = ecs
        .app
        .world_mut()
        .query_filtered::<(&Order, &Parts), With<Utterance>>();
    let mut messages: Vec<_> = query.iter(ecs.app.world()).collect();
    messages.sort_by_key(|(order, _)| order.0);
    messages
        .into_iter()
        .map(|(_, parts)| result_names(parts, &slots))
        .filter(|names| !names.is_empty())
        .collect()
}

#[tokio::test]
async fn serial_serving_reproduces_the_recorded_request_order() {
    with_anthropic_cassette("streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order", |client| async move {
        let mut ecs = run(client, true).await;
        assert!(observation(&ecs).errors.is_empty());
        assert!(observation(&ecs).got_final_response);
        assert_eq!(history(&mut ecs).into_iter().flatten().collect::<Vec<_>>(), ["lookup_harbor_label", "lookup_orchard_label"]);
    }).await;
    assert_cassette_groups_multiple_tool_results(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}

#[tokio::test]
async fn streaming_tool_concurrency_surfaces_results_in_call_order_after_batch_settles() {
    with_anthropic_cassette("streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order", |client| async move {
        let mut ecs = run(client, false).await;
        let expected = ["lookup_harbor_label", "lookup_orchard_label"];
        let seen = observation(&ecs);
        assert!(seen.errors.is_empty());
        assert!(seen.got_final_response);
        assert_eq!(seen.tool_calls, expected);
        assert_events_emit_all_tool_calls_before_results(&seen.events);
        assert_contains_all_case_insensitive(seen.final_response_text.as_deref().expect("final response"), &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT]);
        assert_eq!(ecs.app.world().resource::<Published>().0, [expected]);
        let messages = history(&mut ecs);
        assert_eq!(messages.iter().flatten().cloned().collect::<Vec<_>>(), expected);
        assert_eq!(messages.last().expect("tool result message"), &expected);
    }).await;
    assert_cassette_tool_results_follow_assistant_tool_use_order(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}
