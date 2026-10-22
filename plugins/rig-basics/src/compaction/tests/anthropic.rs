mod pending;

use super::*;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};
use serde_json::{Value, json};

const FACTS: &str = "The observatory key is amber-731. The archive shelf is Kestrel-42.";
const SUMMARY: &str =
    "## Goal\nRemember the observatory key amber-731 and archive shelf Kestrel-42.";

fn response(text: &str, stop: &str) -> MockHttpResponse {
    sse(response_events(text, stop))
}

fn response_events(text: &str, stop: &str) -> [Value; 6] {
    [
        json!({"type":"message_start", "message":{
            "id":"msg_synthetic", "type":"message", "role":"assistant",
            "model":"claude-opus-4-6", "content":[], "stop_reason":null,
            "usage":{"input_tokens":100,"output_tokens":0}
        }}),
        json!({"type":"content_block_start", "index":0,
            "content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta", "index":0,
            "delta":{"type":"text_delta","text":text}}),
        json!({"type":"content_block_stop", "index":0}),
        json!({"type":"message_delta", "delta":{"stop_reason":stop},
            "usage":{"output_tokens":20}}),
        json!({"type":"message_stop"}),
    ]
}

fn sse(events: impl IntoIterator<Item = Value>) -> MockHttpResponse {
    let body = events
        .into_iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            )
        })
        .collect::<String>();
    MockHttpResponse::success_typed(body, "text/event-stream")
}

fn connect_anthropic(
    app: &mut App,
    http: &(impl rig_core::http_client::HttpClientExt + Clone + 'static),
    window: u32,
) -> Option<Entity> {
    let name = ToolName::new("read").ok()?;
    app.world_mut().spawn(ToolDef(
        rig_core::completion::ToolDefinition {
            name,
            description: "Read a synthetic file.".to_owned(),
            parameters: json!({"type":"object", "properties":{"path":{"type":"string"}}}),
            annotations: Default::default(),
        }
        .into(),
    ));
    let served = rig_core::providers::anthropic::AnthropicConfig::new("synthetic-key")
        .connect(http.clone())
        .completion("claude-opus-4-6");
    let spec = rig_core::catalog::Catalog::builtin().resolve("anthropic/claude-opus-4-6");
    assert!(spec.is_ok(), "catalog model: {spec:?}");
    let spec = spec.ok()?.shared();
    let spec = Arc::new(spec.as_ref().clone().with_context_window(window));
    let handler = rig_core::serve::adapters::ModelAdapter::<rig_core::operation::Completion>::new(
        spec.reference(),
        served,
    );
    let agent = restored_agent(app);
    assert!(agent.is_some(), "one agent");
    let agent = agent?;
    app.world_mut().entity_mut(agent).insert(Connection {
        spec,
        handler: rig_ecs::effects::Handler(rig_core::serve::ErasedHandler::new(handler)),
    });
    Some(agent)
}

fn seed(app: &mut App, agent: Entity, turns: usize) {
    let mut messages = long_conversation(turns);
    messages.insert(0, Message::user(FACTS));
    let mut exchange = read("toolu_synthetic", "synthetic-facts.txt", "Read completed.");
    if let Some(Message::Assistant(message)) = exchange.first_mut() {
        message.origin = Some(rig_core::message::Origin::new(
            "anthropic.messages",
            "anthropic",
            "claude-opus-4-6",
        ));
        message.content.insert(0, AssistantContent::reasoning("Inspect the synthetic file.")
            .with_native(json!({"type":"thinking", "thinking":"Inspect the synthetic file.", "signature":"synthetic-signature"})));
    }
    messages.extend(exchange);
    messages.push(Message::assistant("Recorded. Ready for the next task."));
    for message in messages {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
}

fn compact(app: &mut App, wakes: &Receiver<()>, agent: Entity) {
    app.world_mut().resource_mut::<Ended>().0.clear();
    app.world_mut().trigger(RunCommand::new(
        agent,
        "compact preserve the synthetic facts".to_owned(),
    ));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

#[test]
fn manual_compaction_repeats_and_resumes_without_losing_facts() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let http = SequencedHttpClient::new([
        response(SUMMARY, "end_turn"),
        response("Ready.", "end_turn"),
        response(SUMMARY, "end_turn"),
        response("Recalled.", "end_turn"),
    ]);
    let Some(agent) = connect_anthropic(&mut app, &http, 1_000_000) else {
        return;
    };
    seed(&mut app, agent, 2);
    compact(&mut app, &wakes, agent);
    assert!(entries(app.world(), agent).iter().any(holds_summary));
    app.world_mut().resource_mut::<Ended>().0.clear();
    answered_turns(&mut app, &wakes, agent, 1);
    compact(&mut app, &wakes, agent);
    let compacted = entries(app.world(), agent);
    assert_eq!(
        compacted
            .iter()
            .filter(|entry| holds_summary(entry))
            .count(),
        1
    );
    drop(app);

    let (mut resumed, wakes) = start(&store);
    let Some(agent) = connect_anthropic(&mut resumed, &http, 1_000_000) else {
        return;
    };
    assert_eq!(entries(resumed.world(), agent), compacted);
    answered_turns(&mut resumed, &wakes, agent, 1);
    let requests = bodies(&http);
    assert_eq!(
        requests.len(),
        4,
        "no repeated automatic summaries: {requests:?}"
    );
    for request in &requests {
        assert!(
            request.contains("amber-731") && request.contains("Kestrel-42"),
            "{request}"
        );
    }
    assert!(
        requests
            .first()
            .is_some_and(|request| asks_summary(request) && request.contains("Read completed."))
    );
    assert!(
        requests
            .get(1)
            .is_some_and(|request| !request.contains("\"type\":\"tool_use\"")
                && !request.contains("\"type\":\"tool_result\"")
                && !request.contains("synthetic-signature"))
    );
    assert!(requests.get(2).is_some_and(|request| asks_summary(request)));
    assert!(
        requests
            .last()
            .is_some_and(|request| !asks_summary(request))
    );
    // Scripted usage exercises accounting only, not server cache reuse.
    assert_eq!(
        resumed
            .world()
            .get::<AgentUsage>(agent)
            .map(|usage| usage.total.calls),
        Some(4)
    );
}

#[test]
fn automatic_compaction_keeps_the_entire_newest_message() {
    let (mut app, wakes) = start(&MemoryStore::default());
    let http = SequencedHttpClient::new([
        response(SUMMARY, "end_turn"),
        response("Ready.", "end_turn"),
        response("Still ready.", "end_turn"),
    ]);
    let Some(agent) = connect_anthropic(&mut app, &http, 40_000) else {
        return;
    };
    seed(&mut app, agent, 6);
    let newest = format!("NEWEST-BEGIN {} NEWEST-END", "neutral-padding ".repeat(500));
    app.world_mut()
        .trigger(Deliver::new(agent, newest.clone(), DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("Ready."));
    let requests = bodies(&http);
    assert_eq!(requests.len(), 2, "{requests:?}");
    assert!(
        requests
            .first()
            .is_some_and(|request| asks_summary(request) && !request.contains("NEWEST-BEGIN"))
    );
    assert!(
        requests
            .get(1)
            .is_some_and(|request| request.contains(&newest)
                && request.contains("amber-731")
                && request.contains("synthetic-signature")
                && request.contains("Read completed.")),
        "wire retention: {:?}",
        requests
            .iter()
            .map(|request| (
                request.contains("synthetic-signature"),
                request.contains("toolu_synthetic"),
                request.contains("Read completed.")
            ))
            .collect::<Vec<_>>()
    );
    let wire = requests
        .get(1)
        .and_then(|body| serde_json::from_str::<Value>(body).ok());
    assert!(wire.is_some());
    let blocks = wire
        .as_ref()
        .and_then(|wire| wire.get("messages"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .collect::<Vec<_>>();
    let calls = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        .filter_map(|block| block.get("id"))
        .collect::<Vec<_>>();
    let results = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
        .filter_map(|block| block.get("tool_use_id"))
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls, results, "retained calls and results stay paired");
    app.world_mut().resource_mut::<Ended>().0.clear();
    answered_turns(&mut app, &wakes, agent, 1);
    assert_eq!(
        http.requests().len(),
        3,
        "the reduced history must not compact again"
    );
}

#[test]
fn cancelling_an_http_summary_preserves_the_original_entries() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    let http = SequencedHttpClient::new([response(SUMMARY, "end_turn")]);
    let Some(agent) = connect_anthropic(&mut app, &http, 1_000_000) else {
        return;
    };
    seed(&mut app, agent, 1);
    let original = entries(app.world(), agent);
    app.world_mut()
        .trigger(RunCommand::new(agent, "compact".to_owned()));
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Reviewing>())
    });
    app.world_mut().trigger(StopTurn::new(agent));
    app.world_mut().flush();
    app.update();
    assert_eq!(entries(app.world(), agent), original);
    assert_eq!(http.requests().len(), 1);
    drop(app);
    let (mut resumed, _) = start(&store);
    let agent = restored_agent(&mut resumed);
    assert!(agent.is_some(), "restored agent");
    let Some(agent) = agent else {
        return;
    };
    assert_eq!(entries(resumed.world(), agent), original);
}

#[test]
fn an_oversized_newest_message_fails_after_one_summary_and_can_be_recovered() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let Ok(status) = "400".parse() else {
        return;
    };
    let refusal = MockHttpResponse::error(
        status,
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens"}}"#,
    );
    let http = SequencedHttpClient::new([refusal.clone(), response(SUMMARY, "end_turn"), refusal]);
    let Some(agent) = connect_anthropic(&mut app, &http, 40_000) else {
        return;
    };
    for message in [Message::user(FACTS), Message::assistant("Recorded.")] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    let original = entries(app.world(), agent);
    let newest = format!("NEWEST-BEGIN {} NEWEST-END", "word ".repeat(56_000));
    app.world_mut()
        .trigger(Deliver::new(agent, newest.clone(), DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let requests = bodies(&http);
    assert_eq!(requests.len(), 3, "one summary and one retry, never a loop");
    assert!(requests.get(1).is_some_and(|request| asks_summary(request)));
    assert!(
        requests
            .get(2)
            .is_some_and(|request| !asks_summary(request) && request.contains(&newest))
    );
    assert!(
        requests
            .first()
            .is_some_and(|request| request.contains(&newest) && !asks_summary(request))
    );
    assert!(entries(app.world(), agent).iter().any(holds_summary));
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Failed(why)] if why.contains("context window")),
        "{ended:?}"
    );
    app.world_mut().trigger(Uncompact { entity: agent });
    app.world_mut().flush();
    app.update();
    assert_eq!(
        entries(app.world(), agent),
        original,
        "undo recovers original committed history"
    );
    drop(app);
    let (mut resumed, _) = start(&store);
    let agent = restored_agent(&mut resumed);
    assert!(agent.is_some());
    let Some(agent) = agent else {
        return;
    };
    assert_eq!(entries(resumed.world(), agent), original);
}

#[test]
fn overflow_still_summarizes_when_less_than_a_quarter_can_be_removed() {
    recoverable_overflow(200_000, false, 152_000);
}

#[test]
fn overflow_recovery_uses_the_routed_request_not_the_agents_smaller_window() {
    recoverable_overflow(40_000, true, 152_000);
}

#[test]
fn an_overestimated_retained_message_can_still_recover_from_overflow() {
    recoverable_overflow(200_000, false, 168_000);
}

fn recoverable_overflow(window: u32, routed: bool, words: usize) {
    let (mut app, wakes) = start(&MemoryStore::default());
    let Ok(status) = "400".parse() else {
        return;
    };
    let refusal = MockHttpResponse::error(
        status,
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens"}}"#,
    );
    let http = SequencedHttpClient::new([
        refusal,
        response(SUMMARY, "end_turn"),
        response("Recovered.", "end_turn"),
    ]);
    let Some(agent) = connect_anthropic(&mut app, &http, window) else {
        return;
    };
    if routed {
        let connection = app.world().get::<Connection>(agent).cloned();
        assert!(connection.is_some());
        let Some(mut connection) = connection else {
            return;
        };
        connection.spec = Arc::new(
            connection
                .spec
                .as_ref()
                .clone()
                .with_context_window(200_000),
        );
        app.insert_resource(Routed(connection))
            .add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    }
    for message in [
        Message::user(format!("{FACTS} {}", "word ".repeat(16_000))),
        Message::assistant("Recorded."),
    ] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    let newest = format!("NEWEST-BEGIN {} NEWEST-END", "word ".repeat(words));
    let old = entries(app.world(), agent)
        .into_iter()
        .map(|entry| entry.message)
        .collect::<Vec<_>>();
    let kept = [Arc::new(Message::user(newest.clone()))];
    assert!(estimate_tokens(&old) * 4 < estimate_tokens(&kept));
    if words == 168_000 {
        // The mock accepts the compacted request despite this estimate.
        // Its usage is scripted accounting, not a tokenizer accuracy claim.
        assert!(estimate_tokens(&kept) > 200_000);
    } else {
        assert!(estimate_tokens(&kept) < 200_000);
    }
    app.world_mut()
        .trigger(Deliver::new(agent, newest.clone(), DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("Recovered."));
    let requests = bodies(&http);
    assert_eq!(
        requests.len(),
        3,
        "one rejected request, one summary, one recovery"
    );
    assert!(
        requests
            .first()
            .is_some_and(|request| !asks_summary(request) && request.contains(&newest))
    );
    assert!(
        requests
            .get(1)
            .is_some_and(|request| asks_summary(request) && !request.contains("NEWEST-BEGIN"))
    );
    assert!(requests.get(2).is_some_and(|request| !asks_summary(request)
        && request.contains(&newest)
        && request.contains("amber-731")));
    assert!(entries(app.world(), agent).iter().any(holds_summary));
}

#[test]
fn a_transport_timeout_during_the_summary_preserves_history() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let partial = response_events("Partial text before timeout", "end_turn")
        .into_iter()
        .take(4)
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            )
        })
        .collect::<String>();
    let timeout = rig_core::http_client::Error::Instance(Box::new(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "synthetic transport timeout",
    )));
    let http = rig_core::test_utils::SequencedStreamingHttpClient::new(vec![
        Ok(partial.into()),
        Err(timeout),
    ]);
    let Some(agent) = connect_anthropic(&mut app, &http, 1_000_000) else {
        return;
    };
    seed(&mut app, agent, 1);
    let original = entries(app.world(), agent);
    compact(&mut app, &wakes, agent);
    assert_eq!(entries(app.world(), agent), original);
    drop(app);
    let (mut resumed, wakes) = start(&store);
    let http = SequencedHttpClient::new([response("Ready.", "end_turn")]);
    let Some(agent) = connect_anthropic(&mut resumed, &http, 1_000_000) else {
        return;
    };
    assert_eq!(entries(resumed.world(), agent), original);
    answered_turns(&mut resumed, &wakes, agent, 1);
    assert!(
        bodies(&http)
            .last()
            .is_some_and(|request| request.contains(FACTS))
    );
}

#[test]
fn failed_summaries_preserve_history_and_restore_it() {
    let failures = [
        (
            "provider error",
            sse([
                json!({"type":"error", "error":{"type":"invalid_request_error", "message":"synthetic rejection"}}),
            ]),
        ),
        (
            "malformed frame",
            MockHttpResponse::success_typed(
                "event: message_start\ndata: {invalid\n\n",
                "text/event-stream",
            ),
        ),
        ("empty summary", response("", "end_turn")),
        ("output limit", response("Partial summary", "max_tokens")),
    ];
    for (case, failure) in failures {
        let store = MemoryStore::default();
        let (mut app, wakes) = start(&store);
        let http = SequencedHttpClient::new([failure, response("Ready.", "end_turn")]);
        let Some(agent) = connect_anthropic(&mut app, &http, 1_000_000) else {
            return;
        };
        seed(&mut app, agent, 1);
        let original = entries(app.world(), agent);
        compact(&mut app, &wakes, agent);
        assert_eq!(entries(app.world(), agent), original, "{case}");
        assert_eq!(http.requests().len(), 1, "{case}: no retry loop");
        drop(app);
        let (mut resumed, wakes) = start(&store);
        let Some(agent) = connect_anthropic(&mut resumed, &http, 1_000_000) else {
            return;
        };
        assert_eq!(entries(resumed.world(), agent), original, "{case}: restore");
        answered_turns(&mut resumed, &wakes, agent, 1);
        assert!(
            bodies(&http)
                .last()
                .is_some_and(|request| request.contains(FACTS)),
            "{case}: original facts sent after resume"
        );
    }
}
