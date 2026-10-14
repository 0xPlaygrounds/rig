use rig_core::completion::{CompletionRequest, Usage};
use rig_core::providers::anthropic::{AnthropicConfig, Messages};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::wire::{Body, Mode, Wire};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Collected, Ended, app_on, collect, connect, log_ended, run_until};
use rig_tui::TuiPlugin;
use serde_json::Value;

use super::*;

/// The "current time" section: a new text at each turn.
#[derive(Resource)]
struct Clock(Entity, u32);

fn tick(_: On<Add<TurnOf>>, mut clock: ResMut<Clock>, mut sections: Query<&mut PromptSection>) {
    clock.1 += 1;
    if let Ok(mut section) = sections.get_mut(clock.0) {
        section.text = format!("It is {}:00.", 9 + clock.1);
    }
}

fn answered(input: u64, cached: u64) -> Vec<MockStreamEvent> {
    let mut usage = Usage::default();
    usage.input_tokens = Some(input);
    usage.cached_input_tokens = Some(cached);
    vec![
        MockStreamEvent::text("ok"),
        MockStreamEvent::final_response(usage),
    ]
}

fn noop(_: On<ToolCalled<Value>>) {}

/// `request` as an Anthropic Messages body.
fn on_the_wire(request: &CompletionRequest) -> Value {
    let wire = Messages {
        provider: AnthropicConfig::new("test-key"),
        model: "claude-sonnet-4-6".to_owned(),
        default_max_tokens: Some(1024),
        prompt_caching: false,
        static_prefix_cache_ttl: None,
        strict_tools: false,
        tool_input_streaming: Default::default(),
        facts: Default::default(),
    };
    let Ok(encoded) = wire.encode(request.clone(), Mode::Streaming) else {
        return Value::Null;
    };
    let Body::Bytes(bytes) = encoded.request.into_body() else {
        return Value::Null;
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn marked(block: &Value) -> bool {
    block.get("cache_control").is_some()
}

/// The indexes of the wire messages with a marked block.
fn marked_messages(body: &Value) -> Vec<usize> {
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    (0..messages.len())
        .filter(|&at| {
            messages
                .get(at)
                .and_then(|message| message["content"].as_array())
                .is_some_and(|blocks| blocks.iter().any(marked))
        })
        .collect()
}

#[test]
fn requests_keep_their_prefix_and_breaks_are_named() {
    let model = MockCompletionModel::from_stream_turns([
        answered(1000, 0),
        answered(1000, 500),
        answered(1000, 900),
    ]);
    let connection = connect(&model);
    assert!(connection.is_some(), "the scripted model connects");
    let Some(connection) = connection else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, TuiPlugin, CacheShapingPlugin))
        .add_open_tool("zoom", "Zoom in.", ToolOptions::default(), noop)
        .add_open_tool("aim", "Aim.", ToolOptions::default(), noop)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(tick);
    collect::<Notice>(&mut app);
    let clock = app
        .world_mut()
        .spawn(PromptSection::new(
            PromptSection::ORDER_ENVIRONMENT,
            "time",
            "",
        ))
        .id();
    app.world_mut().spawn(PromptSection::new(
        PromptSection::ORDER_PROJECT,
        "project",
        "Be brief.",
    ));
    app.insert_resource(Clock(clock, 0));
    app.finish();
    app.update();
    let agent = app.world_mut().spawn((Agent, connection)).id();
    for (turn, text) in ["one", "two", "three"].into_iter().enumerate() {
        if turn == 2 {
            app.add_open_tool("lsp_hover", "Hover.", ToolOptions::default(), noop);
        }
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == turn + 1
        });
    }
    app.update();

    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    let bodies: Vec<Value> = requests.iter().map(on_the_wire).collect();
    let [first, second, third] = bodies.as_slice() else {
        return;
    };
    // The clock is no part of the system prompt: it ends each request's
    // last message, after every marker.
    for (request, body) in requests.iter().zip(&bodies) {
        let system = request.system_instructions().unwrap_or_default();
        assert!(
            !system.contains("<time>") && system.contains("<project>"),
            "{system}"
        );
        let messages = body["messages"].as_array().cloned().unwrap_or_default();
        let last = messages.last().map(Value::to_string).unwrap_or_default();
        assert!(
            last.contains("<time>") && !last.contains("cache_control"),
            "{last}"
        );
        assert!(
            marked_messages(body)
                .iter()
                .all(|&at| at + 1 < messages.len())
        );
    }
    // Requests 2 and 3: the last tool, the system prompt and the end of
    // the previous request's messages are marked.
    for body in [second, third] {
        let tools = body["tools"].as_array().cloned().unwrap_or_default();
        assert!(tools.last().is_some_and(marked), "{body}");
        assert!(
            body["system"]
                .as_array()
                .and_then(|s| s.last())
                .is_some_and(marked)
        );
        assert_eq!(marked_messages(body).len(), 1, "{body}");
    }
    // The end of the previous request's messages, without the clock.
    assert_eq!(marked_messages(second), [0]);
    assert_eq!(marked_messages(third), [2]);
    // The prefix up to the system prompt's marker is the same bytes.
    assert_eq!(first["tools"].to_string(), second["tools"].to_string());
    assert_eq!(first["system"].to_string(), second["system"].to_string());
    assert_eq!(first["system"].to_string(), third["system"].to_string());
    let names = |body: &Value| -> Vec<String> {
        let tools = body["tools"].as_array().cloned().unwrap_or_default();
        tools.iter().map(|tool| tool["name"].to_string()).collect()
    };
    assert_eq!(names(third), ["\"aim\"", "\"lsp_hover\"", "\"zoom\""]);

    // The conversation is untouched.
    let conversation = app.world().get::<Conversation>(agent);
    let kept = serde_json::to_string(&conversation.map(|c| c.messages().to_vec()));
    assert!(!kept.unwrap_or_default().contains("<time>"));

    // 900 of 1000 cached last; 1400 of 3000 in the session.
    let shown = app.world().get::<StatusItems>(agent).and_then(|items| {
        items
            .iter()
            .find(|item| (item.side, item.order) == (RATE.side, RATE.order))
            .map(|item| item.text.clone())
    });
    assert_eq!(shown.as_deref(), Some("cache 90% / 47%"));

    app.world_mut()
        .trigger(RunCommand::new(agent, "cache why".to_owned()));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let why = notices
        .iter()
        .find(|notice| notice.text.starts_with("request "));
    let why = why.map(|notice| notice.text.as_str()).unwrap_or_default();
    assert_eq!(why, "request 3: tool list changed: `lsp_hover` added");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CacheShapingPlugin);
}
