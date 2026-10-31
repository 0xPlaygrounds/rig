use rig_core::message::Message;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, collect, first_agent, log_ended, log_tool_outputs,
    reply, run_until, scratch_home, scripted_models,
};
use schemars::JsonSchema;

use super::*;

/// The test tool's arguments.
#[derive(Deserialize, JsonSchema)]
struct Pick {
    path: String,
    count: i64,
}

/// `pick` says what it was given.
fn picked(called: On<ToolCalled<Pick>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let text = format!("{} x{}", called.args.path, called.args.count);
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("test"), ToolResult::success(text.into()));
    }
}

/// A call of `pick` streamed as argument fragments.
fn streamed(id: &str, fragments: &[&str]) -> Vec<MockStreamEvent> {
    let mut events = vec![MockStreamEvent::tool_call_name_delta(id, "pick")];
    events.extend(
        fragments
            .iter()
            .map(|fragment| MockStreamEvent::tool_call_arguments_delta(id, *fragment)),
    );
    events.push(MockStreamEvent::tool_call_end(id));
    events
}

/// The valid call, exactly as the model sends it.
const VALID: &str = r#"{"count":4,"path":"d.txt"}"#;

#[test]
fn broken_arguments_run_repaired_and_are_kept_repaired() {
    let mut turn = streamed("c1", &["{'path': 'a.txt', ", "'count': '2',}"]);
    turn.extend(streamed(
        "c2",
        &[r#""{\"path\":\"b.txt\","#, r#"\"count\":1}""#],
    ));
    turn.extend(streamed("c3", &[r#"{"path": "#]));
    turn.extend(streamed("c4", &[VALID]));
    turn.push(MockStreamEvent::final_response_with_default_usage());
    let model = MockCompletionModel::from_stream_turns([turn, reply("done")]);

    let home = scratch_home("corpus-ho-tool-arg-repair");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ToolArgRepairPlugin))
        .add_open_tool("pick", "Pick a file.", ToolOptions::default(), picked)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "Pick.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    assert_eq!(output("c1"), "a.txt x2");
    assert_eq!(output("c2"), "b.txt x1");
    assert_eq!(output("c4"), "d.txt x4");
    let refused = output("c3");
    assert!(refused.contains("are not a JSON object"), "{refused}");
    assert!(
        refused.contains(
            r#"arguments must be a JSON object like: {"path": <string>, "count": <integer>}"#
        ),
        "{refused}"
    );

    // The conversation holds valid objects, and the valid call as sent.
    let conversation = app.world().get::<Conversation>(agent);
    let stored: Vec<(String, Option<String>, String)> = conversation
        .map(|conversation| conversation.messages())
        .unwrap_or_default()
        .iter()
        .filter_map(|message| match &**message {
            Message::Assistant(reply) => Some(reply.tool_calls().cloned().collect::<Vec<_>>()),
            _ => None,
        })
        .flatten()
        .map(|call| {
            let function = call.function;
            (
                call.id.to_string(),
                function.invalid_arguments.clone(),
                function.arguments_value().to_string(),
            )
        })
        .collect();
    let args = |id: &str| stored.iter().find(|(call, ..)| call == id).cloned();
    let typed = r#"{"path":"a.txt","count":2}"#.to_owned();
    assert_eq!(args("c1"), Some(("c1".to_owned(), None, typed)));
    let unwrapped = r#"{"path":"b.txt","count":1}"#.to_owned();
    assert_eq!(args("c2"), Some(("c2".to_owned(), None, unwrapped)));
    assert_eq!(args("c4"), Some(("c4".to_owned(), None, VALID.to_owned())));
    assert!(
        args("c3").is_some_and(|(_, raw, _)| raw.is_some()),
        "{stored:?}"
    );

    // `/repairs` counts the kinds per model.
    app.world_mut().trigger(RunCommand::new(agent, "repairs"));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let counted = notices.iter().any(|notice| {
        notice.text.contains("test/model:")
            && notice.text.contains("single-quotes 1")
            && notice.text.contains("trailing-comma 1")
            && notice.text.contains("string-to-integer 1")
    });
    assert!(
        counted,
        "{:?}",
        notices.iter().map(|n| &n.text).collect::<Vec<_>>()
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ToolArgRepairPlugin);
}
