use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_on, collect, first_agent, log_ended, log_tool_outputs,
    reply, run_until, scripted_models,
};

use super::*;

const LOOP: &str = "spam line spam line spam\n";

/// What the provider's side of each model call was read of: the items
/// pulled from it, and whether it was read to its end.
#[derive(Resource, Clone, Default)]
struct Read {
    items: Arc<AtomicUsize>,
    to_end: Arc<AtomicBool>,
}

/// Puts a counter under the cutter on every call, so it sees what was
/// pulled from the provider.
fn count_reads(mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>, read: Res<Read>) {
    for mut adapted in &mut calls {
        let read = read.clone();
        let count = move |_| {
            let read = read.clone();
            move |item: Relayed| {
                read.items.fetch_add(1, Ordering::SeqCst);
                if matches!(item, Relayed::Done(_)) {
                    read.to_end.store(true, Ordering::SeqCst);
                }
                vec![item]
            }
        };
        let handler = adapted.0.handler.0.clone().map_stream("count", count);
        adapted.0.handler = Handler(handler);
    }
}

/// A `read` that answers at once.
fn read_file(
    called: On<ToolCalled<serde_json::Value>>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("test"),
            ToolResult::success("file text".into()),
        );
    }
}

/// The text of each assistant message, in order.
fn answers(world: &World, agent: Entity) -> Vec<String> {
    let messages = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages());
    let assistant = messages
        .unwrap_or_default()
        .iter()
        .filter_map(|message| match &**message {
            message::Message::Assistant(answer) => Some(answer),
            _ => None,
        });
    assistant
        .map(|answer| {
            answer
                .content
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect()
        })
        .collect()
}

#[test]
fn looping_and_role_play_replies_are_cut_where_they_go_wrong() {
    let mut looping = vec![MockStreamEvent::text("Answer.\n")];
    looping.extend(std::iter::repeat_n(MockStreamEvent::text(LOOP), 10));
    looping.push(MockStreamEvent::final_response_with_default_usage());
    // A finished `read`, a second call still streaming its arguments, then
    // looping text.
    let mut calling = vec![
        MockStreamEvent::tool_call("c1", "read", serde_json::json!({ "path": "a.rs" })),
        MockStreamEvent::tool_call_name_delta("c2", "read"),
        MockStreamEvent::tool_call_arguments_delta("c2", r#"{"path":"#),
    ];
    calling.extend(std::iter::repeat_n(MockStreamEvent::text(LOOP), 6));
    calling.push(MockStreamEvent::final_response_with_default_usage());
    let model = MockCompletionModel::from_stream_turns([
        looping,
        reply("Done.\nUser: thanks"),
        calling,
        reply("Read it."),
    ]);

    let (mut app, wakes) = app_on(&Default::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, StreamCutterPlugin))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read_file)
        .init_resource::<Read>()
        .add_systems(
            AgentPass,
            count_reads
                .in_set(PrepareSystems::Adapt)
                .before(StreamCutterSystems),
        )
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model_name = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
    });
    let ask = |app: &mut App, text: &str, turns: usize| {
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() >= turns
        });
    };

    // A line repeated: the reply keeps its first time, and the provider's
    // stream is left unread.
    ask(&mut app, "Go.", 1);
    let read = app.world().resource::<Read>().clone();
    assert!(
        !read.to_end.load(Ordering::SeqCst),
        "the stream was read to its end"
    );
    assert!(
        read.items.load(Ordering::SeqCst) < 12,
        "the stream was read on"
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let cut = notices
        .iter()
        .filter(|notice| notice.text.starts_with("cut a"))
        .count();
    assert_eq!(cut, 1, "{notices:?}");
    assert_eq!(
        answers(app.world(), agent),
        ["Answer.\nspam line spam line spam\n"]
    );

    // A role marker: everything before it.
    ask(&mut app, "And?", 2);
    assert_eq!(
        answers(app.world(), agent).get(1).map(String::as_str),
        Some("Done.")
    );

    // A finished call before the cut runs; the one still streaming does not.
    ask(&mut app, "Read a.rs.", 3);
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert!(
        outputs.contains_key("c1") && !outputs.contains_key("c2"),
        "{outputs:?}"
    );
    let answers = answers(app.world(), agent);
    assert_eq!(
        answers.get(2..).unwrap_or_default(),
        [LOOP.to_owned(), "Read it.".to_owned()]
    );
    assert_eq!(app.world().resource::<Cuts>().0.get("test/model"), Some(&3));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(StreamCutterPlugin);
}
