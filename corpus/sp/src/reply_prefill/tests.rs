use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use rig_core::catalog::{Compat, ModelSpec};
use rig_core::message::{AssistantMessage, Message as Said, UserContent};
use rig_core::providers::registry::ProviderId;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, collect, log_ended, reply, run_until, scripted_models,
    tui_view,
};
use rig_tui::{ShowAgent, TuiTestScreen};
use serde_json::json;

use super::*;

/// Holds every model stream's items until opened.
#[derive(Default)]
struct Valve {
    open: Mutex<bool>,
    opened: Condvar,
}

impl Valve {
    fn wait(&self) {
        let open = self
            .open
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ten = Duration::from_secs(10);
        self.opened
            .wait_timeout_while(open, ten, |open| !*open)
            .ok();
    }

    fn open(&self) {
        *self
            .open
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        self.opened.notify_all();
    }
}

/// An app with the plugin, an agent on `test/a` (which continues a
/// prefill) or `test/b` (which does not), a `read` tool, the terminal view
/// and the notices kept; each model's stream is held until `valve` opens.
fn app(
    a: &MockCompletionModel,
    b: &MockCompletionModel,
    valve: &Arc<Valve>,
) -> (App, Entity, std::sync::mpsc::Receiver<()>) {
    let mut models = scripted_models([("test/a", a), ("test/b", b)]);
    if let Ok(test) = ProviderId::declare("test") {
        let prefills = Compat::default().with_prefill(true);
        models.0.catalog_mut().insert(
            ModelSpec::new(test, "a")
                .with_tools(true)
                .with_compat(prefills),
        );
    }
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let valve = valve.clone();
    let held = move |mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>| {
        for mut adapted in &mut calls {
            let valve = valve.clone();
            let hold = move |_| {
                let valve = valve.clone();
                move |item| {
                    valve.wait();
                    vec![item]
                }
            };
            let handler = adapted.0.handler.0.clone().map_stream("held", hold);
            adapted.0.handler = Handler(handler);
        }
    };
    app.insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, ReplyPrefillPlugin))
        .add_systems(AgentPass, held.in_set(PrepareSystems::Adapt))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .insert_resource(TuiTestScreen::new(80, 30));
    tui_view(&mut app);
    collect::<Notice>(&mut app);
    let agent = app
        .world_mut()
        .spawn((Agent, ModelChoice("test/a".to_owned())))
        .id();
    app.finish();
    app.cleanup();
    app.update();
    app.world_mut().trigger(ShowAgent { entity: agent });
    (app, agent, wakes)
}

/// A `read` that answers with the file's text.
fn read(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("read"),
            ToolResult::success("fn main() {}".into()),
        );
    }
}

/// Runs `line` as typed, sends `prompt` unless empty, and runs until
/// `turns` turns ended.
fn ask(
    app: &mut App,
    wakes: &std::sync::mpsc::Receiver<()>,
    agent: Entity,
    line: &str,
    prompt: &str,
    turns: usize,
) {
    if !line.is_empty() {
        app.world_mut().trigger(RunCommand::new(agent, line));
        app.update();
    }
    if !prompt.is_empty() {
        app.world_mut()
            .trigger(Deliver::new(agent, prompt, DeliveryMode::Steer));
    }
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// The assistant messages of `agent`'s conversation.
fn answers(world: &World, agent: Entity) -> Vec<AssistantMessage> {
    let messages = world.get::<Conversation>(agent).map(|it| {
        it.messages()
            .iter()
            .map(|m| (**m).clone())
            .collect::<Vec<_>>()
    });
    let messages = messages.unwrap_or_default().into_iter();
    messages
        .filter_map(|message| match message {
            Said::Assistant(answer) => Some(answer),
            _ => None,
        })
        .collect()
}

/// The last message of `model`'s request `at`.
fn last_sent(model: &MockCompletionModel, at: usize) -> Option<Said> {
    let requests = model.requests();
    requests
        .get(at)
        .and_then(|request| request.chat_history.last().map(|m| (**m).clone()))
}

/// The text of `message` if it is the model's.
fn assistant_text(message: Option<&Said>) -> Option<String> {
    match message {
        Some(Said::Assistant(answer)) => Some(answer.text()),
        _ => None,
    }
}

fn screen(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

fn notices(app: &App) -> String {
    let notices = app.world().resource::<Collected<Notice>>().0.iter();
    notices
        .map(|notice| notice.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_prefill_starts_one_stored_reply_and_is_cleared_after_it() {
    let read_then_done = vec![
        MockStreamEvent::text(" 1. read"),
        MockStreamEvent::tool_call("r1", "read", json!({ "path": "src/main.rs" })),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let failed = vec![MockStreamEvent::Error(MockError::request(
        "the model broke",
    ))];
    let a = MockCompletionModel::from_stream_turns([
        reply(r#"a": 1}"#),
        read_then_done,
        reply("Done."),
        failed,
        reply(r#"x": 3}"#),
    ]);
    let b = MockCompletionModel::from_stream_turns([reply("Sure: nothing")]);
    let valve = Arc::new(Valve::default());
    let (mut app, agent, wakes) = app(&a, &b, &valve);

    // The prefill shows as the reply's start once the request is sent,
    // before any token.
    app.world_mut()
        .trigger(RunCommand::new(agent, r#"prefill {""#));
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "Give me JSON.", DeliveryMode::Steer));
    let mut sent = app
        .world_mut()
        .query_filtered::<(), (With<TurnCall>, With<Started>, Without<Settled>)>();
    run_until(&mut app, &wakes, |_| a.request_count() == 1);
    app.update();
    let shown = screen(&app);
    let started = sent.iter(app.world()).count();
    assert_eq!(started, 1, "{shown}");
    assert!(shown.lines().any(|row| row.trim() == r#"{""#), "{shown}");
    valve.open();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });

    assert_eq!(
        assistant_text(last_sent(&a, 0).as_ref()).as_deref(),
        Some(r#"{""#)
    );
    let texts: Vec<String> = answers(app.world(), agent)
        .iter()
        .map(AssistantMessage::text)
        .collect();
    assert_eq!(texts, [r#"{"a": 1}"#]);
    assert!(app.world().get::<Prefill>(agent).is_none());

    // A continuation's tool call runs; the request after its result
    // carries no prefill.
    ask(
        &mut app,
        &wakes,
        agent,
        "prefill Plan:",
        "Plan the change.",
        2,
    );
    assert_eq!(
        assistant_text(last_sent(&a, 1).as_ref()).as_deref(),
        Some("Plan:")
    );
    assert!(
        assistant_text(last_sent(&a, 2).as_ref()).is_none(),
        "{:?}",
        last_sent(&a, 2)
    );
    let stored = answers(app.world(), agent);
    let plan = stored.get(1);
    assert_eq!(
        plan.map(AssistantMessage::text).as_deref(),
        Some("Plan: 1. read")
    );
    let called = plan.is_some_and(|plan| plan.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(call) if call.id.to_string() == "r1")));
    let messages = app
        .world()
        .get::<Conversation>(agent)
        .map(|it| {
            it.messages()
                .iter()
                .map(|m| (**m).clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let answered = messages.iter().any(|message| match message {
        Said::User { content } => content.iter().any(|part| matches!(part, UserContent::ToolResult(result) if result.call.to_string() == "r1")),
        _ => false,
    });
    assert!(called && answered, "{messages:#?}");

    // B cannot continue one: refused by default, nothing sent, the prefill
    // kept.
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/b".to_owned(),
    });
    ask(&mut app, &wakes, agent, r#"prefill {""#, "Give me JSON.", 3);
    assert_eq!(b.request_count(), 0);
    ask(&mut app, &wakes, agent, "prefill", "", 3);
    app.update();
    let told = notices(&app);
    assert!(told.contains("cannot continue a prefilled reply"), "{told}");
    assert!(screen(&app).contains("cannot continue"), "{}", screen(&app));
    let shown = app
        .world()
        .resource::<Collected<Notice>>()
        .0
        .last()
        .map(|it| it.text.clone());
    assert_eq!(shown.as_deref(), Some(r#"The next reply starts with: {""#));
    assert_eq!(
        app.world().get::<Prefill>(agent),
        Some(&Prefill(r#"{""#.to_owned()))
    );

    // Asked to instruct, B is told to begin with it when the refused prompt
    // is sent again, and its reply is kept as it wrote it.
    app.world_mut().resource_mut::<PrefillSettings>().fallback = Fallback::Instruct;
    ask(&mut app, &wakes, agent, "", "Give me JSON.", 4);
    let asked = last_sent(&b, 0);
    assert!(
        format!("{asked:?}").contains(
            r#"exactly the following text, character for character, then continue it:\n{\""#
        ),
        "{asked:?}"
    );
    assert!(assistant_text(asked.as_ref()).is_none());
    let last = answers(app.world(), agent)
        .last()
        .map(AssistantMessage::text);
    assert_eq!(last.as_deref(), Some("Sure: nothing"));
    assert!(app.world().get::<Prefill>(agent).is_none());

    // On A, a request that fails before a token stores nothing and keeps
    // the prefill; sending the prompt again sends the prefill again.
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/a".to_owned(),
    });
    let before = answers(app.world(), agent).len();
    ask(&mut app, &wakes, agent, r#"prefill {""#, "Again.", 5);
    assert_eq!(answers(app.world(), agent).len(), before);
    assert!(app.world().get::<Prefill>(agent).is_some());
    ask(&mut app, &wakes, agent, "", "Again.", 6);
    assert_eq!(
        assistant_text(last_sent(&a, 4).as_ref()).as_deref(),
        Some(r#"{""#)
    );
    let last = answers(app.world(), agent)
        .last()
        .map(AssistantMessage::text);
    assert_eq!(last.as_deref(), Some(r#"{"x": 3}"#));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ReplyPrefillPlugin);
}
