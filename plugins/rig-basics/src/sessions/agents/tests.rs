use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockHttpResponse, SequencedHttpClient};
use rig_ecs::AgentPlugin;
use rig_ecs::fork::ForkedFrom;
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::model::{Connection, ModelChoice};
use rig_ecs::prelude::{Agent, AgentId, Conversation, MessageId, TurnOf, TurnOutcome};
use rig_ecs::session_log::SessionLogPlugin;
use rig_harness::commands::RunCommand;
use rig_harness::pick::{PickPlugin, Picker};
use rig_harness::prelude::{MainAgentQuery, ShownAgent, main_agent};
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, log_ended, press, reply, run_until, scripted_models, tui_view,
};
use rig_tui::{
    Editor, MainEditor, MainPane, ShowAgent, TranscriptItem, TranscriptView, TuiTestScreen,
};

use super::*;

#[test]
fn agents_shows_the_columns_other_plugins_add_to_an_agents_line() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, AgentsPlugin));
    let world = app.world_mut();
    let agent = world.spawn((Agent, Name::new("main"))).id();
    let short = world.get::<AgentId>(agent).map(|id| id.short().to_owned());
    world.spawn((AgentColumn::new("spent", "$0.10"), ColumnOf(agent)));
    world.spawn((AgentColumn::new("queued", "2 queued"), ColumnOf(agent)));
    world.trigger(RunCommand::new(agent, "agents".to_owned()));
    world.flush();
    let mut pickers = world.query::<&Picker>();
    let labels: Vec<&str> = pickers
        .iter(world)
        .flat_map(|picker| &picker.items)
        .map(|item| item.label.as_str())
        .collect();
    let short = short.unwrap_or_default();
    assert_eq!(
        labels,
        [format!(
            "main {short} · no model · idle · current · 2 queued · $0.10"
        )]
    );
}

/// The lines of an open `/agents` follow the agents: which is current,
/// which a fork came from, and whether one works.
#[test]
fn agents_marks_the_current_agent_and_forks_and_follows_their_state() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, AgentsPlugin));
    let first = AgentId("00000000-0000-4000-8000-000000000000".to_owned());
    let source = app
        .world_mut()
        .spawn((Agent, Name::new("agent"), first))
        .id();
    let id = |app: &App, agent: Entity| {
        let id = app.world().get::<AgentId>(agent);
        id.map(|id| (id.0.clone(), id.short().to_owned()))
            .unwrap_or_default()
    };
    let (source_id, source_short) = id(&app, source);
    let fork = (app.world_mut())
        .spawn((Agent, Name::new("agent"), ForkedFrom(source_id)))
        .id();
    let (_, fork_short) = id(&app, fork);
    app.update();
    app.world_mut()
        .trigger(RunCommand::new(source, "agents".to_owned()));
    let labels = |app: &mut App| {
        app.update();
        let world = app.world_mut();
        let mut pickers = world.query::<&Picker>();
        let mut labels: Vec<String> = (pickers.iter(world))
            .flat_map(|picker| &picker.items)
            .map(|item| item.label.clone())
            .collect();
        labels.sort();
        labels
    };
    let line =
        |short: &str, more: &str, state: &str| format!("agent {short} · no model · {state}{more}");
    let lines = |more: [&str; 2], state: &str| {
        let fork_of = format!("{} · fork of {source_short}", more[1]);
        let mut lines = vec![
            line(&source_short, more[0], "idle"),
            line(&fork_short, &fork_of, state),
        ];
        lines.sort();
        lines
    };
    assert_eq!(labels(&mut app), lines([" · current", ""], "idle"));

    app.world_mut().resource_mut::<ShownAgent>().0 = Some(fork);
    let turn = app.world_mut().spawn(TurnOf(fork)).id();
    assert_eq!(labels(&mut app), lines(["", " · current"], "working"));
    app.world_mut().despawn(turn);
    assert_eq!(labels(&mut app), lines(["", " · current"], "idle"));
}

/// When a line above the selected one leaves an open `/agents`, such as a
/// helper's that retired, Enter shows the agent that is selected on screen.
#[test]
fn enter_in_agents_shows_the_agent_selected_after_a_line_above_left() {
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (mut app, _wakes) = on(&MemoryStore::default(), &model);
    app.insert_resource(TuiTestScreen::new(100, 30));
    tui_view(&mut app);
    app.add_plugins((PickPlugin, AgentsPlugin));
    let ids = [1, 2, 3, 4].map(|n| AgentId(format!("0000000{n}-0000-4000-8000-000000000000")));
    let agents = ids.map(|id| app.world_mut().spawn((Agent, Name::new("agent"), id)).id());
    app.finish();
    app.cleanup();
    app.update();
    let [first, gone, selected, _] = agents;
    app.world_mut().trigger(ShowAgent { entity: first });
    app.world_mut()
        .trigger(RunCommand::new(first, "agents".to_owned()));
    app.update();
    press(&mut app, "down down");
    app.world_mut().despawn(gone);
    app.update();
    press(&mut app, "enter");
    assert_eq!(shown(&app), Some(selected));
}

/// Sends `text` to `agent` and runs frames until its turn, the `n`th of
/// the app, ended.
fn ask(app: &mut App, wakes: &Receiver<()>, (agent, text): (Entity, &str), n: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| world.resource::<Ended>().0.len() >= n);
    let ended = app.world().resource::<Ended>().0.last().cloned();
    assert!(matches!(ended, Some(TurnOutcome::Answered(_))), "{ended:?}");
}

/// The agent the user looks at.
fn shown(app: &App) -> Option<Entity> {
    app.world().resource::<ShownAgent>().0
}

/// An app on `store` with the agents plugin, keeping how turns end.
fn on(store: &MemoryStore, model: &MockCompletionModel) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::default())
        .insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    (app, wakes)
}

/// Enter on an earlier message of the user's forks the shown agent just
/// before it, shows the fork with the message back in the input, and a
/// resumed session shows the fork again and sends to it, though the
/// source's id sorts first.
#[test]
fn rewind_forks_before_a_message_shows_the_fork_and_a_resume_continues_it() {
    let model = MockCompletionModel::from_stream_turns([reply("One."), reply("Two.")]);
    let store = MemoryStore::default();
    let (mut app, wakes) = on(&store, &model);
    app.insert_resource(TuiTestScreen::new(100, 30));
    tui_view(&mut app);
    app.add_plugins(AgentsPlugin);
    let first = AgentId("00000000-0000-4000-8000-000000000000".to_owned());
    let agent = app
        .world_mut()
        .spawn((Agent, first, ModelChoice("test/model".to_owned())))
        .id();
    app.finish();
    app.cleanup();
    app.update();
    app.world_mut().trigger(ShowAgent { entity: agent });
    ask(&mut app, &wakes, (agent, "first question"), 1);
    ask(&mut app, &wakes, (agent, "second question"), 2);
    let second: Option<MessageId> = app.world().get::<Conversation>(agent).and_then(|c| c.id(2));
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut TranscriptView, With<MainPane>>();
    if let Ok(mut view) = main.single_mut(world) {
        view.selected = second.map(TranscriptItem::Message);
    }
    press(&mut app, "enter");
    app.update();

    let fork = shown(&app).filter(|fork| *fork != agent);
    assert!(fork.is_some(), "the fork is shown");
    let Some(fork) = fork else { return };
    let texts = |entity: Entity| -> Vec<String> {
        let conversation = app.world().get::<Conversation>(entity);
        let messages = conversation
            .map(|c| c.messages().to_vec())
            .unwrap_or_default();
        let text = |message: &rig_core::completion::Message| {
            let json = serde_json::to_value(message).unwrap_or_default();
            json.pointer("/content/0/text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        messages.iter().map(|message| text(message)).collect()
    };
    assert_eq!(texts(fork), ["first question", "One."]);
    assert_eq!(texts(agent).len(), 4, "the source is left as it was");
    let world = app.world_mut();
    let mut input = world.query_filtered::<&Editor, With<MainEditor>>();
    let typed = input
        .single(world)
        .ok()
        .map(|input| input.text().to_owned());
    assert_eq!(typed.as_deref(), Some("second question"));
    let fork_id = app.world().get::<AgentId>(fork).cloned();
    app.update();
    drop(app);

    // The session resumed shows the agent shown when it stopped, and talks
    // to it.
    let (mut again, _wakes) = on(&store, &model);
    again.add_plugins(AgentsPlugin);
    again.finish();
    again.update();
    let id = |agent: Option<Entity>| agent.and_then(|agent| again.world().get::<AgentId>(agent));
    assert_eq!(id(shown(&again)), fork_id.as_ref());
    let main = again
        .world_mut()
        .run_system_cached(|agents: MainAgentQuery| main_agent(&agents))
        .ok()
        .flatten();
    let id = |agent: Option<Entity>| agent.and_then(|agent| again.world().get::<AgentId>(agent));
    assert_eq!(id(main), fork_id.as_ref());
    drop(again);

    // A program without the app reads the same main agent.
    let saved = rig_harness::rig_memory::Session::read(&store).ok();
    let main = saved.as_ref().and_then(|saved| saved.main_agent());
    assert_eq!(
        main.map(|agent| agent.id.as_str()),
        fork_id.as_ref().map(|id| id.0.as_str())
    );
}

/// A Responses reply streamed with `items` as its output.
fn streamed(items: Vec<serde_json::Value>) -> MockHttpResponse {
    let mut body = String::new();
    for (index, item) in items.iter().enumerate() {
        let event = serde_json::json!({
            "type": "response.output_item.done", "output_index": index, "item": item,
        });
        body.push_str(&format!(
            "event: response.output_item.done\ndata: {event}\n\n"
        ));
    }
    let usage = serde_json::json!({
        "input_tokens": 900, "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 905,
    });
    let done = serde_json::json!({"type": "response.completed", "response": {
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.4",
        "output": items, "usage": usage,
    }});
    body.push_str(&format!("event: response.completed\ndata: {done}\n\n"));
    MockHttpResponse::success_typed(body, "text/event-stream")
}

/// A reply's reasoning item `n`, with its ciphertext, and its text.
fn reasoned(n: usize, text: &str) -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "reasoning", "id": format!("rs_{n}"), "summary": [], "content": [],
            "encrypted_content": format!("ciphertext-{n}"),
        }),
        serde_json::json!({
            "type": "message", "id": format!("msg_{n}"), "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }),
    ]
}

/// The catalog's `model` through the real Responses wire of `dialect`
/// over `http`, a scripted transport.
fn responses(
    (dialect, model): (&rig_core::providers::openai::wire::Dialect, &str),
    http: &SequencedHttpClient,
) -> Option<Connection> {
    let served = rig_core::providers::openai::OpenAIConfig::with_key(dialect, "key")
        .connect(http.clone())
        .completion(model);
    let spec = rig_core::catalog::Catalog::builtin()
        .resolve(&format!("{}/{model}", dialect.name))
        .ok()?
        .shared();
    let handler = rig_core::serve::adapters::ModelAdapter::<rig_core::operation::Completion>::new(
        spec.reference(),
        served,
    );
    Some(Connection {
        spec,
        handler: rig_ecs::effects::Handler(rig_core::serve::ErasedHandler::new(handler)),
    })
}

/// Whether `next` is `previous` with items appended to its last field: its
/// bytes up to that field's closing bracket, then more.
fn extends(previous: &str, next: &str) -> bool {
    previous.strip_suffix("]}").is_some_and(|open| {
        next.strip_prefix(open)
            .is_some_and(|more| more.starts_with(','))
    })
}

/// A fork's first request starts with the bytes of its source's last one,
/// through the real OpenAI and ChatGPT Responses wires: same system prompt,
/// tools, options and prompt-cache key, the same items, reasoning
/// ciphertext included, so it reads what the source's requests cached.
#[test]
fn a_forks_first_request_starts_with_the_bytes_of_its_sources_last() {
    use rig_core::providers::{chatgpt, openai};
    for (dialect, model) in [
        (&openai::wire::OPENAI, "gpt-5.4"),
        (&chatgpt::DIALECT, "gpt-5.4"),
    ] {
        let name = dialect.name;
        let http = SequencedHttpClient::new([
            streamed(reasoned(1, "One.")),
            streamed(reasoned(2, "Two.")),
            streamed(reasoned(3, "Three.")),
        ]);
        let unused = MockCompletionModel::from_stream_turns([reply("unused")]);
        let (mut app, wakes) = on(&MemoryStore::default(), &unused);
        app.add_plugins(AgentsPlugin);
        let Some(connection) = responses((dialect, model), &http) else {
            return;
        };
        let agent = app.world_mut().spawn((Agent, connection)).id();
        app.finish();
        app.update();
        ask(&mut app, &wakes, (agent, "first question"), 1);
        ask(&mut app, &wakes, (agent, "second question"), 2);
        app.world_mut()
            .trigger(RunCommand::new(agent, "fork".to_owned()));
        app.update();
        let fork = shown(&app).filter(|fork| *fork != agent);
        assert!(fork.is_some(), "{name}: the fork is shown");
        let Some(fork) = fork else { return };
        ask(&mut app, &wakes, (fork, "third question"), 3);
        let bodies: Vec<String> = http
            .requests()
            .iter()
            .map(|request| String::from_utf8_lossy(&request.body).into_owned())
            .collect();
        let [first, second, forked] = bodies.as_slice() else {
            assert_eq!(bodies.len(), 3, "{name}: {bodies:#?}");
            return;
        };
        assert!(extends(first, second), "{name}:\n{first}\n{second}");
        assert!(extends(second, forked), "{name}:\n{second}\n{forked}");
        assert!(
            forked.contains(r#""encrypted_content":"ciphertext-2""#),
            "{name}: {forked}"
        );
        let key = app.world().get::<AgentId>(agent).map(|id| id.0.clone());
        let key = key.map(|key| format!(r#""prompt_cache_key":"{key}""#));
        assert!(
            key.is_some_and(|key| forked.contains(&key)),
            "{name}: the source's cache key: {forked}"
        );
    }
}
