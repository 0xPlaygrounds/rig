use std::sync::mpsc::Receiver;

use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{app_on, calls, reply, run_until, scripted_models};

use rig_harness::view::mirror;
use rig_harness::view::protocol::event::{Event, TurnStarted};
use rig_harness::view::protocol::{AgentId as WireId, EventRecord, Seq, SessionId};
use rig_tui::{Editor, MainEditor};

use super::*;

/// A tool the test answers itself, or never.
fn open_tool(_: On<ToolCalled<serde_json::Value>>) {}

/// A `shell` call's arguments.
#[derive(Deserialize, schemars::JsonSchema)]
struct Command {
    /// The command.
    #[expect(dead_code, reason = "only declared")]
    command: String,
}

/// `shell`, whose calls the test holds open and answers itself.
fn held_shell(_: On<ToolCalled<Command>>) {}

/// The kernel, the agent half (which adds subagents itself), a worker's
/// outbox and four open tools on `store`, the
/// models `test/main` and `test/side`, after the first frame; and the main
/// agent, on `test/main`.
fn start(
    store: &MemoryStore,
    main: &MockCompletionModel,
    side: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    let reads = || ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    app.insert_resource(Settings::from_toml("[btw]\nmodel = \"test/side\""))
        .insert_resource(scripted_models([("test/main", main), ("test/side", side)]))
        .init_resource::<WorkerOutbox>()
        .add_plugins((AgentPlugin, SessionLogPlugin, SideQuestionPlugin))
        .add_open_tool("read", "Read a file.", reads(), open_tool)
        .add_open_tool("search", "Search files.", reads(), open_tool)
        .add_open_tool(
            "shell",
            "Run a command.",
            ToolOptions::default(),
            held_shell,
        )
        .add_open_tool("write", "Write a file.", ToolOptions::default(), open_tool);
    app.finish();
    app.update();
    let world = app.world_mut();
    let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    let agent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    (app, wakes, agent)
}

/// Whether `agent` asked `model` `requests` times and its turn ended.
fn idle_after(
    model: &MockCompletionModel,
    requests: usize,
    agent: Entity,
) -> impl Fn(&World) -> bool {
    move |world: &World| {
        model.request_count() >= requests && world.get::<ActiveTurn>(agent).is_none()
    }
}

/// The text of every message `request` sends.
fn sent(request: &CompletionRequest) -> String {
    format!("{:?}", request.chat_history)
}

/// The open `shell` call, if any.
fn open_shell(world: &mut World) -> Option<Entity> {
    let mut calls =
        world.query_filtered::<(Entity, &ToolRequest), (With<Started>, Without<Outcome>)>();
    calls
        .iter(world)
        .find(|(_, request)| request.call.function.name.as_str() == "shell")
        .map(|(call, _)| call)
}

/// The side conversations in `agent`'s conversation: the message each
/// follows, and its note.
fn side_entries(world: &World, agent: Entity) -> Vec<(Option<MessageId>, Note)> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    (conversation.asides())
        .filter(|(_, aside)| aside.author == Author::plugin(TITLE))
        .filter_map(|(_, aside)| Note::of(aside).map(|note| (aside.after, note)))
        .collect()
}

/// The last side conversation the agent half published.
fn published(app: &App) -> Option<BtwState> {
    let outbox = app.world().resource::<WorkerOutbox>();
    let last = outbox
        .events
        .iter()
        .rev()
        .find(|event| event.kind == BtwState::KIND);
    last.and_then(|event| serde_json::from_value(event.data.clone()).ok())
}

/// Runs `line` as the agent's command, as a view's key sends it.
fn command(app: &mut App, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
}

#[test]
fn a_side_question_is_answered_beside_a_running_turn_and_kept_apart() {
    let store = MemoryStore::default();
    let main = MockCompletionModel::from_stream_turns([
        calls("c1", "shell", serde_json::json!({ "command": "migrate" })),
        reply("Migrated."),
    ]);
    let side = MockCompletionModel::from_stream_turns([reply("X is Y"), reply("Z is W")]);
    let (mut app, wakes, agent) = start(&store, &main, &side);

    // The main turn's `shell` call runs and is held open.
    let migrate = Deliver::new(agent, "Migrate the DB.", DeliveryMode::Steer);
    app.world_mut().trigger(migrate);
    run_until(&mut app, &wakes, |world| {
        let calls = world.try_query_filtered::<&ToolRequest, (With<Started>, Without<Outcome>)>();
        calls.is_some_and(|mut calls| {
            calls
                .iter(world)
                .any(|request| request.call.function.name.as_str() == "shell")
        })
    });
    assert!(open_shell(app.world_mut()).is_some());

    // `/btw` while it runs: the side agent answers in the panel.
    let btw = Deliver::new(agent, "/btw what is X?", DeliveryMode::Steer);
    app.world_mut().trigger(btw);
    run_until(&mut app, &wakes, |world| {
        world.resource::<SideChat>().last_answer() == Some("X is Y")
    });
    assert!(
        open_shell(app.world_mut()).is_some(),
        "the main agent's call still runs"
    );
    app.update();
    let chat = app.world().resource::<SideChat>();
    assert_eq!(chat.last_answer(), Some("X is Y"));
    let id = |entity: Option<Entity>| {
        entity
            .and_then(|entity| app.world().get::<AgentId>(entity))
            .map(|id| id.0.clone())
    };
    let state = BtwState {
        main: id(Some(agent)),
        side: id(chat.agent),
        last_answer: Some("X is Y".to_owned()),
    };
    assert_eq!(published(&app), Some(state), "the views are told");
    let asked = side.requests();
    let first = asked.first().map(sent).unwrap_or_default();
    for part in ["Migrate the DB.", "migrate", "what is X?"] {
        assert!(first.contains(part), "{part} not in {first}");
    }
    let mut offered: Vec<String> = asked
        .first()
        .map(|request| {
            request
                .tools
                .iter()
                .map(|tool| tool.name.to_string())
                .collect()
        })
        .unwrap_or_default();
    offered.sort();
    assert_eq!(offered, ["read", "search"]);

    // The held call is released: the main agent's next request knows
    // nothing of the side question.
    if let Some(call) = open_shell(app.world_mut())
        && let Some(mut decision) = app.world_mut().get_mut::<Decision>(call)
    {
        decision.decide(
            Decider::plugin("fh-side-question"),
            ToolResult::success("migrated".into()),
        );
    }
    run_until(&mut app, &wakes, idle_after(&main, 2, agent));
    let main_requests = main.requests();
    assert_eq!(main_requests.len(), 2);
    for request in &main_requests {
        let text = sent(request);
        assert!(
            !text.contains("what is X") && !text.contains("X is Y"),
            "{text}"
        );
    }

    // A follow-up, as the view's editor sends it, goes to the same side
    // agent.
    command(&mut app, agent, "btw and Z?");
    run_until(&mut app, &wakes, |world| {
        world.resource::<SideChat>().last_answer() == Some("Z is W")
    });
    let follow_up = side.requests().last().map(sent).unwrap_or_default();
    assert!(
        follow_up.contains("what is X?") && follow_up.contains("and Z?"),
        "{follow_up}"
    );

    // Closed, as the view's Ctrl+Y closes it: nothing was sent.
    let before = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| c.messages().len());
    command(&mut app, agent, "btw close");
    assert_eq!(published(&app), Some(BtwState::default()));
    for _ in 0..3 {
        app.update();
    }
    let after = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| c.messages().len());
    assert_eq!(before, after);
    assert_eq!(main.requests().len(), 2);
    assert!(app.world().resource::<SideChat>().main.is_none());
    // The side conversation is an entry after the message it was asked at,
    // the reply calling `shell`.
    let at = app.world().get::<Conversation>(agent).and_then(|c| c.id(1));
    let entries = side_entries(app.world(), agent);
    assert_eq!(entries.len(), 1);
    let entry = entries.first().cloned();
    assert_eq!(entry.as_ref().map(|entry| entry.0), Some(at));
    drop(app);

    // Resumed, the side exchange is its own entry, and the main agent's
    // next request does not carry it.
    let main = MockCompletionModel::from_stream_turns([reply("Next.")]);
    let side = MockCompletionModel::from_stream_turns(Vec::<
        Vec<rig_core::test_utils::MockStreamEvent>,
    >::new());
    let (mut app, wakes, agent) = start(&store, &main, &side);
    let kept = side_entries(app.world(), agent);
    assert_eq!(kept.first(), entry.as_ref());
    assert_eq!(kept.len(), 1);
    let kept = kept
        .first()
        .map(|(_, note)| (note.title.as_str(), note.text.as_str()));
    let text = "› what is X?\nX is Y\n\n› and Z?\nZ is W";
    assert_eq!(kept, Some(("btw: what is X? (+1 follow-ups)", text)));
    let world = app.world_mut();
    let mut sides = world.query::<&SideAgent>();
    assert_eq!(sides.iter(world).count(), 0, "the side agent was retired");
    app.world_mut()
        .trigger(Deliver::new(agent, "And now?", DeliveryMode::Steer));
    run_until(&mut app, &wakes, idle_after(&main, 1, agent));
    let requests = main.requests();
    let text = requests.first().map(sent).unwrap_or_default();
    assert!(
        text.contains("And now?") && text.contains("Migrate the DB."),
        "{text}"
    );
    assert!(
        !text.contains("what is X") && !text.contains("X is Y"),
        "{text}"
    );
}

/// The view half opens its panels as the agent half's state says and
/// sends what is typed and pressed back as `/btw` commands; Ctrl+Y puts
/// the last answer in the main input.
#[test]
fn the_view_shows_the_side_agent_and_sends_its_keys_as_commands() {
    let mut view = App::new();
    view.add_plugins((rig_tui::TuiPlugin, SideQuestionView));
    view.finish();
    let sent = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let log = sent.clone();
    view.add_observer(move |run: On<RunCommand>| {
        if let Ok(mut log) = log.lock() {
            log.push(run.line.clone());
        }
    });
    let started = |agent: &str| {
        let event = Event::TurnStarted(TurnStarted {
            agent_id: WireId(agent.to_owned()),
            turn_id: format!("t-{agent}"),
        });
        (event.kind().to_owned(), event.encode())
    };
    let state = BtwState {
        main: Some("main".to_owned()),
        side: Some("side".to_owned()),
        last_answer: Some("X is Y".to_owned()),
    };
    let state = serde_json::to_value(&state).unwrap_or_default();
    let records = [
        started("main"),
        started("side"),
        (BtwState::KIND.to_owned(), state),
    ];
    let records = records
        .into_iter()
        .zip(1..)
        .map(|((kind, data), seq)| EventRecord {
            session: SessionId("s-1".into()),
            seq: Seq(seq),
            at_ms: 0,
            source: None,
            kind,
            data,
        });
    mirror::apply(view.world_mut(), records);
    view.update();

    let world = view.world();
    let shown = world.resource::<SideView>();
    let side = world.resource::<Mirror>().agent("side");
    assert_eq!(
        world.resource::<KeyFocus>().0,
        Some(shown.editor),
        "the keys"
    );
    assert!(world.get::<TuiPanel>(shown.editor).is_some());
    let pane = world.get::<TranscriptPane>(shown.pane);
    let drawn = pane
        .map(|pane| matches!(pane.source, TranscriptSource::Agent(agent) if Some(agent) == side));
    assert_eq!(drawn, Some(true), "the side agent is drawn in its pane");

    let editor = shown.editor;
    if let Some(mut typed) = view.world_mut().get_mut::<Editor>(editor) {
        typed.set("and Z?");
    }
    press(&mut view, "Ask the side agent");
    press(&mut view, "Send the side answer to the agent");
    let world = view.world_mut();
    let input = world
        .query_filtered::<&Editor, With<MainEditor>>()
        .single(world)
        .map(|editor| editor.text().to_owned())
        .ok();
    assert_eq!(input.as_deref(), Some("X is Y"));
    let sent = sent.lock().map(|sent| sent.clone()).unwrap_or_default();
    assert_eq!(sent, ["btw and Z?", "btw close"]);
}

/// Presses the view's binding labelled `label`.
fn press(app: &mut App, label: &str) {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let found: Vec<Entity> = bindings
        .iter(world)
        .filter(|(_, binding)| binding.label == label)
        .map(|(entity, _)| entity)
        .collect();
    assert_eq!(found.len(), 1, "{label}");
    for entity in found {
        world.trigger(KeyPressed {
            entity,
            agent: None,
            released: false,
        });
    }
    app.update();
}
