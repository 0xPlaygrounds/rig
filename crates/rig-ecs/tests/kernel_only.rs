//! The kernel on its own: a headless app with Bevy's task pools, the agent
//! runtime and its session journal, and nothing else: no effect log. A
//! scripted model asks for one tool call, then answers; every call is
//! recorded as it settles, with ids after the ones given; a second app on the same store gets the
//! conversation back; a failed call is sent again once its backoff passed
//! on Bevy's clock; a reasoning setting the model does not take is reset;
//! a plugin's saved component changed on the exit frame is saved.
//! Only rig-ecs's public API is used, as a third-party plugin would.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use bevy_time::{Time, TimeUpdateStrategy, Virtual};
use rig_core::ProviderResponseError;
use rig_core::catalog::Catalog;
use rig_core::completion::{Message, Reasoning, Section, ToolDefinition};
use rig_core::effect::{EffectId, EffectRecord, HandlerKey, tool_key};
use rig_core::message::{ToolName, UserContent};
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::{ModelAdapter, ToolAdapter};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_ecs::effects::{Effects, Handler};
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_memory::MemoryStore;
use serde::Deserialize;

/// How long a turn may take before the test gives up.
const DEADLINE: Duration = Duration::from_secs(10);

/// Adds two numbers, counting its calls.
struct Add(Arc<AtomicU32>);

#[derive(Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Adds two integers.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
            "required": ["a", "b"],
        })
    }

    async fn call(&self, _: &mut ToolContext, args: AddArgs) -> Result<i64, ToolExecutionError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(args.a + args.b)
    }
}

/// How each turn ended, in order.
#[derive(Resource, Default)]
struct Ended(Vec<TurnOutcome>);

/// The records of the effects that settled, such as tool calls.
#[derive(Resource, Default)]
struct Settles(Vec<EffectRecord>);

/// A headless kernel app on `store`, with its loop's wakes.
fn kernel(store: &MemoryStore) -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .init_resource::<Settles>()
        .add_systems(
            Last,
            |mut recorded: MessageReader<EffectRecorded>, mut settles: ResMut<Settles>| {
                settles
                    .0
                    .extend(recorded.read().map(|record| record.0.clone()));
            },
        )
        .add_observer(
            |settled: On<bevy_ecs::lifecycle::Add<Settled>>,
             turns: Query<&TurnOutcome>,
             mut log: ResMut<Ended>| {
                log.0.extend(turns.get(settled.entity).cloned());
            },
        );
    app.finish();
    // A loop that sleeps for seconds counts its frames in full, as
    // rig-harness's does; Bevy's default stops at 250 ms.
    if let Some(mut time) = app.world_mut().get_resource_mut::<Time<Virtual>>() {
        time.set_max_delta(Duration::from_secs(60));
    }
    (app, wakes)
}

/// Runs frames, sleeping until a wake between them, as a windowless loop
/// does, until `done` or the deadline passed.
fn run_until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&mut World) -> bool) {
    let started = Instant::now();
    while started.elapsed() < DEADLINE {
        app.update();
        if done(app.world_mut()) {
            return;
        }
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
}

fn a_turn_ended(world: &mut World) -> bool {
    !world.resource::<Ended>().0.is_empty()
}

/// `model` as the built-in catalog's DeepSeek model.
fn connection(model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .shared();
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(
        spec.reference(),
        model.clone(),
    ));
    Some(Connection {
        spec,
        handler: Handler(handler),
    })
}

/// An agent of `app` on `model`.
fn connected(app: &mut App, model: &MockCompletionModel) -> Option<Entity> {
    let connection = connection(model)?;
    Some(app.world_mut().spawn((Agent, connection)).id())
}

fn ask(app: &mut App, agent: Entity) {
    // The first frame restores the (empty) session and starts the journal.
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "What is 2 + 3?", DeliveryMode::Steer));
}

fn messages(app: &App, agent: Entity) -> Vec<Message> {
    app.world()
        .get::<Conversation>(agent)
        .map(|conversation| {
            conversation
                .messages()
                .iter()
                .map(|message| (**message).clone())
                .collect()
        })
        .unwrap_or_default()
}

/// The agents of `app`, with their ids.
fn agents(app: &mut App) -> Vec<(Entity, AgentId)> {
    let mut agents = app
        .world_mut()
        .query_filtered::<(Entity, &AgentId), With<Agent>>();
    agents
        .iter(app.world())
        .map(|(entity, id)| (entity, id.clone()))
        .collect()
}

/// A kernel app on `store` and `effects` whose agent, on a scripted model, was asked a
/// question; it ran the frames until the turn ended. The model's first
/// reply also calls a tool that does not exist. Returns the app, the agent,
/// the model and how often the tool ran.
fn one_turn(
    store: &MemoryStore,
    effects: Effects,
) -> Option<(App, Entity, MockCompletionModel, u32)> {
    let (mut app, wakes) = kernel(store);
    app.insert_resource(effects);
    let calls = Arc::new(AtomicU32::new(0));
    app.add_tool(Add(calls.clone()));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("call-1", "add", serde_json::json!({ "a": 2, "b": 3 })),
            MockStreamEvent::tool_call("call-2", "sub", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = connected(&mut app, &model)?;
    ask(&mut app, agent);
    run_until(&mut app, &wakes, a_turn_ended);
    let ran = calls.load(Ordering::Relaxed);
    Some((app, agent, model, ran))
}

#[test]
fn a_turn_ends_answered_after_one_tool_round() {
    let turn = one_turn(&MemoryStore::default(), Effects::default());
    assert!(turn.is_some(), "the built-in catalog lists the model");
    let Some((mut app, agent, model, ran)) = turn else {
        return;
    };
    let ended = &app.world().resource::<Ended>().0;
    assert_eq!(ended.len(), 1, "{ended:?}");
    let answer = match ended.first() {
        Some(TurnOutcome::Answered(message)) => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("5"), "{ended:?}");
    assert_eq!(ran, 1, "the tool ran once");
    assert_eq!(model.request_count(), 2, "one tool round, then the answer");
    // The question, the tool calls, their results in call order and the answer.
    let said = messages(&app, agent);
    assert_eq!(said.len(), 4, "{said:?}");
    let results: Vec<(bool, String)> = match said.get(2) {
        Some(Message::User { content }) => content
            .iter()
            .filter_map(|item| match item {
                UserContent::ToolResult(result) => Some((result.is_error, format!("{result:?}"))),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let refused = |text: &String| {
        text.contains(
            "There is no tool named `sub`. Your tools are: `add`. Call one by its exact name.",
        )
    };
    assert!(
        matches!(results.as_slice(), [(false, _), (true, sub)] if refused(sub)),
        "{results:?}"
    );
    // The model takes effort levels, not a reasoning budget: set with the
    // model, as a restore does, the budget is reset.
    let world = app.world_mut();
    let choice = world.get::<Connection>(agent).map(|c| c.spec.reference());
    let budget = ReasoningSetting(Some(Reasoning::Budget { tokens: 1000 }));
    let choice = ModelChoice(choice.unwrap_or_default());
    world.entity_mut(agent).insert(budget).insert(choice);
    world.flush();
    assert_eq!(
        world.get::<ReasoningSetting>(agent),
        Some(&ReasoningSetting(None))
    );
}

#[test]
fn a_new_app_on_the_store_restores_the_conversation() {
    let store = MemoryStore::default();
    let last = Some(EffectId::from_raw(41));
    let turn = one_turn(&store, Effects::continue_after(last));
    assert!(turn.is_some(), "the built-in catalog lists the model");
    let Some((mut app, agent, _, _)) = turn else {
        return;
    };
    let said = messages(&app, agent);
    assert_eq!(said.len(), 4, "{said:?}");
    // The user's message, both model calls and both tool calls, the
    // refused one too, were recorded as they settled, numbered after the
    // given id.
    let mut log = app.world().resource::<Settles>().0.clone();
    log.sort_by_key(|record| record.id);
    let recorded: Vec<_> = log
        .iter()
        .map(|record| (record.id.as_u64(), &record.key, record.outcome.is_ok()))
        .collect();
    let tool = |key: &HandlerKey, name| *key == tool_key(name);
    let message = HandlerKey::new("message");
    assert!(
        matches!(
            recorded.as_slice(),
            [(42, said, true), (43, model, true), (44, add, true), (45, sub, false), (46, again, true)]
                if **said == message && tool(add, "add") && tool(sub, "sub") && model == again
        ),
        "{recorded:?}"
    );
    let before = agents(&mut app);
    drop(app);

    let (mut restored, _wakes) = kernel(&store);
    restored.update();
    let after = agents(&mut restored);
    let ids = |agents: &[(Entity, AgentId)]| -> Vec<AgentId> {
        agents.iter().map(|(_, id)| id.clone()).collect()
    };
    assert_eq!(ids(&after), ids(&before), "the same one agent");
    let Some((again, _)) = after.first() else {
        return;
    };
    assert_eq!(messages(&restored, *again), said);
}

/// A kernel app on a model whose first call fails with an error worth
/// retrying and whose second answers, on a clock that moves only when the
/// test moves it. It ran until the failed call's [`Backoff`] began.
fn backing_off() -> Option<(App, Receiver<()>, Entity, MockCompletionModel)> {
    let (mut app, wakes) = kernel(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    let overloaded = ProviderResponseError::without_status("overloaded").with_transient(Some(true));
    let model = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::Error(MockError::ProviderResponse(
            overloaded,
        ))],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = connected(&mut app, &model)?;
    ask(&mut app, agent);
    run_until(&mut app, &wakes, |world| {
        world.query::<&Backoff>().iter(world).next().is_some() || a_turn_ended(world)
    });
    Some((app, wakes, agent, model))
}

/// Runs one frame `by` later on the app's clock, then a few with the
/// clock still.
fn advance(app: &mut App, by: Duration) {
    app.insert_resource(TimeUpdateStrategy::ManualDuration(by));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    for _ in 0..3 {
        app.update();
    }
}

#[test]
fn a_failed_call_is_sent_again_once_its_backoff_passed_on_the_clock() {
    let started = backing_off();
    assert!(started.is_some(), "the built-in catalog lists the model");
    let Some((mut app, wakes, _, model)) = started else {
        return;
    };
    assert_eq!(
        model.request_count(),
        1,
        "{:?}",
        app.world().resource::<Ended>().0
    );
    // The first backoff is 1.5 to 2 seconds; no time passes on its own.
    advance(&mut app, Duration::from_secs(1));
    assert_eq!(model.request_count(), 1, "still waiting");
    advance(&mut app, Duration::from_secs(1));
    run_until(&mut app, &wakes, a_turn_ended);
    let ended = &app.world().resource::<Ended>().0;
    let answer = match ended.as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("5"), "{ended:?}");
    assert_eq!(model.request_count(), 2);
}

#[test]
fn an_interrupt_during_a_backoff_cancels_the_retry() {
    let started = backing_off();
    assert!(started.is_some(), "the built-in catalog lists the model");
    let Some((mut app, _wakes, agent, model)) = started else {
        return;
    };
    app.world_mut().trigger(StopTurn::new(agent));
    advance(&mut app, Duration::from_secs(5));
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Stopped(None)]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    assert_eq!(model.request_count(), 1, "the call was not sent again");
}

/// A plugin's task of its own output type ends as a `Done` with no system
/// of the plugin's; one still running when the app exits is cancelled.
#[test]
fn a_plugin_task_ends_as_done_and_the_exit_cancels_it() {
    let (mut app, wakes) = kernel(&MemoryStore::default());
    let pool = bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let wake = app.world().resource::<Wake>().clone();
    let world = app.world_mut();
    let done = world
        .spawn(Running::spawn(pool, &wake, async { 7_u8 }))
        .id();
    let stuck = world
        .spawn(Running::spawn(pool, &wake, std::future::pending::<u8>()))
        .id();
    run_until(&mut app, &wakes, |world| {
        world.get::<Done<u8>>(done).is_some()
    });
    assert_eq!(
        app.world().get::<Done<u8>>(done).map(|done| done.0),
        Some(7)
    );
    app.world_mut().write_message(AppExit::Success);
    app.update();
    assert!(app.world().get::<Running>(stuck).is_none());
}

/// The entity of the registered tool or prompt section that
/// `is` picks.
fn find<D: Component>(app: &mut App, is: impl Fn(&D) -> bool) -> Option<Entity> {
    let mut found = app.world_mut().query::<(Entity, &D)>();
    let found = found.iter(app.world()).find(|(_, data)| is(data));
    found.map(|(entity, _)| entity)
}

/// What another plugin added is turned off by inserting Bevy's
/// `Disabled` on its entity, which frees its name: a tool and a prompt
/// section give way to their replacements, the tool one registered while
/// the app runs, as an MCP plugin would.
#[test]
fn a_disabled_tool_or_section_gives_way_to_its_replacement() {
    let (mut app, wakes) = kernel(&MemoryStore::default());
    let (old, new) = (Arc::new(AtomicU32::new(0)), Arc::new(AtomicU32::new(0)));
    app.add_tool(Add(old.clone()));
    app.world_mut()
        .spawn(PromptSection::new(200, "rules", "the old rules"));
    let picked = [
        find::<ToolDef>(&mut app, |def| def.0.name.as_str() == "add"),
        find::<PromptSection>(&mut app, |section| section.tag == "rules"),
    ];
    for entity in picked.into_iter().flatten() {
        app.world_mut().entity_mut(entity).insert(Disabled);
    }
    let (tool, name) = (Add(new.clone()), ToolName::new(<Add as Tool>::NAME));
    assert!(name.is_ok());
    let Ok(name) = name else { return };
    let schema = Tool::parameters(&tool);
    let definition = ToolDefinition::new(name, Tool::description(&tool), schema);
    let handler = ErasedHandler::new(ToolAdapter::new(tool));
    // From a system: `commands.queue(move |world: &mut World| { .. })`.
    let world = app.world_mut();
    world.spawn_tool(definition, handler, ToolOptions::default());
    app.world_mut()
        .spawn(PromptSection::new(200, "rules", "the new rules"));
    let add = MockStreamEvent::tool_call("call-1", "add", serde_json::json!({ "a": 2, "b": 3 }));
    let end = MockStreamEvent::final_response_with_default_usage;
    let model = MockCompletionModel::from_stream_turns([
        vec![add, end()],
        vec![MockStreamEvent::text("5"), end()],
    ]);
    let agent = connected(&mut app, &model);
    assert!(agent.is_some(), "the built-in catalog lists the model");
    let Some(agent) = agent else { return };
    ask(&mut app, agent);
    run_until(&mut app, &wakes, a_turn_ended);
    let calls = (old.load(Ordering::Relaxed), new.load(Ordering::Relaxed));
    assert_eq!(calls, (0, 1));
    let prompt = model
        .requests()
        .first()
        .and_then(|request| request.system_instructions().map(str::to_owned));
    let prompt = prompt.unwrap_or_default();
    assert!(
        prompt.contains("the new rules") && !prompt.contains("the old rules"),
        "{prompt}"
    );
}

/// A Prepare system changes the system prompt, the tools and the options,
/// and sends the turn's call to another model than the agent's; a
/// reasoning setting the other model does not take is left out.
#[test]
fn a_plugin_reshapes_the_request_and_sends_the_turn_to_another_model() {
    let (mut app, wakes) = kernel(&MemoryStore::default());
    app.add_tool(Add(Arc::default()));
    let answer = || {
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    };
    let (own, routed) = (
        MockCompletionModel::from_stream_turns([answer()]),
        MockCompletionModel::from_stream_turns([answer()]),
    );
    let route = connection(&routed);
    let reshape = move |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            if let Some(prompt) = &mut call.prompt {
                prompt
                    .sections
                    .push(Section::new("plan_mode", "Plan only."));
            }
            let request = call.request_mut();
            request.tools.clear();
            request.options.seed = Some(7);
            request.options.reasoning = Some(Reasoning::Budget { tokens: 1000 });
            call.connection = route.clone();
        }
    };
    app.add_systems(AgentPass, reshape.in_set(Stage::Prepare));
    let agent = connected(&mut app, &own);
    assert!(agent.is_some(), "the built-in catalog lists the model");
    let Some(agent) = agent else { return };
    ask(&mut app, agent);
    run_until(&mut app, &wakes, a_turn_ended);
    assert_eq!((own.request_count(), routed.request_count()), (0, 1));
    let sent = routed.requests();
    let sent = sent.first();
    let prompt = sent.and_then(|request| request.system_instructions());
    assert!(
        prompt.is_some_and(|prompt| prompt.ends_with("<plan_mode>\nPlan only.\n</plan_mode>")),
        "{prompt:?}"
    );
    assert!(sent.is_some_and(|request| request.tools.is_empty()
        && request.options.seed == Some(7)
        && request.options.reasoning.is_none()));
}

#[test]
fn plugins_edit_or_take_over_what_the_user_typed_before_it_is_delivered() {
    let (mut app, _) = kernel(&MemoryStore::default());
    let edit = |mut typed: Query<&mut Incoming, Added<Admitted>>| {
        for mut typed in &mut typed {
            if typed.author.kind == AuthorKind::User && typed.text() == "hi" {
                *typed.text_mut() = "hello".to_owned();
            }
        }
    };
    // Taken over: nothing is sent.
    let take_over =
        |mut typed: Query<(&Incoming, &mut Decision), (ReadyToStart, IncomingMessages)>| {
            for (typed, mut decision) in &mut typed {
                if typed.author.kind == AuthorKind::User && typed.text().starts_with('!') {
                    decision.decide(Decider::plugin("test"), Outcome::custom("shell"));
                }
            }
        };
    app.add_systems(
        AgentPass,
        (edit.in_set(Stage::Prepare), take_over.in_set(Stage::Gate)),
    );
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    let note = |text: &str| Deliver::new(agent, text, DeliveryMode::Inform);
    // The kernel has no slash commands: a `/` line is text to it.
    for typed in [note("hi"), note("!ls"), note("/echo typed")] {
        app.world_mut().trigger(typed);
    }
    app.update();
    let sent: Vec<String> = messages(&app, agent)
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content.to_vec(),
            _ => Vec::new(),
        })
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(text.text),
            _ => None,
        })
        .collect();
    assert_eq!(sent, ["hello", "/echo typed"]);
}

/// What Gate read of each turn request's history: who changed it, and
/// whether it was a splice at the start.
#[derive(Resource, Default)]
struct Attributed(Vec<(Option<Author>, bool)>);

/// A Prepare edit of a request is attributed to its plugin, goes to the
/// model after the system prompt, and never reaches the conversation or
/// its log.
#[test]
fn a_prepare_edit_of_a_request_is_attributed_and_never_saved() {
    let store = MemoryStore::default();
    let (mut app, wakes) = kernel(&store);
    let note = |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            if let Some(history) = &mut call.history {
                let entry = Entry::new(Message::user("Be brief.")).by(Author::plugin("x"));
                history.splice(0..0, [entry], Author::plugin("x"));
            }
        }
    };
    let read = |calls: Query<&ModelRequest, (ReadyToStart, ModelCalls)>,
                mut seen: ResMut<Attributed>| {
        for call in &calls {
            let changes = call.history.iter().flat_map(|history| history.unsaved());
            seen.0.extend(changes.map(|change| {
                let at_start = matches!(change.change, Change::Splice { from: 0, to: 0, .. });
                (change.by.clone(), at_start)
            }));
        }
    };
    app.init_resource::<Attributed>().add_systems(
        AgentPass,
        (note.in_set(PrepareSystems::Edit), read.in_set(Stage::Gate)),
    );
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("5"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let Some(agent) = connected(&mut app, &model) else {
        return;
    };
    app.world_mut()
        .entity_mut(agent)
        .insert(SystemPrompt("You add.".into()));
    ask(&mut app, agent);
    run_until(&mut app, &wakes, a_turn_ended);
    let seen = &app.world().resource::<Attributed>().0;
    assert_eq!(seen, &[(Some(Author::plugin("x")), true)]);
    let sent = model.requests();
    let sent = sent.first().map(|request| request.chat_history.as_slice());
    // The model's wire joins the two user messages.
    let asked = [
        Message::system("You add."),
        Message::User {
            content: vec![
                UserContent::text("Be brief."),
                UserContent::text("What is 2 + 3?"),
            ],
        },
    ]
    .map(std::sync::Arc::new);
    assert_eq!(sent, Some(asked.as_slice()));
    // Neither the conversation nor its log has the plugin's entry.
    let by_x = |conversation: &rig_core::conversation::Conversation| {
        let mut entries = conversation.entries();
        entries.any(|entry| entry.author == Some(&Author::plugin("x")))
    };
    let kept = app.world().get::<Conversation>(agent);
    assert!(kept.is_some_and(|kept| kept.messages().len() == 2 && !by_x(kept)));
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let saved = Session::read(&store).map(|session| session.agents);
    assert!(saved.is_ok_and(
        |agents| !agents.is_empty() && agents.iter().all(|agent| !by_x(&agent.conversation))
    ),);
}

/// A plugin's mode, kept with its agent.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct Mode(u32);

/// A plugin's last word: on the frame the app exits, it changes every
/// agent's saved `Mode` in `Last`, before the kernel saves it.
fn last_word(mut modes: Query<&mut Mode>) {
    for mut mode in &mut modes {
        mode.0 = 2;
    }
}

#[test]
fn a_saved_component_a_plugin_changes_in_last_on_the_exit_frame_is_saved() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = kernel(&store);
    app.add_systems(
        Last,
        last_word
            .before(rig_ecs::SaveSystems)
            .run_if(on_message::<AppExit>),
    );
    app.update();
    let agent = agents(&mut app).first().map(|(agent, _)| *agent);
    assert!(agent.is_some(), "the kernel spawns the first agent");
    let Some(agent) = agent else {
        return;
    };
    if let Ok(mut agent) = app.world_mut().get_entity_mut(agent) {
        agent.insert(Mode(1));
    }
    app.update();
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);

    let (mut restored, _wakes) = kernel(&store);
    restored.update();
    let mut modes = restored.world_mut().query::<&Mode>();
    let modes: Vec<u32> = modes.iter(restored.world()).map(|mode| mode.0).collect();
    assert_eq!(modes, [2]);
}

#[test]
fn unknown_tools_get_filtered_feedback_and_fail_the_turn_on_the_third_call() {
    let (mut app, wakes) = kernel(&MemoryStore::default());
    let ran = Arc::new(AtomicU32::new(0));
    app.add_tool(Add(ran.clone()));
    app.add_open_tool::<serde_json::Value, _>(
        "secret",
        "",
        ToolOptions::default(),
        |_: On<ToolCalled<serde_json::Value>>| {},
    );
    let call = |id: &str, name: &str| {
        MockStreamEvent::tool_call(id, name, serde_json::json!({"a": 2, "b": 3}))
    };
    let reply = |calls: Vec<MockStreamEvent>| {
        let mut events = calls;
        events.push(MockStreamEvent::final_response_with_default_usage());
        events
    };
    // 1. `bash`, missing, three times with good calls between: `secret`,
    // registered but denied, reads and counts as missing; `plan`, offered
    // with nothing serving it, does not count.
    // 2. `bash` twice, then an answer: the count started over.
    // 3. `bash` three times in one reply.
    // 4. `secret` three times: a denied name fails a turn as a missing one.
    let single = |names: &[&str]| {
        let calls = names.iter().map(|name| reply(vec![call("call", name)]));
        calls.collect::<Vec<_>>()
    };
    let mut turns = single(&[
        "bash", "add", "plan", "plan", "plan", "secret", "add", "bash", "add", "bash",
    ]);
    turns.extend(single(&["bash", "bash"]));
    turns.push(reply(vec![MockStreamEvent::text("done")]));
    turns.push(reply(["1", "2", "3"].map(|id| call(id, "bash")).to_vec()));
    turns.extend(single(&["secret", "secret", "secret"]));
    let model = MockCompletionModel::from_stream_turns(turns);
    let agent = connected(&mut app, &model);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    // A definition no handler serves is offered all the same.
    let plan =
        ToolName::new("plan").map(|name| ToolDefinition::new(name, "", serde_json::json!({})));
    assert!(plan.is_ok());
    let Ok(plan) = plan else { return };
    app.world_mut().spawn(ToolDef(Arc::new(plan)));
    app.world_mut()
        .entity_mut(agent)
        .insert(ToolAccess::all().narrow(["add", "plan"]));
    let reason =
        |name: &str| format!("the model kept calling `{name}`, which is not one of its tools");
    let feedback = |name: &str| {
        format!(
            "There is no tool named `{name}`. Your tools are: `add`, `plan`. Call one by its \
             exact name."
        )
    };
    // Each turn: the requests made so far, `add`'s runs, the names refused,
    // and the name that failed it, if one did.
    let turns = [
        (
            10,
            3,
            vec!["bash", "plan", "plan", "plan", "secret", "bash", "bash"],
            Some("bash"),
        ),
        (13, 3, vec!["bash", "bash"], None),
        (14, 3, vec!["bash", "bash", "bash"], Some("bash")),
        (17, 3, vec!["secret", "secret", "secret"], Some("secret")),
    ];
    let mut expected = Vec::new();
    for (turn, (requests, added, refused, failed)) in turns.into_iter().enumerate() {
        ask(&mut app, agent);
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == turn + 1
        });
        assert_eq!(model.request_count(), requests, "turn {turn}");
        assert_eq!(ran.load(Ordering::Relaxed) as usize, added);
        let ended = app.world().resource::<Ended>().0.last().cloned();
        match failed.map(reason) {
            Some(reason) => {
                assert!(
                    matches!(&ended, Some(TurnOutcome::Failed(why)) if *why == reason),
                    "{ended:?}"
                );
                let notices = app
                    .world_mut()
                    .resource_mut::<bevy_ecs::message::Messages<Notice>>()
                    .drain()
                    .any(|notice| {
                        notice.level == NoticeLevel::Error
                            && notice.text.to_lowercase().contains(&reason)
                    });
                assert!(notices, "turn {turn}");
            }
            None => assert!(matches!(ended, Some(TurnOutcome::Answered(_))), "{ended:?}"),
        }
        expected.extend(refused.into_iter().map(feedback));
        let errors: Vec<String> = app
            .world()
            .resource::<Settles>()
            .0
            .iter()
            .filter_map(|record| record.outcome.as_ref().err())
            .map(|error| error.message.clone())
            .collect();
        assert_eq!(errors, expected, "turn {turn}");
    }
}
