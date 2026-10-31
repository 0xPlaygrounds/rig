use std::cell::Cell;
use std::collections::HashSet;
use std::sync::mpsc::Receiver;

use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::{CompletionResponse, Message, Usage};
use rig_core::message::{AssistantContent, Origin, Text};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use rig_ecs::fork::{ForkAgent, ForkedFrom};
use rig_harness_test_support::{
    MemoryStore, app_on, calls, commit_message, connect, reply, run_until, start_call,
};

/// Each message delivered: the agent it went to, the request it names, its
/// mode, its text and who wrote it.
#[derive(Resource, Default)]
struct Delivered(Vec<(Entity, String, DeliveryMode, String, Author)>);

/// The arguments of `look`, a read-only tool every agent may call.
#[derive(Deserialize, JsonSchema)]
struct LookArgs {}

/// Answers a `look` call at once.
fn look(called: On<ToolCalled<LookArgs>>, mut commands: Commands) {
    let seen = Outcome::from(ToolResult::success("seen".into()));
    commands.entity(called.call).insert_if_new(seen);
}

/// An app on a session store with the subagents and `look`, after its first frame
/// restored the session, and the agent nothing spawned. No catalog model
/// connects: agents have only the connections a test gives them, and a
/// subagent on its parent's model shares the parent's.
struct Session {
    app: App,
    wakes: Receiver<()>,
    parent: Entity,
}

impl Session {
    /// A session on `store`; a new parent answered by `model`, when given.
    fn new(store: &MemoryStore, model: Option<&MockCompletionModel>) -> Self {
        Self::with(store, model, |_| {})
    }

    /// As [`Session::new`], with what `configure` adds before the first
    /// frame.
    fn with(
        store: &MemoryStore,
        model: Option<&MockCompletionModel>,
        configure: impl FnOnce(&mut App),
    ) -> Self {
        let (mut app, wakes) = app_on(store);
        app.insert_resource(Models(Connector::new(Catalog::default())))
            .add_plugins((AgentPlugin, SessionLogPlugin))
            .add_open_tool("look", "Look.", read_only(), look)
            .init_resource::<Delivered>()
            .add_observer(
                |sent: On<Add<Incoming>>,
                 messages: Query<(&Incoming, &ForAgent)>,
                 mut delivered: ResMut<Delivered>| {
                    let Ok((message, &ForAgent(to))) = messages.get(sent.entity) else {
                        return;
                    };
                    let request = message.author.request.clone().map(|request| request.0);
                    let text = message.text().to_owned();
                    let author = message.author.clone();
                    let entry = (to, request.unwrap_or_default(), message.mode, text, author);
                    delivered.0.push(entry);
                },
            );
        rig_harness::load::<SubagentsPlugin>(&mut app, "rig-basics", "path plugins/rig-basics");
        if let Some(connection) = model.and_then(connect) {
            let choice = ModelChoice(connection.spec.reference());
            app.world_mut().spawn((Agent, connection, choice));
        }
        configure(&mut app);
        app.finish();
        app.update();
        let world = app.world_mut();
        let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let parent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        Self { app, wakes, parent }
    }

    /// `agent` calls `tool` with `args` in call `id`, as the model call 1
    /// asked, run by passes of the stages until it is answered and nothing
    /// else of a frame, and gets its output once it has one.
    fn call(&mut self, agent: Entity, tool: &str, id: &str, args: serde_json::Value) -> String {
        let started = start_call(self.app.world_mut(), agent, (tool, id), args);
        self.passes(|world| started.is_some_and(|call| world.get::<Outcome>(call).is_some()));
        let output = started.and_then(|call| self.app.world().get::<Outcome>(call));
        output
            .map(|output| output.tool_result().output().render())
            .unwrap_or_default()
    }

    /// `parent`'s `task` call `id` with `peers`, and the subagent it started.
    fn task(&mut self, id: &str) -> Entity {
        let args = json!({ "description": id, "prompt": "Agree on a name.", "peers": true });
        self.call(self.parent, TASK, id, args);
        let spawned = self.app.world().get::<Spawned>(self.parent);
        let child = spawned.and_then(|spawned| spawned.iter().last());
        child.unwrap_or(Entity::PLACEHOLDER)
    }

    /// Runs passes of the stages, and nothing else of a frame, until
    /// `done` or 8 passes ran.
    fn passes(&mut self, done: impl Fn(&World) -> bool) {
        for _ in 0..8 {
            if done(self.app.world()) {
                return;
            }
            self.app.world_mut().run_schedule(AgentPass);
        }
    }

    /// The subagents and helpers of `agent`, in the order they were spawned.
    fn children(&self, agent: Entity) -> Vec<Entity> {
        let spawned = self.app.world().get::<Spawned>(agent);
        spawned
            .map(|spawned| spawned.iter().collect())
            .unwrap_or_default()
    }

    /// The short id of `agent`.
    fn short(&self, agent: Entity) -> String {
        let id = self.app.world().get::<AgentId>(agent);
        id.map(|id| id.short().to_owned()).unwrap_or_default()
    }

    /// Answers `agent`'s model calls by `model`.
    fn answer_by(&mut self, agent: Entity, model: &MockCompletionModel) {
        if let Some(connection) = connect(model) {
            self.app.world_mut().entity_mut(agent).insert(connection);
        }
    }

    /// The messages delivered to `to`, as (request, mode, text).
    fn to(&self, to: Entity) -> Vec<(&str, DeliveryMode, &str)> {
        let delivered = self.app.world().resource::<Delivered>().0.iter();
        let delivered = delivered.filter(|(entity, ..)| *entity == to);
        delivered
            .map(|(_, request, mode, text, _)| (request.as_str(), *mode, text.as_str()))
            .collect()
    }

    /// Who wrote each message delivered to `to`.
    fn authors(&self, to: Entity) -> Vec<&Author> {
        let delivered = self.app.world().resource::<Delivered>().0.iter();
        let delivered = delivered.filter(|(entity, ..)| *entity == to);
        delivered.map(|(.., author)| author).collect()
    }
}

#[test]
fn peers_agree_and_each_reports_as_it_is_done() {
    let parent = MockCompletionModel::from_stream_turns([reply("Noted."), reply("Settled.")]);
    let mut session = Session::new(&MemoryStore::default(), Some(&parent));
    let section = PromptSection::new(PromptSection::ORDER_PROJECT, "project", "One-word names.");
    session.app.world_mut().spawn(section);
    let (a, b) = (session.task("Propose"), session.task("Critique"));
    // The proposer asks, ends a turn while the critic owes it the answer,
    // and answers once that came; the critic waits for the proposal.
    let ask = json!({ "agent": session.short(b), "text": "How about Lumen?" });
    let proposer = MockCompletionModel::from_stream_turns([
        calls("m1", MESSAGE, ask),
        reply("Waiting for the critic."),
        reply("We agreed on Lumen."),
    ]);
    let wait = json!({ "agent": session.short(a) });
    let critic =
        MockCompletionModel::from_stream_turns([calls("w1", WAIT, wait), reply("Lumen is good.")]);
    session.answer_by(a, &proposer);
    session.answer_by(b, &critic);
    let parent_agent = session.parent;
    run_until(&mut session.app, &session.wakes, |world| {
        let finished =
            |agent| matches!(world.get::<Lifecycle>(agent), Some(Lifecycle::Finished(_)));
        let idle = world.get::<ActiveTurn>(parent_agent).is_none();
        finished(a) && finished(b) && idle && parent.request_count() > 0
    });

    // Each request got one report, as soon as it was done: the critic's
    // first.
    let reports = session.to(session.parent);
    let expected = [
        ("Critique", DeliveryMode::Steer, "Lumen is good."),
        ("Propose", DeliveryMode::Steer, "We agreed on Lumen."),
    ];
    assert_eq!(reports, expected);
    let finished = |agent| session.app.world().get::<Lifecycle>(agent).cloned();
    let answer = |text: &str| Some(Lifecycle::Finished(text.to_owned()));
    assert_eq!(finished(a), answer("We agreed on Lumen."));
    assert_eq!(finished(b), answer("Lumen is good."));
    // The proposer's brief named its peer, and the project's section was in
    // its prompt; the parent read the reports headed by who sent them.
    let asked = format!("{:?}", proposer.requests().first());
    assert!(asked.contains(&session.short(b)), "{asked}");
    assert!(asked.contains("One-word names."), "{asked}");
    // Its role is a section of its own, not its saved system prompt.
    assert!(asked.contains("<peer_role>"), "{asked}");
    // Its brief is headed by the parent alone, not by its own task's title.
    let from = format!("Output of agent {}, not", session.short(session.parent));
    assert!(asked.contains(&from), "{asked}");
    let prompts = |agent| {
        session
            .app
            .world()
            .get::<SystemPrompt>(agent)
            .map(|p| p.0.clone())
    };
    assert_eq!(prompts(a), prompts(session.parent));
    let read = format!("{:?}", parent.requests());
    let header = format!("Output of agent {} \\\"Critique\\\"", session.short(b));
    assert!(read.contains(&header), "{read}");
    assert!(!read.contains("Waiting for the critic."), "{read}");

    // A peer started later has the others in its brief, and they learn of
    // it by a note.
    let c = session.task("Judge");
    session.app.update();
    let judge = session.short(c);
    let plugin = AuthorKind::Plugin(ORIGIN.to_owned());
    for peer in [a, b] {
        let note = session.to(peer).last().copied();
        let told =
            |(_, mode, text): (_, _, &str)| mode == DeliveryMode::Inform && text.contains(&judge);
        assert!(note.is_some_and(told), "{note:?}");
        let author = session.authors(peer).last().map(|author| &author.kind);
        assert_eq!(author, Some(&plugin));
    }
    let brief = session.to(c).first().map(|(.., text)| text.to_string());
    let brief = brief.unwrap_or_default();
    let names = brief.contains(&session.short(a)) && brief.contains(&session.short(b));
    assert!(names, "{brief}");
    // The brief is headed by the parent alone; the task's title is the
    // subagent's own.
    let author = session.authors(c).first().map(|author| author.label());
    assert_eq!(
        author,
        Some(format!("agent {}", session.short(session.parent)))
    );
    let title = session
        .app
        .world()
        .get::<AgentTitle>(c)
        .map(|title| title.0.as_str());
    assert_eq!(title, Some("Judge"));
}

/// The open calls of `slow`, a tool that answers only when a test does.
#[derive(Resource, Default)]
struct SlowCalls(Vec<Entity>);

/// Keeps a `slow` call open, for the test to answer.
fn slow(called: On<ToolCalled<LookArgs>>, mut open: ResMut<SlowCalls>) {
    open.0.push(called.call);
}

#[test]
fn a_subagents_question_is_a_message_and_only_its_final_answer_reports() {
    let unread = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&MemoryStore::default(), Some(&unread));
    let args = json!({ "description": "Fix", "prompt": "Fix the bug." });
    session.call(session.parent, TASK, "t1", args);
    let child = session.children(session.parent).last().copied();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let (parent_id, child_id) = (session.short(session.parent), session.short(child));
    // The subagent asks the parent which file, waits for the answer, and
    // reports once it is done.
    let ask = json!({ "agent": parent_id, "text": "Which file?" });
    let child_model = MockCompletionModel::from_stream_turns([
        calls("c1", MESSAGE, ask),
        calls("c2", WAIT, json!({ "agent": parent_id })),
        reply("Fixed src/lib.rs."),
    ]);
    let answer = json!({ "agent": child_id, "text": "Use src/lib.rs." });
    let parent_model = MockCompletionModel::from_stream_turns([
        calls("p1", MESSAGE, answer),
        reply("Told it."),
        reply("Thanks."),
    ]);
    session.answer_by(session.parent, &parent_model);
    session.answer_by(child, &child_model);
    let parent = session.parent;
    run_until(&mut session.app, &session.wakes, |world| {
        let finished = matches!(world.get::<Lifecycle>(child), Some(Lifecycle::Finished(_)));
        let idle = world.get::<ActiveTurn>(parent).is_none();
        finished && idle && parent_model.request_count() >= 2
    });

    // The question woke the parent as a message on no request; the final
    // answer is the one report, on the parent's last request.
    let to_parent = session.to(session.parent);
    let expected = [
        ("", DeliveryMode::Steer, "Which file?"),
        ("p1", DeliveryMode::Steer, "Fixed src/lib.rs."),
    ];
    assert_eq!(to_parent, expected);
    let read = format!("{:?}", child_model.requests().last());
    assert!(read.contains("Use src/lib.rs."), "{read}");
    assert!(read.contains("not your report"), "{read}");
}

#[test]
fn a_correction_reaches_a_working_subagent_with_its_next_model_call() {
    let parent = MockCompletionModel::from_stream_turns([reply("Noted.")]);
    let mut session = Session::with(&MemoryStore::default(), Some(&parent), |app| {
        app.init_resource::<SlowCalls>()
            .add_open_tool("slow", "Slow.", read_only(), slow);
    });
    let args = json!({ "description": "Build", "prompt": "Build it with plan A." });
    session.call(session.parent, TASK, "t1", args);
    let child = session.children(session.parent).last().copied();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let child_model = MockCompletionModel::from_stream_turns([
        calls("s1", "slow", json!({})),
        reply("Built with plan B."),
        reply("Built again."),
    ]);
    session.answer_by(child, &child_model);
    run_until(&mut session.app, &session.wakes, |world| {
        !world.resource::<SlowCalls>().0.is_empty()
    });
    let correct = json!({ "agent": session.short(child), "text": "Use plan B." });
    let said = session.call(session.parent, MESSAGE, "m1", correct);
    assert!(said.contains("with its next step"), "{said}");
    let open = session.app.world().resource::<SlowCalls>().0.clone();
    for call in open {
        let done = Outcome::from(ToolResult::success("done".into()));
        session.app.world_mut().entity_mut(call).insert(done);
    }
    run_until(&mut session.app, &session.wakes, |_| {
        parent.request_count() > 0
    });

    // The model call after the slow one read the correction, and the
    // subagent's answer reported on both requests at once.
    let read = format!("{:?}", child_model.requests().get(1));
    assert!(read.contains("Use plan B."), "{read}");
    assert_eq!(child_model.request_count(), 2);
    let reports = session.to(session.parent);
    assert_eq!(reports, [("m1", DeliveryMode::Steer, "Built with plan B.")]);
}

/// Holds every model call of a turn, as an approval nobody gave.
fn hold_turn_calls(calls: Query<Entity, (ReadyToStart, With<TurnCall>)>, mut commands: Commands) {
    for call in &calls {
        commands.spawn(HoldOf(call));
    }
}

#[test]
fn requests_that_would_deadlock_are_refused_and_a_restart_answers_open_ones_as_interrupted() {
    let store = MemoryStore::default();
    // Its turns are never read: every model call is held, and the session
    // stops before any is sent.
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&store, Some(&model));
    session
        .app
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let asked = Message::user("Find a name.");
    let added = commit_message(session.app.world_mut(), session.parent, asked);
    assert!(added.is_ok(), "{added:?}");
    let (a, b, c) = (session.task("ta"), session.task("tb"), session.task("tc"));
    // `a` waits on `c` through `b`, so `c` can neither ask nor wait for it;
    // the parent is idle, so nothing would come from it.
    let ask = |agent| json!({ "agent": session.short(agent), "text": "Hi." });
    let wait = |agent| json!({ "agent": session.short(agent) });
    // A refused message to a guessed name sends nothing and closes nothing.
    let guess = json!({ "agent": "critic", "text": "Hi." });
    let calls = [
        (a, MESSAGE, guess, "not an agent you can reach"),
        (a, MESSAGE, ask(b), "in the background"),
        (b, MESSAGE, ask(c), "in the background"),
        (c, MESSAGE, ask(a), "waiting for your report"),
        (c, WAIT, wait(a), "waiting for you"),
        (a, WAIT, wait(session.parent), "is idle"),
    ];
    for (at, (from, tool, args, meant)) in calls.into_iter().enumerate() {
        let said = session.call(from, tool, &format!("m{at}"), args);
        assert!(said.contains(meant), "{said}");
    }
    session.app.update();
    drop(session);

    // Each report is a note that starts no turn of the parent, which gets
    // a model only now and calls it for nothing.
    let mut session = Session::new(&store, None);
    let unused = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    session.answer_by(session.parent, &unused);
    for _ in 0..5 {
        session.app.update();
    }
    let reports = session.to(session.parent);
    let interrupted = reports.iter().filter(|(_, mode, text)| {
        *mode == DeliveryMode::Inform && text.starts_with("Stopped: the session restarted")
    });
    let mut requests: Vec<&str> = interrupted.map(|(request, ..)| *request).collect();
    requests.sort_unstable();
    assert_eq!(requests, ["ta", "tb", "tc"], "{reports:?}");
    assert_eq!(unused.request_count(), 0);
    // The restored subagents have their roles again.
    let world = session.app.world();
    let spawned = world.get::<Spawned>(session.parent);
    let children: Vec<Entity> = spawned
        .map(|spawned| spawned.iter().collect())
        .unwrap_or_default();
    let sections = |child| {
        world
            .get::<AgentSections>(child)
            .map_or(0, |of| of.iter().count())
    };
    let roles: Vec<usize> = children.into_iter().map(sections).collect();
    assert_eq!(roles, [2, 2, 2]);
}

#[test]
fn stopping_an_idle_parent_stops_its_subagents_and_their_reports_start_nothing() {
    // The subagents' model calls are held, so they work until stopped; the
    // parent's model is never asked.
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&MemoryStore::default(), Some(&model));
    session
        .app
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let (a, b) = (session.task("ta"), session.task("tb"));
    let working = |world: &World| {
        [a, b]
            .iter()
            .all(|&child| world.get::<ActiveTurn>(child).is_some())
    };
    run_until(&mut session.app, &session.wakes, working);
    let parent = session.parent;
    assert!(
        session.app.world().get::<ActiveTurn>(parent).is_none(),
        "idle"
    );
    session.app.world_mut().trigger(StopTurn::new(parent));
    for _ in 0..3 {
        session.app.update();
    }
    let world = session.app.world();
    for child in [a, b] {
        assert!(world.get::<ActiveTurn>(child).is_none(), "stopped");
        let finished = world.get::<Lifecycle>(child);
        let stopped = matches!(finished, Some(Lifecycle::Finished(text))
            if text.starts_with("Stopped: the user stopped the subagent before it answered. \
                                 It may have partly run"));
        assert!(stopped, "{finished:?}");
    }
    // Each report is a note from the plugin, and the parent stays idle.
    let reports = session.to(parent);
    let modes: Vec<(&str, DeliveryMode)> = reports
        .iter()
        .map(|(request, mode, _)| (*request, *mode))
        .collect();
    assert_eq!(
        modes,
        [("ta", DeliveryMode::Inform), ("tb", DeliveryMode::Inform)]
    );
    let plugin = AuthorKind::Plugin(ORIGIN.to_owned());
    let authors: Vec<_> = session
        .authors(parent)
        .into_iter()
        .map(|a| &a.kind)
        .collect();
    assert_eq!(authors, [&plugin, &plugin]);
    assert!(
        world.get::<ActiveTurn>(parent).is_none(),
        "no report started it"
    );
    assert_eq!(model.request_count(), 0);
}

#[test]
fn what_a_parent_knows_of_the_environment_goes_to_every_subagent_below_it() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&MemoryStore::default(), Some(&model));
    session
        .app
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let known = "No local toolchain; build on the remote with `koh exec bench -- CMD`.";
    let args = json!({ "description": "ta", "prompt": "Fix it.", "environment": known });
    session.call(session.parent, TASK, "t1", args);
    let child = session.children(session.parent).last().copied();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    // Its own subagent, started without one, gets the same.
    let args = json!({ "description": "tb", "prompt": "Test it." });
    session.call(child, TASK, "t2", args);
    let grandchild = session.children(child).last().copied();
    let grandchild = grandchild.unwrap_or(Entity::PLACEHOLDER);
    let world = session.app.world_mut();
    let mut sections = world.query::<(&SectionOf, &PromptSection)>();
    let mut told = |agent: Entity| {
        let of = sections.iter(world).filter(|(of, _)| of.0 == agent);
        let mut known_of = of.filter(|(_, section)| section.tag == "known_environment");
        known_of.any(|(_, section)| section.text.contains(known))
    };
    assert!(told(child) && told(grandchild));
    assert!(!told(session.parent));
}

/// A read-only tool's options.
fn read_only() -> ToolOptions {
    ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    }
}

/// Adds the argument `name` to `task`'s parameters, as a plugin's
/// `finish` would.
fn add_task_argument(world: &mut World, name: &str) {
    let mut tools = world.query::<&mut ToolDef>();
    for mut tool in tools.iter_mut(world) {
        let tool = std::sync::Arc::make_mut(&mut tool.0);
        if tool.name.as_str() == TASK
            && let Some(properties) = tool.parameters.get_mut("properties")
            && let Some(properties) = properties.as_object_mut()
        {
            properties.insert(name.to_owned(), json!({ "type": "string" }));
        }
    }
}

#[test]
fn a_plugin_shapes_a_tasks_subagent_through_the_stages_and_the_record_shows_it() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&MemoryStore::default(), Some(&model));
    // A plugin's own `task` argument, `profile`, reaches the request, and
    // picks its tools and system prompt; the agent spawned gets the
    // profile's prompt section.
    add_task_argument(session.app.world_mut(), "profile");
    let apply = |mut requests: Query<&mut SpawnSubagent, Added<Admitted>>| {
        for mut request in &mut requests {
            if request.extra.get("profile") == Some(&json!("reviewer")) {
                request.tools = Some(vec!["look".to_owned()]);
                request.system_prompt = Some("You review.".to_owned());
            }
        }
    };
    let section = |spawned: Query<&SpawnedAgent, Added<SpawnedAgent>>, mut commands: Commands| {
        for agent in &spawned {
            let order = PromptSection::ORDER_ROLE + 5;
            let section = PromptSection::new(order, "reviewer", "Be terse.");
            commands.spawn((section, SectionOf(agent.0)));
        }
    };
    session.app.add_systems(
        AgentPass,
        (apply.in_set(Stage::Prepare), section.in_set(Stage::Review)),
    );
    let args = json!({ "description": "Review", "prompt": "Review src.", "profile": "reviewer" });
    let said = session.call(session.parent, TASK, "t1", args);
    assert!(said.starts_with("Started subagent"), "{said}");

    let child = session.children(session.parent).last().copied();
    let world = session.app.world();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let access = world.get::<ToolAccess>(child);
    let only_look =
        access.and_then(ToolAccess::names) == Some(&["look", MESSAGE, WAIT].map(str::to_owned)[..]);
    assert!(
        only_look,
        "its profile's tools, and those that reach agents"
    );
    let prompt = world
        .get::<SystemPrompt>(child)
        .map(|prompt| prompt.0.to_string());
    assert_eq!(prompt.as_deref(), Some("You review."));
    let sections = world
        .get::<AgentSections>(child)
        .map_or(0, |of| of.iter().count());
    assert_eq!(sections, 2, "its role and the profile's");
    // The spawn is an effect of its own, nested under the call, recorded
    // with the request as it started and the agent's id.
    let id = world.get::<AgentId>(child).map(|id| id.0.clone());
    let records = world.resource::<Messages<EffectRecorded>>();
    let spawn = records
        .iter_current_update_messages()
        .find_map(
            |EffectRecorded(record)| match (&record.kind, &record.outcome) {
                (
                    rig_core::effect::EffectKind::Custom { kind, payload },
                    Ok(rig_core::effect::Answered::Custom { payload: answer }),
                ) if &**kind == SUBAGENT => Some((payload.clone(), answer.clone(), record.parent)),
                _ => None,
            },
        );
    let (request, answer, parent) = spawn.unwrap_or_default();
    assert_eq!(request.get("tools"), Some(&json!(["look"])));
    assert_eq!(
        request.get("extra"),
        Some(&json!({ "profile": "reviewer" }))
    );
    let agent = answer.get("agent").and_then(|agent| agent.as_str());
    assert_eq!(agent.map(str::to_owned), id);
    let call = world.get::<EffectParent>(child).map(|parent| parent.0);
    assert!(
        parent.is_some() && call.is_some() && parent != call,
        "{parent:?} {call:?}"
    );
    // The plugin's spawns are marked its own.
    let mine = world.resource::<PluginEntity<SubagentsPlugin>>().entity();
    assert_eq!(world.get::<ProvidedBy>(child).map(|by| by.0), Some(mine));
}

#[test]
fn the_policy_limits_depth_and_a_gate_holds_spawns_until_a_slot_frees() {
    /// On a hold of a request until its parent has a slot free.
    #[derive(Component)]
    struct Slot;
    const SLOTS: usize = 2;
    let working = |children: &Query<(&SpawnedBy, &Lifecycle)>, parent: Entity| {
        let working = |(by, step): &(&SpawnedBy, &Lifecycle)| {
            by.0 == parent && matches!(step, Lifecycle::Working | Lifecycle::WaitingOn(_))
        };
        children.iter().filter(working).count()
    };
    // At most two subagents of an agent work at once: a Gate system holds
    // the others' requests, and a Review system releases one as a slot
    // frees.
    let queue = move |requests: Query<(Entity, &ForAgent), (With<SpawnSubagent>, ReadyToStart)>,
                      children: Query<(&SpawnedBy, &Lifecycle)>,
                      mut commands: Commands| {
        let mut admitted = 0;
        for (request, &ForAgent(parent)) in &requests {
            if working(&children, parent) + admitted < SLOTS {
                admitted += 1;
            } else {
                commands.spawn((Slot, HoldOf(request)));
            }
        }
    };
    let release = move |holds: Query<(Entity, &HoldOf), With<Slot>>,
                        requests: Query<&ForAgent>,
                        children: Query<(&SpawnedBy, &Lifecycle)>,
                        mut commands: Commands| {
        let mut released = 0;
        for (hold, HoldOf(request)) in &holds {
            let Ok(&ForAgent(parent)) = requests.get(*request) else {
                continue;
            };
            if working(&children, parent) + released < SLOTS {
                released += 1;
                commands.entity(hold).despawn();
            }
        }
    };
    let model = MockCompletionModel::from_stream_turns((0..5).map(|_| reply("Done.")));
    let mut session = Session::new(&MemoryStore::default(), Some(&model));
    session
        .app
        .insert_resource(SubagentPolicy { max_depth: 1 })
        .add_systems(
            AgentPass,
            (queue.in_set(Stage::Gate), release.in_set(Stage::Review)),
        );
    let mut tasks = Vec::new();
    for at in 0..4 {
        let args = json!({ "description": format!("t{at}"), "prompt": "Count the files." });
        let id = format!("t{at}");
        tasks.extend(start_call(
            session.app.world_mut(),
            session.parent,
            (TASK, &id),
            args,
        ));
    }
    session.app.update();
    let held = |world: &mut World| world.query::<&HoldOf>().iter(world).count();
    let spawned = session.children(session.parent).len();
    assert_eq!((spawned, held(session.app.world_mut())), (2, 2));
    // A call that ends first, as an interrupt ends it, ends its held
    // request and the hold.
    let last = tasks.last().copied().unwrap_or(Entity::PLACEHOLDER);
    let ended = Outcome::refused("interrupted");
    session.app.world_mut().entity_mut(last).insert(ended);
    session.passes(|world| world.get::<Settled>(last).is_some());
    let requests = |world: &mut World| {
        let mut open = world.query_filtered::<(), (With<SpawnSubagent>, Without<Settled>)>();
        open.iter(world).count()
    };
    let world = session.app.world_mut();
    assert_eq!((held(world), requests(world)), (1, 1));

    let most = Cell::new(0);
    let finished = |world: &World| {
        let children = world.get::<Spawned>(session.parent).into_iter();
        let children = children.flat_map(|spawned| spawned.iter());
        let steps: Vec<_> = children
            .map(|child| world.get::<Lifecycle>(child))
            .collect();
        let at_work = steps
            .iter()
            .filter(|step| matches!(step, Some(Lifecycle::Working)));
        most.set(most.get().max(at_work.count()));
        steps
            .iter()
            .filter(|step| matches!(step, Some(Lifecycle::Finished(_))))
            .count()
            == 3
    };
    run_until(&mut session.app, &session.wakes, finished);
    assert_eq!(session.children(session.parent).len(), 3);
    assert_eq!(most.get(), 2, "never more than two at work");
    let reports = session.to(session.parent);
    let mut requests: Vec<&str> = reports.iter().map(|(request, ..)| *request).collect();
    requests.sort_unstable();
    assert_eq!(requests, ["t0", "t1", "t2"]);

    // At the policy's depth, a subagent has no `task`, and a request for an
    // agent of its is refused.
    let child = session.children(session.parent).first().copied();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let access = session.app.world().get::<ToolAccess>(child);
    assert!(
        access.is_some_and(|access| !access.allows(TASK)),
        "no `task`"
    );
    let deeper = SpawnSubagent::helper("deeper");
    let request = session
        .app
        .world_mut()
        .spawn((deeper, ForAgent(child)))
        .id();
    session.passes(|world| world.get::<Outcome>(request).is_some());
    let outcome = session.app.world().get::<Outcome>(request);
    let refused = outcome.and_then(|outcome| outcome.0.as_ref().err());
    let why = refused
        .map(|report| report.message.as_str())
        .unwrap_or_default();
    assert!(why.contains("at most 1 deep"), "{why}");
}

#[test]
fn a_helper_works_only_for_its_plugin_and_a_stopped_subagent_reports_why() {
    let parent = MockCompletionModel::from_stream_turns([reply("Noted.")]);
    let mut session = Session::new(&MemoryStore::default(), Some(&parent));
    // A plugin spawns a helper of its own, with its own system prompt.
    let helping = SpawnSubagent::helper("pro")
        .system_prompt("Argue for.")
        .brief("Is Rust good?");
    let request = session
        .app
        .world_mut()
        .spawn((helping, ForAgent(session.parent)))
        .id();
    session.passes(|world| world.get::<Settled>(request).is_some());
    let helper = session
        .app
        .world()
        .get::<SpawnedAgent>(request)
        .map(|a| a.0);
    let helper = helper.unwrap_or(Entity::PLACEHOLDER);
    let world = session.app.world();
    assert_eq!(world.get::<Name>(helper).map(Name::as_str), Some("pro"));
    let prompt = world.get::<SystemPrompt>(helper).map(|p| p.0.to_string());
    assert_eq!(prompt.as_deref(), Some("Argue for."));
    assert!(world.get::<Lifecycle>(helper).is_none());
    // The parent's model cannot reach it.
    let ask = json!({ "agent": session.short(helper), "text": "Hi." });
    let said = session.call(session.parent, MESSAGE, "m1", ask);
    assert!(said.contains("not an agent you can reach"), "{said}");

    // A plugin stops a subagent's second model call, as a budget would, and
    // marks every report it reads.
    let args = json!({ "description": "Count", "prompt": "Count the files." });
    session.call(session.parent, TASK, "t1", args);
    let child = session.children(session.parent).last().copied();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let budget =
        move |calls: Query<(Entity, &CallOf, &ForAgent), (With<TurnCall>, ReadyToStart)>,
              mut seen: Local<HashSet<Entity>>,
              mut commands: Commands| {
            for (call, &CallOf(turn), &ForAgent(agent)) in &calls {
                if agent == child && seen.insert(call) && seen.len() > 1 {
                    let stopped = TurnOutcome::Stopped(Some("over budget".to_owned()));
                    commands.entity(turn).insert(stopped);
                }
            }
        };
    let mark = |mut reports: Query<&mut Incoming, (With<Report>, Added<Admitted>)>| {
        for mut report in &mut reports {
            report.text_mut().insert_str(0, "[report] ");
        }
    };
    session.app.add_systems(
        AgentPass,
        (budget.in_set(Stage::Gate), mark.in_set(Stage::Prepare)),
    );
    let helper_model = MockCompletionModel::from_stream_turns([reply("Rust is good.")]);
    let child_model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("Halfway there."),
        MockStreamEvent::tool_call("l1", "look", json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    session.answer_by(helper, &helper_model);
    session.answer_by(child, &child_model);
    run_until(&mut session.app, &session.wakes, |world| {
        let helper_idle = world.get::<ActiveTurn>(helper).is_none();
        parent.request_count() > 0 && helper_model.request_count() > 0 && helper_idle
    });

    // Only the subagent reported, with why it stopped and what it had said,
    // which carried the parent on; the helper's answer stayed its own.
    let reports = session.to(session.parent);
    assert_eq!(reports.len(), 1, "{reports:?}");
    let (request, mode, text) = reports.first().copied().unwrap_or_default();
    assert_eq!((request, mode), ("t1", DeliveryMode::Steer));
    assert!(text.starts_with("Stopped: over budget."), "{text}");
    assert!(text.contains("It had said: Halfway there."), "{text}");
    assert_eq!(child_model.request_count(), 1);
    let read = format!("{:?}", parent.requests());
    assert!(read.contains("[report] Stopped: over budget."), "{read}");
}

/// The answers of the turns of helpers, as they ended.
#[derive(Resource, Default)]
struct HelperAnswers(Vec<Option<String>>);

#[test]
fn a_helper_for_one_job_retires_after_its_turn_and_is_not_restored() {
    let store = MemoryStore::default();
    let parent = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&store, Some(&parent));
    // The plugin reads the answer as the helper's turn ends.
    session.app.init_resource::<HelperAnswers>().add_observer(
        |settled: On<Add<Settled>>,
         turns: Query<(&ForAgent, &TurnOutcome)>,
         helpers: Query<(), With<Helper>>,
         mut answers: ResMut<HelperAnswers>| {
            if let Ok((&ForAgent(agent), outcome)) = turns.get(settled.entity)
                && helpers.contains(agent)
            {
                let answer = match outcome {
                    TurnOutcome::Answered(message) => message.final_answer(),
                    _ => None,
                };
                answers.0.push(answer);
            }
        },
    );
    let job = SpawnSubagent::helper("summarizer")
        .brief("Sum up the session.")
        .once();
    let request = session
        .app
        .world_mut()
        .spawn((job, ForAgent(session.parent)))
        .id();
    session.passes(|world| world.get::<Settled>(request).is_some());
    let helper = session
        .app
        .world()
        .get::<SpawnedAgent>(request)
        .map(|a| a.0);
    let helper = helper.unwrap_or(Entity::PLACEHOLDER);
    let model = MockCompletionModel::from_stream_turns([reply("A short session.")]);
    session.answer_by(helper, &model);
    run_until(&mut session.app, &session.wakes, |world| {
        world.get_entity(helper).is_err()
    });
    let answers = &session.app.world().resource::<HelperAnswers>().0;
    assert_eq!(answers, &[Some("A short session.".to_owned())]);
    assert!(session.app.world().get_entity(helper).is_err(), "retired");
    session.app.update();
    drop(session);

    let mut session = Session::new(&store, None);
    let world = session.app.world_mut();
    let agents = world
        .query_filtered::<(), With<Agent>>()
        .iter(world)
        .count();
    assert_eq!(agents, 1, "only the parent comes back");
}

/// Forks `agent` at its end; the fork.
fn fork(session: &mut Session, agent: Entity) -> Entity {
    let world = session.app.world_mut();
    world.trigger(ForkAgent {
        agent,
        through: None,
    });
    world.flush();
    let mut forks = world.query_filtered::<Entity, With<ForkedFrom>>();
    let fork = forks.iter(world).next();
    fork.unwrap_or(Entity::PLACEHOLDER)
}

/// A fork of a working subagent is no subagent and owes no report: the
/// task the subagent was given is reported once, by the subagent.
#[test]
fn a_fork_of_a_working_subagent_never_reports_on_its_task() {
    let parent_model = MockCompletionModel::from_stream_turns([reply("Noted."), reply("Noted.")]);
    let mut session = Session::new(&MemoryStore::default(), Some(&parent_model));
    let args = json!({ "description": "Kid", "prompt": "Reply ORIGINAL." });
    session.call(session.parent, TASK, "t1", args);
    let child = session.children(session.parent).last().copied();
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let fork = fork(&mut session, child);
    let world = session.app.world();
    assert!(
        world.get::<Subtask>(fork).is_none(),
        "the fork is no subagent"
    );
    let owed = world.get::<OpenRequests>(fork).map(|open| open.0.len());
    assert_eq!(owed, Some(0), "the fork owes no report");

    let fork_model = MockCompletionModel::from_stream_turns([reply("FORK")]);
    let child_model = MockCompletionModel::from_stream_turns([reply("ORIGINAL")]);
    session.answer_by(fork, &fork_model);
    session.answer_by(child, &child_model);
    let deliver = Deliver::new(fork, "Reply FORK now.", DeliveryMode::Queue);
    session.app.world_mut().trigger(deliver);
    let parent = session.parent;
    run_until(&mut session.app, &session.wakes, |world| {
        let finished = matches!(world.get::<Lifecycle>(child), Some(Lifecycle::Finished(_)));
        let idle = |agent| world.get::<ActiveTurn>(agent).is_none();
        finished && idle(fork) && idle(parent) && fork_model.request_count() > 0
    });
    session.app.update();
    let reports = session.to(session.parent);
    assert_eq!(reports, [("t1", DeliveryMode::Steer, "ORIGINAL")]);
}

/// A fork of a helper for one job goes on after its turn.
#[test]
fn a_fork_of_a_helper_for_one_job_is_not_retired() {
    let parent = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&MemoryStore::default(), Some(&parent));
    let job = SpawnSubagent::helper("summarizer")
        .brief("Sum up the session.")
        .once();
    let request = (session.app.world_mut())
        .spawn((job, ForAgent(session.parent)))
        .id();
    session.passes(|world| world.get::<Settled>(request).is_some());
    let helper = session.app.world().get::<SpawnedAgent>(request);
    let helper = helper.map_or(Entity::PLACEHOLDER, |helper| helper.0);
    let fork = fork(&mut session, helper);
    let model = MockCompletionModel::from_stream_turns([reply("Mine.")]);
    session.answer_by(fork, &model);
    let deliver = Deliver::new(fork, "Go on.", DeliveryMode::Queue);
    session.app.world_mut().trigger(deliver);
    run_until(&mut session.app, &session.wakes, |world| {
        model.request_count() > 0 && world.get::<ActiveTurn>(fork).is_none()
    });
    session.app.update();
    assert!(session.app.world().get_entity(fork).is_ok(), "not retired");
}

#[test]
fn a_quit_stops_every_subagent_and_their_reports_reach_the_parent_after_resume() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&store, Some(&model));
    session
        .app
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let asked = commit_message(
        session.app.world_mut(),
        session.parent,
        Message::user("Find a name."),
    );
    assert!(asked.is_ok(), "{asked:?}");
    let (a, b) = (session.task("ta"), session.task("tb"));
    // `a` asks `b` and waits on it, so `b` owes `a` a report as the app
    // quits.
    let ask = json!({ "agent": session.short(b), "text": "Hi." });
    let said = session.call(a, MESSAGE, "m1", ask);
    assert!(said.contains("in the background"), "{said}");
    let wait = json!({ "agent": session.short(b) });
    start_call(session.app.world_mut(), a, (WAIT, "w1"), wait);
    session.app.update();
    session.app.world_mut().write_message(AppExit::Success);
    session.app.update();
    drop(session);

    // Each stopped as the user's stop would: a note to the parent, kept
    // for the restart, whatever order the exit stopped them in.
    let session = Session::new(&store, None);
    let reports = session.to(session.parent);
    let stopped = reports.iter().filter(|(_, mode, text)| {
        *mode == DeliveryMode::Inform && text.starts_with("Stopped: the run ended")
    });
    let mut requests: Vec<&str> = stopped.map(|(request, ..)| *request).collect();
    requests.sort_unstable();
    assert_eq!(requests, ["ta", "tb"], "{reports:?}");
}

#[test]
fn a_restored_subagent_turn_a_plugin_stops_reports_as_a_note() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&store, Some(&model));
    session
        .app
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let asked = commit_message(
        session.app.world_mut(),
        session.parent,
        Message::user("Find a name."),
    );
    assert!(asked.is_ok(), "{asked:?}");
    session.task("ta");
    session.app.update();
    drop(session);

    // After a crash, a plugin stops the restored turn its own way, such as
    // for a budget: the report is a note all the same, and the parent's
    // model is not called.
    let mut session = Session::with(&store, None, |app| {
        let over_budget = |turns: Query<Entity, (With<Restored>, Without<TurnOutcome>)>,
                           mut commands: Commands| {
            for turn in &turns {
                let stopped = TurnOutcome::Stopped(Some("over budget".to_owned()));
                commands.entity(turn).insert(stopped);
            }
        };
        app.add_systems(AgentPass, over_budget.in_set(Stage::Gate));
    });
    let unused = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    session.answer_by(session.parent, &unused);
    for _ in 0..5 {
        session.app.update();
    }
    let reports = session.to(session.parent);
    let noted = reports.iter().any(|(request, mode, text)| {
        *request == "ta" && *mode == DeliveryMode::Inform && text.contains("over budget")
    });
    assert!(noted, "{reports:?}");
    assert_eq!(unused.request_count(), 0);
}

/// The agent whose model calls [`hold_or_answer`] answers in place.
#[derive(Resource, Default)]
struct AnswerInPlace(Option<Entity>);

/// Holds every model call of a turn, but answers those of the agent
/// [`AnswerInPlace`] names with "Waiting.", as a plugin standing in for the
/// model does.
fn hold_or_answer(
    mut calls: Query<(Entity, &ForAgent, &mut Decision), (ReadyToStart, With<TurnCall>)>,
    answer: Res<AnswerInPlace>,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent), mut decision) in &mut calls {
        if answer.0 != Some(agent) {
            commands.spawn(HoldOf(call));
            continue;
        }
        let said = vec![AssistantContent::Text(Text::new("Waiting.".to_owned()))];
        let made = Origin::new("plugin", "test", "answer");
        let reply = CompletionResponse::new(said, Usage::default(), made, json!(null));
        let reply = Outcome(Ok(rig_core::effect::Answered::Completion(reply.into())));
        decision.decide(Decider::plugin("test"), reply);
    }
}

#[test]
fn a_subagent_that_waits_on_a_stopped_one_reports_as_stopped_after_a_quit_or_a_crash() {
    for crash in [false, true] {
        let store = MemoryStore::default();
        let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
        let mut session = Session::with(&store, Some(&model), |app| {
            app.init_resource::<AnswerInPlace>()
                .add_systems(AgentPass, hold_or_answer.in_set(Stage::Gate));
        });
        let asked = commit_message(
            session.app.world_mut(),
            session.parent,
            Message::user("Find a name."),
        );
        assert!(asked.is_ok(), "{asked:?}");
        let b = session.task("tb");
        // `a` asks `b`, then its turn ends: it waits on `b`, idle.
        let a = session.task("ta");
        let ask = json!({ "agent": session.short(b), "text": "Hi." });
        let said = session.call(a, MESSAGE, "m1", ask);
        assert!(said.contains("in the background"), "{said}");
        let world = session.app.world_mut();
        world.resource_mut::<AnswerInPlace>().0 = Some(a);
        let holds: Vec<(Entity, Entity)> = world
            .query::<(Entity, &HoldOf)>()
            .iter(world)
            .map(|(hold, &HoldOf(call))| (hold, call))
            .collect();
        for (hold, call) in holds {
            if world
                .get::<ForAgent>(call)
                .is_some_and(|&ForAgent(of)| of == a)
            {
                world.despawn(hold);
            }
        }
        let wakes = &session.wakes;
        run_until(
            &mut session.app,
            wakes,
            |world| matches!(world.get::<Lifecycle>(a), Some(Lifecycle::WaitingOn(on)) if *on == b),
        );
        let waiting = session.app.world().get::<Lifecycle>(a);
        assert!(
            matches!(waiting, Some(Lifecycle::WaitingOn(_))),
            "{waiting:?}"
        );
        if !crash {
            session.app.world_mut().write_message(AppExit::Success);
            session.app.update();
        }
        drop(session);

        // Each owed report comes as a note: `b`'s to `a`, and `a`'s, as
        // stopped with it, to the parent.
        let session = Session::new(&store, None);
        let reports = session.to(session.parent);
        let from_a = reports.iter().any(|(request, mode, text)| {
            *request == "ta" && *mode == DeliveryMode::Inform && text.starts_with("Stopped:")
        });
        assert!(from_a, "crash: {crash} {reports:?}");
    }
}

#[test]
fn a_restored_subagent_whose_work_a_plugin_carries_on_is_not_stopped() {
    for claimed in [false, true] {
        let store = MemoryStore::default();
        let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
        let mut session = Session::new(&store, Some(&model));
        session
            .app
            .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
        let asked = commit_message(
            session.app.world_mut(),
            session.parent,
            Message::user("Go."),
        );
        assert!(asked.is_ok(), "{asked:?}");
        let child = session.task("ta");
        session.app.update();
        let named = session.app.world().get::<Name>(child);
        assert_eq!(named.map(Name::as_str), Some("ta"));
        drop(session);
        // A library program reads the subagent's title and finds the main
        // agent by the keys and the order the harness uses.
        assert_eq!(AgentTitle::type_path(), rig_harness::rig_memory::TITLE);
        let saved = rig_harness::rig_memory::Session::read(&store);
        let saved = saved.as_ref().ok();
        let main = saved.and_then(|saved| saved.main_agent());
        assert!(main.is_some_and(|main| main.parent.is_none()));
        let sub = saved.and_then(|saved| saved.agents.iter().find(|agent| agent.parent.is_some()));
        assert_eq!(sub.and_then(|sub| sub.title()).as_deref(), Some("ta"));

        // A plugin that asks the person first lets the restored turns
        // carry on and holds their calls.
        let session = Session::with(&store, None, |app| {
            if claimed {
                let carry_on = |turns: Query<Entity, (With<Restored>, Without<CarryOn>)>,
                                mut commands: Commands| {
                    for turn in &turns {
                        commands.entity(turn).insert(CarryOn);
                    }
                };
                app.add_systems(AgentPass, (carry_on, hold_turn_calls).in_set(Stage::Gate));
            }
        });
        let reports = session.to(session.parent);
        let interrupted = reports.iter().any(|(request, _, text)| {
            *request == "ta" && text.starts_with("Stopped: the session restarted")
        });
        assert_eq!(interrupted, !claimed, "{reports:?}");
    }
}
