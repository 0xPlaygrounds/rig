use std::sync::mpsc::Receiver;

use bevy_ecs::system::RunSystemOnce;
use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::Message;
use rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::journal::commit_message;
use rig_ecs::turn::ToolStarter;
use serde_json::json;

use super::*;
use crate::plugins::testing::{app_on, calls, connect, reply, run_until};

/// Each message delivered: the agent it went to, the request it names, its
/// mode and its text.
#[derive(Resource, Default)]
struct Delivered(Vec<(Entity, String, DeliveryMode, String)>);

/// An app on a session store with the subagents, after its first frame
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
        let (mut app, wakes) = app_on(store);
        app.insert_resource(Models(Connector::new(Catalog::default())))
            .add_plugins((AgentPlugin, JournalPlugin, SubagentsPlugin))
            .init_resource::<Delivered>()
            .add_observer(|sent: On<Deliver>, mut delivered: ResMut<Delivered>| {
                let request = sent.origin.request.clone().map(|request| request.0);
                let text = sent.text.clone();
                let entry = (sent.entity, request.unwrap_or_default(), sent.mode, text);
                delivered.0.push(entry);
            });
        if let Some(connection) = model.and_then(connect) {
            let choice = ModelChoice(connection.spec.reference());
            app.world_mut().spawn((Agent, connection, choice));
        }
        app.finish();
        app.update();
        let world = app.world_mut();
        let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let parent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        Self { app, wakes, parent }
    }

    /// `agent` calls `tool` with `args` in call `id`, as the model call 1
    /// asked, and gets its output once it has one.
    fn call(&mut self, agent: Entity, tool: &str, id: &str, args: serde_json::Value) -> String {
        let Ok(name) = ToolName::new(tool) else {
            return String::new();
        };
        let call = ToolCall::from_wire(id, ToolFunction::new(name, args));
        let started = self.app.world_mut().run_system_once(
            move |starter: ToolStarter, mut commands: Commands| {
                let run = starter.run(call.clone(), Some(EffectId::from_raw(1)));
                let entity = commands.spawn(run.clone()).id();
                starter.start(&mut commands, entity, agent, &run);
                entity
            },
        );
        let output = started
            .ok()
            .and_then(|call| self.app.world().get::<ToolOutput>(call));
        output
            .map(|output| output.0.output().render())
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
            .map(|(_, request, mode, text)| (request.as_str(), *mode, text.as_str()))
            .collect()
    }
}

#[test]
fn peers_agree_and_the_one_that_waited_reports_its_later_answer_with_the_batch() {
    let parent = MockCompletionModel::from_stream_turns([reply("Settled.")]);
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
    run_until(&mut session.app, &session.wakes, |_| {
        parent.request_count() > 0
    });

    // Each request got one report; the critic's came first, as a note read
    // with the proposer's, which carried the parent on.
    let reports = session.to(session.parent);
    let expected = [
        ("Critique", DeliveryMode::Note, "Lumen is good."),
        ("Propose", DeliveryMode::Queue, "We agreed on Lumen."),
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
    let read = format!("{:?}", parent.requests());
    let header = format!("Output of agent {} \\\"Critique\\\"", session.short(b));
    assert!(read.contains(&header), "{read}");
    assert!(!read.contains("Waiting for the critic."), "{read}");
}

#[test]
fn requests_that_would_deadlock_are_refused_and_a_restart_answers_open_ones_as_interrupted() {
    let store = MemoryStore::default();
    // Its turns are never read: the session stops before a reply arrives.
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut session = Session::new(&store, Some(&model));
    let asked = Message::user("Find a name.");
    commit_message(session.app.world_mut(), session.parent, asked);
    let (a, b, c) = (session.task("ta"), session.task("tb"), session.task("tc"));
    // `a` waits on `c` through `b`, so `c` can neither ask nor wait for it;
    // the parent is idle, so nothing would come from it.
    let ask = |agent| json!({ "agent": session.short(agent), "text": "Hi." });
    let wait = |agent| json!({ "agent": session.short(agent) });
    let calls = [
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

    let session = Session::new(&store, None);
    let reports = session.to(session.parent);
    let interrupted = reports
        .iter()
        .filter(|(_, mode, text)| *mode == DeliveryMode::Queue && text.starts_with("Interrupted"));
    let mut requests: Vec<&str> = interrupted.map(|(request, ..)| *request).collect();
    requests.sort_unstable();
    assert_eq!(requests, ["ta", "tb", "tc"], "{reports:?}");
}
