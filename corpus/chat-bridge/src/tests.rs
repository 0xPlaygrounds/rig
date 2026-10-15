use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{app_on, calls, connect, first_agent, reply, run_until, start_call};
use schemars::JsonSchema;

use super::*;

/// What the fake network holds: messages to receive, and every post sent.
#[derive(Default)]
struct Network {
    incoming: Vec<ChatMessage>,
    sent: Vec<Post>,
}

/// An in-memory transport over a shared [`Network`].
#[derive(Clone, Default)]
struct Fake(Arc<Mutex<Network>>);

impl Fake {
    fn say(&self, thread: &str, user: &str, text: &str, mentions_bot: bool) {
        if let Ok(mut network) = self.0.lock() {
            network.incoming.push(ChatMessage {
                channel: "C1".to_owned(),
                thread: thread.to_owned(),
                user: user.to_owned(),
                text: text.to_owned(),
                mentions_bot,
            });
        }
    }

    /// Whether a post in `thread` contains `text`.
    fn posted(&self, thread: &str, text: &str) -> bool {
        self.0.lock().is_ok_and(|network| {
            network
                .sent
                .iter()
                .any(|post| post.thread == thread && post.text.contains(text))
        })
    }
}

impl ChatTransport for Fake {
    fn start(&mut self, _wake: Wake) {}

    fn receive(&mut self) -> Vec<ChatMessage> {
        self.0
            .lock()
            .map(|mut network| std::mem::take(&mut network.incoming))
            .unwrap_or_default()
    }

    fn send(&mut self, post: &Post) -> Result<(), Disconnected> {
        if let Ok(mut network) = self.0.lock() {
            network.sent.push(post.clone());
        }
        Ok(())
    }
}

/// `read`'s arguments.
#[derive(Deserialize, JsonSchema)]
struct ReadArgs {}

/// An app on `store` with the bridge on `fake`, allowing `U1` in `C1`, and
/// a `read` tool; a terminal's agent answered by `model`, when given.
fn bridge_app(
    store: &MemoryStore,
    fake: &Fake,
    model: Option<&MockCompletionModel>,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let settings = "[chat]\nchannels = [\"C1\"]\nusers = [\"U1\"]";
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(ChatBridge::new(fake.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, ChatBridgePlugin))
        .add_open_tool(
            "read",
            "Read a file.",
            ToolOptions::default(),
            |called: On<ToolCalled<ReadArgs>>, mut calls: Query<&mut Decision, ToolCalls>| {
                if let Ok(mut call) = calls.get_mut(called.call) {
                    call.decide(
                        Decider::plugin("test"),
                        ToolResult::success("fn main() {}".into()),
                    );
                }
            },
        );
    if let Some(connection) = model.and_then(connect) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut().spawn((Agent, connection, choice));
    }
    app.finish();
    app.update();
    (app, wakes)
}

fn agents_of(world: &mut World, thread: &str) -> Vec<Entity> {
    let mut agents = world.query::<(Entity, &ChatThread)>();
    agents
        .iter(world)
        .filter(|(_, bound)| bound.thread == thread)
        .map(|(agent, _)| agent)
        .collect()
}

/// Whether `agent`'s conversation has `text` from the user, by the chat
/// user `U1`.
fn said(world: &World, agent: Entity, text: &str) -> bool {
    world
        .get::<Conversation>(agent)
        .is_some_and(|conversation| {
            conversation
                .messages()
                .iter()
                .enumerate()
                .any(|(at, message)| {
                    let message::Message::User { content } = &**message else {
                        return false;
                    };
                    content.iter().any(
                        |content| matches!(content, UserContent::Text(said) if said.text == text),
                    ) && conversation
                        .author(at)
                        .is_some_and(|author| author.title.as_deref() == Some("U1"))
                })
        })
}

#[test]
fn a_thread_gets_its_own_agent_its_answers_and_its_questions() {
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", serde_json::json!({})),
        reply("fixed"),
        reply("documented"),
    ]);
    let fake = Fake::default();
    let store = MemoryStore::default();
    let (mut app, wakes) = bridge_app(&store, &fake, Some(&model));
    let terminal = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);

    fake.say("T1", "U1", "fix the bug", true);
    run_until(&mut app, &wakes, |_| fake.posted("T1", "fixed"));
    assert!(fake.posted("T1", "working… read 1 file"));
    assert!(fake.posted("T1", "fixed"));
    let agents = agents_of(app.world_mut(), "T1");
    assert_eq!(agents.len(), 1);
    let agent = agents.first().copied().unwrap_or(Entity::PLACEHOLDER);
    assert_ne!(agent, terminal);
    assert!(said(app.world(), agent, "fix the bug"));

    // A reply in the thread, without a mention, goes to the same agent.
    fake.say("T1", "U1", "and the docs", false);
    run_until(&mut app, &wakes, |_| fake.posted("T1", "documented"));
    assert_eq!(agents_of(app.world_mut(), "T1"), vec![agent]);
    assert!(said(app.world(), agent, "and the docs"));

    // Someone not allowed starts no agent and reaches none.
    fake.say("T2", "U2", "delete everything", true);
    fake.say("T1", "U2", "delete everything", false);
    for _ in 0..5 {
        app.update();
    }
    assert!(agents_of(app.world_mut(), "T2").is_empty());
    assert!(!said(app.world(), agent, "delete everything"));
    assert!(!said(app.world(), terminal, "delete everything"));
    assert_eq!(model.request_count(), 3);

    // Another plugin's question about the agent is the thread's to answer.
    let ask = Ask::yes_no("Run `rm -rf build`?");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    app.update();
    assert!(fake.posted("T1", "Reply `yes` or `no`."));
    fake.say("T1", "U1", "no", false);
    app.update();
    let answer = app.world().get::<Outcome>(ask).and_then(Answer::of);
    assert_eq!(answer, Some(Answer::choice("no")));
    drop(app);

    // A restart keeps the thread's agent, and its questions.
    let (mut app, _wakes) = bridge_app(&store, &fake, None);
    let agents = agents_of(app.world_mut(), "T1");
    assert_eq!(agents.len(), 1);
    let restored = agents.first().copied().unwrap_or(Entity::PLACEHOLDER);
    let answers = app.world().get::<Answerer>(restored).copied();
    assert_eq!(
        answers,
        Some(Answerer {
            agent: Some(restored)
        })
    );
}

#[test]
fn a_turn_ended_in_the_frame_its_calls_started_still_posts_its_work() {
    let fake = Fake::default();
    let (mut app, _wakes) = bridge_app(&MemoryStore::default(), &fake, None);
    let world = app.world_mut();
    let thread = ChatThread {
        channel: "C1".to_owned(),
        thread: "T1".to_owned(),
    };
    let agent = world.spawn((Agent, thread)).id();
    let call = start_call(world, agent, ("read", "r1"), serde_json::json!({}));
    assert!(call.is_some());
    let answered = TurnOutcome::Answered(message::Message::assistant("fixed"));
    world.spawn((answered, ForAgent(agent))).insert(Settled);
    app.update();
    assert!(fake.posted("T1", "working… read 1 file"));
    assert!(fake.posted("T1", "fixed"));
}
