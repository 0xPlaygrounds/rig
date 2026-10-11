use std::sync::{Arc, Mutex};

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

/// What the fake network holds: messages to receive, and every post sent.
#[derive(Default)]
struct Network {
    incoming: Vec<Incoming>,
    sent: Vec<Post>,
}

/// An in-memory transport over a shared [`Network`].
#[derive(Clone, Default)]
struct Fake(Arc<Mutex<Network>>);

impl Fake {
    fn say(&self, thread: &str, user: &str, text: &str, mentions_bot: bool) {
        if let Ok(mut network) = self.0.lock() {
            network.incoming.push(Incoming {
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

    fn receive(&mut self) -> Vec<Incoming> {
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

/// The command lines the approval's choices ran.
#[derive(Resource, Default)]
struct Chosen(Vec<String>);

fn agents_of(world: &mut World, thread: &str) -> Vec<Entity> {
    let mut agents = world.query::<(Entity, &ChatThread)>();
    agents
        .iter(world)
        .filter(|(_, bound)| bound.thread == thread)
        .map(|(agent, _)| agent)
        .collect()
}

fn said(world: &World, agent: Entity, text: &str) -> bool {
    world.get::<Conversation>(agent).is_some_and(|conversation| {
        conversation.messages().iter().any(|message| {
            matches!(message, message::Message::User { content }
                if content.iter().any(|item| matches!(item, UserContent::Text(said) if said.text == text)))
        })
    })
}

#[test]
fn a_thread_gets_its_own_agent_and_its_answers() {
    let model = MockCompletionModel::from_stream_turns([reply("fixed"), reply("documented")]);
    let fake = Fake::default();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(ChatConfig {
            channels: vec!["C1".to_owned()],
            users: vec!["U1".to_owned()],
        })
        .insert_resource(ChatBridge::new(fake.clone()))
        .init_resource::<Chosen>()
        .add_plugins((AgentPlugin, JournalPlugin, ChatBridgePlugin))
        .add_command(
            "probe",
            "Records its arguments",
            |In(args): In<CommandArgs>, mut chosen: ResMut<Chosen>| chosen.0.push(args.args),
        );
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        let terminal = app.world_mut().spawn((Agent, connection, choice)).id();
        app.finish();
        app.update();

        fake.say("T1", "U1", "fix the bug", true);
        run_until(&mut app, &wakes, |_| fake.posted("T1", "fixed"));
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
        assert_eq!(model.request_count(), 2);

        // A question of another plugin is answered by the thread.
        app.world_mut().write_message(PickRequest {
            agent,
            title: "Allow `rm -rf build`?".to_owned(),
            items: ["Allow once", "Deny"]
                .map(|label| PickItem {
                    label: label.to_owned(),
                    command: format!("probe {label}"),
                })
                .to_vec(),
            selected: 0,
        });
        app.update();
        assert!(fake.posted("T1", "Reply `yes` or `no`."));
        fake.say("T1", "U1", "no", false);
        app.update();
        let chosen = &app.world().resource::<Chosen>().0;
        assert_eq!(chosen, &vec!["Deny".to_owned()]);
    }
    assert!(
        model.request_count() > 0,
        "the scripted model never answered"
    );
}
