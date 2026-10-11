use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

const EXAMPLE: &str = "---
when: commit
---
## user
Commit the parser fix.
## assistant
fix(parser): accept a trailing comma
";

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// A project of its own beside the test binary, holding `commit.md`.
fn project() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let root = exe
        .parent()?
        .join(format!("few-shot-examples-{}", std::process::id()));
    let examples = root.join(".rig").join("examples");
    std::fs::create_dir_all(&examples).ok()?;
    std::fs::write(examples.join("commit.md"), EXAMPLE).ok()?;
    Some(root)
}

/// Says `text` to `agent` and runs frames until turn `turns` ended.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    let said = Deliver::new(agent, text, DeliveryMode::Queue).with_origin(Origin::user());
    app.world_mut().trigger(said);
    run_until(app, wakes, |world| world.resource::<Ended>().0 >= turns);
}

#[test]
fn a_fitting_example_goes_first_in_the_request_and_nowhere_else() {
    let root = project();
    assert!(root.is_some(), "the project was written");
    let Some(root) = root else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, JournalPlugin, FewShotExamplesPlugin))
        .insert_resource(ExamplesProject(root.clone()))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("Committed."), reply("Hi.")]);
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let (agent, connection) = (agent.ok(), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "one agent, and the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    say(&mut app, &wakes, agent, "commit this", 1);
    let requests = model.requests();
    let first = requests.first().map(|request| request.chat_history.clone());
    assert!(
        matches!(
            first.as_deref(),
            Some([
                Message::System { .. },
                example_user,
                example_assistant,
                real,
            ]) if *example_user == Message::user("Commit the parser fix.")
                && *example_assistant == Message::assistant("fix(parser): accept a trailing comma")
                && *real == Message::user("commit this")
        ),
        "{first:#?}"
    );
    let conversation = app.world().get::<Conversation>(agent);
    let messages = conversation.map(|conversation| conversation.messages().to_vec());
    assert!(
        matches!(messages.as_deref(), Some([user, _answer]) if *user == Message::user("commit this")),
        "{messages:#?}"
    );
    let sent = app.world().get::<SentExamples>(agent).cloned();
    assert_eq!(sent, Some(SentExamples(vec!["commit".to_owned()])));

    say(&mut app, &wakes, agent, "hello", 2);
    let requests = model.requests();
    let second = requests.get(1).map(|request| request.chat_history.clone());
    let second = second.unwrap_or_default();
    let commit = Message::user("Commit the parser fix.");
    assert_eq!(second.len(), 4, "{second:#?}");
    assert!(!second.contains(&commit), "{second:#?}");
    assert_eq!(second.last(), Some(&Message::user("hello")));
    std::fs::remove_dir_all(root).ok();
}
