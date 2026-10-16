use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, collect, connect, first_agent, log_ended, reply, run_until,
};

use super::*;

const EXAMPLE: &str = "---
when: commit
---
## user
Commit the parser fix.
## assistant
fix(parser): accept a trailing comma
";

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
    let said = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(said);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// The messages of request `index` to `model` but the system prompt,
/// wherever the model's adapter keeps that.
fn sent_messages(model: &MockCompletionModel, index: usize) -> Vec<Message> {
    let requests = model.requests();
    let history = requests
        .get(index)
        .map(|request| request.chat_history.clone());
    let history = history.unwrap_or_default().into_iter();
    history
        .filter(|message| !matches!(&**message, Message::System { .. }))
        .map(std::sync::Arc::unwrap_or_clone)
        .collect()
}

#[test]
fn a_fitting_example_goes_first_in_the_request_and_nowhere_else() {
    let root = project();
    assert!(root.is_some(), "the project was written");
    let Some(root) = root else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(root.clone()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            WatchPlugin,
            FewShotExamplesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<FileChanges>(&mut app);
    app.finish();
    app.update();
    // The examples are read on the watch's first listing.
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Collected<FileChanges>>().0.is_empty()
    });
    let model = MockCompletionModel::from_stream_turns([reply("Committed."), reply("Hi.")]);
    let world = app.world_mut();
    let (agent, connection) = (first_agent(world), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "one agent, and the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    say(&mut app, &wakes, agent, "commit this", 1);
    let first = sent_messages(&model, 0);
    assert!(
        matches!(
            first.as_slice(),
            [
                example_user,
                example_assistant,
                real,
            ] if *example_user == Message::user("Commit the parser fix.")
                && *example_assistant == Message::assistant("fix(parser): accept a trailing comma")
                && *real == Message::user("commit this")
        ),
        "{first:#?}"
    );
    let conversation = app.world().get::<Conversation>(agent);
    let messages = conversation.map(|conversation| conversation.messages().to_vec());
    assert!(
        matches!(messages.as_deref(), Some([user, _answer]) if *user == Message::user("commit this").into()),
        "{messages:#?}"
    );
    let sent = app.world().get::<SentExamples>(agent).cloned();
    assert_eq!(sent, Some(SentExamples(vec!["commit".to_owned()])));

    say(&mut app, &wakes, agent, "hello", 2);
    let second = sent_messages(&model, 1);
    let commit = Message::user("Commit the parser fix.");
    assert_eq!(second.len(), 3, "{second:#?}");
    assert!(!second.contains(&commit), "{second:#?}");
    assert_eq!(second.last(), Some(&Message::user("hello")));
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(FewShotExamplesPlugin);
}
