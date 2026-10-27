use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

/// A session on a fresh store in the rig home `home`, its project beside
/// it, whose first agent `turns` answer, after its first frame.
fn session(
    home: &Home,
    turns: Vec<Vec<MockStreamEvent>>,
) -> (App, Receiver<()>, MockCompletionModel) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(home.root().join("project")))
        .add_plugins((AgentPlugin, SessionLogPlugin, LongTermMemoryPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns(turns);
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(&model)) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, model)
}

/// The user types `text` to the first agent; frames run until the turn
/// ended.
fn ask(app: &mut App, wakes: &Receiver<()>, text: &str) {
    let Some(agent) = first_agent(app.world_mut()) else {
        return;
    };
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

#[test]
fn a_fact_remembered_in_one_session_is_recalled_in_the_next_but_not_saved() {
    let home = scratch_home("corpus-long-term-memory");

    let fact = serde_json::json!({ "fact": "Run tests with cargo nextest", "scope": "project" });
    let (mut first, wakes, _model) =
        session(&home, vec![calls("1", "remember", fact), reply("Noted.")]);
    ask(
        &mut first,
        &wakes,
        "Remember that tests run with cargo nextest.",
    );
    let stored = fs::read_to_string(home.root().join("project/.rig/memory.md")).unwrap_or_default();
    assert!(
        stored.contains("- Run tests with cargo nextest"),
        "{stored}"
    );
    drop(first);

    let (mut second, wakes, model) = session(&home, vec![reply("Like this.")]);
    ask(&mut second, &wakes, "how do I run the tests?");
    let requests = model.requests();
    // The block is in the turn the cache point ends before.
    let carrying = requests.first().and_then(|request| {
        let turns = request.cache_points.iter().find_map(|point| match point {
            CachePoint::Messages(turns) => Some(*turns),
            _ => None,
        })?;
        let mut conversation = request
            .chat_history
            .iter()
            .filter(|message| !matches!(&***message, message::Message::System { .. }));
        serde_json::to_string(conversation.nth(turns)?).ok()
    });
    assert!(
        carrying
            .as_deref()
            .is_some_and(|message| message.contains("how do I run the tests?")
                && message.contains("<memory>\\nWhat you remember")
                && message.contains("] Run tests with cargo nextest\\n</memory>")),
        "{requests:?}"
    );
    let world = second.world_mut();
    let saved: Vec<String> = world
        .query::<&Conversation>()
        .iter(world)
        .flat_map(|conversation| conversation.messages())
        .map(|message| serde_json::to_string(message).unwrap_or_default())
        .collect();
    assert!(
        saved
            .iter()
            .any(|message| message.contains("how do I run the tests?"))
    );
    assert!(
        !saved.iter().any(|message| message.contains("nextest")),
        "{saved:?}"
    );
    fs::remove_dir_all(home.root()).ok();
}

/// A failed `/memory edit` says why, and what the editor saved before it
/// failed is read all the same.
#[cfg(unix)]
#[test]
fn a_failed_editor_says_why_and_the_files_are_read_again() {
    use std::os::unix::process::ExitStatusExt;

    use rig_harness_test_support::{Collected, collect};

    let home = scratch_home("corpus-long-term-memory-editor");
    let (mut app, _wakes, _model) = session(&home, Vec::new());
    collect::<Notice>(&mut app);
    let project = home.root().join("project/.rig/memory.md");
    fs::create_dir_all(home.root().join("project/.rig")).ok();
    fs::write(&project, "- Saved before the editor failed\n").ok();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "no agent");
    let Some(agent) = agent else {
        return;
    };
    let failed = TerminalExit(Ok(std::process::ExitStatus::from_raw(3 << 8)));
    app.world_mut().spawn((Editing(agent), failed));
    app.update();
    fs::remove_dir_all(home.root()).ok();

    let notices = &app.world().resource::<Collected<Notice>>().0;
    let failure = notices
        .iter()
        .find(|notice| notice.text == "/memory edit: the editor failed (exit status 3).");
    assert!(
        failure.is_some_and(|notice| notice.agent == Some(agent)),
        "{notices:?}"
    );
    let listed = app.world().resource::<Memories>().list().join("\n");
    assert!(
        listed.contains("Saved before the editor failed"),
        "{listed}"
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LongTermMemoryPlugin);
}
