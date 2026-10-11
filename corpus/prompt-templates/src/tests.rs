use rig_basics::BasicCommandsPlugin;
use rig_cassette::journal::MemoryStore;
use rig_core::completion::Message;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The text of a user message.
fn user_text(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };
    let texts: Vec<String> = content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect();
    Some(texts.join("\n"))
}

/// The notices written since the last frame.
fn notices(app: &App) -> Vec<String> {
    let notices = app.world().resource::<Messages<Notice>>();
    notices
        .iter_current_update_messages()
        .map(|notice| notice.text.clone())
        .collect()
}

#[test]
fn a_template_is_a_command_that_sends_its_expansion() {
    let dir = std::env::temp_dir().join(format!("corpus-prompt-templates-{}", std::process::id()));
    let prompts = dir.join("prompts");
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(TemplateDirs(vec![prompts.clone()]))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            BasicCommandsPlugin,
            PromptTemplatesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("No bugs.")]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let ready = agent.zip(connect(&model));
    assert!(ready.is_some(), "an agent, and the catalog lists the model");
    let Some((agent, connection)) = ready else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    app.update();

    // A template written while the agent runs.
    let written = std::fs::create_dir_all(&prompts).and_then(|()| {
        std::fs::write(
            prompts.join("review.md"),
            "---\ndescription: Review a file\n---\nReview $1 for bugs.\n",
        )
    });
    assert!(written.is_ok(), "{written:?}");

    // `/help` lists the template with its description.
    let typed = |text: &str| Deliver::new(agent, text, DeliveryMode::Steer);
    app.world_mut().trigger(typed("/help"));
    app.world_mut().flush();
    let help = notices(&app).join("\n");
    let listed = help
        .lines()
        .any(|line| line.starts_with("/review ") && line.ends_with(" Review a file"));
    assert!(listed, "{help}");

    // Without its argument it is refused with its usage, and nothing is sent.
    app.world_mut().trigger(typed("/review"));
    app.world_mut().flush();
    let recalled = app.world().resource::<Messages<Recalled>>();
    let why: Vec<String> = recalled
        .iter_current_update_messages()
        .filter_map(|recalled| recalled.why.clone())
        .collect();
    assert_eq!(why, ["Usage: /review <arg1>"]);
    app.update();
    assert!(model.requests().is_empty());

    // With it, the expansion is the user's message to the model.
    app.world_mut().trigger(typed("/review src/lib.rs"));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);
    let requests = model.requests();
    let last = requests
        .first()
        .and_then(|request| request.chat_history.last())
        .and_then(user_text);
    assert_eq!(requests.len(), 1);
    assert_eq!(last.as_deref(), Some("Review src/lib.rs for bugs."));
    std::fs::remove_dir_all(&dir).ok();
}
