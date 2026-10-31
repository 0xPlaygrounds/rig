use rig_basics::BasicCommandsPlugin;
use rig_core::completion::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, collect, connect, log_ended, reply, run_until,
};

use super::*;

/// The text of the last message of the last request `model` got, if it is
/// the user's.
fn last_user_text(model: &MockCompletionModel) -> Option<String> {
    let requests = model.requests();
    let Some(Message::User { content }) = requests
        .last()?
        .chat_history
        .last()
        .map(|message| (**message).clone())
    else {
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

#[test]
fn typed_commands_are_expanded_in_parallel_with_a_timeout() {
    let model = MockCompletionModel::from_stream_turns([reply("Three."), reply("Seen.")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml("[inline_shell]\ntimeout_secs = 0.2"))
        .insert_resource(ProjectRoot(env!("CARGO_MANIFEST_DIR").into()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            BasicCommandsPlugin,
            InlineShellPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
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

    // The user's commands run side by side: the slow one times out, the
    // escaped one stays as typed.
    let typed = r"count: !{printf 3} and !{sleep 30} and \!{x}";
    app.world_mut()
        .trigger(Deliver::new(agent, typed, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| !model.requests().is_empty());
    let sent = last_user_text(&model).unwrap_or_default();
    assert!(sent.contains('3'), "{sent}");
    assert!(sent.contains("(timed out)"), "{sent}");
    assert!(sent.contains("!{x}"), "{sent}");
    assert_eq!(sent.matches("!{").count(), 1, "{sent}");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    // The conversation keeps the expanded text.
    let kept = app
        .world()
        .get::<Conversation>(agent)
        .and_then(
            |conversation| match conversation.messages().first().map(|message| &**message) {
                Some(Message::User { content }) => Some(format!("{content:?}")),
                _ => None,
            },
        )
        .unwrap_or_default();
    assert!(kept.contains("(timed out)"), "{kept}");

    // A plugin's message reaches the model as it was sent.
    let from_plugin = Deliver::new(agent, "!{printf 3}", DeliveryMode::Steer)
        .with_author(Author::plugin("tester"));
    app.world_mut().trigger(from_plugin);
    run_until(&mut app, &wakes, |_| model.requests().len() == 2);
    let sent = last_user_text(&model).unwrap_or_default();
    assert!(sent.contains("!{printf 3}"), "{sent}");

    // A slash command runs as typed, nothing expanded and nothing sent:
    // `/help`, which takes no text, refuses the text as it came.
    app.world_mut()
        .trigger(Deliver::new(agent, "/help !{x}", DeliveryMode::Steer));
    for _ in 0..3 {
        app.update();
    }
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    let help = recalled
        .iter()
        .any(|recalled| recalled.text == "/help !{x}");
    assert!(help, "{recalled:?}");
    let mut jobs = app.world_mut().query::<&Job>();
    assert_eq!(jobs.iter(app.world()).count(), 0);
    assert_eq!(model.requests().len(), 2);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(InlineShellPlugin);
}
