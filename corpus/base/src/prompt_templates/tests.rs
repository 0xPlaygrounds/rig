use rig_basics::BasicCommandsPlugin;
use rig_core::catalog::{Catalog, Connector, ModelSpec, Transport};
use rig_core::completion::Message;
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

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

/// The catalog, with `templates/big`, answered by `big`.
fn models_with(big: &MockCompletionModel) -> Option<Models> {
    let vendor = ProviderId::declare("templates").ok()?;
    let mut connector = Connector::default();
    connector
        .catalog_mut()
        .insert(ModelSpec::new(vendor, "big").with_tools(true));
    let big = big.clone();
    connector.provide(
        vendor,
        move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
            let model = ModelAdapter::<Completion>::new(spec.reference(), big.clone());
            Ok(ErasedHandler::new(model))
        },
    );
    Some(Models(connector))
}

/// Whether the command `/name` exists.
fn has_command(world: &World, name: &str) -> bool {
    let commands = world.try_query_filtered::<&Name, With<SlashCommand>>();
    commands
        .is_some_and(|mut commands| commands.iter(world).any(|command| command.as_str() == name))
}

#[test]
fn a_template_is_a_command_that_sends_its_expansion() {
    let root = std::env::temp_dir().join(format!("corpus-prompt-templates-{}", std::process::id()));
    let prompts = root.join(".rig").join("prompts");
    let write = |name: &str, text: &str| {
        let written = std::fs::create_dir_all(&prompts)
            .and_then(|()| std::fs::write(prompts.join(name), text));
        assert!(written.is_ok(), "{written:?}");
    };
    write(
        "review.md",
        "---\ndescription: Review a file\n---\nReview $1 for bugs.\n",
    );

    let model = MockCompletionModel::from_stream_turns([reply("No bugs.")]);
    let big = MockCompletionModel::from_stream_turns([reply("They are scopes.")]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    if let Some(models) = models_with(&big) {
        app.insert_resource(models);
    }
    app.insert_resource(ProjectRoot(root.clone()))
        .insert_resource(RigHome(Home::at(root.join("home"))))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            WatchPlugin,
            BasicCommandsPlugin,
            PromptTemplatesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
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
    let typed = |app: &mut App, text: &str| {
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
        app.update();
    };

    // The template is a command once its directory's watch is armed, and
    // `/help` lists it with its description.
    run_until(&mut app, &wakes, |world| has_command(world, "/review"));
    typed(&mut app, "/help");
    let notices = app.world().resource::<Messages<Notice>>();
    let help: Vec<String> = notices
        .iter_current_update_messages()
        .map(|notice| notice.text.clone())
        .collect();
    let help = help.join("\n");
    let listed = help
        .lines()
        .any(|line| line.starts_with("/review ") && line.ends_with(" Review a file"));
    assert!(listed, "{help}");

    // Without its argument it is refused with its usage, and nothing is sent.
    typed(&mut app, "/review");
    let recalled = app.world().resource::<Messages<Recalled>>();
    let why: Vec<String> = recalled
        .iter_current_update_messages()
        .filter_map(|recalled| recalled.why.clone())
        .collect();
    assert_eq!(why, ["Usage: /review <arg1>"]);
    app.update();
    assert!(model.requests().is_empty());

    // With it, the expansion is the user's message to the model.
    typed(&mut app, "/review src/lib.rs");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert_eq!(model.requests().len(), 1);
    assert_eq!(
        last_user_text(&model).as_deref(),
        Some("Review src/lib.rs for bugs.")
    );
    // The conversation keeps the template's name on the message.
    let titled = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.author(0)?.title.clone());
    assert_eq!(titled.as_deref(), Some("review"));

    // A template written while the agent runs is a command once seen, and
    // its turn runs on the model it names.
    write(
        "explain.md",
        "---\nmodel: templates/big\nargs: [topic]\n---\nExplain ${topic}.",
    );
    run_until(&mut app, &wakes, |world| has_command(world, "/explain"));
    assert!(has_command(app.world(), "/explain"));
    typed(&mut app, "/explain lifetimes");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    assert_eq!((model.requests().len(), big.requests().len()), (1, 1));
    assert_eq!(last_user_text(&big).as_deref(), Some("Explain lifetimes."));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(PromptTemplatesPlugin);
}
