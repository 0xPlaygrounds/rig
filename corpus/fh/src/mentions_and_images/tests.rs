use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, collect, first_agent, log_ended, reply, run_until, scripted_models,
};

use std::fs;

use rig_basics::coding_tools::AttachPlugin;

use super::*;

const TYPED: &str = "fix @src/a.rs see @shot.png and @missing.rs";
const A_RS: &str = "fn a() {\n    broken();\n}\n";

/// A project of its own beside the test binary: `src/a.rs` and `shot.png`.
fn project(name: &str) -> PathBuf {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name(name);
    fs::remove_dir_all(&dir).ok();
    assert!(fs::create_dir_all(dir.join("src")).is_ok());
    assert!(fs::write(dir.join("src/a.rs"), A_RS).is_ok());
    assert!(fs::write(dir.join("shot.png"), b"\x89PNG\r\n\x1a\nnot really pixels").is_ok());
    dir
}

/// The scripted models, `test/seeing` and `test/vision` reading images.
fn models(
    seeing: &MockCompletionModel,
    blind: &MockCompletionModel,
    vision: &MockCompletionModel,
) -> Models {
    let mut models = scripted_models([
        ("test/seeing", seeing),
        ("test/blind", blind),
        ("test/vision", vision),
    ]);
    for name in ["test/seeing", "test/vision"] {
        if let Ok(found) = models.0.catalog().resolve(name) {
            let mut spec = found.spec.clone();
            spec.input.image = true;
            models.0.catalog_mut().insert(spec);
        }
    }
    models
}

/// An app on `store` working in `root`, with the plugin beside
/// `rig_basics::coding_tools`' `AttachPlugin`, which it turns off, after its first
/// frame, and its agent.
fn app(store: &MemoryStore, root: &Path, models: Models) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(ProjectRoot(root.to_owned()))
        .insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, AttachPlugin, MentionsPlugin))
        .insert_resource(Mentions {
            vision_model: Some("test/vision".to_owned()),
            ..Mentions::default()
        })
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// Sends the message on `model` and runs until its turn ended.
fn send(app: &mut App, wakes: &Receiver<()>, agent: Entity, model: &str) {
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model.to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, TYPED, DeliveryMode::Steer));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

/// The content of the last user message of `messages`.
fn last_user(messages: &[impl std::borrow::Borrow<ChatMessage>]) -> Vec<UserContent> {
    let user = messages
        .iter()
        .map(std::borrow::Borrow::<ChatMessage>::borrow)
        .rev()
        .find_map(|message| match message {
            ChatMessage::User { content } => Some(content.clone()),
            _ => None,
        });
    user.unwrap_or_default()
}

/// The texts of `content`, and how many images it has.
fn texts(content: &[UserContent]) -> (Vec<String>, usize) {
    let texts = content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text().to_owned()),
            _ => None,
        })
        .collect();
    let images = content
        .iter()
        .filter(|item| matches!(item, UserContent::Image(_)))
        .count();
    (texts, images)
}

fn warned_about_missing(app: &App) -> bool {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .iter()
        .any(|notice| notice.level == NoticeLevel::Warn && notice.text.contains("@missing.rs"))
}

#[test]
fn mentions_go_with_the_message_as_sent() {
    let root = project("fh-mentions-seeing");
    let seeing = MockCompletionModel::from_stream_turns([reply("on it")]);
    let unused = MockCompletionModel::from_stream_turns(Vec::<
        Vec<rig_core::test_utils::MockStreamEvent>,
    >::new());
    let store = MemoryStore::default();
    let (mut first, wakes, agent) = app(&store, &root, models(&seeing, &unused, &unused));
    send(&mut first, &wakes, agent, "test/seeing");

    let requests = seeing.requests();
    let sent = last_user(
        requests
            .first()
            .map(|request| request.chat_history.as_slice())
            .unwrap_or_default(),
    );
    let (sent_texts, images) = texts(&sent);
    let file = format!("<file path=\"src/a.rs\">\n{A_RS}</file>");
    assert_eq!(
        sent_texts.first(),
        Some(&TYPED.to_owned()),
        "the text goes first, as typed"
    );
    assert!(
        matches!(sent.last(), Some(UserContent::Image(_))),
        "then src/a.rs and shot.png, in order: {sent_texts:?}"
    );
    assert!(sent_texts.contains(&file), "{sent_texts:?}");
    assert_eq!(images, 1, "{sent_texts:?}");
    assert_eq!(
        sent_texts.len(),
        2,
        "the text and src/a.rs once, nothing for missing.rs: {sent_texts:?}"
    );
    assert!(warned_about_missing(&first));
    drop(first);

    // The file changes; the resumed session keeps what was sent.
    assert!(fs::write(root.join("src/a.rs"), "fn a() {}\n").is_ok());
    let (mut second, _, agent) = app(&store, &root, models(&unused, &unused, &unused));
    second.update();
    let saved = second
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| last_user(conversation.messages()));
    let (saved_texts, images) = texts(&saved.unwrap_or_default());
    fs::remove_dir_all(&root).ok();
    assert!(saved_texts.contains(&file), "{saved_texts:?}");
    assert_eq!(images, 1, "the image is stored with the session");
}

#[test]
fn a_model_that_reads_no_images_gets_a_description() {
    let root = project("fh-mentions-blind");
    let blind = MockCompletionModel::from_stream_turns([reply("on it")]);
    let vision = MockCompletionModel::from_stream_turns([reply("a red error dialog")]);
    let unused = MockCompletionModel::from_stream_turns(Vec::<
        Vec<rig_core::test_utils::MockStreamEvent>,
    >::new());
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = app(&store, &root, models(&unused, &blind, &vision));
    send(&mut app, &wakes, agent, "test/blind");
    fs::remove_dir_all(&root).ok();

    let requests = blind.requests();
    let sent = last_user(
        requests
            .first()
            .map(|request| request.chat_history.as_slice())
            .unwrap_or_default(),
    );
    let (sent_texts, images) = texts(&sent);
    let described = "[description of shot.png by test/vision]\na red error dialog";
    assert!(
        sent_texts.iter().any(|text| text == described),
        "{sent_texts:?}"
    );
    assert_eq!(images, 0);
    assert!(warned_about_missing(&app));
    let asked = vision.requests();
    let (_, seen) = texts(&last_user(
        asked
            .first()
            .map(|request| request.chat_history.as_slice())
            .unwrap_or_default(),
    ));
    assert_eq!(seen, 1, "the vision model got the image");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(MentionsPlugin);
}
