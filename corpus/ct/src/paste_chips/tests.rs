use std::sync::mpsc::Receiver;

use rig_core::completion::CompletionRequest;
use rig_core::message::{Message, UserContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models, tui_view,
};
use rig_tui::MainEditor;
use rig_tui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::*;

/// Puts `items` in the view's keys, then runs a frame.
fn push(app: &mut App, items: impl IntoIterator<Item = KeyInput>) {
    app.world_mut()
        .resource_mut::<PendingInput>()
        .0
        .extend(items);
    app.update();
}

/// Types `text`, a key for each character.
fn type_text(app: &mut App, text: &str) {
    let key = |character| Event::Key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
    push(
        app,
        text.chars()
            .map(|character| KeyInput::Event(key(character))),
    );
}

/// Pastes `text` as a terminal with bracketed paste reports it.
fn paste(app: &mut App, text: &str) {
    push(app, [KeyInput::Event(Event::Paste(text.to_owned()))]);
}

/// Presses `keys`, each in a frame of its own.
fn press(app: &mut App, keys: &str) {
    for key in keys.split_whitespace() {
        rig_harness_test_support::press(app, key);
    }
}

/// The main editor; an empty one without it.
fn editor(world: &World) -> Editor {
    let main = world.try_query_filtered::<&Editor, With<MainEditor>>();
    main.and_then(|mut main| main.single(world).ok().cloned())
        .unwrap_or_default()
}

fn input(app: &App) -> (String, usize) {
    let input = editor(app.world());
    (input.text().to_owned(), input.cursor())
}

/// The content of the last user message `request` sends.
fn last_user(request: Option<&CompletionRequest>) -> Vec<UserContent> {
    let history = request.map(|request| request.chat_history.as_slice());
    let user = history
        .unwrap_or_default()
        .iter()
        .rev()
        .find_map(|message| match &**message {
            Message::User { content } => Some(content.to_vec()),
            _ => None,
        });
    user.unwrap_or_default()
}

fn warnings(app: &App) -> usize {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warned = |notice: &&Notice| notice.level == NoticeLevel::Warn;
    notices.iter().filter(warned).count()
}

fn ended(app: &App) -> usize {
    app.world().resource::<Ended>().0.len()
}

/// A clipboard holding a 1×1 PNG.
fn fake_clipboard(_: &Launcher, into: &Path) -> io::Result<Option<PathBuf>> {
    std::fs::create_dir_all(into)?;
    let path = into.join("pasted.png");
    image::RgbaImage::new(1, 1)
        .save(&path)
        .map_err(io::Error::other)?;
    Ok(Some(path))
}

/// The terminal view on a scratch session with the plugin and the
/// agent's attachments, `test/model`
/// reading no images and `test/vision` reading them, showing its agent.
fn app(model: &MockCompletionModel, vision: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let home = scratch_home("ct-paste-chips");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let mut models = scripted_models([("test/model", model), ("test/vision", vision)]);
    if let Ok(found) = models.0.catalog().resolve("test/vision") {
        let mut spec = found.spec.clone();
        spec.input.image = true;
        models.0.catalog_mut().insert(spec);
    }
    app.insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, PasteChipsPlugin))
        // The agent half that reads `@path` into an attachment.
        .add_plugins(rig_basics::coding_tools::AttachPlugin)
        .insert_resource(ClipboardImages::new(Arc::new(fake_clipboard)))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    tui_view(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
    });
    app.insert_resource(ShownAgent(Some(agent)));
    app.update();
    (app, wakes, agent)
}

#[test]
fn long_pastes_are_chips_sent_as_their_contents() {
    let model = MockCompletionModel::from_stream_turns([reply("Fixed.")]);
    let vision = MockCompletionModel::from_stream_turns([reply("A pixel.")]);
    let (mut app, wakes, agent) = app(&model, &vision);
    let log: Vec<String> = (1..=50).map(|n| format!("line {n}")).collect();
    let typed = "fix this: [Pasted #1 · 50 lines] thanks";

    // One chip between the typed texts; Left from the end skips it whole.
    type_text(&mut app, "fix this: ");
    paste(&mut app, &log.join("\n"));
    type_text(&mut app, " thanks");
    assert_eq!(input(&app), (typed.to_owned(), typed.len()));
    press(&mut app, "left left left left left left left left");
    assert_eq!(input(&app).1, "fix this: ".len());

    // Another paste, then Backspace: gone whole.
    press(&mut app, "ctrl+e");
    let other: Vec<String> = (1..=20).map(|n| format!("other {n}")).collect();
    paste(&mut app, &other.join("\n"));
    assert!(input(&app).0.ends_with("thanks[Pasted #2 · 20 lines]"));
    press(&mut app, "backspace");
    assert_eq!(input(&app).0, typed);

    // Opened before the chip, line 7 changed, closed again.
    press(&mut app, "ctrl+a");
    press(&mut app, &"right ".repeat(10));
    press(&mut app, "alt+e");
    let opened = input(&app).0;
    assert!(
        opened.starts_with("fix this: [Pasted #1 ▾]\nline 1\n"),
        "{opened}"
    );
    let at = opened.find("line 7\n").unwrap_or_default() + "line 7".len();
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut Editor, With<MainEditor>>();
    if let Ok(mut input) = main.single_mut(world) {
        input.set_cursor(at);
    }
    type_text(&mut app, "x");
    press(&mut app, "alt+e");
    assert_eq!(input(&app).0, typed);

    // Sent with the 50 lines, as edited, in a fence between the texts.
    press(&mut app, "enter");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let edited = log.join("\n").replace("line 7\n", "line 7x\n");
    let sent = format!("fix this: \n```\n{edited}\n```\n thanks");
    let requests = model.requests();
    let content = last_user(requests.first());
    assert!(
        matches!(content.as_slice(), [UserContent::Text(text)] if text.text == sent),
        "{content:?}"
    );

    // The clipboard's image: a chip, a warning on Enter for a model that
    // reads no images, then sent as an image to one that does.
    press(&mut app, "ctrl+v");
    run_until(&mut app, &wakes, |world| {
        editor(world).text().contains("[Image #3 · 1×1]")
    });
    type_text(&mut app, " what is it?");
    let before = warnings(&app);
    press(&mut app, "enter");
    assert_eq!(warnings(&app), before + 1);
    assert_eq!(input(&app).0, "[Image #3 · 1×1] what is it?");
    assert_eq!(ended(&app), 1, "not sent");
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/vision".to_owned(),
    });
    app.update();
    press(&mut app, "enter");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let requests = vision.requests();
    let content = last_user(requests.first());
    // The text, and the image `@path` attached.
    let image = content
        .iter()
        .any(|part| matches!(part, UserContent::Image(_)));
    let text = content
        .iter()
        .any(|part| matches!(part, UserContent::Text(_)));
    assert!(image && text, "{content:?}");

    // A short paste is plain text.
    let short = "a\nb\nc\nd\ne";
    paste(&mut app, short);
    assert_eq!(input(&app).0, short);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(PasteChipsPlugin);
}
