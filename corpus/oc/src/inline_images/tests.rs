use std::sync::mpsc::Receiver;

use image::{ImageBuffer, Rgba};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_core::message::ImageMediaType;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, commit_message, first_agent, log_ended, reply, run_until,
    scripted_models, tui_view,
};
use rig_tui::TuiTestScreen;

use super::*;

/// A 100×50 PNG: red above, blue below.
fn screenshot() -> Image {
    let pixels = ImageBuffer::from_fn(100, 50, |_, y| {
        if y < 25 {
            Rgba([255, 0, 0, 255])
        } else {
            Rgba([0, 0, 255, 255])
        }
    });
    let mut png = Vec::new();
    let encoded =
        DynamicImage::ImageRgba8(pixels).write_to(&mut Cursor::new(&mut png), ImageFormat::Png);
    assert!(encoded.is_ok(), "{encoded:?}");
    Image {
        data: DocumentSourceKind::Base64(STANDARD.encode(png)),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    }
}

/// A `screenshot` tool that returns the PNG.
fn shoot(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        let output = ToolOutput::one(ToolResultContent::Image(screenshot()));
        call.decide(Decider::plugin("screenshot"), ToolResult::success(output));
    }
}

/// The kernel, the terminal view on an 80×24 screen in memory, the
/// `screenshot` tool and the plugin on `store`, with `settings` and a
/// terminal that draws `graphics`; its first frame restored the session.
fn front(
    store: &MemoryStore,
    settings: &str,
    graphics: Option<Protocol>,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(TerminalGraphics {
            protocol: graphics,
            ..TerminalGraphics::default()
        })
        .insert_resource(TuiTestScreen::new(80, 24))
        .insert_resource(TerminalSize(Rect::new(0, 0, 80, 24)))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_open_tool(
            "screenshot",
            "Take a screenshot.",
            ToolOptions::default(),
            shoot,
        );
    // Another plugin's decorator, drawn after this one's under the call.
    tui_view(&mut app)
        .add_plugins(InlineImagesPlugin)
        .add_systems(
            TranscriptRender,
            note.after(decorate_calls).in_set(RenderSystems::Decorate),
        );
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().insert_resource(ShownAgent(Some(agent)));
    app.update();
    (app, wakes, agent)
}

/// Another plugin's decoration under every call.
fn note(mut batch: ResMut<RenderBatch>) {
    for (_, output) in batch.calls_mut() {
        output.decorate("note", vec![Line::from("  (a note under the call)")]);
    }
}

/// What the plugin wrote to the terminal so far.
fn written(app: &App) -> String {
    String::from_utf8_lossy(app.world().resource::<TuiTestScreen>().written()).into_owned()
}

/// A number after `key=` in a kitty sequence.
fn number(sequence: &str, key: &str) -> Option<u16> {
    let start = sequence.find(&format!(",{key}="))? + key.len() + 2;
    let digits: String = sequence
        .get(start..)?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// What the main pane drew; nothing without it.
fn main_drawn(app: &App) -> DrawnTranscript {
    let world = app.world();
    let panes = world.try_query_filtered::<&DrawnTranscript, With<MainPane>>();
    let drawn = panes.and_then(|mut panes| panes.single(world).ok().cloned());
    drawn.unwrap_or_default()
}

/// The rows of the whole transcript that are an image in half-blocks
/// (13 columns of `▀`).
fn block_rows(app: &App) -> usize {
    let drawn = main_drawn(app);
    let blocks = "▀".repeat(13);
    drawn
        .rows()
        .iter()
        .filter(|row| row.text.trim() == blocks)
        .count()
}

#[test]
fn a_screenshot_is_placed_under_its_call_taken_off_when_scrolled_away_and_drawn_again_on_resume() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "screenshot", serde_json::json!({})),
        reply("That is the login page."),
    ]);
    let (mut app, wakes, agent) = front(&store, "", Some(Protocol::Kitty));
    app.insert_resource(scripted_models([("test/model", &model)]))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
    });
    app.world_mut().trigger(Deliver::new(
        agent,
        "Show me the login page.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.update();

    // Sent once and placed on the screen row below its caption, within
    // the bounds (60% of 80 columns less the indent; 20 rows), keeping
    // the 2:1 shape of 100×50 pixels in 8×16 cells.
    let out = written(&app);
    assert_eq!(out.matches("\x1b_Ga=t,f=100,t=d,").count(), 1, "{out:?}");
    let place = out
        .split("\x1b_G")
        .find(|part| part.starts_with("a=p,"))
        .unwrap_or_default();
    assert_eq!(
        (number(place, "c"), number(place, "r")),
        (Some(13), Some(4))
    );
    let rows = app.world().resource::<TuiTestScreen>().rows();
    let caption = rows.iter().position(|row| row.contains("image 100×50 png"));
    let cup = out
        .split("\x1b7\x1b[")
        .nth(1)
        .and_then(|cup| cup.split(';').next()?.parse::<usize>().ok());
    assert!(
        cup.zip(caption)
            .is_some_and(|(cup, caption)| cup == caption + 2),
        "{out:?} on {rows:#?}"
    );
    // The caption is one of this plugin's decoration rows, though another
    // decorator draws after it.
    let (drawn, frame) = (&main_drawn(&app), app.world().resource::<TuiFrame>());
    let call = drawn.items().iter().find_map(|(item, rows)| match item {
        TranscriptItem::Decoration(_, name) if name == DECORATOR => Some(rows.clone()),
        _ => None,
    });
    let on_rows = call.clone().into_iter().flatten().filter_map(|row| {
        let at = frame.screen_row(drawn, row)?;
        Some(usize::from(at))
    });
    let on_rows: Vec<usize> = on_rows.collect();
    assert!(
        caption.is_some_and(|caption| on_rows.contains(&caption)),
        "{call:?} {on_rows:?}"
    );
    drop(app);

    // Resumed in a fresh front: drawn again from the saved conversation.
    let (mut app, _, agent) = front(&store, "", Some(Protocol::Kitty));
    let out = written(&app);
    assert!(out.contains("\x1b_Ga=t,f=100,t=d,"), "{out:?}");
    assert!(out.contains("\x1b_Ga=p,"), "{out:?}");

    // Scrolled off by a long reply: taken off the screen.
    let long = (1..=40)
        .map(|n| format!("line {n}\n\n"))
        .collect::<String>();
    let committed = commit_message(app.world_mut(), agent, ChatMessage::assistant(long));
    assert!(committed.is_ok(), "{committed:?}");
    let before = written(&app).len();
    app.update();
    let out = written(&app);
    let new = out.get(before..).unwrap_or_default();
    let rows = app.world().resource::<TuiTestScreen>().rows();
    assert!(new.contains("\x1b_Ga=d,d=i,"), "{new:?} {rows:#?}");
    drop(app);

    // No graphics, `mode = "auto"`: half-blocks under the call and under
    // an image the user attached, and no image written to the terminal.
    let (mut app, _, agent) = front(&store, "", None);
    let attached = ChatMessage::User {
        content: vec![UserContent::Image(screenshot())],
    };
    // Answered, so a restart has nothing to carry on.
    for message in [attached, ChatMessage::assistant("A login page.")] {
        let committed = commit_message(app.world_mut(), agent, message);
        assert!(committed.is_ok(), "{committed:?}");
    }
    app.update();
    assert_eq!(block_rows(&app), 8);
    let out = written(&app);
    assert!(
        !out.contains("\x1b_G") && !out.contains("\x1b]1337;"),
        "{out:?}"
    );
    drop(app);

    // iTerm2 draws the attachment in the cells: scrolled off, the view writes those
    // cells again.
    let (mut app, _, agent) = front(&store, "", Some(Protocol::Iterm2));
    assert!(written(&app).contains("\x1b]1337;File=inline=1;"));
    let areas: Vec<Rect> = app
        .world()
        .resource::<Placed>()
        .spots
        .values()
        .map(Spot::area)
        .collect();
    assert_eq!(areas.len(), 1, "{areas:?}");
    let long = ChatMessage::assistant("more\n\n".repeat(30));
    let committed = commit_message(app.world_mut(), agent, long);
    assert!(committed.is_ok(), "{committed:?}");
    app.update();
    app.update();
    let repainted = app.world().resource::<TuiTestScreen>().repainted();
    assert_eq!(repainted, areas.as_slice());
    drop(app);

    // `mode = "off"`: one line with the size, no blocks.
    let (app, _, _) = front(&store, "[images]\nmode = \"off\"\n", Some(Protocol::Kitty));
    let drawn = main_drawn(&app);
    let line = drawn
        .rows()
        .iter()
        .find(|row| row.text.contains("[image 100×50 png,"));
    assert!(
        line.is_some_and(|row| row.text.contains("B — press o to open]")),
        "{:#?}",
        drawn.rows()
    );
    assert_eq!(block_rows(&app), 0);
}
