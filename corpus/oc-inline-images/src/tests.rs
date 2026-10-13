use std::sync::mpsc::Receiver;

use bevy_ecs::system::RunSystemOnce;
use image::{ImageBuffer, Rgba};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_core::message::ImageMediaType;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, first_agent, log_ended, reply, run_until, scripted_models,
    tui_view,
};
use rig_tui::{Expanded, TranscriptLayout};

use super::*;

/// A 100×50 PNG: red above, blue below.
fn screenshot() -> Vec<u8> {
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
    png
}

/// A `screenshot` tool that returns the PNG.
fn shoot(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let image = Image {
        data: DocumentSourceKind::Base64(STANDARD.encode(screenshot())),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    if let Ok(mut call) = calls.get_mut(called.call) {
        let output = ToolOutput::one(ToolResultContent::Image(image));
        call.decide(Decider::plugin("screenshot"), ToolResult::success(output));
    }
}

/// The kernel, the terminal view without a terminal, the `screenshot` tool
/// and the plugin on `store`, with `settings` and a terminal that draws
/// `graphics`; its first frame restored the session.
fn front(store: &MemoryStore, settings: &str, graphics: Option<Protocol>) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(TerminalGraphics {
            protocol: graphics,
            ..TerminalGraphics::default()
        })
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_open_tool(
            "screenshot",
            "Take a screenshot.",
            ToolOptions::default(),
            shoot,
        );
    tui_view(&mut app).add_plugins(InlineImagesPlugin);
    app.finish();
    app.update();
    (app, wakes)
}

/// The shown agent's transcript, laid out 80 columns wide.
fn laid_out(app: &mut App) -> DrawnTranscript {
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let lay_out = move |layout: TranscriptLayout, view: Res<TranscriptView>| {
        layout.lay_out(&view, agent, 80, Expanded::AsShown)
    };
    app.world_mut()
        .run_system_once(lay_out)
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// The rows of the screenshot call.
fn call_rows(drawn: &DrawnTranscript) -> Range<usize> {
    let call = drawn
        .items()
        .iter()
        .find(|(item, _)| matches!(item, TranscriptItem::Call(_)));
    call.map(|(_, rows)| rows.clone()).unwrap_or_default()
}

/// What the plugin writes to the terminal for `drawn` with `on_screen`
/// on screen, the transcript at the top of the screen.
fn frame(app: &mut App, drawn: &DrawnTranscript, on_screen: Range<usize>) -> String {
    let world = app.world_mut();
    let (images, graphics) = (
        world.resource::<Images>().clone(),
        *world.resource::<TerminalGraphics>(),
    );
    let view = Viewport {
        drawn,
        on_screen,
        origin: (0, 0),
        covered: false,
    };
    let shared = images.lock();
    let wanted = wanted(&shared.slots, &view);
    writes(
        &mut world.resource_mut::<Placed>(),
        wanted,
        &shared,
        &view,
        &graphics,
    )
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

#[test]
fn a_screenshot_is_placed_under_its_call_taken_off_when_scrolled_away_and_drawn_again_on_resume() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "screenshot", serde_json::json!({})),
        reply("That is the login page."),
    ]);
    let (mut app, wakes) = front(&store, "", Some(Protocol::Kitty));
    app.insert_resource(scripted_models([("test/model", &model)]))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().insert_resource(Shown(Some(agent)));
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

    // The call reserves blank rows under its caption, after the result.
    let drawn = laid_out(&mut app);
    let rows = call_rows(&drawn);
    let texts: Vec<&str> = drawn.rows().iter().map(|row| row.text.as_str()).collect();
    let caption = texts
        .iter()
        .position(|text| text.contains("image 100×50 png"));
    assert!(caption.is_some_and(|at| rows.contains(&at)), "{texts:#?}");

    // On screen: sent once, placed below the caption, within the bounds
    // (60% of 80 columns less the indent; 20 rows), keeping the 2:1 shape.
    let out = frame(&mut app, &drawn, 0..drawn.rows().len());
    assert!(out.contains("\x1b_Ga=t,f=100,t=d,"), "{out:?}");
    let place = out
        .split("\x1b_G")
        .find(|part| part.starts_with("a=p,"))
        .unwrap_or_default();
    let (cols, image_rows) = (number(place, "c"), number(place, "r"));
    assert_eq!((cols, image_rows), (Some(13), Some(4)), "{place:?}");
    let row = out
        .split("\x1b[")
        .nth(1)
        .and_then(|cup| cup.split(';').next()?.parse::<usize>().ok());
    assert!(
        row.zip(caption)
            .is_some_and(|(row, caption)| row == caption + 2),
        "{out:?}"
    );

    // The same frame again writes nothing; scrolled away, it comes off.
    assert_eq!(frame(&mut app, &drawn, 0..drawn.rows().len()), "");
    let below = rows.end..rows.end + 10;
    let out = frame(&mut app, &drawn, below);
    assert!(out.contains("\x1b_Ga=d,d=i,"), "{out:?}");
    drop(app);

    // Resumed in a fresh front on a terminal without graphics: half-blocks
    // where the image was, red above blue.
    let (mut app, _) = front(&store, "", None);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().insert_resource(Shown(Some(agent)));
    let drawn = laid_out(&mut app);
    let blocks = drawn
        .rows()
        .iter()
        .filter(|row| row.text.trim() == "▀".repeat(13))
        .count();
    assert_eq!(blocks, 4, "{:#?}", drawn.rows());
    drop(app);

    // `mode = "off"`: one line with the size, no blocks.
    let (mut app, _) = front(&store, "[images]\nmode = \"off\"\n", Some(Protocol::Kitty));
    let drawn = laid_out(&mut app);
    let line = drawn
        .rows()
        .iter()
        .find(|row| row.text.contains("[image 100×50 png,"));
    assert!(
        line.is_some_and(|row| row.text.contains("B — press o to open]")),
        "{:#?}",
        drawn.rows()
    );
    assert!(!drawn.rows().iter().any(|row| row.text.contains('▀')));
}
