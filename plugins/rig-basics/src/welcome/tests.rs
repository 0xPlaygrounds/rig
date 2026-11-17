use rig_harness::prelude::*;
use rig_harness_test_support::{MemoryStore, app_on, first_agent};

use rig_tui::ratatui::style::Color;

use super::*;

/// The kernel and the welcome on a fresh session, after its first frame,
/// with its agent.
fn start() -> (App, Option<Entity>) {
    let (mut app, _) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, WelcomePlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    (app, agent)
}

/// The welcome asides of `agent`'s conversation, with their ids.
fn welcomes(app: &App, agent: Option<Entity>) -> Vec<(AsideId, Aside)> {
    let conversation = agent.and_then(|agent| app.world().get::<Conversation>(agent));
    let asides = conversation
        .into_iter()
        .flat_map(|conversation| conversation.asides());
    asides
        .filter(|(_, aside)| aside.kind == WELCOME)
        .map(|(id, aside)| (id, aside.clone()))
        .collect()
}

#[test]
fn the_main_agent_has_one_unsaved_welcome_kept_current() {
    let (mut app, agent) = start();
    let shown = welcomes(&app, agent);
    assert_eq!(shown.len(), 1, "{shown:?}");
    let Some((id, aside)) = shown.first() else {
        return;
    };
    assert!(!aside.saved && aside.after.is_none());
    let says: Option<Welcome> = serde_json::from_value(aside.value.clone()).ok();
    assert_eq!(
        says.as_ref().map(|says| says.version.as_str()),
        Some(env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(says.and_then(|says| says.model), None);

    let Some(agent_entity) = agent else {
        return;
    };
    app.world_mut()
        .entity_mut(agent_entity)
        .insert(ModelChoice("acme/fast".to_owned()));
    app.update();
    app.update();
    let shown = welcomes(&app, agent);
    assert_eq!(shown.len(), 1, "{shown:?}");
    let Some((again, aside)) = shown.first() else {
        return;
    };
    // Replaced in its place, not added again.
    assert_eq!(again, id);
    let says: Option<Welcome> = serde_json::from_value(aside.value.clone()).ok();
    assert_eq!(
        says.and_then(|says| says.model).as_deref(),
        Some("acme/fast")
    );
}

#[cfg(feature = "tui")]
#[test]
fn the_welcome_is_the_mark_beside_the_version_model_and_directory() {
    let says = Welcome {
        version: "1.2.3".to_owned(),
        model: Some("acme/fast".to_owned()),
        reasoning: "high".to_owned(),
        directory: "~/code/rig".to_owned(),
    };
    let aside = Aside::new(
        WELCOME,
        Author::plugin(PLUGIN),
        serde_json::to_value(&says).unwrap_or_default(),
    );
    let (mut app, agent) = start();
    let Some(agent) = agent else {
        return;
    };
    let id = app
        .world_mut()
        .get_mut::<Conversation>(agent)
        .map(|mut conversation| conversation.add_aside(aside.clone()));
    let Some(id) = id else {
        return;
    };
    let theme = rig_tui::Theme::on((0x1c, 0x1c, 0x1c), true);
    let lines = tui::lines(&rig_tui::AsideRender { id, aside, theme });
    let rows: Vec<String> = lines.iter().map(ToString::to_string).collect();
    let row = |at: usize| rows.get(at).map(String::as_str).unwrap_or_default();
    assert!(
        row(0).starts_with(" █▙ █▙ █▙  ▟█ ▟█ ▟█") && row(0).ends_with("rig v1.2.3"),
        "{rows:?}"
    );
    assert!(row(1).ends_with("acme/fast · reasoning high"), "{rows:?}");
    assert!(row(2).ends_with("~/code/rig"), "{rows:?}");
    // Gray, and a little stronger from the outer panels in.
    let shades: Vec<_> = tui::mark(1, &theme, None)
        .iter()
        .filter_map(|span| span.style.fg)
        .collect();
    let gray = |color: &Color| match color {
        Color::Rgb(r, g, b) if r == g && g == b => Some(*r),
        _ => None,
    };
    let levels: Vec<_> = shades.iter().filter_map(gray).collect();
    assert_eq!(levels.len(), 6, "{shades:?}");
    let ([outer, middle, inner, ..], [.., inner_r, middle_r, outer_r]) =
        (levels.as_slice(), levels.as_slice())
    else {
        return;
    };
    assert!(outer < middle && middle < inner, "{levels:?}");
    assert!(
        (outer, middle, inner) == (outer_r, middle_r, inner_r),
        "{levels:?}"
    );
}

/// The welcome on a terminal view's test screen, with activity, after its
/// first frame, with its agent shown.
#[cfg(feature = "tui")]
fn on_screen(width: u16) -> (App, Option<Entity>) {
    let (mut app, _) = app_on(&MemoryStore::default());
    app.insert_resource(rig_tui::TuiTestScreen::new(width, 30));
    rig_harness_test_support::tui_view(&mut app);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        WelcomePlugin,
        crate::activity::ActivityPlugin,
    ));
    crate::activity::animate(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    app.world_mut()
        .resource_mut::<rig_harness::main_agent::ShownAgent>()
        .0 = agent;
    (app, agent)
}

/// Steps Bevy's real clock by `after`, runs a frame, and returns the
/// color of the first cell of each of the mark's three rows on screen.
#[cfg(feature = "tui")]
fn mark_colors(app: &mut App, after: std::time::Duration) -> Vec<Option<Color>> {
    app.world_mut()
        .resource_mut::<Time<Real>>()
        .update_with_duration(after);
    app.update();
    let buffer = app.world().resource::<rig_tui::TuiTestScreen>().buffer();
    let area = buffer.area;
    let rows: Vec<Vec<_>> = (area.top()..area.bottom())
        .map(|y| {
            let cells = (area.left()..area.right()).filter_map(|x| buffer.cell((x, y)));
            cells.collect()
        })
        .collect();
    ["█▙ █▙ █▙", "██ ██ ██", "█▛ █▛ █▛"]
        .into_iter()
        .map(|panels| {
            let row = rows.iter().find(|row| {
                let text: String = row.iter().map(|cell| cell.symbol()).collect();
                text.contains(panels)
            })?;
            let first = row.iter().find(|cell| cell.symbol() == "█")?;
            Some(first.fg)
        })
        .collect()
}

#[cfg(feature = "tui")]
#[test]
fn the_mark_pulses_while_the_agent_works_and_rests_when_idle() {
    pulses_while_working(100);
}

#[cfg(feature = "tui")]
#[test]
fn the_whole_mark_pulses_when_the_line_beside_it_wraps() {
    // Too narrow for "no model: /model picks one" beside the mark.
    pulses_while_working(40);
}

/// On a screen `width` columns wide, the whole mark is still while idle,
/// every row of it pulses while the agent works, and it rests again after.
#[cfg(feature = "tui")]
fn pulses_while_working(width: u16) {
    let (mut app, agent) = on_screen(width);
    assert!(agent.is_some(), "the session has its agent");
    let Some(agent) = agent else {
        return;
    };
    let still = mark_colors(&mut app, crate::activity::STEP);
    assert!(
        still.len() == 3 && still.iter().all(Option::is_some),
        "the whole mark is on screen: {still:?}"
    );
    assert_eq!(mark_colors(&mut app, crate::activity::STEP), still);

    // A plugin's work for the agent, which runs until it has an outcome.
    let review = EffectRequest {
        kind: rig_harness::rig_core::effect::EffectKind::Custom {
            kind: "review".into(),
            payload: serde_json::Value::Null,
        },
        handler: None,
    };
    let review = app
        .world_mut()
        .spawn((review, ForAgent(agent), Name::new("reviewing")))
        .id();
    let colors: Vec<_> = (0..4)
        .map(|_| mark_colors(&mut app, crate::activity::STEP))
        .collect();
    // Every row of the mark pulses, so the mark never looks split.
    for row in 0..3 {
        let shown: Vec<_> = colors
            .iter()
            .map(|colors| colors.get(row).copied())
            .collect();
        assert!(
            shown
                .iter()
                .all(|color| color.is_some_and(|color| color.is_some()))
                && shown.iter().any(|color| *color != still.get(row).copied()),
            "row {row} of the mark stays on screen and pulses: {shown:?}"
        );
    }

    app.world_mut()
        .entity_mut(review)
        .insert(Outcome::custom("done"));
    mark_colors(&mut app, crate::activity::STEP);
    assert_eq!(mark_colors(&mut app, crate::activity::STEP), still);
}
