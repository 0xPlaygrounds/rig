//! The welcome at the top of the main agent's transcript: rig's mark, its
//! version, the agent's model and reasoning setting, and the project's
//! directory. It is an unsaved aside before the first entry
//! ([`WELCOME`]), so it never reaches the model and scrolls away with the
//! conversation, kept current as the model or the setting changes. While
//! the shown agent works, the mark on screen pulses with the
//! [working animation](crate::activity::animation); without it the mark
//! stays still.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_basics::welcome::WelcomePlugin);
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::agent::Conversation;
use rig_ecs::model::{Connection, ModelChoice, ReasoningSetting};
use rig_harness::main_agent::{MainAgentQuery, main_agent};
use rig_harness::rig_core::conversation::{Aside, AsideId, Author};
use rig_harness::services::ProjectRoot;
use serde::{Deserialize, Serialize};

use crate::models::{ModelLabel, effort_name};
use crate::session_picker::tilde;

/// The welcome aside's kind.
pub const WELCOME: &str = "rig.welcome";

/// The welcome's plugin, its author.
const PLUGIN: &str = "welcome";

/// What the welcome says, its aside's value.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Welcome {
    /// rig's version.
    pub version: String,
    /// The model, as the status line names it; `None` with none.
    pub model: Option<String>,
    /// The reasoning setting, such as `high`.
    pub reasoning: String,
    /// The project's directory, under `~` when it is in the home.
    pub directory: String,
}

/// Keeps the welcome before the main agent's first entry, and draws it in
/// the terminal view.
#[derive(Default)]
pub struct WelcomePlugin;

impl Plugin for WelcomePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, welcome);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// Puts the welcome before the main agent's first entry, again when a
/// restored conversation replaced it, and replaces it when what it says
/// changed.
fn welcome(
    agents: MainAgentQuery,
    mut conversations: Query<(
        &mut Conversation,
        Option<&ModelChoice>,
        Option<&ModelLabel>,
        Option<&ReasoningSetting>,
        Option<&Connection>,
    )>,
    root: Option<Res<ProjectRoot>>,
    mut placed: Local<Option<AsideId>>,
) {
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    let Ok((mut conversation, model, label, reasoning, connection)) = conversations.get_mut(agent)
    else {
        return;
    };
    let says = Welcome {
        version: rig_harness::harness_protocol::long_version(),
        model: label
            .map(|label| label.0.clone())
            .or_else(|| model.map(|model| model.0.clone())),
        reasoning: effort_name(connection, reasoning.copied().unwrap_or_default()),
        directory: root.map(|root| tilde(&root.0)).unwrap_or_default(),
    };
    let value = serde_json::to_value(&says).unwrap_or_default();
    // Read first: a conversation only borrowed mutably is not changed.
    let current = placed.filter(|id| {
        conversation
            .aside(*id)
            .is_some_and(|aside| aside.kind == WELCOME)
    });
    if current
        .and_then(|id| conversation.aside(id))
        .is_some_and(|aside| aside.value == value)
    {
        return;
    }
    let aside = Aside::new(WELCOME, Author::plugin(PLUGIN), value).unsaved();
    match current {
        Some(id) => {
            conversation.replace_aside(id, aside);
        }
        None => *placed = Some(conversation.add_aside(aside)),
    }
}

#[cfg(feature = "tui")]
pub use tui::WelcomeLooks;

#[cfg(feature = "tui")]
mod tui {
    use bevy_app::App;
    use rig_harness::prelude::*;
    use rig_tui::ratatui::buffer::Buffer;
    use rig_tui::ratatui::style::{Color, Stylize};
    use rig_tui::ratatui::text::{Line, Span};
    use rig_tui::{
        AsideRender, MainPane, PanelCanvas, RenderBatch, RenderSystems, Theme, TranscriptRender,
        TuiSystems,
    };

    use super::{WELCOME, Welcome};
    use crate::activity::animation::{Pulse, level};

    /// A panel of rig's mark, its rows from the top: one left of the gap,
    /// its inner corners (right) cut, and its mirror right of it.
    const LEFT: [&str; 3] = ["█▙", "██", "█▛"];
    const RIGHT: [&str; 3] = ["▟█", "██", "▜█"];

    /// How far each pair of panels leans from the background, from the
    /// outer pair in: a little stronger toward the middle.
    const SHADES: [f32; 3] = [0.6, 0.72, 0.85];

    /// Row `row` of rig's mark in `theme`'s grays: six panels around a
    /// gap, three each side, at the working animation's step `step`, or
    /// still in [`SHADES`].
    pub(super) fn mark(row: usize, theme: &Theme, step: Option<usize>) -> Vec<Span<'static>> {
        let panel = |cells: &[&'static str; 3], pair: usize| {
            let cell = cells.get(row).copied().unwrap_or_default();
            let lift = f32::from(level(pair, step)) - f32::from(level(pair, None));
            let shade = SHADES.get(pair).copied().unwrap_or_default() + 0.07 * lift;
            Span::from(cell).fg(theme.shade(shade))
        };
        let left = (0..SHADES.len()).map(|pair| panel(&LEFT, pair));
        let right = (0..SHADES.len()).rev().map(|pair| panel(&RIGHT, pair));
        let mut spans = vec![Span::from(" ")];
        for (at, span) in left.chain(right).enumerate() {
            if at > 0 {
                spans.push(Span::from(if at == SHADES.len() { "  " } else { " " }));
            }
            spans.push(span);
        }
        spans
    }

    /// The system that draws the welcome aside, in
    /// [`RenderSystems::Render`]. A plugin that draws it its own way runs
    /// after it.
    #[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct WelcomeLooks;

    pub(super) fn add(app: &mut App) {
        app.add_systems(
            TranscriptRender,
            draw.in_set(WelcomeLooks).in_set(RenderSystems::Render),
        )
        .add_systems(
            PostUpdate,
            pulse
                .run_if(resource_exists::<Pulse>)
                .after(TuiSystems::Draw)
                .before(TuiSystems::Render),
        );
    }

    /// Paints the mark in the main pane in the working animation's step,
    /// over the drawn transcript, so the transcript is not laid out again
    /// for it. It looks down the pane from the top until it has painted
    /// the mark's last row, so each row of the mark still on screen pulses
    /// wherever a title or wrapped lines put it, and only rows that start
    /// with a row of the mark change. Under `NO_COLOR` the mark stays as
    /// drawn.
    fn pulse(
        pulse: Res<Pulse>,
        mut panes: Query<&mut PanelCanvas, With<MainPane>>,
        theme: Res<Theme>,
    ) {
        let Some(step) = pulse.step() else {
            return;
        };
        if theme.plain {
            return;
        }
        for mut canvas in &mut panes {
            // Painting changes only the buffer, not a change to redraw for.
            let buffer = canvas.bypass_change_detection().buffer_mut();
            let area = buffer.area;
            let last = LEFT.len().saturating_sub(1);
            for y in area.top()..area.bottom() {
                let painted = (0..LEFT.len()).find(|row| paint(buffer, (y, *row), step, &theme));
                if painted == Some(last) {
                    break;
                }
            }
        }
    }

    /// Colors screen row `y` of `buffer` as row `row` of the mark at
    /// animation step `step`, when the row starts with that row of the
    /// mark; whether it did.
    fn paint(buffer: &mut Buffer, (y, row): (u16, usize), step: usize, theme: &Theme) -> bool {
        let mut x = buffer.area.left();
        let mut cells = Vec::new();
        for span in mark(row, theme, Some(step)) {
            for glyph in span.content.chars() {
                let cell = buffer.cell((x, y));
                if !cell.is_some_and(|cell| cell.symbol().chars().eq([glyph])) {
                    return false;
                }
                cells.extend(span.style.fg.map(|color| (x, color)));
                x = x.saturating_add(1);
            }
        }
        for (x, color) in cells {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_fg(color);
            }
        }
        true
    }

    /// Draws the welcome asides.
    fn draw(mut batch: ResMut<RenderBatch>) {
        for (aside, output) in batch.asides_mut(WELCOME) {
            output.lines = lines(aside).into_iter().map(Into::into).collect();
        }
    }

    /// The mark in grays, and beside it rig's version, the model and the
    /// directory, on the terminal's own background.
    pub(super) fn lines(view: &AsideRender) -> Vec<Line<'static>> {
        let says: Welcome = rig_core::json_utils::from_value(&view.aside.value).unwrap_or_default();
        let model = match says.model {
            Some(model) => vec![
                Span::from(model).bold(),
                Span::from(format!(" · reasoning {}", says.reasoning)).dim(),
            ],
            None => vec![Span::from("no model: /model picks one").dim()],
        };
        let beside = [
            vec![Span::from(format!("rig v{}", says.version)).bold()],
            model,
            vec![Span::from(says.directory).dim()],
        ];
        let rows = beside.into_iter().enumerate().map(|(row, beside)| {
            let mut spans = mark(row, &view.theme, None);
            spans.push(Span::from("   "));
            spans.extend(beside);
            // Its own background, not the band asides are drawn on.
            Line::from(spans).bg(Color::Reset)
        });
        rows.chain(std::iter::once(Line::default().bg(Color::Reset)))
            .collect()
    }
}

#[cfg(test)]
mod tests;
