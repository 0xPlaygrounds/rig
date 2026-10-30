//! The welcome at the top of the main agent's transcript: rig's mark, its
//! version, the agent's model and reasoning setting, and the project's
//! directory. It is an unsaved aside before the first entry
//! ([`WELCOME`]), so it never reaches the model and scrolls away with the
//! conversation, kept current as the model or the setting changes.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_basics::welcome::WelcomePlugin);
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::agent::Conversation;
use rig_ecs::model::{ModelChoice, ReasoningSetting};
use rig_harness::main_agent::{MainAgentQuery, main_agent};
use rig_harness::rig_core::conversation::{Aside, AsideId, Author};
use rig_harness::services::ProjectRoot;
use serde::{Deserialize, Serialize};

use crate::models::ModelLabel;

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
    )>,
    root: Option<Res<ProjectRoot>>,
    mut placed: Local<Option<AsideId>>,
) {
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    let Ok((mut conversation, model, label, reasoning)) = conversations.get_mut(agent) else {
        return;
    };
    let says = Welcome {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        model: label
            .map(|label| label.0.clone())
            .or_else(|| model.map(|model| model.0.clone())),
        reasoning: reasoning.copied().unwrap_or_default().to_string(),
        directory: root.map(|root| shown(&root.0)).unwrap_or_default(),
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

/// `path`, under `~` when it is in the home.
fn shown(path: &std::path::Path) -> String {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    match home
        .as_deref()
        .and_then(|home| path.strip_prefix(home).ok())
    {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_owned(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

#[cfg(feature = "tui")]
pub use tui::WelcomeLooks;

#[cfg(feature = "tui")]
mod tui {
    use bevy_app::App;
    use rig_harness::prelude::*;
    use rig_tui::ratatui::style::{Color, Stylize};
    use rig_tui::ratatui::text::{Line, Span};
    use rig_tui::{AsideView, RenderBatch, RenderSystems, Theme, TranscriptRender};

    use super::{WELCOME, Welcome};

    /// A panel of rig's mark, its rows from the top: one left of the gap,
    /// its inner corners (right) cut, and its mirror right of it.
    const LEFT: [&str; 3] = ["█▙", "██", "█▛"];
    const RIGHT: [&str; 3] = ["▟█", "██", "▜█"];

    /// How far each pair of panels leans from the background, from the
    /// outer pair in: a little stronger toward the middle.
    const SHADES: [f32; 3] = [0.6, 0.72, 0.85];

    /// Row `row` of rig's mark in `theme`'s grays: six panels around a
    /// gap, three each side.
    pub(super) fn mark(row: usize, theme: &Theme) -> Vec<Span<'static>> {
        let panel = |cells: &[&'static str; 3], shade: f32| {
            let cell = cells.get(row).copied().unwrap_or_default();
            Span::from(cell).fg(theme.shade(shade))
        };
        let left = SHADES.iter().map(|shade| panel(&LEFT, *shade));
        let right = SHADES.iter().rev().map(|shade| panel(&RIGHT, *shade));
        let mut spans = vec![Span::from(" ")];
        for (at, span) in left.enumerate() {
            if at > 0 {
                spans.push(Span::from(" "));
            }
            spans.push(span);
        }
        spans.push(Span::from("  "));
        for (at, span) in right.enumerate() {
            if at > 0 {
                spans.push(Span::from(" "));
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
        );
    }

    /// Draws the welcome asides.
    fn draw(mut batch: ResMut<RenderBatch>) {
        for (aside, output) in batch.asides_mut(WELCOME) {
            output.lines = lines(&aside.view()).into_iter().map(Into::into).collect();
        }
    }

    /// The mark in grays, and beside it rig's version, the model and the
    /// directory, on the terminal's own background.
    pub(super) fn lines(view: &AsideView<'_>) -> Vec<Line<'static>> {
        let says: Welcome = serde_json::from_value(view.aside.value.clone()).unwrap_or_default();
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
            let mut spans = mark(row, &view.theme);
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
