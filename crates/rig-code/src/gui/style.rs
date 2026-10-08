//! The window's colours and the small node builders every panel uses.

use bevy::picking::Pickable;
use bevy::prelude::*;

pub(super) const BACKGROUND: Color = Color::srgb(0.075, 0.08, 0.095);
pub(super) const PANEL: Color = Color::srgb(0.105, 0.11, 0.13);
pub(super) const RAISED: Color = Color::srgb(0.15, 0.16, 0.19);
pub(super) const SELECTED: Color = Color::srgb(0.2, 0.26, 0.38);
pub(super) const BORDER: Color = Color::srgb(0.22, 0.23, 0.27);
pub(super) const TEXT: Color = Color::srgb(0.88, 0.89, 0.91);
pub(super) const DIM: Color = Color::srgb(0.55, 0.57, 0.62);
pub(super) const BLUE: Color = Color::srgb(0.36, 0.56, 0.95);
pub(super) const GREEN: Color = Color::srgb(0.38, 0.78, 0.45);
pub(super) const RED: Color = Color::srgb(0.92, 0.38, 0.38);
pub(super) const YELLOW: Color = Color::srgb(0.95, 0.8, 0.3);
pub(super) const ORANGE: Color = Color::srgb(0.95, 0.58, 0.25);
pub(super) const MAGENTA: Color = Color::srgb(0.78, 0.45, 0.9);
pub(super) const CYAN: Color = Color::srgb(0.35, 0.8, 0.85);
pub(super) const TURN: Color = Color::srgb(0.25, 0.27, 0.33);

pub(super) const SMALL: f32 = 11.0;
pub(super) const BODY: f32 = 13.0;
pub(super) const TITLE: f32 = 15.0;

/// A run of text in one colour that clicks pass through, so the node
/// under it gets them.
pub(super) fn text(content: impl Into<String>, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font_size: FontSize::Px(size),
            ..default()
        },
        TextColor(color),
        Pickable::IGNORE,
    )
}

/// Like [`text`], on one line: whatever does not fit is clipped by the
/// node around it.
pub(super) fn label(content: impl Into<String>, size: f32, color: Color) -> impl Bundle {
    (text(content, size, color), TextLayout::no_wrap())
}

/// A panel's heading.
pub(super) fn heading(content: impl Into<String>) -> impl Bundle {
    (
        label(content, TITLE, TEXT),
        Node {
            margin: UiRect::bottom(px(6)),
            ..default()
        },
    )
}

/// A button's look: a raised box with its label.
pub(super) fn button_node(selected: bool) -> impl Bundle {
    (
        Node {
            padding: UiRect::axes(px(8), px(3)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
            align_items: AlignItems::Center,
            ..default()
        },
        BorderColor::all(BORDER),
        BackgroundColor(if selected { SELECTED } else { RAISED }),
    )
}

/// A horizontal bar `fraction` full, in `color`, `width` wide.
pub(super) fn meter(fraction: f32, color: Color, width: Val) -> impl Bundle {
    (
        Node {
            width,
            height: px(6),
            border_radius: BorderRadius::all(px(3)),
            ..default()
        },
        BackgroundColor(RAISED),
        Pickable::IGNORE,
        children![(
            Node {
                width: percent(fraction.clamp(0.0, 1.0) * 100.0),
                height: percent(100),
                border_radius: BorderRadius::all(px(3)),
                ..default()
            },
            BackgroundColor(color),
            Pickable::IGNORE,
        )],
    )
}

/// The colour of a context share: green, then yellow past 70%, red past
/// 90%, as the terminal's meter.
pub(super) fn context_color(percent: u64) -> Color {
    match percent {
        0..70 => GREEN,
        70..90 => YELLOW,
        _ => RED,
    }
}

/// `text`'s first `lines` lines, each cut to `width` characters, and how
/// many lines were left out.
pub(super) fn excerpt(text: &str, lines: usize, width: usize) -> (Vec<String>, usize) {
    let total = text.lines().count();
    let shown = text
        .lines()
        .take(lines)
        .map(|line| {
            let mut cut: String = line.chars().take(width).collect();
            if cut.len() < line.len() {
                cut.push('…');
            }
            cut
        })
        .collect();
    (shown, total.saturating_sub(lines))
}

/// A duration as `850ms`, `12.3s`, `4m05s` or `1h02m`.
pub(super) fn duration(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    match seconds {
        0 => format!("{}ms", elapsed.as_millis()),
        1..60 => format!("{:.1}s", elapsed.as_secs_f32()),
        60..3600 => format!("{}m{:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h{:02}m", seconds / 3600, seconds % 3600 / 60),
    }
}
