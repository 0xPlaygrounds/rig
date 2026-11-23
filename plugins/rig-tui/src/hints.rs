//! The keys an empty input lists over its text: Enter sends, `/` for
//! commands and the rest, or while a turn runs, Enter steers and Esc
//! stops. Without [`KeyHintsPlugin`] that row stays empty, and the input
//! still says how to get out of where the keys are: a panel that has
//! them, a question answered in it, an earlier prompt shown.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins((rig_tui::TuiPlugin, rig_tui::hints::KeyHintsPlugin));
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use unicode_width::UnicodeWidthStr;

/// Lists over an empty input the keys it takes. The terminal view works
/// without it; [`TuiPlugins`](crate::TuiPlugins) adds it with the view.
/// Without the [`TuiViewPlugin`](crate::TuiViewPlugin), added before it
/// or after, it adds nothing.
#[derive(Default)]
pub struct KeyHintsPlugin;

impl Plugin for KeyHintsPlugin {
    fn build(&self, _app: &mut App) {}

    // In `finish`, after every plugin's `build`, so the order they were
    // added in does not matter.
    fn finish(&self, app: &mut App) {
        if app.is_plugin_added::<crate::TuiViewPlugin>() {
            app.init_resource::<KeyHints>();
        }
    }
}

/// Says the [`KeyHintsPlugin`] is on.
#[derive(Resource, Default)]
pub(crate) struct KeyHints;

/// A key hint, with how much it matters: a row too narrow for every hint
/// drops those that matter least first.
pub(crate) type Key = (&'static str, u8);

/// The keys of an empty input that has them, for a message: with Enter
/// steering and Tab and Esc for the turn while one runs.
pub(crate) fn keys(turn_running: bool) -> &'static [Key] {
    match turn_running {
        true => &[
            ("Enter steers", 9),
            ("Tab sends after the turn", 7),
            ("Esc stops", 10),
            ("Shift+Enter or Ctrl+J new line", 5),
            ("Ctrl+V image", 1),
            ("/ commands", 4),
            ("@ files", 2),
            ("Ctrl+D quits", 3),
        ],
        false => &[
            ("Enter sends", 9),
            ("Shift+Enter or Ctrl+J new line", 7),
            ("Ctrl+V image", 4),
            ("/ commands", 8),
            ("@ files", 5),
            ("Ctrl+D quits", 6),
        ],
    }
}

/// `keys` in `width` columns, joined by ` · `: all of them, or else the
/// most that fit after dropping the least important, in their order and
/// ending in `…`.
pub(crate) fn fit_keys(keys: &[Key], width: usize) -> String {
    let mut kept = keys.to_vec();
    loop {
        let mut items: Vec<&str> = kept.iter().map(|(item, _)| *item).collect();
        if kept.len() < keys.len() {
            items.push("…");
        }
        if items.is_empty() {
            return String::new();
        }
        let row = format!(" {} ", items.join(" · "));
        if row.width() <= width {
            return row;
        }
        // The least important; of equals, the last.
        let least = kept
            .iter()
            .enumerate()
            .min_by_key(|(at, (_, keep))| (*keep, std::cmp::Reverse(*at)))
            .map(|(at, _)| at);
        match least {
            Some(at) => {
                kept.remove(at);
            }
            None => return String::new(),
        }
    }
}

#[cfg(test)]
mod tests;
