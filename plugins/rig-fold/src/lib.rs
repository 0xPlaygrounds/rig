//! Folded tool output in the terminal view. Each finished tool call, an MCP
//! server's included, shows its header, the first and last lines of the
//! rest, and a line that counts the lines between and names the key that
//! shows them. A failed or stopped call is drawn whole, as is every call of
//! an export. Only the terminal view's rows change: the model, the session
//! log, `--json` and ACP get every line as before.
//!
//! The key (Ctrl+O unless `[fold] key` names another) shows the selected
//! call (Alt+Up selects) whole, or folds it again; with no call selected it
//! shows every call whole, or folds them all again.
//!
//! ```no_run
//! use rig_fold::FoldPlugin;
//! use rig_harness::prelude::*;
//!
//! let mut app = App::new();
//! app.insert_resource(Settings::from_toml("[fold]\nhead = 2\ntail = 6\nkey = \"alt+o\""))
//!     .add_plugins((rig_tui::TuiPlugin, FoldPlugin));
//! ```

use std::collections::HashSet;

use rig_harness::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{
    AppKeyBindingsExt, CallItem, CallState, KeyBinding, KeyPress, KeyPressed, KeyScope, Keys,
    LookLine, MainPane, OUTPUT_INDENT, RenderBatch, RenderRevision, RenderSystems, ToolRender,
    TranscriptItem, TranscriptRender, TranscriptView,
};
use serde::{Deserialize, Deserializer};

#[cfg(test)]
mod tests;

/// Folds finished tool calls in the terminal view and binds the key that
/// shows them whole ([`FoldSettings`]).
#[derive(Default)]
pub struct FoldPlugin;

/// `[fold]` in `settings.toml`.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct FoldSettings {
    /// The lines kept after the header.
    pub head: usize,
    /// The last lines kept.
    pub tail: usize,
    /// The key that shows the selected call, or every call, whole.
    #[serde(deserialize_with = "keys")]
    pub key: Keys,
}

impl Default for FoldSettings {
    fn default() -> Self {
        Self {
            head: 4,
            tail: 8,
            key: KeyPress::ctrl('o').into(),
        }
    }
}

/// Keys written as `rig_tui` parses them, such as `ctrl+o`.
fn keys<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Keys, D::Error> {
    let text = String::deserialize(deserializer)?;
    text.parse().map_err(serde::de::Error::custom)
}

/// Which calls are drawn whole: every call when `all` is set, but those in
/// `flipped`, and only those in `flipped` when it is not.
#[derive(Resource, Default, Debug)]
struct Unfolded {
    /// Whether the key showed every call.
    all: bool,
    /// The calls the key flipped from what `all` says.
    flipped: HashSet<CallItem>,
}

impl Unfolded {
    /// Whether the call `item` is drawn whole.
    fn shows(&self, item: Option<CallItem>) -> bool {
        self.all != item.is_some_and(|item| self.flipped.contains(&item))
    }

    /// Flips `item`, or every call with none.
    fn flip(&mut self, item: Option<CallItem>) {
        match item {
            Some(item) if !self.flipped.remove(&item) => {
                self.flipped.insert(item);
            }
            Some(_) => {}
            None => {
                self.all = !self.all;
                self.flipped.clear();
            }
        }
    }
}

impl Plugin for FoldPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<FoldSettings>("fold")
            .init_resource::<Unfolded>()
            .add_systems(TranscriptRender, fold.in_set(RenderSystems::Edit));
        let key = app.world().resource::<FoldSettings>().key.clone();
        let binding = KeyBinding::new(key, "Show the selected tool call whole, or every call");
        app.add_key_binding(binding.in_scope(KeyScope::Input), flip);
    }
}

/// The call the main transcript has selected, if a call or one of its
/// decorations is.
fn selected_call(view: &TranscriptView) -> Option<CallItem> {
    match view.selected.as_ref()? {
        TranscriptItem::Call(item) | TranscriptItem::Decoration(item, _) => Some(*item),
        _ => None,
    }
}

/// Flips the selected call, or every call, and draws the transcript again.
fn flip(
    pressed: On<KeyPressed>,
    panes: Query<&TranscriptView, With<MainPane>>,
    mut unfolded: ResMut<Unfolded>,
    mut revision: ResMut<RenderRevision>,
) {
    if pressed.released {
        return;
    }
    let selected = panes.single().ok().and_then(selected_call);
    unfolded.flip(selected);
    revision.0 = revision.0.wrapping_add(1);
}

/// Folds every finished call of a pane's batch that is not drawn whole.
fn fold(mut batch: ResMut<RenderBatch>, settings: Res<FoldSettings>, unfolded: Res<Unfolded>) {
    // An export has no pane, and keeps every line.
    if batch.pane.is_none() {
        return;
    }
    for (call, output) in batch.calls_mut() {
        if !folds(call) || unfolded.shows(call.item) {
            continue;
        }
        if let Some(lines) = folded(&output.lines, &settings) {
            output.lines = lines;
        }
    }
}

/// Whether `call` is answered and neither failed nor was stopped.
fn folds(call: &ToolRender) -> bool {
    let view = call.view();
    call.state == CallState::Done && call.result.is_some() && !view.failed()
}

/// `lines`, a call's look, as its first line, the `head` lines after it, a
/// line counting those left out, and the last `tail`; `None` when that
/// leaves out fewer than two lines.
fn folded(lines: &[LookLine], settings: &FoldSettings) -> Option<Vec<LookLine>> {
    let (header, body) = lines.split_first()?;
    let hidden = body.len().checked_sub(settings.head + settings.tail)?;
    if hidden < 2 {
        return None;
    }
    let (head, rest) = body.split_at_checked(settings.head)?;
    let tail = rest.get(hidden..)?;
    let marker = format!("{OUTPUT_INDENT}… {hidden} lines · {} shows", settings.key);
    let marker = Line::styled(marker, Style::new().dim().italic());
    let mut kept = Vec::with_capacity(settings.head + settings.tail + 2);
    kept.push(header.clone());
    kept.extend_from_slice(head);
    kept.push(marker.into());
    kept.extend_from_slice(tail);
    Some(kept)
}
