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

use bevy_log::warn;
use rig_harness::main_agent::ShownAgent;
use rig_harness::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{
    AppKeyBindingsExt, CallItem, CallState, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope,
    Keys, LookLine, MainPane, OUTPUT_INDENT, RenderBatch, RenderRevision, RenderSystems,
    ToolRender, TranscriptItem, TranscriptPane, TranscriptRender, TranscriptSource, TranscriptView,
};
use serde::{Deserialize, Deserializer};

#[cfg(test)]
mod tests;

/// The fold key's label, which key listings show.
const LABEL: &str = "Show the selected tool call whole, or every call";

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

/// A call of a transcript: the pane, the agent it shows (none for a
/// conversation no agent holds) and the call's place there. Message ids
/// rise from 0 in each conversation, so the place alone names calls of
/// several agents. An entity's generation keeps one that is gone from
/// naming a later one.
type Shown = (Entity, Option<Entity>, CallItem);

/// Which calls are drawn whole: every call when `all` is set, but those in
/// `flipped`, and only those in `flipped` when it is not.
#[derive(Resource, Default, Debug)]
struct Unfolded {
    /// Whether the key showed every call.
    all: bool,
    /// The calls the key flipped from what `all` says.
    flipped: HashSet<Shown>,
}

impl Unfolded {
    /// Whether the call `call` is drawn whole.
    fn shows(&self, call: Option<Shown>) -> bool {
        self.all != call.is_some_and(|call| self.flipped.contains(&call))
    }

    /// Flips `call`, or every call with none.
    fn flip(&mut self, call: Option<Shown>) {
        match call {
            Some(call) if !self.flipped.remove(&call) => {
                self.flipped.insert(call);
            }
            Some(_) => {}
            None => {
                self.all = !self.all;
                self.flipped.clear();
            }
        }
    }
}

/// On the fold key's binding. Folding needs it there and not disabled, so
/// folded lines always have a key that shows them.
#[derive(Component)]
struct FoldKey;

impl Plugin for FoldPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<FoldSettings>("fold")
            .init_resource::<Unfolded>()
            .add_systems(TranscriptRender, fold.in_set(RenderSystems::Edit));
        let key = app.world().resource::<FoldSettings>().key.clone();
        // An input key a global binding would take unseen, such as Enter.
        let world = app.world_mut();
        let mut bindings = world.query::<&KeyBinding>();
        let input = bindings.iter(world).find(|bound| {
            let (bound_keys, keys) = (bound.keys.presses(), key.presses());
            bound.scope == KeyScope::Input
                && (bound_keys.starts_with(keys) || keys.starts_with(bound_keys))
        });
        if let Some(bound) = input {
            warn!(
                "[fold] key `{key}` is the input's `{}` ({}): nothing is folded",
                bound.keys, bound.label
            );
            return;
        }
        // Global, so no other binding of the key shadows it unseen: one
        // bound first refuses it, with a warning, instead.
        let binding = KeyBinding::new(key, LABEL).in_scope(KeyScope::Global);
        if let Some(bound) = app.add_key_binding(binding, flip) {
            app.world_mut().entity_mut(bound).insert(FoldKey);
        }
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

/// Flips the call selected in the transcript pane that has the keys, else
/// in the main pane, or every call with none selected, and draws the
/// transcript again.
fn flip(
    pressed: On<KeyPressed>,
    (focus, shown): (Option<Res<KeyFocus>>, Option<Res<ShownAgent>>),
    panes: Query<(Entity, &TranscriptPane, &TranscriptView, Has<MainPane>)>,
    mut unfolded: ResMut<Unfolded>,
    mut revision: ResMut<RenderRevision>,
) {
    if pressed.released {
        return;
    }
    let focused = focus.and_then(|focus| panes.get(focus.0?).ok());
    let pane = focused.or_else(|| panes.iter().find(|(.., main)| *main));
    let selected = pane.and_then(|(pane, transcript, view, _)| {
        let agent = match &transcript.source {
            TranscriptSource::Shown => shown.as_ref().and_then(|shown| shown.0),
            TranscriptSource::Agent(agent) => Some(*agent),
            TranscriptSource::Value(_) => None,
        };
        Some((pane, agent, selected_call(view)?))
    });
    unfolded.flip(selected);
    revision.0 = revision.0.wrapping_add(1);
}

/// Folds every finished call of a pane's batch that is not drawn whole,
/// while the fold key is bound.
fn fold(
    mut batch: ResMut<RenderBatch>,
    settings: Res<FoldSettings>,
    unfolded: Res<Unfolded>,
    key: Query<(), With<FoldKey>>,
) {
    // An export has no pane, and keeps every line.
    let (Some(pane), agent) = (batch.pane, batch.agent) else {
        return;
    };
    if key.is_empty() {
        return;
    }
    for (call, output) in batch.calls_mut() {
        let shown = call.item.map(|item| (pane, agent, item));
        if !folds(call) || unfolded.shows(shown) {
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
    let kept = settings.head.saturating_add(settings.tail);
    let hidden = body.len().checked_sub(kept)?;
    if hidden < 2 {
        return None;
    }
    let (head, rest) = body.split_at_checked(settings.head)?;
    let tail = rest.get(hidden..)?;
    let marker = format!("{OUTPUT_INDENT}… {hidden} lines · {} shows", settings.key);
    let marker = Line::styled(marker, Style::new().dim().italic());
    let mut lines = Vec::with_capacity(kept.saturating_add(2));
    lines.push(header.clone());
    lines.extend_from_slice(head);
    lines.push(marker.into());
    lines.extend_from_slice(tail);
    Some(lines)
}
