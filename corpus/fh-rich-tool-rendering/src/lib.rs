//! FH-19: rich tool call rendering in the terminal view (`rig-tui`).
//!
//! The built-in tools' calls get looks of their own, in place of the
//! default plugins' ([`TOOLS`]): `edit` and `write` as a coloured,
//! highlighted diff that collapses after [`COLLAPSED_LINES`] lines, `shell`
//! as `$ command` with the last [`SHELL_TAIL`] lines of its output and
//! `exit N · 4.2s`, `read` as one line with the lines read, `search` as a
//! count with the first matches. Ctrl+O expands or collapses every call
//! ([`ToolViews`]). While a reply streams, a panel under the transcript
//! shows what is known of the tool calls it is writing
//! ([`StreamingCalls`]). Any other tool keeps its own renderer or the
//! default look; the renderers never change what is saved or sent.

mod highlight;
mod looks;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::SystemTime;

use rig_harness::prelude::*;
use rig_harness::rig_core::message::CallId;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    AppKeyBindingsExt, AppToolRenderersExt, KeyBinding, KeyPress, KeyPressed, PanelCanvas,
    Placement, RequestRedraw, ToolRenderer, TuiPanel, TuiPlugin, TuiSystems,
};

pub use looks::{COLLAPSED_LINES, SHELL_TAIL};

/// The tools this plugin draws: the default plugins' looks of them are
/// turned off.
pub const TOOLS: [&str; 5] = ["edit", "write", "shell", "read", "search"];

/// Draws the built-in tools' calls the way they read best.
#[derive(Default)]
pub struct RichToolRenderingPlugin;

impl Plugin for RichToolRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ToolViews>()
            .init_resource::<StreamingCalls>()
            .add_observer(record_duration)
            .add_systems(PostUpdate, collect_arguments);
    }

    fn finish(&self, app: &mut App) {
        // Every plugin's `build` ran: the default looks exist and are
        // turned off here, so their names are free.
        let world = app.world_mut();
        let mut renderers = world.query::<(Entity, &ToolRenderer)>();
        let taken: Vec<Entity> = renderers
            .iter(world)
            .filter(|(_, renderer)| TOOLS.contains(&renderer.tool.as_str()))
            .map(|(entity, _)| entity)
            .collect();
        for renderer in taken {
            world.entity_mut(renderer).insert(Disabled);
        }
        let views = world.resource::<ToolViews>().clone();
        for tool in TOOLS {
            let views = views.clone();
            app.add_tool_renderer(tool, move |call| looks::render(call, &views));
        }
        let world = app.world_mut();
        let mut renderers = world.query::<(Entity, &ToolRenderer)>();
        let ours: Vec<Entity> = renderers
            .iter(world)
            .filter(|(_, renderer)| TOOLS.contains(&renderer.tool.as_str()))
            .map(|(entity, _)| entity)
            .collect();
        for renderer in ours {
            world.entity_mut(renderer).insert(RichRenderer);
        }
        // What only the terminal view uses: its messages are there then.
        if app.is_plugin_added::<TuiPlugin>() {
            let toggle = KeyBinding::new(KeyPress::ctrl('o'), "Expand or collapse tool calls");
            app.add_key_binding(toggle, toggle_all);
            app.world_mut().spawn((
                PreviewPanel,
                TuiPanel::new(Placement::Bottom(Constraint::Length(0))),
            ));
            app.add_systems(
                PostUpdate,
                (
                    size_preview
                        .after(collect_arguments)
                        .before(TuiSystems::Layout),
                    draw_preview.in_set(TuiSystems::Draw),
                ),
            );
        }
    }
}

/// What the renderers read besides the call: whether every call is
/// expanded (Ctrl+O), and how long each call that ended in this run took.
/// Shared with the renderers, which are plain functions of a call.
#[derive(Resource, Clone, Default)]
pub struct ToolViews(Arc<RwLock<Views>>);

/// [`ToolViews`]' state.
#[derive(Default)]
struct Views {
    expanded: bool,
    durations: HashMap<CallId, Duration>,
}

impl ToolViews {
    /// Whether every call is drawn whole.
    pub fn expanded(&self) -> bool {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .expanded
    }

    /// Expands every call when they were collapsed, and the other way
    /// round; returns whether they are expanded now.
    pub fn toggle(&self) -> bool {
        let mut views = self.0.write().unwrap_or_else(PoisonError::into_inner);
        views.expanded = !views.expanded;
        views.expanded
    }

    /// How long the call `call` ran, if it ended in this run.
    pub fn duration(&self, call: &CallId) -> Option<Duration> {
        let views = self.0.read().unwrap_or_else(PoisonError::into_inner);
        views.durations.get(call).copied()
    }

    fn record(&self, call: CallId, took: Duration) {
        let mut views = self.0.write().unwrap_or_else(PoisonError::into_inner);
        views.durations.insert(call, took);
    }
}

/// On a renderer this plugin added.
#[derive(Component)]
pub struct RichRenderer;

/// The panel under the transcript with the tool calls being streamed.
#[derive(Component)]
struct PreviewPanel;

/// Keeps how long each tool call ran, from its start to its answer.
fn record_duration(
    answered: On<Add<Outcome>>,
    calls: Query<(&ToolRequest, &Started)>,
    views: Res<ToolViews>,
) {
    if let Ok((request, started)) = calls.get(answered.entity) {
        let took = SystemTime::now()
            .duration_since(started.at)
            .unwrap_or_default();
        views.record(request.call.id.clone(), took);
    }
}

/// Ctrl+O: every call expanded or collapsed. The transcript keeps each
/// message's rows until its look changes, so the renderers are marked
/// changed to draw every call again.
fn toggle_all(
    _: On<KeyPressed>,
    views: Res<ToolViews>,
    mut renderers: Query<&mut ToolRenderer, With<RichRenderer>>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    views.toggle();
    for mut renderer in &mut renderers {
        renderer.set_changed();
    }
    redraw.write(RequestRedraw);
}

/// The argument JSON of the tool calls the models are streaming, by model
/// call and part, until the reply is in.
#[derive(Resource, Default)]
pub struct StreamingCalls {
    calls: BTreeMap<(Entity, usize), String>,
}

impl StreamingCalls {
    /// Adds `json`, a fragment of the arguments of part `part` of the
    /// model call `call`.
    pub fn push(&mut self, call: Entity, part: usize, json: &str) {
        self.calls.entry((call, part)).or_default().push_str(json);
    }

    /// What is known of each call being streamed so far: the file it
    /// names or its command, and the end of the text it writes.
    pub fn preview(&self) -> Vec<Line<'static>> {
        self.calls
            .values()
            .flat_map(|json| looks::streaming(json))
            .collect()
    }

    /// Whether no call is being streamed.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

/// Collects the tool calls' argument fragments, and forgets a reply's once
/// it is in: its calls are in the transcript then.
fn collect_arguments(
    mut deltas: MessageReader<StreamDelta>,
    mut streaming: ResMut<StreamingCalls>,
    answered: Query<Has<Outcome>>,
) {
    for delta in deltas.read() {
        if let StreamEvent::Arguments { part, json } = &delta.event {
            streaming.push(delta.effect, part.index(), json);
        }
    }
    if streaming.calls.is_empty() {
        return;
    }
    streaming
        .bypass_change_detection()
        .calls
        .retain(|(call, _), _| answered.get(*call).is_ok_and(|done| !done));
}

/// Most rows the streaming preview takes.
const PREVIEW_ROWS: usize = 8;

/// Sizes the preview panel to what it shows; a change of the panel draws a
/// frame.
fn size_preview(
    streaming: Res<StreamingCalls>,
    mut panels: Query<&mut TuiPanel, With<PreviewPanel>>,
) {
    if !streaming.is_changed() {
        return;
    }
    let rows = streaming.preview().len().min(PREVIEW_ROWS);
    let placement = Placement::Bottom(Constraint::Length(u16::try_from(rows).unwrap_or(0)));
    for mut panel in &mut panels {
        if panel.placement != placement {
            panel.placement = placement;
        }
    }
}

fn draw_preview(
    streaming: Res<StreamingCalls>,
    mut panels: Query<&mut PanelCanvas, With<PreviewPanel>>,
) {
    let mut lines = streaming.preview();
    let skip = lines.len().saturating_sub(PREVIEW_ROWS);
    lines.drain(..skip);
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.clone()).style(Style::new().dim()));
    }
}

#[cfg(test)]
mod tests;
