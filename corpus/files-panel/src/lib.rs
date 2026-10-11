//! Plugin corpus: 52. Touched-files panel (spec `52-files-panel.md`).
//!
//! Every file an agent of the session read or changed, with the lines its
//! changes added and removed, kept as saved resources so `/resume` brings
//! the same list back. Changes are what tools publish ([`FilesChanged`]),
//! whatever the tool; a file's text before its first change is kept when a
//! call that may change it is about to run, so its session diff is that
//! text against the file now. In the terminal view, `ctrl+f` opens the
//! panel on the right with the keys, up and down select, Enter shows the
//! selected file's session diff over the screen and Esc closes it;
//! `/files` lists the files in any front.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ChangeKind, FilesChanged};
use rig_tools::ReadArgs;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::buffer::Buffer;
use rig_tui::ratatui::layout::{Constraint, Rect};
use rig_tui::ratatui::style::{Modifier, Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::{
    Block, List, ListItem, ListState, Paragraph, StatefulWidget, Widget, Wrap,
};
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, Keys, PanelCanvas,
    Placement, RequestRedraw, TuiPanel, TuiPlugin, TuiSystems,
};

/// The panel's width.
pub const WIDTH: u16 = 32;

/// The largest file whose text is kept for its session diff.
const MAX_KEPT: u64 = 256 * 1024;

/// A file an agent touched.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub struct TouchedFile {
    /// The file's absolute path.
    pub path: String,
    /// The subagent that touched it last, by name; empty for a top agent.
    pub by: String,
    /// Lines its changes added.
    pub added: usize,
    /// Lines its changes removed.
    pub removed: usize,
    /// When it last changed, on [`TouchedFiles::clock`].
    pub changed_at: Option<u64>,
    /// When it was last read, on [`TouchedFiles::clock`].
    pub read_at: Option<u64>,
    /// Whether it was gone when a turn last ended.
    pub gone: bool,
}

/// Every file the session's agents read or changed, saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct TouchedFiles {
    /// The files, by absolute path.
    pub files: HashMap<String, TouchedFile>,
    /// Counts touches, to order the lists newest first.
    pub clock: u64,
}

impl TouchedFiles {
    /// The changed files, newest first.
    pub fn changed(&self) -> Vec<&TouchedFile> {
        self.newest(|file| file.changed_at)
    }

    /// The files read, newest first.
    pub fn read(&self) -> Vec<&TouchedFile> {
        self.newest(|file| file.read_at)
    }

    /// The panel's rows, which the selection counts: the changed files,
    /// then the files read.
    pub fn rows(&self) -> Vec<&TouchedFile> {
        let mut rows = self.changed();
        rows.extend(self.read());
        rows
    }

    fn newest(&self, at: impl Fn(&TouchedFile) -> Option<u64>) -> Vec<&TouchedFile> {
        let mut files: Vec<_> = self.files.values().filter(|f| at(f).is_some()).collect();
        files.sort_by_key(|file| std::cmp::Reverse(at(file)));
        files
    }

    /// The file at `path`, its entry made if new, and the time of this
    /// touch.
    fn touch(&mut self, path: String) -> (&mut TouchedFile, u64) {
        self.clock += 1;
        let file = self
            .files
            .entry(path)
            .or_insert_with_key(|path| TouchedFile {
                path: path.clone(),
                ..TouchedFile::default()
            });
        (file, self.clock)
    }

    /// The list as text, for `/files`.
    pub fn text(&self, root: &Path) -> String {
        let mut lines = vec!["changed".to_owned()];
        lines.extend(
            self.changed()
                .iter()
                .map(|f| format!("  {}", label(f, root))),
        );
        lines.push("read".to_owned());
        lines.extend(self.read().iter().map(|f| format!("  {}", name(f, root))));
        lines.join("\n")
    }
}

/// What a changed file held before its first change this session.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
pub enum Before {
    /// It did not exist.
    Missing,
    /// This text.
    Text(String),
    /// More than is kept.
    TooLarge,
}

/// Each file's text before its first change this session, by absolute
/// path, for its session diff; saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct Originals(pub HashMap<String, Before>);

/// `path` as an absolute path under `root`, without `.` parts.
fn absolute(root: &Path, path: &str) -> String {
    let path: PathBuf = root.join(path).components().collect();
    path.display().to_string()
}

/// A file's path, relative to `root` when under it, with the subagent that
/// touched it.
fn name(file: &TouchedFile, root: &Path) -> String {
    let path = shown(&file.path, root);
    if file.by.is_empty() {
        path
    } else {
        format!("{path} ({})", file.by)
    }
}

fn shown(path: &str, root: &Path) -> String {
    Path::new(path)
        .strip_prefix(root)
        .map_or_else(|_| path.to_owned(), |path| path.display().to_string())
}

/// A changed file's name with its counts, such as `a.txt +1 −1`.
fn label(file: &TouchedFile, root: &Path) -> String {
    format!("{} +{} −{}", name(file, root), file.added, file.removed)
}

/// The session diff of the file at `path`: what it held before its first
/// change against what it holds now. Empty when nothing was kept.
pub fn session_diff(path: &str, root: &Path, originals: &Originals) -> String {
    let before = match originals.0.get(path) {
        Some(Before::Text(text)) => text.as_str(),
        Some(Before::Missing) => "",
        Some(Before::TooLarge) | None => return String::new(),
    };
    let now = std::fs::read_to_string(path).unwrap_or_default();
    let shown = shown(path, root);
    similar::TextDiff::from_lines(before, &now)
        .unified_diff()
        .header(&shown, &shown)
        .to_string()
}

/// Draws the files into `area` of `buffer`, the row `selected` marked.
pub fn draw_files(
    files: &TouchedFiles,
    root: &Path,
    selected: Option<usize>,
    area: Rect,
    buffer: &mut Buffer,
) {
    let inner = usize::from(area.width.saturating_sub(2));
    let entry = |text: String, gone: bool| {
        let style = if gone {
            Style::new().add_modifier(Modifier::CROSSED_OUT).dim()
        } else {
            Style::new()
        };
        ListItem::new(Line::styled(fit(&text, inner), style))
    };
    let changed = files.changed();
    let mut items = vec![ListItem::new(Line::from("changed").bold())];
    items.extend(changed.iter().map(|f| entry(label(f, root), f.gone)));
    items.push(ListItem::new(Line::from("read").bold()));
    items.extend(files.read().iter().map(|f| entry(name(f, root), f.gone)));
    // The rows count past the headers: one before the changed, two before
    // the read.
    let row = selected.map(|row| row + if row < changed.len() { 1 } else { 2 });
    let list = List::new(items)
        .block(Block::bordered().title("files"))
        .highlight_style(Style::new().reversed());
    StatefulWidget::render(
        list,
        area,
        buffer,
        &mut ListState::default().with_selected(row),
    );
}

/// `text` cut from the left to `width` characters.
fn fit(text: &str, width: usize) -> String {
    let count = text.chars().count();
    if count <= width {
        return text.to_owned();
    }
    let tail: String = text.chars().skip(count + 1 - width).collect();
    format!("…{tail}")
}

/// The touched-files panel.
#[derive(Default)]
pub struct FilesPanelPlugin;

/// The panel's entity, there while it is closed too, with its selected row.
/// Open, it is a [`TuiPanel`].
#[derive(Component, Default)]
struct FilesPanel(usize);

/// The diff box's entity: a file's session diff and how far it is
/// scrolled. Open, it is a [`TuiPanel`].
#[derive(Component, Default)]
struct DiffBox {
    path: String,
    diff: String,
    scroll: u16,
}

impl Plugin for FilesPanelPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        let panel = app.world_mut().spawn(FilesPanel::default()).id();
        let diff = app.world_mut().spawn(DiffBox::default()).id();
        let key = |code| Keys::from(KeyPress::key(code));
        let in_panel =
            |code, label: &str| KeyBinding::new(key(code), label).in_scope(KeyScope::Panel(panel));
        let in_diff =
            |code, label: &str| KeyBinding::new(key(code), label).in_scope(KeyScope::Panel(diff));
        app.init_resource::<TouchedFiles>()
            .init_resource::<Originals>()
            .init_resource::<ProjectRoot>()
            .register_type::<TouchedFiles>()
            .register_type::<Originals>()
            .add_command(
                "files",
                "The files the agents read and changed",
                files_command,
            )
            .add_observer(track)
            .add_observer(check_gone)
            .add_systems(AgentPass, keep_original.in_set(Stage::Gate))
            .add_systems(
                PostUpdate,
                refresh
                    .run_if(resource_changed::<TouchedFiles>)
                    .in_set(TuiSystems::Prepare),
            )
            .add_systems(PostUpdate, (draw_panel, draw_diff).in_set(TuiSystems::Draw));
        app.add_key_binding(in_panel(KeyCode::Up, "Select the file above"), select(-1));
        app.add_key_binding(in_panel(KeyCode::Down, "Select the file below"), select(1));
        app.add_key_binding(
            in_panel(KeyCode::Enter, "Show the file's session diff"),
            open_diff,
        );
        app.add_key_binding(in_diff(KeyCode::Up, "Scroll the diff up"), scroll(-1));
        app.add_key_binding(in_diff(KeyCode::Down, "Scroll the diff down"), scroll(1));
        app.add_key_binding(in_diff(KeyCode::Esc, "Close the diff"), close_diff);
    }

    /// Once every plugin was built, the view's own `ctrl+f` (a character
    /// forward, which the right arrow does too) is there to free.
    fn finish(&self, app: &mut App) {
        let ctrl_f = Keys::from(KeyPress::ctrl('f'));
        let world = app.world_mut();
        let mut bindings = world.query::<(Entity, &KeyBinding)>();
        let taken: Vec<Entity> = bindings
            .iter(world)
            .filter(|(_, binding)| binding.keys == ctrl_f)
            .map(|(entity, _)| entity)
            .collect();
        for binding in taken {
            world.entity_mut(binding).insert(Disabled);
        }
        app.add_key_binding(KeyBinding::new(ctrl_f, "The files panel"), toggle);
    }
}

/// Keeps the text of the file a call names in `path` before the session
/// first changes it, when a call that may change files is about to run:
/// the calls before it in its reply are done by then.
fn keep_original(
    calls: Query<(&ToolRequest, &Footprint), Undecided>,
    root: Res<ProjectRoot>,
    mut originals: ResMut<Originals>,
) {
    for (request, footprint) in &calls {
        let path = request.call.function.arguments.get("path");
        let Some(path) = path.and_then(|path| path.as_str()) else {
            continue;
        };
        let path = absolute(&root.0, path);
        if *footprint != Footprint::Exclusive || originals.0.contains_key(&path) {
            continue;
        }
        let before = match std::fs::metadata(&path) {
            Err(_) => Before::Missing,
            Ok(meta) if meta.len() > MAX_KEPT => Before::TooLarge,
            Ok(_) => std::fs::read_to_string(&path).map_or(Before::TooLarge, Before::Text),
        };
        originals.0.insert(path, before);
    }
}

/// Records each tool call as it settles: the files it changed, and the
/// file a successful `read` read.
fn track(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome, Option<&Published>, &ForAgent)>,
    subagents: Query<&Name, With<SpawnedBy>>,
    root: Res<ProjectRoot>,
    mut files: ResMut<TouchedFiles>,
) {
    let Ok((request, outcome, published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let by = subagents
        .get(agent)
        .map(|name| name.as_str().to_owned())
        .unwrap_or_default();
    let changes = published.and_then(|p| p.0.get::<FilesChanged>().ok().flatten());
    for change in changes.unwrap_or_default().0 {
        let (file, at) = files.touch(absolute(&root.0, &change.path));
        file.by.clone_from(&by);
        file.added += change.added;
        file.removed += change.removed;
        file.changed_at = Some(at);
        file.gone = change.kind == ChangeKind::Deleted;
    }
    let function = &request.call.function;
    if function.name.as_str() != "read" || !outcome.tool_result().is_success() {
        return;
    }
    if let Ok(args) = serde_json::from_value::<ReadArgs>(function.arguments.clone().into()) {
        let (file, at) = files.touch(absolute(&root.0, &args.path));
        file.by = by;
        file.read_at = Some(at);
        file.gone = false;
    }
}

/// When a turn ends, marks the files that are gone, such as one `shell`
/// deleted, and those back again.
fn check_gone(
    settled: On<Add<Settled>>,
    turns: Query<(), With<TurnOutcome>>,
    mut files: ResMut<TouchedFiles>,
) {
    if !turns.contains(settled.entity) {
        return;
    }
    let gone = |file: &TouchedFile| !Path::new(&file.path).exists();
    if files.files.values().any(|file| file.gone != gone(file)) {
        for file in files.files.values_mut() {
            file.gone = gone(file);
        }
    }
}

/// `/files`: the list.
fn files_command(
    In(args): In<CommandArgs>,
    files: Res<TouchedFiles>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    notices.write(Notice::info(args.agent, files.text(&root.0)));
}

/// `ctrl+f`: opens the panel with the keys; gives it the keys back when
/// it is open without them; closes it when it has them.
fn toggle(
    _: On<KeyPressed>,
    panels: Query<(Entity, Has<TuiPanel>), With<FilesPanel>>,
    boxes: Query<Entity, With<DiffBox>>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    let Ok((panel, open)) = panels.single() else {
        return;
    };
    if !open {
        let right = Placement::Right(Constraint::Length(WIDTH));
        commands.entity(panel).insert(TuiPanel::new(right));
        focus.0 = Some(panel);
    } else if focus.0 == Some(panel) {
        commands.entity(panel).remove::<TuiPanel>();
        for diff in &boxes {
            commands.entity(diff).remove::<TuiPanel>();
        }
        focus.0 = None;
    } else {
        focus.0 = Some(panel);
    }
}

/// Up and down in the panel.
fn select(
    by: isize,
) -> impl FnMut(On<KeyPressed>, Query<&mut FilesPanel>, Res<TouchedFiles>, MessageWriter<RequestRedraw>)
{
    move |_, mut panels, files, mut redraw| {
        let last = files.rows().len().saturating_sub(1);
        for mut panel in &mut panels {
            panel.0 = panel.0.saturating_add_signed(by).min(last);
        }
        redraw.write(RequestRedraw);
    }
}

/// Enter in the panel: the selected file's session diff, with the keys.
fn open_diff(
    _: On<KeyPressed>,
    panels: Query<&FilesPanel>,
    mut boxes: Query<(Entity, &mut DiffBox)>,
    files: Res<TouchedFiles>,
    originals: Res<Originals>,
    root: Res<ProjectRoot>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    let row = panels
        .iter()
        .next()
        .map(|panel| panel.0)
        .unwrap_or_default();
    let (Some(file), Ok((entity, mut diff))) = (files.rows().get(row).copied(), boxes.single_mut())
    else {
        return;
    };
    diff.path.clone_from(&file.path);
    diff.diff = session_diff(&file.path, &root.0, &originals);
    diff.scroll = 0;
    let over = Placement::Over {
        width: Constraint::Percentage(80),
        height: Constraint::Percentage(80),
    };
    commands.entity(entity).insert(TuiPanel::new(over));
    focus.0 = Some(entity);
}

/// Up and down in the diff box.
fn scroll(
    by: i16,
) -> impl FnMut(On<KeyPressed>, Query<&mut DiffBox>, MessageWriter<RequestRedraw>) {
    move |_, mut boxes, mut redraw| {
        for mut diff in &mut boxes {
            diff.scroll = diff.scroll.saturating_add_signed(by);
        }
        redraw.write(RequestRedraw);
    }
}

/// Esc in the diff box: closes it, and gives the panel the keys.
fn close_diff(
    _: On<KeyPressed>,
    boxes: Query<Entity, With<DiffBox>>,
    panels: Query<Entity, (With<FilesPanel>, With<TuiPanel>)>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    for diff in &boxes {
        commands.entity(diff).remove::<TuiPanel>();
    }
    focus.0 = panels.iter().next();
}

/// When the files changed while the panel or the diff box is open: the
/// diff again, and a frame.
fn refresh(
    open: Query<(), (Or<(With<FilesPanel>, With<DiffBox>)>, With<TuiPanel>)>,
    mut boxes: Query<&mut DiffBox, With<TuiPanel>>,
    originals: Res<Originals>,
    root: Res<ProjectRoot>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    for mut shown in &mut boxes {
        let DiffBox { path, diff, .. } = &mut *shown;
        *diff = session_diff(path, &root.0, &originals);
    }
    if !open.is_empty() {
        redraw.write(RequestRedraw);
    }
}

fn draw_panel(
    mut panels: Query<(Entity, &mut PanelCanvas, &FilesPanel), With<TuiPanel>>,
    files: Res<TouchedFiles>,
    root: Res<ProjectRoot>,
    focus: Res<KeyFocus>,
) {
    for (entity, mut canvas, &FilesPanel(row)) in &mut panels {
        let selected = (focus.0 == Some(entity)).then_some(row);
        let area = canvas.area();
        draw_files(&files, &root.0, selected, area, canvas.buffer_mut());
    }
}

fn draw_diff(
    mut boxes: Query<(&mut PanelCanvas, &DiffBox), With<TuiPanel>>,
    root: Res<ProjectRoot>,
) {
    for (mut canvas, diff) in &mut boxes {
        let lines = rig_tui::diff::unified_lines(&diff.diff, usize::MAX)
            .unwrap_or_else(|| vec![Line::from("No changes kept for this file.").dim()]);
        let title = format!("{} (Esc closes)", shown(&diff.path, &root.0));
        let area = canvas.area();
        Paragraph::new(lines)
            .block(Block::bordered().title(title))
            .wrap(Wrap { trim: false })
            .scroll((diff.scroll, 0))
            .render(area, canvas.buffer_mut());
    }
}

#[cfg(test)]
mod tests;
