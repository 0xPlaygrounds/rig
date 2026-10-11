//! Plugin corpus: 52. Touched-files panel (spec `52-files-panel.md`).
//!
//! Every file an agent of the session read, edited or wrote, with the lines
//! its edits added and removed, kept as a saved resource ([`TouchedFiles`])
//! so `/resume` brings the same list back. The terminal view shows it in a
//! right-hand panel, and a file's session diff in a box over the screen;
//! `/files` lists it in any front.
//!
//! The terminal view gives a plugin no keys, so what the spec binds to keys
//! is a command here: `/files panel` toggles the panel, `/files pick` opens
//! the view's picker over the files (up, down, Enter), whose choice runs
//! `/files diff <path>`; Esc on the agent shown, or `/files close`, closes
//! the diff.

use std::path::Path;

use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Modifier, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::{Block, Paragraph, Wrap};
use rig_tui::{
    Focused, PanelCanvas, PickItem, PickRequest, Placement, RequestRedraw, TuiPanel, TuiSystems,
};

/// The panel's width.
const WIDTH: u16 = 32;

/// What one file's changes added and removed, and their diffs.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub struct Change {
    /// Lines added.
    pub added: u32,
    /// Lines removed.
    pub removed: u32,
    /// Each change's unified diff, oldest first.
    pub diff: String,
    /// When it last changed, on [`TouchedFiles::clock`].
    pub at: u64,
}

/// A file an agent touched.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub struct TouchedFile {
    /// The path, as the tool was given it.
    pub path: String,
    /// The subagent that touched it last, by name; empty for a top agent.
    pub by: String,
    /// Its changes, once one was made.
    pub changed: Option<Change>,
    /// When it was last read, on [`TouchedFiles::clock`].
    pub read_at: Option<u64>,
    /// Whether it was gone when a turn last ended.
    pub gone: bool,
}

/// Every file the session's agents read or changed, saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct TouchedFiles {
    /// The files, in the order first touched.
    pub files: Vec<TouchedFile>,
    /// Counts touches, to order the lists newest first.
    pub clock: u64,
}

impl TouchedFiles {
    /// The changed files, newest first.
    pub fn changed(&self) -> Vec<(&TouchedFile, &Change)> {
        let mut changed: Vec<_> = self
            .files
            .iter()
            .filter_map(|file| Some((file, file.changed.as_ref()?)))
            .collect();
        changed.sort_by_key(|(_, change)| std::cmp::Reverse(change.at));
        changed
    }

    /// The files read, newest first.
    pub fn read(&self) -> Vec<&TouchedFile> {
        let mut read: Vec<_> = self.files.iter().filter(|f| f.read_at.is_some()).collect();
        read.sort_by_key(|file| std::cmp::Reverse(file.read_at));
        read
    }

    /// The file at `path`, its entry made if new, marked as touched by
    /// `by`, and the time of this touch.
    fn touch(&mut self, path: &str, by: String) -> Option<(&mut TouchedFile, u64)> {
        self.clock += 1;
        let at = self.clock;
        if !self.files.iter().any(|file| file.path == path) {
            self.files.push(TouchedFile {
                path: path.to_owned(),
                ..TouchedFile::default()
            });
        }
        let file = self.files.iter_mut().find(|file| file.path == path)?;
        file.by = by;
        file.gone = false;
        Some((file, at))
    }

    /// The list as text lines, for `/files`.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec!["changed".to_owned()];
        for (file, change) in self.changed() {
            lines.push(format!("  {}", changed_label(file, change)));
        }
        lines.push("read".to_owned());
        for file in self.read() {
            lines.push(format!("  {}", label(file)));
        }
        lines
    }
}

/// `path` relative to the working directory when under it.
fn shown(path: &str) -> String {
    let relative = std::env::current_dir().ok().and_then(|dir| {
        Some(
            Path::new(path)
                .strip_prefix(dir)
                .ok()?
                .display()
                .to_string(),
        )
    });
    relative.unwrap_or_else(|| path.trim_start_matches("./").to_owned())
}

/// A file's path, with the subagent that touched it.
fn label(file: &TouchedFile) -> String {
    let path = shown(&file.path);
    if file.by.is_empty() {
        path
    } else {
        format!("{path} ({})", file.by)
    }
}

/// A changed file's label with its counts, such as `a.txt +1 −1`.
fn changed_label(file: &TouchedFile, change: &Change) -> String {
    format!("{} +{} −{}", label(file), change.added, change.removed)
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

/// The panel's contents for `files`, at `width` columns.
pub fn panel(files: &TouchedFiles, width: u16) -> Paragraph<'static> {
    let inner = usize::from(width.saturating_sub(2));
    let entry = |text: String, gone: bool| {
        let style = if gone {
            Style::new().add_modifier(Modifier::CROSSED_OUT).dim()
        } else {
            Style::new()
        };
        Line::from(Span::styled(fit(&text, inner), style))
    };
    let mut lines = vec![Line::from("changed").bold()];
    for (file, change) in files.changed() {
        lines.push(entry(changed_label(file, change), file.gone));
    }
    lines.push(Line::from("read").bold());
    for file in files.read() {
        lines.push(entry(label(file), file.gone));
    }
    Paragraph::new(lines).block(Block::bordered().title("files"))
}

/// The lines `diff` adds and removes, after its `---` header.
fn count(diff: &str) -> (u32, u32) {
    let mut counts = (0, 0);
    for line in diff.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if line.starts_with('+') {
            counts.0 += 1;
        } else if line.starts_with('-') {
            counts.1 += 1;
        }
    }
    counts
}

/// The touched-files panel.
#[derive(Default)]
pub struct FilesPanelPlugin;

impl Plugin for FilesPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TouchedFiles>()
            .register_type::<TouchedFiles>()
            .add_message::<RequestRedraw>()
            .add_message::<PickRequest>()
            .add_command(
                "files",
                "The files read and changed: `panel`, `pick`, `diff <path>`, `close`",
                files_command,
            )
            .add_observer(track)
            .add_observer(check_gone)
            .add_observer(close_on_esc)
            .add_systems(
                PostUpdate,
                redraw
                    .run_if(resource_changed::<TouchedFiles>)
                    .in_set(TuiSystems::Prepare),
            )
            .add_systems(PostUpdate, (draw_panel, draw_diff).in_set(TuiSystems::Draw));
    }
}

/// The panel's entity.
#[derive(Component)]
struct FilesPanel;

/// The diff box's entity, showing the session diff of the file at its path.
#[derive(Component)]
struct DiffBox(String);

/// Records each successful `read`, `edit` and `write` call as it ends.
fn track(
    done: On<Add<ToolOutput>>,
    calls: Query<(&ToolCallRun, &ToolOutput, Option<&CallOf>)>,
    turns: Query<&TurnOf>,
    subagents: Query<&Name, With<SpawnedBy>>,
    mut files: ResMut<TouchedFiles>,
) {
    let Ok((run, ToolOutput(result), of)) = calls.get(done.entity) else {
        return;
    };
    if !result.is_success() {
        return;
    }
    let arguments = &run.call.function.arguments;
    let argument = |key: &str| arguments.get(key).and_then(|value| value.as_str());
    let Some(path) = argument("path") else {
        return;
    };
    let tool = run.call.function.name.as_str();
    if !matches!(tool, "read" | "edit" | "write") {
        return;
    }
    let agent = of.and_then(|CallOf(turn)| turns.get(*turn).ok());
    let by = agent
        .and_then(|TurnOf(agent)| subagents.get(*agent).ok())
        .map(|name| name.as_str().to_owned())
        .unwrap_or_default();
    // The edit's diff is in its text result: the tool returns no data.
    let diff = match tool {
        "read" => None,
        "edit" => {
            let text = result.output().render();
            let start = text.find("\n---").map_or(0, |at| at + 1);
            Some(text.get(start..).unwrap_or_default().to_owned())
        }
        _ => {
            let content = argument("content").unwrap_or_default();
            let mut diff = format!(
                "--- {path}\n+++ {path}\n@@ -0,0 +1,{} @@\n",
                content.lines().count()
            );
            for line in content.lines() {
                diff.push('+');
                diff.push_str(line);
                diff.push('\n');
            }
            Some(diff)
        }
    };
    let Some((file, at)) = files.touch(path, by) else {
        return;
    };
    match diff {
        None => file.read_at = Some(at),
        Some(diff) => {
            let (added, removed) = count(&diff);
            let change = file.changed.get_or_insert_default();
            change.added += added;
            change.removed += removed;
            change.diff.push_str(&diff);
            change.at = at;
        }
    }
}

/// When a turn ends, marks the files that are gone, such as one `shell`
/// deleted.
fn check_gone(_: On<TurnEnded>, mut files: ResMut<TouchedFiles>) {
    let gone: Vec<bool> = files
        .files
        .iter()
        .map(|file| !Path::new(&file.path).exists())
        .collect();
    if files
        .files
        .iter()
        .map(|file| file.gone)
        .eq(gone.iter().copied())
    {
        return;
    }
    for (file, gone) in files.files.iter_mut().zip(gone) {
        file.gone = gone;
    }
}

/// Esc on the agent shown closes the diff box.
fn close_on_esc(
    interrupt: On<Interrupt>,
    focused: Query<(), With<Focused>>,
    boxes: Query<Entity, With<DiffBox>>,
    mut commands: Commands,
) {
    if focused.contains(interrupt.entity) {
        for entity in &boxes {
            commands.entity(entity).despawn();
        }
    }
}

/// `/files`: the list; `panel`, `pick`, `diff <path>` and `close`.
fn files_command(
    In(args): In<CommandArgs>,
    files: Res<TouchedFiles>,
    panels: Query<Entity, With<FilesPanel>>,
    boxes: Query<Entity, With<DiffBox>>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let (verb, rest) = args
        .args
        .trim()
        .split_once(' ')
        .unwrap_or((args.args.trim(), ""));
    match verb {
        "" => {
            notices.write(Notice::info(args.agent, files.lines().join("\n")));
        }
        "panel" => {
            if panels.is_empty() {
                let right = Placement::Right(Constraint::Length(WIDTH));
                commands.spawn((FilesPanel, TuiPanel::new(right)));
            }
            for panel in &panels {
                commands.entity(panel).despawn();
            }
        }
        "pick" => {
            let items = files
                .changed()
                .into_iter()
                .map(|(file, change)| PickItem {
                    label: changed_label(file, change),
                    command: format!("files diff {}", file.path),
                })
                .collect();
            picks.write(PickRequest {
                agent: args.agent,
                title: "Changed files".to_owned(),
                items,
                selected: 0,
            });
        }
        "diff" => {
            let path = rest.trim();
            if !files.changed().iter().any(|(file, _)| file.path == path) {
                notices.write(Notice::error(args.agent, format!("{path} was not changed")));
                return;
            }
            for entity in &boxes {
                commands.entity(entity).despawn();
            }
            let over = Placement::Over {
                width: Constraint::Percentage(80),
                height: Constraint::Percentage(80),
            };
            commands.spawn((DiffBox(path.to_owned()), TuiPanel::new(over)));
        }
        "close" => {
            for entity in &boxes {
                commands.entity(entity).despawn();
            }
        }
        other => {
            let why = format!("/files takes panel, pick, diff <path> or close, not {other}");
            notices.write(Notice::error(args.agent, why));
        }
    }
}

/// A frame for changed data while a panel or diff box shows it.
fn redraw(
    shown: Query<(), Or<(With<FilesPanel>, With<DiffBox>)>>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    if !shown.is_empty() {
        redraw.write(RequestRedraw);
    }
}

fn draw_panel(mut panels: Query<&mut PanelCanvas, With<FilesPanel>>, files: Res<TouchedFiles>) {
    for mut canvas in &mut panels {
        let width = canvas.area().width;
        canvas.render(panel(&files, width));
    }
}

fn draw_diff(mut boxes: Query<(&mut PanelCanvas, &DiffBox)>, files: Res<TouchedFiles>) {
    for (mut canvas, DiffBox(path)) in &mut boxes {
        let diff = files
            .files
            .iter()
            .find(|file| &file.path == path)
            .and_then(|file| file.changed.as_ref())
            .map(|change| change.diff.as_str())
            .unwrap_or_default();
        let limit = usize::from(canvas.area().height);
        let lines = rig_tui::diff::unified_lines(diff, limit).unwrap_or_default();
        let title = format!("{} (Esc closes)", shown(path));
        let block = Block::bordered().title(title);
        canvas.render(
            Paragraph::new(lines)
                .block(block)
                .wrap(Wrap { trim: false }),
        );
    }
}

#[cfg(test)]
mod tests;
