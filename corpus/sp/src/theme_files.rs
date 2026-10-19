//! SP-08: theme files. Colour themes are small TOML files in
//! `~/.config/rig/themes/`, beside the built-in `dark` and `light`; each
//! key names a role ([`ThemeRole`]), not a widget. `/theme <name>` switches
//! at once, `/theme` alone opens a picker that previews the highlighted
//! theme and puts the old one back on Esc. The choice is kept in the
//! user's plugin data across sessions, and editing the active theme's file
//! applies it as it is saved; a file that does not read keeps the colours
//! and says why once. With `NO_COLOR` set, no colours are drawn.
//!
//! Other plugins colour their own panels by role from the [`Theme`]
//! resource. The terminal view itself has fixed ANSI colours and no roles,
//! so this plugin recolours what its widgets drew between
//! [`TuiSystems::Draw`] and [`TuiSystems::Render`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message;
use rig_harness::rig_core::message::UserContent;
use rig_tui::ratatui::buffer::Buffer;
use rig_tui::ratatui::style::{Color, Modifier, Style};
use rig_tui::{
    DrawnTranscript, Editor, PanelCanvas, RequestRedraw, StatusLine, TranscriptItem,
    TranscriptPane, TuiSystems,
};
use serde::Deserialize;

/// The plugin's name, as the author of its notices and its files' prefix.
pub const PLUGIN: &str = "theme-files";

/// The user's choice of theme, in [`PluginData::user`].
const CHOICE: &str = "theme-files/theme.json";

/// What a theme colours. Keys of a theme file are these names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThemeRole {
    /// What the user typed.
    User,
    /// The model's answer text.
    Assistant,
    /// A tool call and its output.
    Tool,
    /// An error, such as a failed call's result.
    Error,
    /// A warning.
    Warning,
    /// Text that matters less, such as a call's output.
    Dim,
    /// What stands out: headings, markers, the app's own state.
    Accent,
    /// The status line's text.
    StatusFg,
    /// The status line's background.
    StatusBg,
    /// A line a diff adds.
    DiffAdd,
    /// A line a diff removes.
    DiffRemove,
    /// The background of what is selected.
    Selection,
}

impl ThemeRole {
    /// Every role, in the order a theme file lists them.
    pub const ALL: [Self; 12] = [
        Self::User,
        Self::Assistant,
        Self::Tool,
        Self::Error,
        Self::Warning,
        Self::Dim,
        Self::Accent,
        Self::StatusFg,
        Self::StatusBg,
        Self::DiffAdd,
        Self::DiffRemove,
        Self::Selection,
    ];

    /// Its key in a theme file, such as `status_bg`.
    pub fn key(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Dim => "dim",
            Self::Accent => "accent",
            Self::StatusFg => "status_fg",
            Self::StatusBg => "status_bg",
            Self::DiffAdd => "diff_add",
            Self::DiffRemove => "diff_remove",
            Self::Selection => "selection",
        }
    }

    /// The role a theme file's `key` names.
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|role| role.key() == key)
    }
}

/// The 16 ANSI colours by number, with their names in a theme file.
const ANSI: [(&str, Color); 16] = [
    ("black", Color::Black),
    ("red", Color::Red),
    ("green", Color::Green),
    ("yellow", Color::Yellow),
    ("blue", Color::Blue),
    ("magenta", Color::Magenta),
    ("cyan", Color::Cyan),
    ("white", Color::Gray),
    ("bright_black", Color::DarkGray),
    ("bright_red", Color::LightRed),
    ("bright_green", Color::LightGreen),
    ("bright_yellow", Color::LightYellow),
    ("bright_blue", Color::LightBlue),
    ("bright_magenta", Color::LightMagenta),
    ("bright_cyan", Color::LightCyan),
    ("bright_white", Color::White),
];

/// A theme file's colour: `#rrggbb`, `ansi:<0-15>` or an ANSI colour's
/// name (`red`, `bright_blue`, `white` is ANSI 7); `None` for anything
/// else.
pub fn parse_color(value: &str) -> Option<Color> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix('#') {
        let channel = |at: usize| u8::from_str_radix(hex.get(at..at + 2)?, 16).ok();
        return match hex.len() {
            6 => Some(Color::Rgb(channel(0)?, channel(2)?, channel(4)?)),
            _ => None,
        };
    }
    if let Some(number) = value.strip_prefix("ansi:") {
        let number: usize = number.parse().ok()?;
        return ANSI.get(number).map(|(_, color)| *color);
    }
    let name = value.to_ascii_lowercase().replace([' ', '-'], "_");
    ANSI.iter()
        .find(|(known, _)| *known == name)
        .map(|(_, color)| *color)
}

/// A colour for each role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Palette(BTreeMap<ThemeRole, Color>);

impl Palette {
    /// The built-in `dark` theme: the terminal view's own colours.
    pub fn dark() -> Self {
        Self(BTreeMap::from([
            (ThemeRole::User, Color::Cyan),
            (ThemeRole::Assistant, Color::Reset),
            (ThemeRole::Tool, Color::Reset),
            (ThemeRole::Error, Color::Red),
            (ThemeRole::Warning, Color::Yellow),
            (ThemeRole::Dim, Color::DarkGray),
            (ThemeRole::Accent, Color::Cyan),
            (ThemeRole::StatusFg, Color::Reset),
            (ThemeRole::StatusBg, Color::Reset),
            (ThemeRole::DiffAdd, Color::Green),
            (ThemeRole::DiffRemove, Color::Red),
            (ThemeRole::Selection, Color::DarkGray),
        ]))
    }

    /// The built-in `light` theme, for a light terminal.
    pub fn light() -> Self {
        Self(BTreeMap::from([
            (ThemeRole::User, Color::Blue),
            (ThemeRole::Assistant, Color::Black),
            (ThemeRole::Tool, Color::Black),
            (ThemeRole::Error, Color::Red),
            (ThemeRole::Warning, Color::Rgb(0xaf, 0x5f, 0x00)),
            (ThemeRole::Dim, Color::DarkGray),
            (ThemeRole::Accent, Color::Magenta),
            (ThemeRole::StatusFg, Color::Black),
            (ThemeRole::StatusBg, Color::Gray),
            (ThemeRole::DiffAdd, Color::Green),
            (ThemeRole::DiffRemove, Color::Red),
            (ThemeRole::Selection, Color::Gray),
        ]))
    }

    /// The built-in theme `name`.
    pub fn built_in(name: &str) -> Option<Self> {
        match name {
            "dark" => Some(Self::dark()),
            "light" => Some(Self::light()),
            _ => None,
        }
    }

    /// The colour of `role`.
    pub fn get(&self, role: ThemeRole) -> Color {
        self.0.get(&role).copied().unwrap_or(Color::Reset)
    }

    /// The theme a file's `text` holds, each role it leaves out as `dark`
    /// has it.
    ///
    /// # Errors
    ///
    /// What is wrong, naming the line of a text that is not TOML, or the
    /// key whose value is no colour or that names no role.
    pub fn parse(text: &str) -> Result<Self, String> {
        let table: toml::Table = text.parse().map_err(|error: toml::de::Error| {
            let line = error.span().map(|span| line_of(text, span.start));
            let at = line
                .map(|line| format!("line {line}: "))
                .unwrap_or_default();
            format!("{at}{}", error.message())
        })?;
        let mut palette = Self::dark();
        for (key, value) in table {
            let role = ThemeRole::from_key(&key).ok_or_else(|| format!("`{key}`: no such role"))?;
            let color = value.as_str().and_then(parse_color);
            let color = color.ok_or_else(|| format!("`{key}`: {value} is not a colour"))?;
            palette.0.insert(role, color);
        }
        Ok(palette)
    }
}

/// The line, from 1, byte `at` of `text` is on.
fn line_of(text: &str, at: usize) -> usize {
    text.get(..at)
        .map_or(0, |before| before.matches('\n').count())
        + 1
}

/// The theme in use: its name and colours. A plugin colours its own
/// panels by role with [`Theme::fg`], which has no colour under
/// `NO_COLOR`, and draws again when it changes.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct Theme {
    name: String,
    palette: Palette,
    no_color: bool,
}

impl Theme {
    /// The theme's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The colour of `role`; `None` for the terminal's own, and under
    /// `NO_COLOR`.
    pub fn color(&self, role: ThemeRole) -> Option<Color> {
        let color = self.palette.get(role);
        (!self.no_color && color != Color::Reset).then_some(color)
    }

    /// A style in the colour of `role` (the background for
    /// [`ThemeRole::StatusBg`] and [`ThemeRole::Selection`]).
    pub fn fg(&self, role: ThemeRole) -> Style {
        let style = Style::new();
        match (role, self.color(role)) {
            (_, None) => style,
            (ThemeRole::StatusBg | ThemeRole::Selection, Some(color)) => style.bg(color),
            (_, Some(color)) => style.fg(color),
        }
    }
}

/// Whether colours are off: `NO_COLOR` set and not empty, as
/// <https://no-color.org> says. A test inserts it before the plugin.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoColor(pub bool);

impl Default for NoColor {
    fn default() -> Self {
        Self(std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()))
    }
}

/// The `[themes]` table: `dir`, where theme files are; by default
/// `$XDG_CONFIG_HOME/rig/themes`, else `~/.config/rig/themes`.
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ThemeSettings {
    /// The directory of theme files.
    pub dir: Option<PathBuf>,
}

impl ThemeSettings {
    /// The directory of theme files.
    pub fn dir(&self) -> Option<PathBuf> {
        let config = || {
            let xdg = std::env::var_os("XDG_CONFIG_HOME").filter(|dir| !dir.is_empty());
            xdg.map(PathBuf::from)
                .or_else(|| Some(PathBuf::from(std::env::var_os("HOME")?).join(".config")))
        };
        self.dir
            .clone()
            .or_else(|| Some(config()?.join("rig").join("themes")))
    }
}

/// The themes read from files, by name, as last read whole, and what is
/// wrong with each file that did not read.
#[derive(Resource, Default)]
struct ThemeFiles {
    palettes: BTreeMap<String, Palette>,
    problems: BTreeMap<String, String>,
}

impl ThemeFiles {
    /// The theme `name`: a file's, else a built-in one.
    fn palette(&self, name: &str) -> Option<Palette> {
        let file = self.palettes.get(name).cloned();
        file.or_else(|| Palette::built_in(name))
    }

    /// Every theme's name: the built-in ones, then the files'.
    fn names(&self) -> Vec<String> {
        let built_in = ["dark", "light"].map(str::to_owned);
        let files = self
            .palettes
            .keys()
            .filter(|name| Palette::built_in(name).is_none());
        built_in.into_iter().chain(files.cloned()).collect()
    }
}

/// On the watch of the theme files.
#[derive(Component)]
struct ThemeWatch;

/// On the `/theme` picker: the theme in use when it opened.
#[derive(Component)]
struct ThemePicker {
    before: String,
}

/// Themes from files, switched with `/theme`, kept across sessions,
/// applied as they are edited.
#[derive(Default)]
pub struct ThemeFilesPlugin;

impl Plugin for ThemeFilesPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ThemeSettings>("themes")
            .init_resource::<NoColor>()
            .init_resource::<ThemeFiles>()
            .add_message::<FileChanges>()
            .add_message::<RequestRedraw>()
            .add_command(
                "theme",
                "Switch the colour theme: /theme <name>, or pick one",
                theme_command,
            )
            .add_systems(Startup, (restore_choice, watch_themes))
            .add_systems(
                Update,
                (
                    note_rows,
                    read_theme_files,
                    preview_highlighted,
                    redraw.run_if(resource_changed::<Theme>),
                )
                    .chain(),
            )
            .add_systems(
                PostUpdate,
                recolour.after(TuiSystems::Draw).before(TuiSystems::Render),
            );
        let no_color = app.world().resource::<NoColor>().0;
        app.insert_resource(Theme {
            name: "dark".to_owned(),
            palette: Palette::dark(),
            no_color,
        });
    }
}

/// Puts back the theme the user chose last, from their plugin data.
fn restore_choice(data: Option<Res<PluginData>>, files: Res<ThemeFiles>, mut theme: ResMut<Theme>) {
    let chosen = data.and_then(|data| data.user.read_file(CHOICE).ok());
    let chosen = chosen.and_then(|bytes| serde_json::from_slice::<String>(&bytes).ok());
    if let Some(name) = chosen {
        // A file's theme gets its colours once the watch reads it.
        theme.palette = files.palette(&name).unwrap_or_else(Palette::dark);
        theme.name = name;
    }
}

/// Watches the theme files' directory.
fn watch_themes(settings: Res<ThemeSettings>, mut commands: Commands) {
    if let Some(dir) = settings.dir() {
        commands.spawn((Name::new("theme files"), ThemeWatch, FileWatch::new(dir)));
    }
}

/// Reads each theme file the watch reports; the theme in use takes a
/// changed file's colours, and a file that does not read keeps them and
/// says why, once for each new problem.
fn read_theme_files(
    mut changes: MessageReader<FileChanges>,
    watches: Query<(), With<ThemeWatch>>,
    mut files: ResMut<ThemeFiles>,
    mut theme: ResMut<Theme>,
    mut notices: MessageWriter<Notice>,
) {
    let paths = changes
        .read()
        .filter(|change| watches.contains(change.watch));
    for path in paths.flat_map(|change| change.paths.iter()) {
        let Some(name) = theme_name(path) else {
            continue;
        };
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                files.palettes.remove(&name);
                continue;
            }
            Err(error) => {
                notices.write(Notice::warn(None, format!("Theme `{name}`: {error}")));
                continue;
            }
        };
        match Palette::parse(&text) {
            Ok(palette) => {
                files.problems.remove(&name);
                if theme.name == name && theme.palette != palette {
                    theme.palette = palette.clone();
                }
                files.palettes.insert(name, palette);
            }
            Err(problem) if files.problems.get(&name) != Some(&problem) => {
                let text = format!("Theme `{name}` not applied: {problem}");
                notices.write(Notice::warn(None, text));
                files.problems.insert(name, problem);
            }
            Err(_) => {}
        }
    }
}

/// The theme a file at `path` holds: its stem, for a `.toml` file.
fn theme_name(path: &Path) -> Option<String> {
    let toml = path
        .extension()
        .is_some_and(|extension| extension == "toml");
    let stem = path.file_stem()?.to_str()?;
    toml.then(|| stem.to_owned())
}

/// `/theme <name>` switches and keeps the choice; `/theme` alone opens a
/// picker of every theme.
fn theme_command(
    In(args): In<CommandArgs>,
    files: Res<ThemeFiles>,
    mut theme: ResMut<Theme>,
    data: Option<Res<PluginData>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let name = args.args.trim();
    if name.is_empty() {
        let names = files.names();
        let picker = Picker::new("Theme").for_agent(args.agent);
        let mut picker = names.iter().fold(picker, |picker, name| {
            picker.item(name.as_str(), format!("theme {name}"))
        });
        picker.selected = names
            .iter()
            .position(|name| *name == theme.name)
            .unwrap_or(0);
        let before = theme.name.clone();
        commands
            .spawn((picker, ThemePicker { before }))
            .observe(closed);
        return;
    }
    let Some(palette) = files.palette(name) else {
        let known = files.names().join(", ");
        notices.write(Notice::error(
            args.agent,
            format!("No theme `{name}`; there are {known}."),
        ));
        return;
    };
    theme.palette = palette;
    theme.name = name.to_owned();
    let kept = serde_json::to_vec(name).map_err(std::io::Error::other);
    let kept = kept.and_then(|json| match &data {
        Some(data) => data.user.write_file(CHOICE, &json),
        None => Ok(()),
    });
    if let Err(error) = kept {
        notices.write(Notice::warn(args.agent, format!("Theme not kept: {error}")));
    }
}

/// Shows the theme highlighted in a `/theme` picker, without keeping it.
fn preview_highlighted(
    pickers: Query<(&Picker, &Highlighted), (With<ThemePicker>, Changed<Highlighted>)>,
    files: Res<ThemeFiles>,
    mut theme: ResMut<Theme>,
) {
    for (picker, &Highlighted(at)) in &pickers {
        let Some(name) = picker.items.get(at).map(|item| item.label.clone()) else {
            continue;
        };
        if theme.name != name
            && let Some(palette) = files.palette(&name)
        {
            theme.palette = palette;
            theme.name = name;
        }
    }
}

/// A `/theme` picker closed without a pick puts the theme back.
fn closed(
    picked: On<Picked>,
    pickers: Query<&ThemePicker>,
    files: Res<ThemeFiles>,
    mut theme: ResMut<Theme>,
) {
    let Ok(ThemePicker { before }) = pickers.get(picked.entity) else {
        return;
    };
    if picked.chosen.is_none()
        && let Some(palette) = files.palette(before)
    {
        theme.palette = palette;
        theme.name.clone_from(before);
    }
}

/// Has each transcript pane keep which item each of its rows is, to
/// colour it by role; panes may be spawned before the plugin is added.
fn note_rows(
    panes: Query<Entity, (With<TranscriptPane>, Without<DrawnTranscript>)>,
    mut commands: Commands,
) {
    for pane in &panes {
        commands.entity(pane).try_insert(DrawnTranscript::default());
    }
}

/// Draws the screen again in the new colours.
fn redraw(mut redraws: MessageWriter<RequestRedraw>) {
    redraws.write(RequestRedraw);
}

/// What the terminal view drew in a cell, read as a role: its widgets
/// have fixed ANSI colours, so the colour stands for the role.
fn role_of_color(color: Color) -> Option<ThemeRole> {
    match color {
        Color::Red => Some(ThemeRole::Error),
        Color::Yellow => Some(ThemeRole::Warning),
        Color::Cyan | Color::Blue | Color::Magenta | Color::LightYellow => Some(ThemeRole::Accent),
        Color::Green => Some(ThemeRole::DiffAdd),
        Color::DarkGray | Color::Gray => Some(ThemeRole::Dim),
        _ => None,
    }
}

/// Recolours the terminal view's own widgets as they were drawn this
/// frame: the transcript by the role of each row's item, the status line
/// on its background, the input; with `NO_COLOR`, every panel loses its
/// colours.
fn recolour(
    theme: Res<Theme>,
    mut canvases: Query<
        (
            &mut PanelCanvas,
            Has<TranscriptPane>,
            Has<StatusLine>,
            Has<Editor>,
            Option<&DrawnTranscript>,
        ),
        Changed<PanelCanvas>,
    >,
    conversations: Query<&Conversation>,
) {
    for (mut canvas, pane, status, editor, drawn) in &mut canvases {
        let buffer = canvas.bypass_change_detection().buffer_mut();
        if theme.no_color {
            strip(buffer);
            continue;
        }
        if editor {
            // An editor with a status line draws it in its bottom row.
            let last = buffer.area.bottom().saturating_sub(1);
            let row = |y| (status && y == last).then_some(ThemeRole::StatusFg);
            recolour_area(&theme, buffer, row, |y| status && y == last);
        } else if status {
            recolour_area(&theme, buffer, |_| Some(ThemeRole::StatusFg), |_| true);
        } else if pane {
            let rows = drawn.map(|drawn| row_roles(drawn, &conversations));
            let rows = rows.unwrap_or_default();
            recolour_area(
                &theme,
                buffer,
                |y| rows.get(&y).copied().flatten(),
                |_| false,
            );
        }
    }
}

/// The role of each screen row of a transcript: the innermost item's,
/// a user's message, a reply or a tool call.
fn row_roles(
    drawn: &DrawnTranscript,
    conversations: &Query<&Conversation>,
) -> BTreeMap<u16, Option<ThemeRole>> {
    let conversation = drawn
        .agent()
        .and_then(|agent| conversations.get(agent).ok());
    let role = |item: &TranscriptItem| match item {
        TranscriptItem::Message(id) => {
            let conversation = conversation?;
            let entry = conversation.entries().nth(conversation.index_of(*id)?)?;
            match entry.message {
                Message::User { content }
                    if content
                        .iter()
                        .any(|part| !matches!(part, UserContent::ToolResult(_))) =>
                {
                    Some(ThemeRole::User)
                }
                Message::Assistant { .. } => Some(ThemeRole::Assistant),
                _ => None,
            }
        }
        TranscriptItem::Call(..) => Some(ThemeRole::Tool),
        _ => None,
    };
    let top = drawn.area().y;
    drawn
        .on_screen()
        .enumerate()
        .filter_map(|(below, row)| {
            let y = top.checked_add(u16::try_from(below).ok()?)?;
            Some((y, drawn.item_at(row).and_then(role)))
        })
        .collect()
}

/// Recolours each cell of `buffer`, `row` giving the role of a row's
/// plain text; on `status_bg` for the status line (`status`).
fn recolour_area(
    theme: &Theme,
    buffer: &mut Buffer,
    row: impl Fn(u16) -> Option<ThemeRole>,
    status: impl Fn(u16) -> bool,
) {
    let area = buffer.area;
    for position in area.positions() {
        let plain = row(position.y);
        let Some(cell) = buffer.cell_mut(position) else {
            continue;
        };
        let dimmed = cell.modifier.contains(Modifier::DIM);
        let role = match cell.fg {
            Color::Reset if dimmed => Some(ThemeRole::Dim),
            Color::Reset => plain,
            // The user's text is the view's cyan.
            Color::Cyan if plain == Some(ThemeRole::User) => plain,
            color => role_of_color(color),
        };
        if let Some(role) = role {
            cell.fg = theme.color(role).unwrap_or(Color::Reset);
        }
        if status(position.y) {
            cell.bg = theme.color(ThemeRole::StatusBg).unwrap_or(Color::Reset);
        } else if cell.bg == Color::DarkGray {
            cell.bg = theme.color(ThemeRole::Selection).unwrap_or(Color::Reset);
        }
    }
}

/// Takes every colour out of `buffer`; bold, underline and the rest stay.
fn strip(buffer: &mut Buffer) {
    for position in buffer.area.positions() {
        if let Some(cell) = buffer.cell_mut(position) {
            cell.fg = Color::Reset;
            cell.bg = Color::Reset;
            cell.underline_color = Color::Reset;
        }
    }
}

#[cfg(test)]
mod tests;
