//! OC-17: images in the terminal transcript. Every tool call whose result
//! holds an image draws it under the call, scaled to at most
//! [`ImagesSettings::max_width_percent`] of the terminal's width and
//! [`ImagesSettings::max_rows`] rows, keeping its aspect ratio:
//!
//! - with a terminal that shows images ([`TerminalGraphics`]: the kitty
//!   graphics protocol or iTerm2's inline images, inside tmux only with its
//!   passthrough on), the call reserves blank rows, and after each frame
//!   the image is placed over them where the view drew them, and taken off
//!   once they scroll away, the screen is resized or a box opens over the
//!   transcript;
//! - otherwise half-block characters coloured as the image
//!   ([`Mode::Blocks`]), or with [`Mode::Off`] one line saying what it is.
//!
//! `o` on a selected call (Alt+Up) opens its image in the system viewer and
//! `y` copies the path it was saved to. The images are drawn from the
//! conversation, so a resumed session draws them again.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Cursor;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageFormat};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message as ChatMessage;
use rig_harness::rig_core::message::{
    CallId, DocumentSourceKind, Image, ToolResultContent, UserContent,
};
use rig_harness::rig_tools::process::Launcher;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::style::{Color, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{
    AppKeyBindingsExt, CopyToClipboard, DrawnTranscript, KeyBinding, KeyPress, KeyPressed,
    KeyScope, PanelCanvas, Placement, RenderToolCall, TerminalWrite, ToolCallView, ToolRenderer,
    TranscriptItem, TranscriptView, TuiInputState, TuiPanel, TuiScreen, TuiSystems,
};
use serde::Deserialize;

/// Columns an image is indented by, under its call's `⎿`.
const INDENT: u16 = 4;

/// What an image looks like in the transcript.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Drawn by the terminal when it shows images, else in half-blocks.
    #[default]
    Auto,
    /// Always in half-block characters.
    Blocks,
    /// One line with its size: `[image 1280×720 png, 84 KB — press o to open]`.
    Off,
}

/// The `[images]` table of `settings.toml`.
#[derive(Resource, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImagesSettings {
    /// How images are drawn.
    pub mode: Mode,
    /// The widest an image is drawn, in percent of the terminal's width.
    pub max_width_percent: u16,
    /// The tallest an image is drawn, in rows.
    pub max_rows: u16,
}

impl Default for ImagesSettings {
    fn default() -> Self {
        Self {
            mode: Mode::Auto,
            max_width_percent: 60,
            max_rows: 20,
        }
    }
}

/// How a terminal draws an image itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    /// The kitty graphics protocol (kitty, WezTerm, Ghostty).
    Kitty,
    /// iTerm2's inline images (OSC 1337).
    Iterm2,
}

/// What the terminal can draw, found when the plugin is built unless a
/// plugin or a test inserted it first.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalGraphics {
    /// How it draws images, if it does (and tmux passes them through).
    pub protocol: Option<Protocol>,
    /// Whether the agent runs inside tmux with its passthrough on: each
    /// sequence is wrapped for tmux.
    pub tmux: bool,
    /// A cell's size in pixels, width and height.
    pub cell: (u16, u16),
}

impl Default for TerminalGraphics {
    fn default() -> Self {
        Self {
            protocol: None,
            tmux: false,
            cell: (8, 16),
        }
    }
}

impl TerminalGraphics {
    /// What this process's terminal says of itself through its environment,
    /// asking tmux, when inside it, whether it passes images through.
    pub fn detect(launcher: &Launcher) -> Self {
        let var = |name: &str| std::env::var(name).unwrap_or_default();
        let program = var("TERM_PROGRAM");
        let kitty = !var("KITTY_WINDOW_ID").is_empty()
            || var("TERM").contains("kitty")
            || matches!(program.as_str(), "WezTerm" | "ghostty");
        let iterm = program == "iTerm.app" || var("LC_TERMINAL") == "iTerm2";
        let in_tmux = !var("TMUX").is_empty();
        let passes = in_tmux && tmux_passes_through(launcher);
        let protocol = match (kitty, iterm) {
            _ if in_tmux && !passes => None,
            (true, _) => Some(Protocol::Kitty),
            (false, true) => Some(Protocol::Iterm2),
            (false, false) => None,
        };
        let size = rig_tui::crossterm::terminal::window_size().ok();
        let cell = size
            .filter(|size| size.width > 0 && size.columns > 0 && size.rows > 0)
            .map_or((8, 16), |size| {
                (size.width / size.columns, size.height / size.rows)
            });
        Self {
            protocol,
            tmux: passes,
            cell: (cell.0.max(1), cell.1.max(1)),
        }
    }
}

/// Whether tmux's `allow-passthrough` is on.
fn tmux_passes_through(launcher: &Launcher) -> bool {
    let mut tmux = launcher.command("tmux");
    tmux.args(["show", "-gv", "allow-passthrough"]);
    let output = tmux.output().ok().filter(|output| output.status.success());
    output.is_some_and(|output| {
        matches!(String::from_utf8_lossy(&output.stdout).trim(), "on" | "all")
    })
}

/// How the transcript draws an image, from the settings and the terminal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Look {
    /// Blank rows the terminal draws the image over.
    Escape(Protocol),
    /// Half-block characters.
    #[default]
    Blocks,
    /// One line.
    Line,
}

/// An image decoded once, by the hash of its bytes.
struct Decoded {
    /// Its bytes as the conversation holds them.
    bytes: Vec<u8>,
    /// Its format's extension, such as `png`.
    format: &'static str,
    /// Its pixels.
    pixels: DynamicImage,
}

impl Decoded {
    /// `[image 100×50 png, 1 KB]`'s inside.
    fn label(&self) -> String {
        let (width, height) = self.pixels.dimensions();
        let size = match self.bytes.len() {
            n if n < 1024 => format!("{n} B"),
            n if n < 1024 * 1024 => format!("{} KB", n.div_ceil(1024)),
            n => format!("{:.1} MB", n as f64 / (1024.0 * 1024.0)),
        };
        format!("image {width}×{height} {}, {size}", self.format)
    }

    /// The image as PNG, which the kitty protocol takes.
    fn png(&self) -> Vec<u8> {
        if self.format == "png" {
            return self.bytes.clone();
        }
        let mut png = Vec::new();
        self.pixels
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .ok();
        png
    }
}

/// Where an image's rows are in its call's rows: they end `from_end` rows
/// before the call's last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    key: u64,
    index: usize,
    from_end: usize,
    cols: u16,
    rows: u16,
}

/// What the renderers and the placing system share: the renderers run
/// inside the view's layout, with no world to read.
#[derive(Default)]
struct Shared {
    /// Every plugin's `build` ran: a tool's own renderer is in by now.
    ready: bool,
    look: Look,
    cell: (u16, u16),
    settings: ImagesSettings,
    /// The terminal's width in columns.
    width: u16,
    decoded: HashMap<u64, Arc<Decoded>>,
    /// Each call's images, as the renderer last reserved rows for them.
    slots: HashMap<CallId, Vec<Slot>>,
    /// Each call's images, by key.
    images_of: HashMap<CallId, Vec<u64>>,
}

/// The state the renderers share with the plugin's systems.
#[derive(Resource, Clone, Default)]
struct Images(Arc<Mutex<Shared>>);

impl Images {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The key of an image's bytes.
fn key_of(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// The bytes of an image the conversation holds, if it holds them.
fn bytes_of(image: &Image) -> Option<Vec<u8>> {
    match &image.data {
        DocumentSourceKind::Base64(data) => STANDARD.decode(data).ok(),
        DocumentSourceKind::Raw(bytes) => Some(bytes.clone()),
        _ => None,
    }
}

/// Cells `width`×`height` pixels take with cells `cell` pixels large, at
/// most `max` cells, keeping the aspect ratio.
fn fit((width, height): (u32, u32), cell: (u16, u16), max: (u16, u16)) -> (u16, u16) {
    let cols = f64::from(width) / f64::from(cell.0.max(1));
    let rows = f64::from(height) / f64::from(cell.1.max(1));
    let (max_cols, max_rows) = (f64::from(max.0.max(1)), f64::from(max.1.max(1)));
    let scale = 1f64
        .min(max_cols / cols.max(1e-9))
        .min(max_rows / rows.max(1e-9));
    let cells = |value: f64, max: f64| (value * scale).ceil().clamp(1.0, max) as u16;
    (cells(cols, max_cols), cells(rows, max_rows))
}

impl Shared {
    /// The image decoded, and its key.
    fn decode(&mut self, image: &Image) -> Option<(u64, Arc<Decoded>)> {
        let bytes = bytes_of(image)?;
        let key = key_of(&bytes);
        if let Some(decoded) = self.decoded.get(&key) {
            return Some((key, decoded.clone()));
        }
        let format = image::guess_format(&bytes).ok()?;
        let pixels = image::load_from_memory_with_format(&bytes, format).ok()?;
        let format = format.extensions_str().first().copied().unwrap_or("image");
        let decoded = Arc::new(Decoded {
            bytes,
            format,
            pixels,
        });
        self.decoded.insert(key, decoded.clone());
        Some((key, decoded))
    }

    /// The most cells an image takes.
    fn bounds(&self) -> (u16, u16) {
        let percent = u32::from(self.settings.max_width_percent.min(100));
        let width = u32::from(self.width.saturating_sub(INDENT)) * percent / 100;
        (
            u16::try_from(width).unwrap_or(u16::MAX).max(1),
            self.settings.max_rows.max(1),
        )
    }

    /// Adds the images of the call's result to its `lines`, noting where
    /// the rows the terminal draws them over are.
    fn add_images(&mut self, view: &ToolCallView<'_>, lines: &mut Vec<Line<'static>>) {
        let images = view.result.into_iter().flat_map(|result| &result.content);
        let images: Vec<&Image> = images
            .filter_map(|content| match content {
                ToolResultContent::Image(image) => Some(image),
                _ => None,
            })
            .collect();
        let mut reserved = Vec::new();
        let mut keys = Vec::new();
        for (index, image) in images.into_iter().enumerate() {
            let Some((key, decoded)) = self.decode(image) else {
                lines.push(Line::from("  ⎿ [image with no data to show]").dim());
                continue;
            };
            keys.push(key);
            let (cols, rows) = fit(decoded.pixels.dimensions(), self.cell, self.bounds());
            match self.look {
                Look::Line => lines
                    .push(Line::from(format!("  ⎿ [{} — press o to open]", decoded.label())).dim()),
                Look::Blocks => {
                    lines.push(Line::from(format!("  ⎿ {}", decoded.label())).dim());
                    lines.extend(half_blocks(&decoded.pixels, cols, rows));
                }
                Look::Escape(_) => {
                    lines.push(Line::from(format!("  ⎿ {}", decoded.label())).dim());
                    let start = lines.len();
                    lines.extend((0..rows).map(|_| Line::default()));
                    reserved.push((start, key, index, cols, rows));
                }
            }
        }
        let total = lines.len();
        let slots = reserved
            .into_iter()
            .map(|(start, key, index, cols, rows)| Slot {
                key,
                index,
                from_end: total - start,
                cols,
                rows,
            });
        let slots: Vec<Slot> = slots.collect();
        let id = view.call.id.clone();
        if keys.is_empty() {
            self.images_of.remove(&id);
        } else {
            self.images_of.insert(id.clone(), keys);
        }
        if slots.is_empty() {
            self.slots.remove(&id);
        } else {
            self.slots.insert(id, slots);
        }
    }
}

/// `pixels` in `cols`×`rows` cells of `▀`, each the colour of two pixels:
/// its top in the foreground, its bottom in the background.
fn half_blocks(pixels: &DynamicImage, cols: u16, rows: u16) -> Vec<Line<'static>> {
    let scaled = pixels
        .resize_exact(u32::from(cols), u32::from(rows) * 2, FilterType::Triangle)
        .to_rgba8();
    let colour = |x: u32, y: u32| match scaled.get_pixel_checked(x, y) {
        Some(pixel) if pixel.0.get(3).is_some_and(|alpha| *alpha >= 128) => {
            let [r, g, b, _] = pixel.0;
            Color::Rgb(r, g, b)
        }
        _ => Color::Reset,
    };
    (0..u32::from(rows))
        .map(|row| {
            let mut spans = vec![Span::from(" ".repeat(usize::from(INDENT)))];
            spans.extend((0..u32::from(cols)).map(|col| {
                let style = Style::new()
                    .fg(colour(col, row * 2))
                    .bg(colour(col, row * 2 + 1));
                Span::styled("▀", style)
            }));
            Line::from(spans)
        })
        .collect()
}

/// Marks a tool renderer this plugin added, wrapping the tool's own look.
#[derive(Component)]
struct WithImages;

/// Draws every tool's calls as they were drawn, with their images under
/// them: the tool's renderer is disabled and wrapped, and a tool without
/// one gets the default look, so a call of any tool shows its images.
fn wrap_renderer(world: &mut World, tool: &str, images: &Images) {
    let mut renderers = world.query::<(Entity, &ToolRenderer, Has<WithImages>)>();
    let found = renderers
        .iter(world)
        .find(|(_, renderer, _)| renderer.tool == tool)
        .map(|(entity, renderer, ours)| (entity, renderer.render.clone(), ours));
    let inner: Option<Arc<RenderToolCall>> = match found {
        Some((_, _, true)) => return,
        Some((entity, render, false)) => {
            world.entity_mut(entity).insert(Disabled);
            Some(render)
        }
        None => None,
    };
    let images = images.clone();
    let render = move |view: &ToolCallView<'_>| {
        let mut lines = match &inner {
            Some(render) => render(view),
            None => view.default_lines(),
        };
        images.lock().add_images(view, &mut lines);
        lines
    };
    world.spawn((
        Name::new(format!("renderer:{tool}")),
        WithImages,
        ToolRenderer {
            tool: tool.to_owned(),
            render: Arc::new(render),
        },
    ));
}

/// A tool added while the app runs gets its images drawn too.
fn wrap_new_tool(
    added: On<Add<ToolDef>>,
    tools: Query<&ToolDef>,
    images: Res<Images>,
    mut commands: Commands,
) {
    let Ok(tool) = tools.get(added.entity) else {
        return;
    };
    // Before `finish`, the tool's plugin may still add its renderer.
    if !images.lock().ready {
        return;
    }
    let (tool, images) = (tool.0.name.to_string(), images.clone());
    commands.queue(move |world: &mut World| wrap_renderer(world, &tool, &images));
}

/// A placed image: where on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spot {
    x: u16,
    y: u16,
    cols: u16,
    rows: u16,
}

/// An image placed for a call: its key, the call and its index there.
type SpotKey = (u64, CallId, usize);

/// What the terminal shows, as this plugin placed it.
#[derive(Resource, Default)]
struct Placed {
    spots: HashMap<SpotKey, Spot>,
    /// The images the terminal holds (kitty keeps them by id).
    sent: Vec<u64>,
    /// The screen they were placed on.
    screen: Option<TuiScreen>,
}

/// The view of the transcript the images are placed in: its rows on
/// screen, and where its first row is on the screen.
struct Viewport<'a> {
    drawn: &'a DrawnTranscript,
    on_screen: Range<usize>,
    origin: (u16, u16),
    /// A box is open over the transcript: no image shows.
    covered: bool,
}

/// Where each image the renderers reserved rows for is wholly on screen.
fn wanted(slots: &HashMap<CallId, Vec<Slot>>, view: &Viewport<'_>) -> HashMap<SpotKey, Spot> {
    let mut wanted = HashMap::new();
    if view.covered {
        return wanted;
    }
    for (call, slots) in slots {
        let Some(rows) = view.drawn.rows_of(&TranscriptItem::Call(call.clone())) else {
            continue;
        };
        for slot in slots {
            let Some(start) = rows.end.checked_sub(slot.from_end) else {
                continue;
            };
            let end = start + usize::from(slot.rows);
            if start < view.on_screen.start || end > view.on_screen.end {
                continue;
            }
            let row = u16::try_from(start - view.on_screen.start).unwrap_or(u16::MAX);
            let spot = Spot {
                x: view.origin.0.saturating_add(INDENT),
                y: view.origin.1.saturating_add(row),
                cols: slot.cols,
                rows: slot.rows,
            };
            wanted.insert((slot.key, call.clone(), slot.index), spot);
        }
    }
    wanted
}

/// A kitty image id from a key: never 0.
fn image_id(key: u64) -> u32 {
    (key as u32).max(1)
}

/// A kitty placement id for an image of a call.
fn placement_id(call: &CallId, index: usize) -> u32 {
    let mut hasher = DefaultHasher::new();
    (call, index).hash(&mut hasher);
    (hasher.finish() as u32).max(1)
}

/// `sequence` as tmux passes it through to the terminal.
fn for_tmux(sequence: &str, tmux: bool) -> String {
    if !tmux {
        return sequence.to_owned();
    }
    format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b"))
}

/// Moves the cursor to `spot`, keeping where it was.
fn at(spot: &Spot, sequence: &str) -> String {
    format!("\x1b7\x1b[{};{}H{sequence}\x1b8", spot.y + 1, spot.x + 1)
}

/// The kitty sequences that send `png` as image `id`, in chunks.
fn kitty_send(id: u32, png: &[u8], tmux: bool) -> String {
    let data = STANDARD.encode(png);
    let chunks: Vec<&[u8]> = data.as_bytes().chunks(4096).collect();
    let last = chunks.len().saturating_sub(1);
    let mut out = String::new();
    for (n, chunk) in chunks.into_iter().enumerate() {
        let more = u8::from(n < last);
        let chunk = String::from_utf8_lossy(chunk);
        let head = if n == 0 {
            format!("a=t,f=100,t=d,i={id},q=2,m={more}")
        } else {
            format!("m={more}")
        };
        out.push_str(&for_tmux(&format!("\x1b_G{head};{chunk}\x1b\\"), tmux));
    }
    out
}

/// The terminal's writes that take `placed` to `wanted`: images no longer
/// wanted where they are come off, and the rest are placed.
fn writes(
    placed: &mut Placed,
    wanted: HashMap<SpotKey, Spot>,
    shared: &Shared,
    view: &Viewport<'_>,
    graphics: &TerminalGraphics,
) -> String {
    let Look::Escape(protocol) = shared.look else {
        return String::new();
    };
    let mut out = String::new();
    let gone: Vec<(SpotKey, Spot)> = placed
        .spots
        .iter()
        .filter(|(key, spot)| wanted.get(*key) != Some(*spot))
        .map(|(key, spot)| (key.clone(), *spot))
        .collect();
    for ((key, call, index), spot) in gone {
        placed.spots.remove(&(key, call.clone(), index));
        match protocol {
            Protocol::Kitty => {
                let (id, placement) = (image_id(key), placement_id(&call, index));
                let delete = format!("\x1b_Ga=d,d=i,i={id},p={placement},q=2\x1b\\");
                out.push_str(&for_tmux(&delete, graphics.tmux));
            }
            // An inline image is the cells' content: the view overwrote
            // those it drew text in, and the rest are blanked, unless a
            // box over the transcript covers them.
            Protocol::Iterm2 if !view.covered => out.push_str(&erase(&spot, view)),
            Protocol::Iterm2 => {}
        }
    }
    for ((key, call, index), spot) in wanted {
        if placed.spots.get(&(key, call.clone(), index)) == Some(&spot) {
            continue;
        }
        let Some(decoded) = shared.decoded.get(&key) else {
            continue;
        };
        match protocol {
            Protocol::Kitty => {
                let id = image_id(key);
                if !placed.sent.contains(&key) {
                    out.push_str(&kitty_send(id, &decoded.png(), graphics.tmux));
                    placed.sent.push(key);
                }
                let placement = placement_id(&call, index);
                let (cols, rows) = (spot.cols, spot.rows);
                let place =
                    format!("\x1b_Ga=p,i={id},p={placement},c={cols},r={rows},C=1,q=2\x1b\\");
                out.push_str(&at(&spot, &for_tmux(&place, graphics.tmux)));
            }
            Protocol::Iterm2 => {
                let data = STANDARD.encode(&decoded.bytes);
                let size = decoded.bytes.len();
                let (cols, rows) = (spot.cols, spot.rows);
                let file = format!(
                    "\x1b]1337;File=inline=1;size={size};width={cols};height={rows};preserveAspectRatio=1:{data}\x07"
                );
                out.push_str(&at(&spot, &for_tmux(&file, graphics.tmux)));
            }
        }
        placed.spots.insert((key, call, index), spot);
    }
    out
}

/// Blanks the cells of `spot` past the text the transcript now draws on
/// each of its rows.
fn erase(spot: &Spot, view: &Viewport<'_>) -> String {
    let mut out = String::new();
    for y in spot.y..spot.y.saturating_add(spot.rows) {
        let row = usize::from(y.saturating_sub(view.origin.1)) + view.on_screen.start;
        let drawn = view
            .drawn
            .rows()
            .get(row)
            .filter(|_| view.on_screen.contains(&row));
        let text = drawn.map_or(0, |row| row.text.chars().count());
        let from = spot.x.max(
            view.origin
                .0
                .saturating_add(u16::try_from(text).unwrap_or(u16::MAX)),
        );
        let end = spot.x.saturating_add(spot.cols);
        if from < end {
            out.push_str(&at(
                &Spot {
                    x: from,
                    y,
                    ..*spot
                },
                &format!("\x1b[{}X", end - from),
            ));
        }
    }
    out
}

/// Where the transcript starts on the screen: below the panels above it
/// and right of those left of it.
fn origin(panels: &Query<(&TuiPanel, &PanelCanvas)>) -> (u16, u16) {
    let (mut x, mut y) = (0, 0);
    for (panel, canvas) in panels {
        let area = canvas.area();
        if area.is_empty() {
            continue;
        }
        match panel.placement {
            Placement::Top(_) => y = y.max(area.bottom()),
            Placement::Left(_) => x = x.max(area.right()),
            _ => {}
        }
    }
    (x, y)
}

/// After each frame, places the images on screen and takes off those that
/// left it, through the view's writer.
fn place_images(
    images: Res<Images>,
    graphics: Res<TerminalGraphics>,
    (drawn, screen, input): (Res<DrawnTranscript>, Res<TuiScreen>, Res<TuiInputState>),
    panels: Query<(&TuiPanel, &PanelCanvas)>,
    mut placed: ResMut<Placed>,
    mut terminal: MessageWriter<TerminalWrite>,
) {
    // A resize clears the screen: everything is placed again.
    if placed.screen != Some(*screen) {
        placed.spots.clear();
        placed.screen = Some(*screen);
    }
    let over = panels.iter().any(|(panel, canvas)| {
        matches!(panel.placement, Placement::Over { .. }) && !canvas.area().is_empty()
    });
    let view = Viewport {
        drawn: &drawn,
        on_screen: drawn.on_screen(),
        origin: origin(&panels),
        covered: over
            || input.picking().is_some()
            || input.answering().is_some()
            || input.completion().is_some(),
    };
    let shared = images.lock();
    let wanted = wanted(&shared.slots, &view);
    let out = writes(&mut placed, wanted, &shared, &view, &graphics);
    if !out.is_empty() {
        terminal.write(TerminalWrite(out.into_bytes()));
    }
}

/// Takes the images off the screen as the app exits.
fn clear_on_exit(
    placed: Res<Placed>,
    graphics: Res<TerminalGraphics>,
    mut terminal: MessageWriter<TerminalWrite>,
) {
    if graphics.protocol != Some(Protocol::Kitty) {
        return;
    }
    let deletes: String = placed
        .sent
        .iter()
        .map(|key| {
            for_tmux(
                &format!("\x1b_Ga=d,d=I,i={},q=2\x1b\\", image_id(*key)),
                graphics.tmux,
            )
        })
        .collect();
    if !deletes.is_empty() {
        terminal.write(TerminalWrite(deletes.into_bytes()));
    }
}

/// Lays the transcript out again for the terminal's new width.
fn follow_width(
    screen: Res<TuiScreen>,
    images: Res<Images>,
    mut renderers: Query<&mut ToolRenderer, With<WithImages>>,
) {
    let width = screen.0.width;
    let mut shared = images.lock();
    if shared.width == width || width == 0 {
        return;
    }
    shared.width = width;
    // A changed renderer has the view lay its calls out again.
    for mut renderer in &mut renderers {
        renderer.set_changed();
    }
}

/// The selected item's first image, saved where the session keeps pasted
/// images, by its key.
fn selected_image(world: &mut World) -> Result<PathBuf, String> {
    let selected = world.resource::<TranscriptView>().selected.clone();
    let images = world.resource::<Images>().clone();
    let decoded = match selected {
        Some(TranscriptItem::Call(call)) => {
            let shared = images.lock();
            let key = shared.images_of.get(&call).and_then(|keys| keys.first());
            key.and_then(|key| shared.decoded.get(key).cloned())
        }
        Some(TranscriptItem::Message(id)) => {
            let shown = world.get_resource::<Shown>().and_then(|shown| shown.0);
            let conversation = shown.and_then(|agent| world.get::<Conversation>(agent));
            let message = conversation.and_then(|conversation| {
                let at = conversation.index_of(id)?;
                conversation.messages().get(at).cloned()
            });
            let first = message.as_ref().and_then(|message| {
                message.images().next().cloned().or_else(|| match message {
                    ChatMessage::User { content } => content.iter().find_map(|item| match item {
                        UserContent::Image(image) => Some(image.clone()),
                        _ => None,
                    }),
                    ChatMessage::Assistant(_) | ChatMessage::System { .. } => None,
                })
            });
            first.and_then(|image| images.lock().decode(&image).map(|(_, decoded)| decoded))
        }
        _ => None,
    };
    let decoded = decoded.ok_or_else(|| "The selected item has no image.".to_owned())?;
    let directory = world.get_resource::<Scratch>().and_then(Scratch::images);
    let directory = directory.ok_or_else(|| "This session keeps no files.".to_owned())?;
    std::fs::create_dir_all(&directory).map_err(|failure| failure.to_string())?;
    let path = directory.join(format!(
        "{:016x}.{}",
        key_of(&decoded.bytes),
        decoded.format
    ));
    if !path.exists() {
        std::fs::write(&path, &decoded.bytes).map_err(|failure| failure.to_string())?;
    }
    Ok(path)
}

/// `o`: opens the selected item's image in the system viewer.
fn open_image(_: On<KeyPressed>, mut commands: Commands) {
    commands.queue(|world: &mut World| {
        let opened = selected_image(world).and_then(|path| {
            let opener = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            let launcher = world.get_resource::<Processes>().map(Processes::launcher);
            let mut command = launcher.unwrap_or_default().command(opener);
            command.arg(&path);
            command
                .spawn()
                .map(drop)
                .map_err(|failure| format!("{opener}: {failure}"))
        });
        if let Err(why) = opened {
            world.write_message(Notice::warn(None, why));
        }
    });
}

/// `y`: copies the path of the selected item's image.
fn copy_path(_: On<KeyPressed>, mut commands: Commands) {
    commands.queue(|world: &mut World| match selected_image(world) {
        Ok(path) => {
            world.write_message(CopyToClipboard(path.display().to_string()));
        }
        Err(why) => {
            world.write_message(Notice::warn(None, why));
        }
    });
}

/// Draws the images of tool results in the terminal view's transcript.
#[derive(Default)]
pub struct InlineImagesPlugin;

impl Plugin for InlineImagesPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ImagesSettings>("images");
        if !app.world().contains_resource::<TerminalGraphics>() {
            let launcher = app
                .world()
                .get_resource::<Processes>()
                .map(Processes::launcher);
            let graphics = TerminalGraphics::detect(&launcher.unwrap_or_default());
            app.insert_resource(graphics);
        }
        app.init_resource::<Images>()
            .init_resource::<Placed>()
            .init_resource::<TuiScreen>()
            .init_resource::<TuiInputState>()
            .add_message::<TerminalWrite>()
            .add_message::<CopyToClipboard>()
            .add_observer(wrap_new_tool)
            .add_systems(PostUpdate, follow_width.in_set(TuiSystems::Prepare))
            .add_systems(Last, clear_on_exit.in_set(OnAppExitSystems));
        let open = KeyBinding::new(KeyPress::key(KeyCode::Char('o')), "Open the image");
        app.add_key_binding(open.in_scope(KeyScope::Transcript), open_image);
        let copy = KeyBinding::new(KeyPress::key(KeyCode::Char('y')), "Copy the image's path");
        app.add_key_binding(copy.in_scope(KeyScope::Transcript), copy_path);
    }

    fn finish(&self, app: &mut App) {
        let settings = app.world().resource::<ImagesSettings>().clone();
        let graphics = *app.world().resource::<TerminalGraphics>();
        let look = match (settings.mode, graphics.protocol) {
            (Mode::Off, _) => Look::Line,
            (Mode::Blocks, _) | (Mode::Auto, None) => Look::Blocks,
            (Mode::Auto, Some(protocol)) => Look::Escape(protocol),
        };
        let images = app.world().resource::<Images>().clone();
        {
            let mut shared = images.lock();
            shared.ready = true;
            shared.look = look;
            shared.cell = graphics.cell;
            shared.settings = settings;
            shared.width = app.world().resource::<TuiScreen>().0.width.max(80);
        }
        if let Look::Escape(_) = look {
            app.init_resource::<DrawnTranscript>().add_systems(
                PostUpdate,
                place_images
                    .after(TuiSystems::Render)
                    .run_if(resource_exists::<DrawnTranscript>),
            );
        }
        let world = app.world_mut();
        let mut tools = world.query::<&ToolDef>();
        let names: Vec<String> = tools
            .iter(world)
            .map(|tool| tool.0.name.to_string())
            .collect();
        for name in names {
            wrap_renderer(world, &name, &images);
        }
    }
}

#[cfg(test)]
mod tests;
