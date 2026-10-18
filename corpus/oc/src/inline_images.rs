//! OC-17: images in the terminal transcript. Every image in the transcript
//! (the user's, the model's, and those in tool results) is drawn below its
//! message or call, scaled to at most [`ImagesSettings::max_width_percent`]
//! of the terminal's width and [`ImagesSettings::max_rows`] rows, keeping
//! its aspect ratio:
//!
//! - with a terminal that shows images ([`TerminalGraphics`]: the kitty
//!   graphics protocol, iTerm2's inline images or sixel, inside tmux only
//!   with its passthrough on), the transcript keeps blank rows, and after
//!   each frame the image is placed over the rows of them on screen, and
//!   taken off once they scroll away, the screen is resized or a box opens
//!   over it;
//! - otherwise half-block characters coloured as the image
//!   ([`Mode::Blocks`]), or with [`Mode::Off`] one line saying what it is.
//!
//! A message's images are drawn by a content renderer and a call's by a
//! call decorator. `o` on a selected image or call opens the image in the
//! system viewer and `y` copies the path it was saved to. The images are
//! drawn from the conversation, so a resumed session draws them again.
//!
//! The plugin is a view half only: `plugins.toml` names it as an entry's
//! `view`, and it draws the mirrored conversation's images.

use std::collections::HashMap;
use std::fmt::Write as _;
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
use rig_harness::rig_core::completion::{AssistantContent, Message as ChatMessage};
use rig_harness::rig_core::message::{
    CallId, DocumentSourceKind, Image, ToolResultContent, UserContent,
};
use rig_harness::rig_tools::process::Launcher;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::layout::Rect;
use rig_tui::ratatui::style::{Color, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{
    AppKeyBindingsExt, AppToolRenderersExt, CallDecorator, ContentBlock, ContentKind,
    CopyToClipboard, DrawnTranscript, KeyBinding, KeyPress, KeyPressed, KeyScope, MainPane,
    RedrawArea, TerminalSize, TerminalWrite, ToolCallView, TranscriptItem, TranscriptView,
    TuiFrame, TuiSystems,
};
use serde::Deserialize;

/// Columns an image is indented by in the transcript.
const INDENT: u16 = 4;

/// The name of this plugin's call decorator.
const DECORATOR: &str = "images";

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
    /// DEC sixel graphics (foot, mlterm, contour).
    Sixel,
}

impl Protocol {
    /// Whether the image becomes the cells' content, which stays on screen
    /// until something is written over it.
    fn in_cells(self) -> bool {
        self != Self::Kitty
    }
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
    /// Terminals are not queried: the view reads every reply as input.
    pub fn detect(launcher: &Launcher) -> Self {
        let var = |name: &str| std::env::var(name).unwrap_or_default();
        let (term, program) = (var("TERM"), var("TERM_PROGRAM"));
        let kitty = !var("KITTY_WINDOW_ID").is_empty()
            || term.contains("kitty")
            || matches!(program.as_str(), "WezTerm" | "ghostty");
        let iterm = program == "iTerm.app" || var("LC_TERMINAL") == "iTerm2";
        let sixel = ["foot", "mlterm", "contour"]
            .iter()
            .any(|name| term.starts_with(name));
        let in_tmux = !var("TMUX").is_empty();
        let passes = in_tmux && tmux_passes_through(launcher);
        let protocol = match (kitty, iterm, sixel) {
            _ if in_tmux && !passes => None,
            (true, _, _) => Some(Protocol::Kitty),
            (false, true, _) => Some(Protocol::Iterm2),
            (false, false, true) => Some(Protocol::Sixel),
            (false, false, false) => None,
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

/// What an image's look depends on: the settings, the terminal, its
/// width and the transcript's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Params {
    look: Look,
    cell: (u16, u16),
    /// The most cells an image takes, columns and rows.
    bounds: (u16, u16),
}

impl Params {
    /// With `widths` the terminal's and the transcript's, which side
    /// panels may narrow (0 before the first frame).
    fn of(settings: &ImagesSettings, graphics: &TerminalGraphics, widths: (u16, u16)) -> Self {
        let look = match (settings.mode, graphics.protocol) {
            (Mode::Off, _) => Look::Line,
            (Mode::Blocks, _) | (Mode::Auto, None) => Look::Blocks,
            (Mode::Auto, Some(protocol)) => Look::Escape(protocol),
        };
        let percent = u32::from(settings.max_width_percent.min(100));
        let cols = u32::from(widths.0.saturating_sub(INDENT)) * percent / 100;
        let cols = match widths.1 {
            0 => cols,
            transcript => cols.min(u32::from(transcript.saturating_sub(INDENT))),
        };
        let bounds = (
            u16::try_from(cols).unwrap_or(u16::MAX).max(1),
            settings.max_rows.max(1),
        );
        Self {
            look,
            cell: graphics.cell,
            bounds,
        }
    }
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

    /// The image as PNG, which kitty and iTerm2 take.
    fn png(&self, pixels: Option<&DynamicImage>) -> Vec<u8> {
        if self.format == "png" && pixels.is_none() {
            return self.bytes.clone();
        }
        let mut png = Vec::new();
        (pixels.unwrap_or(&self.pixels))
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .ok();
        png
    }
}

/// An image as the transcript draws it.
struct Fitted {
    key: u64,
    decoded: Arc<Decoded>,
    /// Its size in cells.
    cols: u16,
    rows: u16,
}

/// What the renderers, which run inside the view's layout with no world,
/// share with the plugin's systems: the look to draw and the decoded
/// images.
#[derive(Default)]
struct Shared {
    params: Params,
    decoded: HashMap<u64, Arc<Decoded>>,
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
    /// The image decoded and sized, if the conversation holds its bytes
    /// and they decode.
    fn fit(&mut self, image: &Image) -> Option<Fitted> {
        let bytes = match &image.data {
            DocumentSourceKind::Base64(data) => STANDARD.decode(data).ok()?,
            DocumentSourceKind::Raw(bytes) => bytes.clone(),
            _ => return None,
        };
        let key = key_of(&bytes);
        let decoded = match self.decoded.get(&key) {
            Some(decoded) => decoded.clone(),
            None => {
                let format = image::guess_format(&bytes).ok()?;
                let pixels = image::load_from_memory_with_format(&bytes, format).ok()?;
                let format = format.extensions_str().first().copied().unwrap_or("image");
                let decoded = Arc::new(Decoded {
                    bytes,
                    format,
                    pixels,
                });
                self.decoded.insert(key, decoded.clone());
                decoded
            }
        };
        let (cols, rows) = fit(
            decoded.pixels.dimensions(),
            self.params.cell,
            self.params.bounds,
        );
        Some(Fitted {
            key,
            decoded,
            cols,
            rows,
        })
    }

    /// The lines of an image, and the image when the terminal draws it
    /// over the blank rows after the first line.
    fn lines(&mut self, image: &Image) -> (Vec<Line<'static>>, Option<Fitted>) {
        let Some(fitted) = self.fit(image) else {
            return (
                vec![Line::from("  [image with no data to show]").dim()],
                None,
            );
        };
        let label = fitted.decoded.label();
        match self.params.look {
            Look::Line => {
                let line = Line::from(format!("  [{label} — press o to open]")).dim();
                (vec![line], None)
            }
            Look::Blocks => {
                let mut lines = vec![Line::from(format!("  {label}")).dim()];
                lines.extend(half_blocks(
                    &fitted.decoded.pixels,
                    fitted.cols,
                    fitted.rows,
                ));
                (lines, None)
            }
            Look::Escape(_) => {
                let mut lines = vec![Line::from(format!("  {label}")).dim()];
                lines.extend((0..fitted.rows).map(|_| Line::default()));
                (lines, Some(fitted))
            }
        }
    }

    /// The lines of `images` one after another, and where among them each
    /// one the terminal draws starts.
    fn all_lines<'a>(
        &mut self,
        images: impl IntoIterator<Item = &'a Image>,
    ) -> (Vec<Line<'static>>, Vec<(usize, Fitted)>) {
        let (mut lines, mut drawn) = (Vec::new(), Vec::new());
        for image in images {
            let (more, fitted) = self.lines(image);
            if let Some(fitted) = fitted {
                drawn.push((lines.len() + 1, fitted));
            }
            lines.extend(more);
        }
        (lines, drawn)
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

/// The images of a tool result.
fn result_images(content: &[ToolResultContent]) -> impl Iterator<Item = &Image> {
    content.iter().filter_map(|content| match content {
        ToolResultContent::Image(image) => Some(image),
        _ => None,
    })
}

/// The images of the result of `call` in `conversation`.
fn images_of_call<'a>(conversation: &'a Conversation, call: &CallId) -> Vec<&'a Image> {
    let result = conversation.messages().iter().rev().find_map(|message| {
        let ChatMessage::User { content } = &**message else {
            return None;
        };
        content.iter().find_map(|content| match content {
            UserContent::ToolResult(result) if result.call == *call => Some(result),
            _ => None,
        })
    });
    result.map_or_else(Vec::new, |result| result_images(&result.content).collect())
}

/// The image that is the `index`-th block of the message `id`.
fn image_in(conversation: &Conversation, id: MessageId, index: usize) -> Option<&Image> {
    let message = conversation.messages().get(conversation.index_of(id)?)?;
    match &**message {
        ChatMessage::User { content } => match content.get(index)? {
            UserContent::Image(image) => Some(image),
            _ => None,
        },
        ChatMessage::Assistant(assistant) => match assistant.content.get(index)? {
            AssistantContent::Image(image) => Some(image),
            _ => None,
        },
        ChatMessage::System { .. } => None,
    }
}

/// The first row of each line among `rows`, as the view wrapped them.
fn line_starts(drawn: &DrawnTranscript, rows: &Range<usize>) -> Vec<usize> {
    let wrapped = drawn.rows().get(rows.clone()).unwrap_or_default();
    let starts = wrapped.iter().enumerate();
    starts
        .filter(|(n, row)| *n == 0 || row.continues.is_none())
        .map(|(n, _)| rows.start + n)
        .collect()
}

/// The images an item of the transcript draws, each with its first row:
/// a message's image block, or this plugin's decoration of a call.
fn drawn_in(
    shared: &mut Shared,
    (conversation, drawn): (&Conversation, &DrawnTranscript),
    item: &TranscriptItem,
    rows: &Range<usize>,
) -> Vec<(usize, Fitted)> {
    let images = match item {
        TranscriptItem::Content(id, index) => {
            image_in(conversation, *id, *index).into_iter().collect()
        }
        TranscriptItem::Decoration(call, name) if name == DECORATOR => {
            images_of_call(conversation, call)
        }
        _ => return Vec::new(),
    };
    let (_, fitted) = shared.all_lines(images);
    let starts = line_starts(drawn, rows);
    let at_row = |(line, fitted): (usize, Fitted)| Some((*starts.get(line)?, fitted));
    fitted.into_iter().filter_map(at_row).collect()
}

/// Has the main transcript pane keep the rows it draws, which images are
/// placed over.
fn keep_drawn(
    panes: Query<Entity, (With<MainPane>, Without<DrawnTranscript>)>,
    mut commands: Commands,
) {
    for pane in &panes {
        commands.entity(pane).insert(DrawnTranscript::default());
    }
}

/// Keeps the look of images in step with the settings, the terminal and
/// its width, and draws the transcript again when it changes.
fn follow_look(
    images: Res<Images>,
    (settings, graphics): (Res<ImagesSettings>, Res<TerminalGraphics>),
    (screen, view): (
        Res<TerminalSize>,
        Option<Single<&TranscriptView, With<MainPane>>>,
    ),
    mut decorators: Query<&mut CallDecorator>,
) {
    let width = match screen.0.width {
        0 => 80,
        width => width,
    };
    let transcript = view.and_then(|view| view.width()).unwrap_or(0);
    let params = Params::of(&settings, &graphics, (width, transcript));
    let mut shared = images.lock();
    if shared.params == params {
        return;
    }
    shared.params = params;
    // A changed look draws the whole transcript again, images and all.
    for mut decorator in &mut decorators {
        if decorator.name == DECORATOR {
            decorator.set_changed();
        }
    }
}

/// An image placed on screen: which one, and where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spot {
    key: u64,
    x: u16,
    y: u16,
    cols: u16,
    rows: u16,
    /// Its rows above the screen's top, cut off.
    skip: u16,
    /// Its rows on screen.
    shown: u16,
}

impl Spot {
    fn area(&self) -> Rect {
        Rect::new(self.x, self.y, self.cols, self.shown)
    }
}

/// An image in the transcript: the item drawing it and its place there.
type At = (TranscriptItem, usize);

/// What the terminal shows, as this plugin placed it.
#[derive(Resource, Default)]
struct Placed {
    spots: HashMap<At, Spot>,
    /// The images the terminal holds (kitty keeps them by id).
    sent: Vec<u64>,
    /// The screen they were placed on.
    screen: Option<Rect>,
    /// Whether an image waits for the cells under it to be written again.
    deferred: bool,
}

/// Where each image the terminal draws is on screen, at least in part
/// and with no overlay over it.
fn wanted(
    shared: &mut Shared,
    conversation: &Conversation,
    drawn: &DrawnTranscript,
    frame: &TuiFrame,
) -> HashMap<At, (Spot, Arc<Decoded>)> {
    let on_screen = drawn.on_screen();
    let mut wanted = HashMap::new();
    for (item, rows) in drawn.items() {
        if rows.end <= on_screen.start || rows.start >= on_screen.end {
            continue;
        }
        for (index, (start, fitted)) in drawn_in(shared, (conversation, drawn), item, rows)
            .into_iter()
            .enumerate()
        {
            let image = start..start + usize::from(fitted.rows);
            let visible = image.start.max(on_screen.start)..image.end.min(on_screen.end);
            let Some(y) = frame.screen_row(drawn, visible.start) else {
                continue;
            };
            let spot = Spot {
                key: fitted.key,
                x: drawn.area().x.saturating_add(INDENT),
                y,
                cols: fitted.cols,
                rows: fitted.rows,
                skip: u16::try_from(visible.start - image.start).unwrap_or(u16::MAX),
                shown: u16::try_from(visible.len()).unwrap_or(u16::MAX),
            };
            if frame.covered(spot.area()) {
                continue;
            }
            wanted.insert((item.clone(), index), (spot, fitted.decoded));
        }
    }
    wanted
}

/// A kitty image id from a key: never 0.
fn image_id(key: u64) -> u32 {
    (key as u32).max(1)
}

/// A kitty placement id for an image in the transcript.
fn placement_id(at: &At) -> u32 {
    let mut hasher = DefaultHasher::new();
    at.hash(&mut hasher);
    (hasher.finish() as u32).max(1)
}

/// `sequence` as tmux passes it through to the terminal.
fn for_tmux(sequence: &str, tmux: bool) -> String {
    if !tmux {
        return sequence.to_owned();
    }
    format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b"))
}

/// Moves the cursor to `(x, y)` for `sequence`, keeping where it was.
fn at((x, y): (u16, u16), sequence: &str) -> String {
    format!("\x1b7\x1b[{};{}H{sequence}\x1b8", y + 1, x + 1)
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

/// The image's pixel rows shown in `spot`: their first and how many.
fn source_rows(spot: &Spot, height: u32) -> (u32, u32) {
    let rows = u32::from(spot.rows.max(1));
    let top = height * u32::from(spot.skip) / rows;
    let shown = (height * u32::from(spot.shown)).div_ceil(rows).max(1);
    (top, shown.min(height.saturating_sub(top)).max(1))
}

/// `pixels` as a sixel image `cols`×`rows` cells of `cell` pixels, in a
/// 6×6×6 colour cube.
fn sixel(pixels: &DynamicImage, cell: (u16, u16), (cols, rows): (u16, u16)) -> String {
    let (width, height) = (
        u32::from(cols) * u32::from(cell.0),
        u32::from(rows) * u32::from(cell.1),
    );
    let scaled = pixels
        .resize_exact(width.max(1), height.max(1), FilterType::Triangle)
        .to_rgba8();
    let mut out = format!("\x1bPq\"1;1;{width};{height}");
    for colour in 0..216u16 {
        let (r, g, b) = (colour / 36, colour / 6 % 6, colour % 6);
        write!(out, "#{colour};2;{};{};{}", r * 20, g * 20, b * 20).ok();
    }
    let cube = |value: u8| u16::from(value) * 5 / 255;
    for band in (0..height).step_by(6) {
        // Each colour's sixels across the band.
        let mut colours: HashMap<u16, Vec<u8>> = HashMap::new();
        for x in 0..width {
            for dy in 0..6 {
                let Some(pixel) = scaled.get_pixel_checked(x, band + dy) else {
                    continue;
                };
                let [r, g, b, alpha] = pixel.0;
                if alpha < 128 {
                    continue;
                }
                let colour = cube(r) * 36 + cube(g) * 6 + cube(b);
                let sixels = colours
                    .entry(colour)
                    .or_insert_with(|| vec![0; width as usize]);
                if let Some(sixel) = sixels.get_mut(x as usize) {
                    *sixel |= 1 << dy;
                }
            }
        }
        let mut colours: Vec<(u16, Vec<u8>)> = colours.into_iter().collect();
        colours.sort_unstable_by_key(|(colour, _)| *colour);
        for (n, (colour, sixels)) in colours.into_iter().enumerate() {
            if n > 0 {
                out.push('$');
            }
            write!(out, "#{colour}").ok();
            for run in sixels.chunk_by(|a, b| a == b) {
                let char = char::from(63 + run.first().copied().unwrap_or(0));
                match run.len() {
                    count @ 4.. => write!(out, "!{count}{char}").ok(),
                    count => write!(out, "{}", String::from(char).repeat(count)).ok(),
                };
            }
        }
        out.push('-');
    }
    out.push_str("\x1b\\");
    out
}

/// The sequence that draws `decoded` in `spot` with a protocol that puts
/// it in the cells, its rows cut off the screen left out.
fn in_cells(protocol: Protocol, decoded: &Decoded, spot: &Spot, cell: (u16, u16)) -> String {
    let (width, height) = decoded.pixels.dimensions();
    let cut = (spot.shown < spot.rows).then(|| {
        let (top, rows) = source_rows(spot, height);
        decoded.pixels.crop_imm(0, top, width, rows)
    });
    match protocol {
        Protocol::Sixel => sixel(
            cut.as_ref().unwrap_or(&decoded.pixels),
            cell,
            (spot.cols, spot.shown),
        ),
        Protocol::Iterm2 | Protocol::Kitty => {
            let png = decoded.png(cut.as_ref());
            let (cols, rows, size) = (spot.cols, spot.shown, png.len());
            let data = STANDARD.encode(&png);
            format!(
                "\x1b]1337;File=inline=1;size={size};width={cols};height={rows};preserveAspectRatio=0:{data}\x07"
            )
        }
    }
}

/// The terminal's writes that take `placed` to `wanted`, and the areas
/// to write again: images no longer wanted where they are come off, and
/// the rest are placed.
fn writes(
    placed: &mut Placed,
    wanted: HashMap<At, (Spot, Arc<Decoded>)>,
    protocol: Protocol,
    graphics: &TerminalGraphics,
) -> (String, Vec<Rect>) {
    let (mut out, mut repaint) = (String::new(), Vec::new());
    let gone: Vec<(At, Spot)> = placed
        .spots
        .iter()
        .filter(|(at, spot)| wanted.get(*at).map(|(want, _)| want) != Some(*spot))
        .map(|(at, spot)| (at.clone(), *spot))
        .collect();
    for (at, spot) in gone {
        placed.spots.remove(&at);
        if protocol.in_cells() {
            repaint.push(spot.area());
            continue;
        }
        let (id, placement) = (image_id(spot.key), placement_id(&at));
        let delete = format!("\x1b_Ga=d,d=i,i={id},p={placement},q=2\x1b\\");
        out.push_str(&for_tmux(&delete, graphics.tmux));
    }
    for (at, (spot, decoded)) in wanted {
        if placed.spots.get(&at) == Some(&spot) {
            continue;
        }
        // The cells repainted after the next frame would cover it: it is
        // placed after that frame.
        if repaint.iter().any(|area| area.intersects(spot.area())) {
            placed.deferred = true;
            continue;
        }
        let sequence = if protocol.in_cells() {
            in_cells(protocol, &decoded, &spot, graphics.cell)
        } else {
            let id = image_id(spot.key);
            if !placed.sent.contains(&spot.key) {
                out.push_str(&kitty_send(id, &decoded.png(None), graphics.tmux));
                placed.sent.push(spot.key);
            }
            let (width, height) = decoded.pixels.dimensions();
            let (top, rows) = source_rows(&spot, height);
            let (placement, cols, shown) = (placement_id(&at), spot.cols, spot.shown);
            format!(
                "\x1b_Ga=p,i={id},p={placement},x=0,y={top},w={width},h={rows},c={cols},r={shown},C=1,q=2\x1b\\"
            )
        };
        out.push_str(&at_spot(&spot, &for_tmux(&sequence, graphics.tmux)));
        placed.spots.insert(at, spot);
    }
    (out, repaint)
}

/// Moves the cursor to `spot`'s top left for `sequence`.
fn at_spot(spot: &Spot, sequence: &str) -> String {
    at((spot.x, spot.y), sequence)
}

/// After each drawn frame that moved the main pane's rows or the panels,
/// or left an image waiting, places the images on screen and takes off
/// those that left it, through the view's writer; the cells an image drawn
/// in them leaves are written again by the view.
fn place_images(
    images: Res<Images>,
    graphics: Res<TerminalGraphics>,
    (drawn, frame): (Single<Ref<DrawnTranscript>, With<MainPane>>, Res<TuiFrame>),
    conversations: Query<&Conversation>,
    mut placed: ResMut<Placed>,
    (mut terminal, mut repaints): (MessageWriter<TerminalWrite>, MessageWriter<RedrawArea>),
) {
    if !drawn.is_changed() && !frame.is_changed() && !placed.deferred {
        return;
    }
    placed.deferred = false;
    // A resize clears the screen: everything is placed again.
    if placed.screen != Some(frame.screen) {
        placed.spots.clear();
        placed.screen = Some(frame.screen);
    }
    let Some(protocol) = graphics.protocol else {
        return;
    };
    let mut shared = images.lock();
    // With another look now, such as after a settings change, every
    // placed image comes off.
    let conversation = drawn
        .agent()
        .and_then(|agent| conversations.get(agent).ok());
    let wanted = match (shared.params.look, conversation) {
        (Look::Escape(_), Some(conversation)) => wanted(&mut shared, conversation, &drawn, &frame),
        _ => HashMap::new(),
    };
    let (out, repaint) = writes(&mut placed, wanted, protocol, &graphics);
    if !out.is_empty() {
        terminal.write(TerminalWrite(out.into_bytes()));
    }
    repaints.write_batch(repaint.into_iter().map(RedrawArea));
}

/// Takes kitty's images off the screen as the app exits.
fn clear_on_exit(
    placed: Res<Placed>,
    graphics: Res<TerminalGraphics>,
    mut terminal: MessageWriter<TerminalWrite>,
) {
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

/// The main pane's selected item's first image, saved where the session
/// keeps pasted images, by its key.
fn selected_image(world: &mut World) -> Result<PathBuf, String> {
    let mut panes = world.query_filtered::<(&TranscriptView, &DrawnTranscript), With<MainPane>>();
    let pane = panes.single(world).ok();
    let selected = pane.and_then(|(view, _)| view.selected.clone());
    let agent = pane.and_then(|(_, drawn)| drawn.agent());
    let conversation = agent.and_then(|agent| world.get::<Conversation>(agent));
    let image = match (selected, conversation) {
        (Some(TranscriptItem::Content(id, index)), Some(conversation)) => {
            image_in(conversation, id, index).cloned()
        }
        (Some(TranscriptItem::Call(call)), Some(conversation)) => {
            images_of_call(conversation, &call)
                .first()
                .copied()
                .cloned()
        }
        _ => None,
    };
    let images = world.resource::<Images>().clone();
    let fitted = image.and_then(|image| images.lock().fit(&image));
    let fitted = fitted.ok_or_else(|| "The selected item has no image.".to_owned())?;
    let directory = world.get_resource::<Scratch>().and_then(Scratch::images);
    let directory = directory.ok_or_else(|| "This session keeps no files.".to_owned())?;
    std::fs::create_dir_all(&directory).map_err(|failure| failure.to_string())?;
    let path = directory.join(format!("{:016x}.{}", fitted.key, fitted.decoded.format));
    if !path.exists() {
        std::fs::write(&path, &fitted.decoded.bytes).map_err(|failure| failure.to_string())?;
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

/// Draws the images of the transcript in the terminal view.
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
        let images = Images::default();
        let (content, call) = (images.clone(), images.clone());
        app.insert_resource(images)
            .init_resource::<Placed>()
            .init_resource::<TuiFrame>()
            .init_resource::<TerminalSize>()
            .add_message::<TerminalWrite>()
            .add_message::<RedrawArea>()
            .add_message::<CopyToClipboard>()
            .add_content_renderer(ContentKind::Image, move |view| match view.block {
                ContentBlock::Image(image) => content.lock().lines(image).0,
                _ => view.default_lines(),
            })
            .add_call_decorator(DECORATOR, move |view: &ToolCallView<'_>| {
                let content = view
                    .result
                    .map_or(&[][..], |result| result.content.as_slice());
                call.lock().all_lines(result_images(content)).0
            })
            .add_systems(
                PostUpdate,
                (
                    (keep_drawn, follow_look).in_set(TuiSystems::Prepare),
                    place_images.after(TuiSystems::Render),
                ),
            )
            .add_systems(
                Last,
                clear_on_exit
                    .in_set(OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            );
        let open = KeyBinding::new(KeyPress::key(KeyCode::Char('o')), "Open the image");
        app.add_key_binding(open.in_scope(KeyScope::Transcript), open_image);
        let copy = KeyBinding::new(KeyPress::key(KeyCode::Char('y')), "Copy the image's path");
        app.add_key_binding(copy.in_scope(KeyScope::Transcript), copy_path);
    }
}

#[cfg(test)]
mod tests;
