//! CT-20: paste chips. A long paste in the terminal view's input becomes a
//! chip, `[Pasted #1 · 412 lines]`, instead of its text; a pasted image (a
//! dropped image file, or the clipboard's image on Ctrl+V) becomes
//! `[Image #2 · 1280×720]`. A chip moves and deletes as one character,
//! Alt+E on it opens it in place for editing and closes it again, keeping
//! the edits, and an identical paste right after a chip may be merged into
//! it as `×2` if the user says so. When the message is sent, each text
//! chip becomes its contents in a fence, and each image chip its file as
//! `@path`, which the agent's attachments (`rig_basics`' `AttachPlugin`)
//! send as an image; Enter on a message with an image warns first when the
//! model reads no images.
//!
//! The input is plain text (an [`Editor`]), so a chip is its token in the
//! text, which the plugin's [`PasteChips`] maps to its contents. The
//! plugin is a view half only: `plugins.toml` names it as an entry's
//! `view`, and the view, which outlives a reload, keeps the chips.

use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use rig_harness::prelude::*;
use rig_harness::rig_tools::attach;
use rig_harness::rig_tools::clipboard;
use rig_harness::rig_tools::process::Launcher;
use rig_tui::crossterm::event::{Event, KeyEventKind};
use rig_tui::{
    AppKeyBindingsExt, Editor, InputAction, InputContext, InputSystems, KeyBinding, KeyInput,
    KeyPress, KeyPressed, KeyScope, MainInput, PendingInput, TuiKeys, TuiPlugin,
};
use serde::{Deserialize, Serialize};

/// Who the plugin is, in notices and decisions.
pub const NAME: &str = "paste-chips";

/// The `[paste_chips]` table: a paste longer than `lines` lines or `chars`
/// characters becomes a chip.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PasteChipSettings {
    /// The most lines pasted as text.
    pub lines: usize,
    /// The most characters pasted as text.
    pub chars: usize,
}

impl Default for PasteChipSettings {
    fn default() -> Self {
        Self {
            lines: 10,
            chars: 1_000,
        }
    }
}

/// What a chip holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChipKind {
    /// Pasted text.
    Text(String),
    /// An image file, with its width and height when they could be read.
    Image {
        /// The file.
        path: PathBuf,
        /// Its width and height in pixels.
        size: Option<(u32, u32)>,
    },
}

/// A paste shown as a chip in the input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chip {
    /// Its number, counted over the session's chips from 1.
    pub id: u32,
    /// What it holds.
    pub kind: ChipKind,
    /// How many identical pastes it stands for.
    pub count: u32,
}

impl Chip {
    /// How it shows in the input: `[Pasted #1 · 412 lines]`,
    /// `[Pasted #1 · 412 lines ×2]` or `[Image #2 · 1280×720]`.
    pub fn token(&self) -> String {
        let times = match self.count {
            0 | 1 => String::new(),
            count => format!(" ×{count}"),
        };
        match &self.kind {
            ChipKind::Text(text) => {
                format!(
                    "[Pasted #{} · {} lines{times}]",
                    self.id,
                    text.lines().count()
                )
            }
            ChipKind::Image {
                size: Some((width, height)),
                ..
            } => format!("[Image #{} · {width}×{height}]", self.id),
            ChipKind::Image { size: None, .. } => format!("[Image #{}]", self.id),
        }
    }

    /// The line an opened chip starts with.
    fn open(&self) -> String {
        format!("[Pasted #{} ▾]\n", self.id)
    }

    /// The line an opened chip ends with.
    fn close(&self) -> String {
        format!("\n[/Pasted #{}]", self.id)
    }
}

/// `text` in a fence longer than any run of backticks in it.
fn fenced(text: &str) -> String {
    let mut longest = 0;
    let mut run = 0;
    for character in text.chars() {
        run = if character == '`' { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    let fence = "`".repeat((longest + 1).max(3));
    format!("{fence}\n{}\n{fence}", text.trim_end_matches('\n'))
}

/// `block` in place of `range` of `text`, on lines of its own: a newline
/// is added before or after only where the text there does not have one.
fn on_own_lines(text: &str, range: Range<usize>, block: &str) -> String {
    let before = text.get(..range.start).unwrap_or_default();
    let after = text.get(range.end..).unwrap_or_default();
    let open = if before.is_empty() || before.ends_with('\n') {
        ""
    } else {
        "\n"
    };
    let close = if after.is_empty() || after.starts_with('\n') {
        ""
    } else {
        "\n"
    };
    format!("{before}{open}{block}{close}{after}")
}

/// The session's chips, by number, and the next number.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct PasteChips {
    /// Every chip made, sent or not, so a prompt brought back from the
    /// history keeps its chips.
    pub chips: Vec<Chip>,
    /// The number of the last chip made.
    pub last: u32,
}

impl PasteChips {
    /// Makes a chip of `kind` and returns its token.
    pub fn add(&mut self, kind: ChipKind) -> String {
        self.last += 1;
        let chip = Chip {
            id: self.last,
            kind,
            count: 1,
        };
        let token = chip.token();
        self.chips.push(chip);
        token
    }

    /// The chip numbered `id`.
    pub fn get(&self, id: u32) -> Option<&Chip> {
        self.chips.iter().find(|chip| chip.id == id)
    }

    /// The chip whose token ends at byte `at` of `text`, and where it is.
    fn ending_at(&self, text: &str, at: usize) -> Option<(Range<usize>, u32)> {
        let before = text.get(..at)?;
        self.chips.iter().find_map(|chip| {
            let token = chip.token();
            before
                .ends_with(&token)
                .then(|| (at - token.len()..at, chip.id))
        })
    }

    /// The chip whose token starts at byte `at` of `text`, and where it is.
    fn starting_at(&self, text: &str, at: usize) -> Option<(Range<usize>, u32)> {
        let after = text.get(at..)?;
        self.chips.iter().find_map(|chip| {
            let token = chip.token();
            after
                .starts_with(&token)
                .then(|| (at..at + token.len(), chip.id))
        })
    }

    /// The opened chip around byte `at` of `text`: where it is, from its
    /// first line to its last, where its contents are, and its number.
    fn opened_around(&self, text: &str, at: usize) -> Option<(Range<usize>, Range<usize>, u32)> {
        self.chips.iter().find_map(|chip| {
            let (open, close) = (chip.open(), chip.close());
            let start = text.get(..at)?.rfind(&open)?;
            let inside = start + open.len();
            let end = inside + text.get(inside..)?.find(&close)?;
            (at <= end + close.len()).then_some((start..end + close.len(), inside..end, chip.id))
        })
    }

    /// The first chip in `text`, collapsed or opened: where it is, the text
    /// it is sent as, and the image it attaches.
    fn first_in(&self, text: &str) -> Option<(Range<usize>, String, Option<&Chip>)> {
        let found = self.chips.iter().filter_map(|chip| {
            let token = chip.token();
            let collapsed = text.find(&token).map(|start| {
                let range = start..start + token.len();
                match &chip.kind {
                    ChipKind::Text(contents) => {
                        let block = vec![fenced(contents); chip.count as usize].join("\n");
                        (range, block, None)
                    }
                    ChipKind::Image { .. } => (range, token.clone(), Some(chip)),
                }
            });
            let opened = text.find(&chip.open()).and_then(|start| {
                let inside = start + chip.open().len();
                let end = inside + text.get(inside..)?.find(&chip.close())?;
                let contents = text.get(inside..end)?;
                let block = vec![fenced(contents); chip.count as usize].join("\n");
                Some((start..end + chip.close().len(), block, None))
            });
            [collapsed, opened]
                .into_iter()
                .flatten()
                .min_by_key(|(range, ..)| range.start)
        });
        found.min_by_key(|(range, ..)| range.start)
    }

    /// `text` as sent: each text chip its contents in a fence, on lines of
    /// their own, the rest as typed; and the image chips it names.
    pub fn expand(&self, text: &str) -> (String, Vec<&Chip>) {
        let (mut done, mut rest, mut images) = (String::new(), text.to_owned(), Vec::new());
        while let Some((range, block, image)) = self.first_in(&rest) {
            let replaced = match image {
                Some(chip) => {
                    images.push(chip);
                    let mut kept = rest.clone();
                    kept.replace_range(range.clone(), &block);
                    kept
                }
                None => on_own_lines(&rest, range.clone(), &block),
            };
            let end = replaced.len() - (rest.len() - range.end);
            done.push_str(replaced.get(..end).unwrap_or_default());
            rest = replaced.get(end..).unwrap_or_default().to_owned();
        }
        done.push_str(&rest);
        (done, images)
    }
}

/// Reads the clipboard's image into a directory: `None` when it holds
/// none. [`clipboard::read_image`] unless a test gives another.
pub type ReadImage = dyn Fn(&Launcher, &Path) -> io::Result<Option<PathBuf>> + Send + Sync;

/// The clipboard's images, read on a thread of their own on Ctrl+V.
#[derive(Resource)]
pub struct ClipboardImages {
    read: Arc<ReadImage>,
    sender: mpsc::Sender<io::Result<Option<PathBuf>>>,
    results: Mutex<mpsc::Receiver<io::Result<Option<PathBuf>>>>,
}

impl ClipboardImages {
    /// Images read by `read`.
    pub fn new(read: Arc<ReadImage>) -> Self {
        let (sender, results) = mpsc::channel();
        Self {
            read,
            sender,
            results: Mutex::new(results),
        }
    }
}

impl Default for ClipboardImages {
    fn default() -> Self {
        Self::new(Arc::new(clipboard::read_image))
    }
}

/// An identical paste the user is asked to merge into the chip before it.
#[derive(Component, Clone, Copy, Debug)]
pub struct MergeAsk {
    /// The chip before.
    pub first: u32,
    /// The identical paste's chip.
    pub second: u32,
}

/// Long pastes and images as chips in the terminal view's input.
#[derive(Default)]
pub struct PasteChipsPlugin;

impl Plugin for PasteChipsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.add_setting::<PasteChipSettings>("paste_chips")
            .init_resource::<PasteChips>()
            .init_resource::<ClipboardImages>()
            .add_systems(TuiKeys, chip_keys.in_set(InputSystems::Filter))
            .add_systems(PreUpdate, receive_images)
            .add_systems(Update, merge_answered);
        let toggle = KeyBinding::new(KeyPress::alt('e'), "Open or close the paste chip");
        app.add_key_binding(toggle.in_scope(KeyScope::Input), open_or_close);
    }
}

/// The input action `item` does, when it is one of the input's own keys.
fn action_of(
    item: &KeyInput,
    bindings: &Query<(&KeyBinding, Option<&InputAction>)>,
) -> Option<InputAction> {
    let key = match item {
        KeyInput::Action(action) => return Some(*action),
        KeyInput::Event(Event::Key(key)) if key.kind != KeyEventKind::Release => key,
        KeyInput::Event(_) => return None,
    };
    let press = [KeyPress::of(key)];
    let bound = |scope: KeyScope| {
        bindings
            .iter()
            .find(|(binding, _)| binding.scope == scope && binding.keys.presses() == press)
    };
    // A global binding of the key shadows the input's.
    if bound(KeyScope::Global).is_some() {
        return None;
    }
    bound(KeyScope::Input).and_then(|(_, action)| action.copied())
}

/// The image file a paste names, as a terminal pastes a dropped file:
/// quoted, or as a `file://` URL.
fn pasted_image(text: &str) -> Option<PathBuf> {
    let text = text.trim();
    let unquoted = ['\'', '"']
        .into_iter()
        .find_map(|quote| text.strip_prefix(quote)?.strip_suffix(quote))
        .unwrap_or(text);
    let path = Path::new(unquoted.strip_prefix("file://").unwrap_or(unquoted));
    (attach::is_image_path(path) && path.is_file()).then(|| path.to_path_buf())
}

/// An image chip for the file at `path`.
fn image_chip(path: PathBuf) -> ChipKind {
    let size = image::ImageReader::open(&path)
        .and_then(|reader| reader.with_guessed_format())
        .ok()
        .and_then(|reader| reader.into_dimensions().ok());
    ChipKind::Image { path, size }
}

/// What the keys may change besides the input: the chips, the clipboard,
/// and who is told.
#[derive(bevy_ecs::system::SystemParam)]
struct Chipping<'w, 's> {
    chips: ResMut<'w, PasteChips>,
    settings: Res<'w, PasteChipSettings>,
    clipboard: Res<'w, ClipboardImages>,
    scratch: Option<Res<'w, Scratch>>,
    processes: Option<Res<'w, Processes>>,
    wake: Res<'w, Wake>,
    shown: Res<'w, ShownAgent>,
    connections: Query<'w, 's, &'static Connection>,
    answerers: Answerers<'w, 's>,
    notices: MessageWriter<'w, Notice>,
    commands: Commands<'w, 's>,
}

/// Takes the key being done when it is about a chip: a long paste or an
/// image file pasted, Ctrl+V, a move or a deletion next to a chip, and
/// Enter on a message with an image the model cannot read, warned once.
fn chip_keys(
    mut events: ResMut<PendingInput>,
    state: Res<InputContext>,
    mut input: MainInput,
    bindings: Query<(&KeyBinding, Option<&InputAction>)>,
    mut chipping: Chipping,
    mut warned: Local<Option<String>>,
) {
    let in_input = state.scope() == KeyScope::Input
        && state.picking().is_none()
        && state.answering().is_none();
    let Some(item) = events.0.first().filter(|_| in_input) else {
        return;
    };
    let taken = match item {
        KeyInput::Event(Event::Paste(text)) => paste(text, &mut input, &mut chipping),
        item => match action_of(item, &bindings) {
            Some(action) => act(action, &state, &mut input, &mut chipping, &mut warned),
            None => false,
        },
    };
    if taken {
        events.0.remove(0);
    }
}

/// A paste: an image file or a long text becomes a chip; a short one is
/// left to the view.
fn paste(text: &str, input: &mut Editor, chipping: &mut Chipping) -> bool {
    if let Some(path) = pasted_image(text) {
        let token = chipping.chips.add(image_chip(path));
        input.insert(&token);
        return true;
    }
    let settings = &chipping.settings;
    if text.lines().count() <= settings.lines && text.chars().count() <= settings.chars {
        return false;
    }
    let before = chipping
        .chips
        .ending_at(input.text(), input.cursor())
        .and_then(|(_, id)| chipping.chips.get(id))
        .filter(|chip| chip.kind == ChipKind::Text(text.to_owned()))
        .map(|chip| chip.id);
    let token = chipping.chips.add(ChipKind::Text(text.to_owned()));
    input.insert(&token);
    let agent = chipping.shown.0;
    if let Some(first) = before
        && chipping.answerers.can_answer(agent)
    {
        let second = chipping.chips.last;
        let ask = Ask::new(format!("Merge paste #{second} into #{first}, as ×2?"))
            .body("The same text was pasted twice in a row.")
            .choice("separate", "Keep them apart")
            .choice("merge", "Merge them")
            .inline();
        let mut spawned = chipping.commands.spawn((ask, MergeAsk { first, second }));
        if let Some(agent) = agent {
            spawned.insert(ForAgent(agent));
        }
    }
    true
}

/// An input action next to a chip, Ctrl+V, or Enter with an image the
/// model cannot read; whether it was taken.
fn act(
    action: InputAction,
    state: &InputContext,
    input: &mut Editor,
    chipping: &mut Chipping,
    warned: &mut Option<String>,
) -> bool {
    let (text, at) = (input.text().to_owned(), input.cursor());
    let chips = &chipping.chips;
    let remove = |input: &mut Editor, range: Range<usize>| {
        let mut left = text.clone();
        left.replace_range(range.clone(), "");
        input.set(left);
        input.set_cursor(range.start);
    };
    match action {
        InputAction::Left => chips
            .ending_at(&text, at)
            .map(|(range, _)| input.set_cursor(range.start))
            .is_some(),
        InputAction::Right => chips
            .starting_at(&text, at)
            .map(|(range, _)| input.set_cursor(range.end))
            .is_some(),
        InputAction::Backspace => chips
            .ending_at(&text, at)
            .map(|(range, _)| remove(input, range))
            .is_some(),
        InputAction::Delete => chips
            .starting_at(&text, at)
            .map(|(range, _)| remove(input, range))
            .is_some(),
        InputAction::PasteImage => read_clipboard(chipping),
        InputAction::Send if state.completion().is_none() => {
            let agent = chipping.shown.0;
            let blind = agent
                .and_then(|agent| chipping.connections.get(agent).ok())
                .is_none_or(|connection| !connection.spec.input.image);
            let (sent, images) = chips.expand(&text);
            if images.is_empty() || !blind || warned.as_deref() == Some(text.as_str()) {
                // What the view sends: the chips' contents, each image as
                // `@path` for the agent to attach.
                if sent != text || !images.is_empty() {
                    input.set(as_mentions(sent, &images));
                }
                return false;
            }
            let names: Vec<String> = images.iter().map(|chip| chip.token()).collect();
            let why = format!(
                "The model reads no images, so {} would go as its name only. Enter again sends it anyway.",
                names.join(", ")
            );
            chipping.notices.write(Notice::warn(agent, why));
            *warned = Some(text);
            true
        }
        _ => false,
    }
}

/// Reads the clipboard's image on a thread, into the session's images;
/// without a place for it, the view says so.
fn read_clipboard(chipping: &Chipping) -> bool {
    let Some(directory) = chipping.scratch.as_deref().and_then(Scratch::images) else {
        return false;
    };
    let launcher = chipping
        .processes
        .as_deref()
        .map(Processes::launcher)
        .unwrap_or_default();
    let (read, sender, wake) = (
        Arc::clone(&chipping.clipboard.read),
        chipping.clipboard.sender.clone(),
        chipping.wake.clone(),
    );
    let spawned = std::thread::Builder::new()
        .name("paste-chips-clipboard".to_owned())
        .spawn(move || {
            sender.send(read(&launcher, &directory)).ok();
            wake.wake();
        });
    if let Err(failure) = spawned {
        chipping.clipboard.sender.send(Err(failure)).ok();
    }
    true
}

/// Puts a chip in the input for each image read from the clipboard.
fn receive_images(
    clipboard: Res<ClipboardImages>,
    mut chips: ResMut<PasteChips>,
    mut input: MainInput,
    shown: Res<ShownAgent>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(results) = clipboard.results.lock() else {
        return;
    };
    for result in results.try_iter() {
        match result {
            Ok(Some(path)) => input.insert(&chips.add(image_chip(path))),
            Ok(None) => {
                notices.write(Notice::info(shown.0, "No image in the clipboard."));
            }
            Err(failure) => {
                let why = format!("Could not paste the clipboard's image: {failure}.");
                notices.write(Notice::info(shown.0, why));
            }
        }
    }
}

/// Alt+E: opens the chip at the cursor in place, its contents between a
/// first and a last line, or closes the opened chip the cursor is in,
/// keeping its edits.
fn open_or_close(_: On<KeyPressed>, mut input: MainInput, mut chips: ResMut<PasteChips>) {
    let (text, at) = (input.text().to_owned(), input.cursor());
    if let Some((range, inside, id)) = chips.opened_around(&text, at) {
        let contents = text.get(inside).unwrap_or_default().to_owned();
        let Some(chip) = chips.chips.iter_mut().find(|chip| chip.id == id) else {
            return;
        };
        chip.kind = ChipKind::Text(contents);
        let token = chip.token();
        let mut closed = text;
        closed.replace_range(range.clone(), &token);
        input.set(closed);
        input.set_cursor(range.start + token.len());
        return;
    }
    let near = chips
        .ending_at(&text, at)
        .or_else(|| chips.starting_at(&text, at));
    let Some((range, chip)) = near.and_then(|(range, id)| Some((range, chips.get(id)?))) else {
        return;
    };
    let ChipKind::Text(contents) = &chip.kind else {
        return;
    };
    let (open, close) = (chip.open(), chip.close());
    let mut opened = text;
    opened.replace_range(range.clone(), &format!("{open}{contents}{close}"));
    input.set(opened);
    input.set_cursor(range.start + open.len());
}

/// An identical paste the user merged: its chip leaves the input and the
/// chip before counts it.
fn merge_answered(
    asks: Query<(&Outcome, &MergeAsk), Added<Outcome>>,
    mut chips: ResMut<PasteChips>,
    mut input: MainInput,
) {
    for (outcome, merge) in &asks {
        if !Answer::of(outcome).is_some_and(|answer| answer.is("merge")) {
            continue;
        }
        let (Some(first), Some(second)) = (chips.get(merge.first), chips.get(merge.second)) else {
            continue;
        };
        let (old, gone) = (first.token(), second.token());
        let Some(chip) = chips.chips.iter_mut().find(|chip| chip.id == merge.first) else {
            continue;
        };
        chip.count += 1;
        let merged = input
            .text()
            .replacen(&gone, "", 1)
            .replacen(&old, &chip.token(), 1);
        input.set(merged);
    }
}

/// `sent` with each image chip's token as `@path`.
fn as_mentions(mut sent: String, images: &[&Chip]) -> String {
    for chip in images {
        if let ChipKind::Image { path, .. } = &chip.kind {
            sent = sent.replace(&chip.token(), &format!("@{}", path.display()));
        }
    }
    sent
}

#[cfg(test)]
mod tests;
