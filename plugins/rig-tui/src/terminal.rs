//! Terminal ownership: raw mode, the alternate screen, keeping stray
//! output off the screen, handing the terminal to a program for a while
//! ([`RunInTerminal`]), whether the terminal has the user's focus
//! ([`TerminalFocus`]), and a plugin's own escape codes, such as a desktop
//! notification ([`TerminalWrite`]) or a copy to the clipboard
//! ([`CopyToClipboard`]), written after each frame, the last one included.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{
    DisableBracketedPaste, DisableFocusChange, EnableBracketedPaste, EnableFocusChange,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{
    BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate, EnterAlternateScreen,
    LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use crossterm::{execute, queue};
use ratatui::backend::{CrosstermBackend, TestBackend};
use ratatui::buffer::Buffer;
use ratatui::layout::{Rect, Size};
use ratatui::{Frame, Terminal};

use super::input::TerminalInput;
use super::keys::KeyBinding;
use super::panel::{RequestRedraw, TerminalSize};
use super::theme::Theme;
use rig_ecs::tasks::Wake;
use rig_harness::harness_protocol::secret_variables;
use rig_harness::prelude::{Exit, ExitReason, Scratch};

/// Runs a program with the terminal, such as `$EDITOR` on a draft: the
/// view leaves the screen, stops reading keys and lets the program have
/// the terminal, its output included, until it exits; then it takes the
/// terminal back, draws again, and inserts [`TerminalExit`] on the entity,
/// which the plugin despawns once it read it. The agent's loop waits
/// while the program runs. Only the terminal view runs it, in the frame
/// it is spawned.
///
/// Ctrl+C in a program that leaves the terminal as it is (not an editor,
/// which takes the keys itself) also stops the agent. The program gets
/// none of the session's secrets.
#[derive(Component, Debug)]
pub struct RunInTerminal(pub Command);

/// How a [`RunInTerminal`] program ended: its exit status, or why it did
/// not start.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct TerminalExit(pub Result<ExitStatus, String>);

impl TerminalExit {
    /// Whether the program ran and exited with success.
    pub fn success(&self) -> bool {
        self.0.as_ref().is_ok_and(ExitStatus::success)
    }
}

/// Whether the terminal has the user's focus, as the terminal reports it
/// where it can: `None` until its first report, and in a run without the
/// terminal view.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerminalFocus(pub Option<bool>);

/// Bytes for the terminal itself, such as an escape code that shows a
/// desktop notification (OSC 9 or 777): the view writes them after its
/// frame, through its own writer, so they never land inside a frame. Bytes
/// sent in Bevy's `OnAppExitSystems` are written before the terminal is
/// given back. Nothing writes them in a run without the terminal view.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct TerminalWrite(pub Vec<u8>);

/// Puts the text on the system clipboard through the terminal (OSC 52),
/// which works over SSH too, where the terminal allows it, and on this
/// machine's clipboard through [`rig_tools::clipboard`], for a terminal
/// that ignores OSC 52.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct CopyToClipboard(pub String);

impl CopyToClipboard {
    /// The escape code that copies it.
    fn escape(&self) -> Vec<u8> {
        format!("\x1b]52;c;{}\x07", STANDARD.encode(&self.0)).into_bytes()
    }
}

/// Bytes buffered before a write to the terminal: a full frame of a large
/// screen.
const FRAME_BUFFER: usize = 1 << 16;

/// The terminal, drawn to through a buffered private copy of stdout, so a
/// frame goes out in a few writes. Dropping it restores the terminal,
/// keeping the alternate screen for a reload so the next build draws over
/// the same screen.
#[derive(Resource)]
pub(crate) struct Tui {
    pub(super) terminal: Terminal<CrosstermBackend<BufWriter<File>>>,
    keep_screen: bool,
    /// Whether the terminal is asked to report key releases.
    releases: bool,
    /// A copy of the screen's file, for a program's output.
    screen: File,
}

/// Where stray output goes when the session keeps no text log.
const NOWHERE: &str = "/dev/null";

/// What the terminal is always asked: keys sent as escape codes, so a
/// modifier it would drop, such as Shift on Enter, comes through.
const KEYS: KeyboardEnhancementFlags = KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES;

/// What the terminal is asked to report while a key binding wants key
/// releases, on top of [`KEYS`].
const RELEASES: KeyboardEnhancementFlags = KEYS.union(KeyboardEnhancementFlags::REPORT_EVENT_TYPES);

impl Tui {
    /// Take over the terminal. On unix, stdout and stderr are then pointed
    /// at `log`, so output from plugins, dependencies or panics lands there
    /// instead of on the screen.
    fn open(log: Option<&Path>) -> io::Result<Self> {
        let screen = screen()?;
        if let Some(log) = log {
            redirect_output(log)?;
        }
        let copy = screen.try_clone()?;
        let backend = CrosstermBackend::new(BufWriter::with_capacity(FRAME_BUFFER, screen));
        let mut tui = Self {
            terminal: Terminal::new(backend)?,
            keep_screen: false,
            releases: false,
            screen: copy,
        };
        tui.enter()?;
        Ok(tui)
    }

    /// Takes the screen: raw mode, then [`take_screen`].
    fn enter(&mut self) -> io::Result<()> {
        enable_raw_mode()?;
        take_screen(&mut self.terminal, self.releases)
    }

    /// Gives the screen back as the shell had it, leaving the alternate
    /// screen unless `keep_screen`.
    fn leave(&mut self, keep_screen: bool) -> io::Result<()> {
        let backend = self.terminal.backend_mut();
        if self.releases {
            execute!(backend, PopKeyboardEnhancementFlags).ok();
        }
        execute!(backend, PopKeyboardEnhancementFlags).ok();
        // Every step is tried, so a failed one leaves no other undone.
        let pastes = execute!(backend, DisableBracketedPaste, DisableFocusChange);
        let alternate = match keep_screen {
            true => Ok(()),
            false => execute!(backend, LeaveAlternateScreen),
        };
        let cursor = self.terminal.show_cursor();
        pastes.and(alternate).and(cursor).and(disable_raw_mode())
    }

    /// Starts a frame: one synchronized update with the cursor hidden, so
    /// the cursor never shows travelling across the screen.
    pub(crate) fn begin_frame(&mut self) -> io::Result<()> {
        queue!(self.terminal.backend_mut(), BeginSynchronizedUpdate, Hide)
    }

    /// Ends the frame, showing the cursor at `cursor`, if any.
    pub(crate) fn end_frame(&mut self, cursor: Option<MoveTo>) -> io::Result<()> {
        let backend = self.terminal.backend_mut();
        if let Some(cursor) = cursor {
            queue!(backend, cursor, Show)?;
        }
        execute!(backend, EndSynchronizedUpdate)
    }

    /// Asks the terminal to report key releases, or no longer.
    fn report_releases(&mut self, releases: bool) {
        if self.releases == releases {
            return;
        }
        self.releases = releases;
        let backend = self.terminal.backend_mut();
        let asked = if releases {
            execute!(backend, PushKeyboardEnhancementFlags(RELEASES))
        } else {
            execute!(backend, PopKeyboardEnhancementFlags)
        };
        if let Err(failure) = asked {
            error!("could not ask the terminal about key releases: {failure}");
        }
    }
}

/// The alternate screen cleared (after a reload it still shows the previous
/// build's frame), pastes reported whole, focus changes reported, keys sent
/// as escape codes, no cursor, and a whole frame drawn next. The terminal is asked nothing: its answer
/// would come on stdin, but stdout may be the session's log by now
/// (`Terminal::clear` asks for the cursor position).
fn take_screen<B>(terminal: &mut Terminal<B>, releases: bool) -> io::Result<()>
where
    B: ratatui::backend::Backend<Error = io::Error> + Write,
{
    let backend = terminal.backend_mut();
    execute!(
        backend,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableFocusChange,
        Clear(ClearType::All)
    )?;
    // A terminal without the keyboard protocol ignores these.
    execute!(backend, PushKeyboardEnhancementFlags(KEYS)).ok();
    if releases {
        execute!(backend, PushKeyboardEnhancementFlags(RELEASES)).ok();
    }
    terminal.hide_cursor()?;
    // Each swap resets the buffer it makes current: both are blank, as the
    // screen is, so the next draw writes every cell.
    terminal.swap_buffers();
    terminal.swap_buffers();
    Ok(())
}

/// A screen in memory that the view draws to, for a test of what a plugin
/// shows: inserted in a run with the view's systems but no terminal (the
/// plugin, then [`add_view`](crate::add_view)), each frame due is drawn
/// to it as it would be to the terminal, and the bytes sent for the
/// terminal ([`TerminalWrite`], [`CopyToClipboard`]) are kept. A test
/// reads what is shown, never a whole screen to compare.
#[derive(Resource)]
pub struct TuiTestScreen {
    terminal: Terminal<TestBackend>,
    written: Vec<u8>,
    repainted: Vec<Rect>,
}

impl TuiTestScreen {
    /// A screen `width` columns wide and `height` rows high.
    pub fn new(width: u16, height: u16) -> Self {
        let Ok(terminal) = Terminal::new(TestBackend::new(width, height));
        Self {
            terminal,
            written: Vec::new(),
            repainted: Vec::new(),
        }
    }

    /// Makes it `width` columns wide and `height` rows high, as a terminal
    /// resized; the next frame is drawn at that size.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.terminal.backend_mut().resize(width, height);
    }

    /// Its size.
    pub fn size(&self) -> Size {
        let area = self.buffer().area;
        Size::new(area.width, area.height)
    }

    /// What the last frame drew.
    pub fn buffer(&self) -> &Buffer {
        self.terminal.backend().buffer()
    }

    /// Each row's text as the last frame drew it.
    pub fn rows(&self) -> Vec<String> {
        let buffer = self.buffer();
        let area = buffer.area;
        (area.top()..area.bottom())
            .map(|y| {
                let cells = (area.left()..area.right()).filter_map(|x| buffer.cell((x, y)));
                cells.map(|cell| cell.symbol()).collect()
            })
            .collect()
    }

    /// The bytes sent for the terminal so far, in order.
    pub fn written(&self) -> &[u8] {
        &self.written
    }

    /// The areas written again after a frame ([`RedrawArea`](crate::RedrawArea)),
    /// clipped to the screen, in order.
    pub fn repainted(&self) -> &[Rect] {
        &self.repainted
    }

    /// Draws a frame with `draw`, then writes `repaint`'s cells again.
    pub(crate) fn draw(&mut self, draw: impl FnOnce(&mut Frame), repaint: &[Rect]) {
        let Ok(completed) = self.terminal.draw(draw);
        let screen = completed.area;
        let areas = repaint.iter().map(|area| area.intersection(screen));
        self.repainted.extend(areas.filter(|area| !area.is_empty()));
    }
}

/// Whether the view has a screen to draw to: the terminal, or a test's.
pub(crate) fn has_screen(tui: Option<Res<Tui>>, test: Option<Res<TuiTestScreen>>) -> bool {
    tui.is_some() || test.is_some()
}

impl Drop for Tui {
    fn drop(&mut self) {
        self.leave(self.keep_screen).ok();
    }
}

/// Opens the terminal and starts reading it at startup, or asks to exit
/// with code 1 when there is none.
pub(crate) fn open_terminal(
    scratch: Option<Res<Scratch>>,
    wake: Res<Wake>,
    mut screen: ResMut<TerminalSize>,
    mut commands: Commands,
) {
    // Asked before the terminal is read, which would take its answer.
    commands.insert_resource(Theme::of_terminal());
    super::highlight::warm_up();
    // Stray output goes to the text log, or nowhere without a place for it.
    let log = scratch.map(|scratch| scratch.log().unwrap_or_else(|| PathBuf::from(NOWHERE)));
    let opened =
        Tui::open(log.as_deref()).and_then(|tui| Ok((tui, TerminalInput::start(wake.clone())?)));
    match opened {
        Ok((tui, input)) => {
            if let Ok(size) = tui.terminal.size() {
                screen.0 = Rect::new(0, 0, size.width, size.height);
            }
            commands.insert_resource(tui);
            commands.insert_resource(input);
        }
        Err(failure) => {
            error!("could not open the terminal: {failure}");
            commands.trigger(Exit(ExitReason::Failed(1)));
        }
    }
}

/// Keeps the alternate screen when the app exits for the launcher to start
/// it again: to reload, or to run another session.
pub(crate) fn keep_screen_on_reload(reason: Option<Res<ExitReason>>, tui: Option<ResMut<Tui>>) {
    if let (Some(mut tui), Some(reason)) = (tui, reason)
        && matches!(*reason, ExitReason::Reload | ExitReason::Switch(_))
    {
        tui.keep_screen = true;
    }
}

/// Writes the [`TerminalWrite`]s and [`CopyToClipboard`]s sent, in order
/// each, after the frame and after Bevy's `OnAppExitSystems`, so bytes
/// sent on the way out are written too.
pub(crate) fn write_to_terminal(
    (tui, test): (Option<ResMut<Tui>>, Option<ResMut<TuiTestScreen>>),
    mut writes: MessageReader<TerminalWrite>,
    mut copies: MessageReader<CopyToClipboard>,
) {
    let bytes: Vec<u8> = writes
        .read()
        .flat_map(|write| write.0.iter().copied())
        .chain(copies.read().flat_map(CopyToClipboard::escape))
        .collect();
    if bytes.is_empty() {
        return;
    }
    let mut tui = match (tui, test) {
        (Some(tui), _) => tui,
        (None, Some(mut test)) => return test.written.extend(bytes),
        (None, None) => return,
    };
    let backend = tui.terminal.backend_mut();
    if let Err(failure) = backend.write_all(&bytes).and_then(|()| backend.flush()) {
        error!("could not write to the terminal: {failure}");
    }
}

/// Asks the terminal to report key releases while a binding wants them.
pub(crate) fn report_releases(mut tui: ResMut<Tui>, bindings: Query<&KeyBinding>) {
    let releases = bindings.iter().any(|binding| binding.releases);
    tui.report_releases(releases);
}

/// Runs each [`RunInTerminal`] program spawned, with the terminal: the
/// input thread stops, the screen is given back, and both are taken again
/// once the programs exited. Without the terminal (a test), the programs
/// run as they are.
pub(crate) fn run_in_terminal(world: &mut World) {
    let mut runs = world.query_filtered::<Entity, (With<RunInTerminal>, Without<TerminalExit>)>();
    let runs: Vec<Entity> = runs.iter(world).collect();
    let input = world.remove_resource::<TerminalInput>();
    let reading = input.is_some();
    if let Some(input) = input {
        input.stop();
    }
    let mut screen = None;
    if let Some(mut tui) = world.get_resource_mut::<Tui>() {
        match tui.leave(false) {
            Ok(()) => screen = tui.screen.try_clone().ok(),
            Err(failure) => error!("could not hand the terminal over: {failure}"),
        }
    }
    for entity in runs {
        let Some(mut run) = world.get_mut::<RunInTerminal>(entity) else {
            continue;
        };
        let exit = run_with(&mut run.0, screen.as_ref());
        world.entity_mut(entity).insert(TerminalExit(exit));
    }
    let size = world.get_resource_mut::<Tui>().and_then(|mut tui| {
        if let Err(failure) = tui.enter() {
            error!("could not take the terminal back: {failure}");
        }
        tui.terminal.size().ok()
    });
    // The terminal may have been resized meanwhile, unseen.
    if let Some(size) = size
        && let Some(mut screen) = world.get_resource_mut::<TerminalSize>()
    {
        screen.set_if_neq(TerminalSize(Rect::new(0, 0, size.width, size.height)));
    }
    let wake = world.get_resource::<Wake>().cloned();
    if reading && let Some(wake) = &wake {
        match TerminalInput::start(wake.clone()) {
            Ok(input) => world.insert_resource(input),
            Err(failure) => error!("could not read the terminal again: {failure}"),
        }
    }
    world.write_message(RequestRedraw);
    // The plugin reads the exit in the next frame.
    if let Some(wake) = wake {
        wake.wake();
    }
}

/// Runs `command` until it exits, its output on `screen` when the view
/// sends its own output elsewhere, without the session's secrets.
fn run_with(command: &mut Command, screen: Option<&File>) -> Result<ExitStatus, String> {
    for secret in secret_variables() {
        command.env_remove(secret);
    }
    if let Some(screen) = screen {
        let output = screen.try_clone().map_err(|failure| failure.to_string())?;
        let errors = screen.try_clone().map_err(|failure| failure.to_string())?;
        command
            .stdout(Stdio::from(output))
            .stderr(Stdio::from(errors));
    }
    let program = command.get_program().to_string_lossy().into_owned();
    command
        .status()
        .map_err(|failure| format!("could not run {program}: {failure}"))
}

#[cfg(unix)]
fn screen() -> io::Result<File> {
    use std::os::fd::AsFd;
    Ok(File::from(io::stdout().as_fd().try_clone_to_owned()?))
}

#[cfg(windows)]
fn screen() -> io::Result<File> {
    use std::os::windows::io::AsHandle;
    Ok(File::from(io::stdout().as_handle().try_clone_to_owned()?))
}

#[cfg(unix)]
fn redirect_output(log: &Path) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    if let Some(directory) = log.parent() {
        std::fs::create_dir_all(directory)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    for target in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
        // SAFETY: both descriptors are open for the duration of the call;
        // `dup2` only replaces `target` with a copy of the log's descriptor.
        if unsafe { libc::dup2(file.as_raw_fd(), target) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Other platforms keep their standard streams; only the screen copy is
/// drawn to.
#[cfg(not(unix))]
fn redirect_output(_log: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests;
