use std::path::PathBuf;
use std::process::Command;

use bevy_app::prelude::*;
use ratatui::{TerminalOptions, Viewport};

use super::*;
use crate::tests::{scratch, view_app};

#[cfg(unix)]
#[test]
fn a_program_runs_with_the_terminal_and_its_exit_comes_back() {
    let (mut app, _) = view_app();
    let draft = scratch("run-in-terminal").join("draft.md");
    let mut editor = Command::new("sh");
    editor
        .arg("-c")
        .arg("printf 'from the editor' > \"$0\"")
        .arg(&draft);
    let edited = app.world_mut().spawn(RunInTerminal(editor)).id();
    let missing = app
        .world_mut()
        .spawn(RunInTerminal(Command::new("rig-tui-no-such-program")))
        .id();
    app.update();
    let exit = |entity| app.world().get::<TerminalExit>(entity).cloned();
    assert!(exit(edited).is_some_and(|exit| exit.success()));
    assert!(exit(missing).is_some_and(|exit| exit.0.is_err()));
    assert_eq!(
        std::fs::read_to_string(&draft).ok().as_deref(),
        Some("from the editor")
    );
    // Run once: another frame runs nothing again.
    std::fs::remove_file(&draft).ok();
    app.update();
    assert!(!draft.exists());
}

/// A failed program's screen stays up: the view says how it ended and
/// waits for Enter, reading no further, or for the keys' end.
#[test]
fn a_failed_program_waits_for_enter() {
    for enter in [b'\n', b'\r', 0x03, 0x04] {
        let typed = [b'x', enter, b'l', b'a', b't', b'e', b'r'];
        let mut keys: &[u8] = &typed;
        let mut screen = Vec::new();
        hold_failure("exit status 2", &mut keys, &mut screen);
        assert_eq!(keys, b"later", "{enter}");
        let said = String::from_utf8_lossy(&screen);
        let line = "[the program exited with exit status 2; press Enter to go back]\r\n";
        assert!(
            said.starts_with(PLAIN_KEYS) && said.ends_with(line),
            "{said:?}"
        );
    }
    let mut ended: &[u8] = b"";
    hold_failure("exit status 2", &mut ended, io::sink());
}

/// Only a program that exited with a failure code holds the screen; every
/// failure has a reason for the caller's notice.
#[cfg(unix)]
#[test]
fn a_failure_is_described() {
    use std::os::unix::process::ExitStatusExt;
    let exited = Ok(ExitStatus::from_raw(2 << 8));
    let killed = Ok(ExitStatus::from_raw(9));
    let missing = Err("could not run vi: not found".to_owned());
    let succeeded = Ok(ExitStatus::from_raw(0));
    assert_eq!(failed_with(&exited).as_deref(), Some("exit status 2"));
    for quiet in [&killed, &missing, &succeeded] {
        assert_eq!(failed_with(quiet), None, "{quiet:?}");
    }
    let why = |exit: Result<ExitStatus, String>| TerminalExit(exit).failure();
    assert_eq!(why(exited).as_deref(), Some("exit status 2"));
    assert!(why(killed).is_some_and(|why| why.contains('9')));
    assert_eq!(why(missing).as_deref(), Some("could not run vi: not found"));
    assert_eq!(why(succeeded), None);
}

/// The terminal view drawing to the file at `path`, as the terminal
/// stands in for it; it never asks the file for its size.
fn tui_on(path: &PathBuf) -> Option<Tui> {
    let screen = File::create(path).ok()?;
    let copy = screen.try_clone().ok()?;
    let viewport = Viewport::Fixed(Rect::new(0, 0, 20, 4));
    let backend = CrosstermBackend::new(BufWriter::new(screen));
    let terminal = Terminal::with_options(backend, TerminalOptions { viewport }).ok()?;
    Some(Tui {
        terminal,
        keep_screen: true,
        releases: false,
        screen: copy,
    })
}

/// Bytes a plugin writes where Bevy says code runs on the way out reach
/// the terminal before the view lets it go.
#[test]
fn a_terminal_write_sent_on_the_way_out_is_written() {
    let (mut app, _) = view_app();
    let path = scratch("exit-write").join("screen");
    let tui = tui_on(&path);
    assert!(tui.is_some());
    let Some(tui) = tui else { return };
    app.insert_resource(tui).add_systems(
        Last,
        (|mut writes: MessageWriter<TerminalWrite>| {
            writes.write(TerminalWrite(b"\x1b]0;bye\x07".to_vec()));
        })
        .in_set(bevy_app::OnAppExitSystems),
    );
    app.update();
    let written = std::fs::read(&path).unwrap_or_default();
    let title: &[u8] = b"\x1b]0;bye\x07";
    assert!(written.windows(title.len()).any(|window| window == title));
}

/// A terminal that answers nothing: it counts the cells drawn and notes a
/// question about the cursor, which a real one answers on stdin.
#[derive(Default)]
struct Probe {
    asked: bool,
    drawn: usize,
}

const PROBE: ratatui::layout::Size = ratatui::layout::Size::new(6, 2);

impl Write for Probe {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ratatui::backend::Backend for Probe {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a ratatui::buffer::Cell)>,
    {
        self.drawn += content.count();
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn get_cursor_position(&mut self) -> io::Result<ratatui::layout::Position> {
        self.asked = true;
        Err(io::Error::other("no answer"))
    }

    fn set_cursor_position<P: Into<ratatui::layout::Position>>(&mut self, _: P) -> io::Result<()> {
        Ok(())
    }

    fn clear(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn clear_region(&mut self, _: ratatui::backend::ClearType) -> io::Result<()> {
        Ok(())
    }

    fn size(&self) -> io::Result<ratatui::layout::Size> {
        Ok(PROBE)
    }

    fn window_size(&mut self) -> io::Result<ratatui::backend::WindowSize> {
        Ok(ratatui::backend::WindowSize {
            columns_rows: PROBE,
            pixels: ratatui::layout::Size::default(),
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Taking the screen (at startup, and back from a program) asks the
/// terminal nothing, and the next frame is drawn whole even when it is
/// the frame drawn before.
#[test]
fn taking_the_screen_asks_nothing_and_the_next_frame_is_whole() {
    let terminal = Terminal::new(Probe::default());
    assert!(terminal.is_ok());
    let Ok(mut terminal) = terminal else { return };
    let frame = |terminal: &mut Terminal<Probe>| {
        let drawn = terminal.draw(|frame| {
            frame.render_widget(ratatui::widgets::Paragraph::new("rig"), frame.area());
        });
        assert!(drawn.is_ok());
        std::mem::take(&mut terminal.backend_mut().drawn)
    };
    let whole = frame(&mut terminal);
    assert_eq!(whole, "rig".len());
    assert_eq!(frame(&mut terminal), 0, "an unchanged frame draws nothing");
    for releases in [false, true] {
        assert!(take_screen(&mut terminal, releases).is_ok());
        assert!(!terminal.backend().asked);
        assert_eq!(frame(&mut terminal), whole);
    }
}
