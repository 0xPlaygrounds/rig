//! The system clipboard, read and written by the programs that own it on
//! this machine: `wl-paste`/`wl-copy` on Wayland, `xclip` on X11, and
//! `pbpaste`/`pbcopy` (`pngpaste` for images) on macOS. Each is started as
//! a [`Launcher`] says, such as one that keeps the session's secrets from
//! it. Blocking: call these off a thread that must not wait.

use std::io::{self, ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use rig_core::message::ImageMediaType;

use crate::process::Launcher;

/// What a clipboard program does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    /// Prints the clipboard's text.
    ReadText,
    /// Prints the clipboard's image as PNG.
    ReadImage,
    /// Replaces the clipboard's text with its input.
    WriteText,
}

/// A clipboard this machine may have.
#[derive(Clone, Copy)]
enum Backend {
    Wayland,
    X11,
    Mac,
}

impl Backend {
    /// The program and arguments that do `what` with this clipboard.
    fn command(self, what: Use) -> (&'static str, &'static [&'static str]) {
        match (self, what) {
            (Self::Wayland, Use::ReadText) => ("wl-paste", &["--no-newline"]),
            (Self::Wayland, Use::ReadImage) => {
                ("wl-paste", &["--no-newline", "--type", "image/png"])
            }
            (Self::Wayland, Use::WriteText) => ("wl-copy", &[]),
            (Self::X11, Use::ReadText) => ("xclip", &["-selection", "clipboard", "-out"]),
            (Self::X11, Use::ReadImage) => (
                "xclip",
                &["-selection", "clipboard", "-target", "image/png", "-out"],
            ),
            (Self::X11, Use::WriteText) => ("xclip", &["-selection", "clipboard", "-in"]),
            (Self::Mac, Use::ReadText) => ("pbpaste", &[]),
            (Self::Mac, Use::ReadImage) => ("pngpaste", &["-"]),
            (Self::Mac, Use::WriteText) => ("pbcopy", &[]),
        }
    }
}

/// The programs that do `what` on this machine, in the order tried.
fn programs(what: Use) -> Vec<(&'static str, &'static [&'static str])> {
    [
        std::env::var_os("WAYLAND_DISPLAY")
            .is_some()
            .then_some(Backend::Wayland),
        std::env::var_os("DISPLAY")
            .is_some()
            .then_some(Backend::X11),
        cfg!(target_os = "macos").then_some(Backend::Mac),
    ]
    .into_iter()
    .flatten()
    .map(|backend| backend.command(what))
    .collect()
}

/// The error for a clipboard none of `programs` could serve.
fn none_of(programs: &[(&str, &[&str])], what: &str) -> io::Error {
    if programs.is_empty() {
        return io::Error::new(
            ErrorKind::Unsupported,
            "no clipboard here: it needs wl-paste and wl-copy, xclip, or pbpaste and pbcopy",
        );
    }
    let names: Vec<&str> = programs.iter().map(|(program, _)| *program).collect();
    io::Error::new(
        ErrorKind::NotFound,
        format!(
            "no {what} in the clipboard (read with {})",
            names.join(", ")
        ),
    )
}

/// What the first program for `what` that succeeds prints.
fn output_of(
    launcher: &Launcher,
    what: Use,
) -> (
    Vec<(&'static str, &'static [&'static str])>,
    Option<Vec<u8>>,
) {
    let programs = programs(what);
    let printed = programs.iter().find_map(|(program, args)| {
        let output = launcher
            .command(program)
            .args(*args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        output.status.success().then_some(output.stdout)
    });
    (programs, printed)
}

/// The clipboard's text.
///
/// # Errors
///
/// [`Unsupported`](ErrorKind::Unsupported) when this machine has no
/// clipboard program, and [`NotFound`](ErrorKind::NotFound) when none of
/// them gave text.
pub fn read_text(launcher: &Launcher) -> io::Result<String> {
    match output_of(launcher, Use::ReadText) {
        (_, Some(bytes)) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        (programs, None) => Err(none_of(&programs, "text")),
    }
}

/// Saves the clipboard's image as a file in the directory `into`, named by
/// the time and the image's type, and returns its path; `None` when the
/// clipboard holds no image.
///
/// # Errors
///
/// [`Unsupported`](ErrorKind::Unsupported) when this machine has no
/// clipboard program, or the file could not be saved.
pub fn read_image(launcher: &Launcher, into: &Path) -> io::Result<Option<PathBuf>> {
    let (programs, printed) = output_of(launcher, Use::ReadImage);
    if programs.is_empty() {
        return Err(none_of(&programs, "image"));
    }
    let Some((bytes, kind)) =
        printed.and_then(|bytes| ImageMediaType::sniff(&bytes).map(|kind| (bytes, kind)))
    else {
        return Ok(None);
    };
    std::fs::create_dir_all(into)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or_default();
    let path = into.join(format!("pasted-{stamp}.{}", kind.extension()));
    std::fs::write(&path, bytes)?;
    Ok(Some(path))
}

/// Replaces the clipboard's text with `text`.
///
/// # Errors
///
/// [`Unsupported`](ErrorKind::Unsupported) when this machine has no
/// clipboard program, else why the last one tried failed.
pub fn write_text(launcher: &Launcher, text: &str) -> io::Result<()> {
    let programs = programs(Use::WriteText);
    let mut failed = none_of(&[], "");
    for (program, args) in programs {
        let written = launcher
            .command(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .and_then(|mut child| {
                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(text.as_bytes())?;
                }
                let status = child.wait()?;
                match status.success() {
                    true => Ok(()),
                    false => Err(io::Error::other(format!("{program} failed: {status}"))),
                }
            });
        match written {
            Ok(()) => return Ok(()),
            Err(error) => failed = error,
        }
    }
    Err(failed)
}
