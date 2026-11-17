//! The system clipboard, read and written by the programs that own it on
//! this machine: `wl-paste`/`wl-copy` on Wayland, `xclip` on X11, and
//! `pbpaste`/`pbcopy` (`pngpaste` for images) on macOS. Each is started as
//! a [`Launcher`] says, such as one that keeps the session's secrets from
//! it. Blocking: call these off a thread that must not wait.

use std::io::{self, ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use rig_core::message::ImageMediaType;

use crate::process::{Launcher, Process, Spawn};

/// What a clipboard program does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    /// Prints the clipboard's text.
    ReadText,
    /// Prints the clipboard's image of this type.
    ReadImage(Image),
    /// Replaces the clipboard's text with its input.
    WriteText,
}

/// An image type a clipboard is asked for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Image {
    Png,
    Jpeg,
    Webp,
}

/// The image types a read asks for, in order, as a clipboard may offer an
/// image only as one of them.
const IMAGES: [Image; 3] = [Image::Png, Image::Jpeg, Image::Webp];

/// A clipboard this machine may have.
#[derive(Clone, Copy)]
enum Backend {
    Wayland,
    X11,
    Mac,
}

impl Backend {
    /// The program and arguments that do `what` with this clipboard.
    fn command(self, what: Use) -> Program {
        match (self, what) {
            (Self::Wayland, Use::ReadText) => ("wl-paste", &["--no-newline", "--type", "text"]),
            (Self::Wayland, Use::ReadImage(Image::Png)) => {
                ("wl-paste", &["--no-newline", "--type", "image/png"])
            }
            (Self::Wayland, Use::ReadImage(Image::Jpeg)) => {
                ("wl-paste", &["--no-newline", "--type", "image/jpeg"])
            }
            (Self::Wayland, Use::ReadImage(Image::Webp)) => {
                ("wl-paste", &["--no-newline", "--type", "image/webp"])
            }
            (Self::Wayland, Use::WriteText) => ("wl-copy", &[]),
            (Self::X11, Use::ReadText) => ("xclip", &["-selection", "clipboard", "-out"]),
            (Self::X11, Use::ReadImage(Image::Png)) => (
                "xclip",
                &["-selection", "clipboard", "-target", "image/png", "-out"],
            ),
            (Self::X11, Use::ReadImage(Image::Jpeg)) => (
                "xclip",
                &["-selection", "clipboard", "-target", "image/jpeg", "-out"],
            ),
            (Self::X11, Use::ReadImage(Image::Webp)) => (
                "xclip",
                &["-selection", "clipboard", "-target", "image/webp", "-out"],
            ),
            (Self::X11, Use::WriteText) => ("xclip", &["-selection", "clipboard", "-in"]),
            (Self::Mac, Use::ReadText) => ("pbpaste", &[]),
            (Self::Mac, Use::ReadImage(_)) => ("pngpaste", &["-"]),
            (Self::Mac, Use::WriteText) => ("pbcopy", &[]),
        }
    }
}

/// A program and its arguments.
type Program = (&'static str, &'static [&'static str]);

/// How long a read waits for the clipboard's program before it gives up.
const READ_LIMIT: Duration = Duration::from_secs(5);

/// The programs that do `what` on this machine, in the order tried.
fn programs(what: Use) -> Vec<Program> {
    [
        std::env::var_os("WAYLAND_DISPLAY")
            .is_some()
            .then_some(Backend::Wayland),
        std::env::var_os("DISPLAY")
            .is_some()
            .then_some(Backend::X11),
        // `pngpaste` prints any image as PNG, so only a PNG read asks it.
        (cfg!(target_os = "macos") && !matches!(what, Use::ReadImage(Image::Jpeg | Image::Webp)))
            .then_some(Backend::Mac),
    ]
    .into_iter()
    .flatten()
    .map(|backend| backend.command(what))
    .collect()
}

/// The error for a machine with no clipboard program to try.
fn no_clipboard() -> io::Error {
    io::Error::new(
        ErrorKind::Unsupported,
        "no clipboard here: it needs wl-paste and wl-copy, xclip, or pbpaste and pbcopy",
    )
}

/// The [`NotFound`](ErrorKind::NotFound) error naming the `missing`
/// programs, and the package of each that has another name.
fn not_installed(missing: &[&str]) -> io::Error {
    let named: Vec<String> = missing
        .iter()
        .map(|program| match *program {
            "wl-paste" | "wl-copy" => format!("{program} (from wl-clipboard)"),
            other => other.to_owned(),
        })
        .collect();
    let verb = if named.len() == 1 { "is" } else { "are" };
    let named = named.join(" and ");
    io::Error::new(ErrorKind::NotFound, format!("{named} {verb} not installed"))
}

/// What the first of `programs` that succeeds prints, waiting at most
/// `limit` for each; `None` when every one that ran failed.
///
/// # Errors
///
/// [`Unsupported`](ErrorKind::Unsupported) when there is none to try,
/// [`NotFound`](ErrorKind::NotFound) naming them when none is installed,
/// and [`TimedOut`](ErrorKind::TimedOut) naming the one that ran past
/// `limit`.
fn output_of(
    launcher: &Launcher,
    programs: &[Program],
    limit: Duration,
) -> io::Result<Option<Vec<u8>>> {
    if programs.is_empty() {
        return Err(no_clipboard());
    }
    let mut missing = Vec::new();
    for (program, args) in programs {
        let mut command = launcher.command(program);
        command.args(*args);
        match Process::start(command, Spawn::default()).and_then(|it| it.output(Some(limit))) {
            Ok(output) if output.status.success() => return Ok(Some(output.stdout)),
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => missing.push(*program),
            Err(error) if error.kind() == ErrorKind::TimedOut => {
                let secs = limit.as_secs_f32();
                let why = format!("{program} did not answer within {secs} s");
                return Err(io::Error::new(ErrorKind::TimedOut, why));
            }
            Err(error) => return Err(io::Error::new(error.kind(), format!("{program}: {error}"))),
        }
    }
    if missing.len() == programs.len() {
        return Err(not_installed(&missing));
    }
    Ok(None)
}

/// The clipboard's text; `None` when it holds none. Each program has
/// five seconds to answer.
///
/// # Errors
///
/// [`Unsupported`](ErrorKind::Unsupported) when this machine has no
/// clipboard program, [`NotFound`](ErrorKind::NotFound) naming the
/// programs when none is installed, and [`TimedOut`](ErrorKind::TimedOut)
/// when one did not answer.
pub fn read_text(launcher: &Launcher) -> io::Result<Option<String>> {
    let printed = output_of(launcher, &programs(Use::ReadText), READ_LIMIT)?;
    Ok(printed.map(|bytes| String::from_utf8_lossy(&bytes).into_owned()))
}

/// Saves the clipboard's image as a file in the directory `into`, named by
/// the time and the image's type, and returns its path; `None` when the
/// clipboard holds no image. It asks for a PNG, then a JPEG, then a WebP;
/// each program has five seconds to answer.
///
/// # Errors
///
/// As [`read_text`], or why the file could not be saved.
pub fn read_image(launcher: &Launcher, into: &Path) -> io::Result<Option<PathBuf>> {
    let mut asked = false;
    for image in IMAGES {
        let programs = programs(Use::ReadImage(image));
        if programs.is_empty() {
            continue;
        }
        asked = true;
        let printed = output_of(launcher, &programs, READ_LIMIT)?;
        let sniffed =
            printed.and_then(|bytes| ImageMediaType::sniff(&bytes).map(|kind| (bytes, kind)));
        if let Some((bytes, kind)) = sniffed {
            std::fs::create_dir_all(into)?;
            let stamp = rig_core::now_ms();
            let path = into.join(format!("pasted-{stamp}.{}", kind.extension()));
            std::fs::write(&path, bytes)?;
            return Ok(Some(path));
        }
    }
    if !asked {
        return Err(no_clipboard());
    }
    Ok(None)
}

/// Replaces the clipboard's text with `text`.
///
/// # Errors
///
/// [`Unsupported`](ErrorKind::Unsupported) when this machine has no
/// clipboard program, [`NotFound`](ErrorKind::NotFound) naming the
/// programs when none is installed, else why the last one tried failed.
pub fn write_text(launcher: &Launcher, text: &str) -> io::Result<()> {
    write_with(launcher, &programs(Use::WriteText), text)
}

/// Gives `text` to the first of `programs` that takes it, as
/// [`write_text`] does.
fn write_with(launcher: &Launcher, programs: &[Program], text: &str) -> io::Result<()> {
    let mut failed = no_clipboard();
    let mut missing = Vec::new();
    for (program, args) in programs {
        let mut command = launcher.command(program);
        command
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let written = Process::start_with_stdio(command).and_then(|mut process| {
            if let Some(mut stdin) = process.take_stdin() {
                stdin.write_all(text.as_bytes())?;
            }
            match process.wait_for(Duration::MAX)? {
                // What it left behind serves the selection after rig.
                Some(status) if status.success() => {
                    process.release();
                    Ok(())
                }
                Some(status) => Err(io::Error::other(format!("{program} failed: {status}"))),
                None => Err(io::Error::other(format!("{program} did not end"))),
            }
        });
        match written {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => missing.push(*program),
            Err(error) => failed = error,
        }
    }
    if !programs.is_empty() && missing.len() == programs.len() {
        return Err(not_installed(&missing));
    }
    Err(failed)
}

#[cfg(test)]
mod tests;
