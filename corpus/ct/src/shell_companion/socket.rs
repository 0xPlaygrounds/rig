//! The session's side of the socket: a listener thread, and per connection
//! a thread that reads the one request line, hands it to the app, then
//! writes the events the app sends back until the app drops their sender.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};

use async_channel::{Receiver, Sender};

use crate::shell_companion::ShellAsk;

/// The longest request line a client may send.
const MAX_LINE: u64 = 256 * 1024;

/// A request from a client and the sender of the events it gets, taken
/// once by the app.
pub(crate) struct Arrived(Mutex<Option<(ShellAsk, mpsc::Sender<String>)>>);

impl Arrived {
    pub(crate) fn take(&self) -> Option<(ShellAsk, mpsc::Sender<String>)> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

/// The socket and its listener thread; dropping it stops the listener and
/// removes the socket.
pub(crate) struct Listening {
    path: PathBuf,
    stop: Arc<AtomicBool>,
}

impl Drop for Listening {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wakes the blocking accept, which then sees `stop`.
        UnixStream::connect(&self.path).ok();
        std::fs::remove_file(&self.path).ok();
    }
}

/// Listens on `path`, the user's alone, unless a live session does;
/// removes a stale socket first.
pub(crate) fn listen(path: &Path) -> io::Result<(Listening, Receiver<Arrived>)> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("no socket directory"))?;
    std::fs::create_dir_all(dir)?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    if path.exists() {
        if UnixStream::connect(path).is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "another session serves this project",
            ));
        }
        std::fs::remove_file(path)?;
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    let (sender, arrivals) = async_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    std::thread::Builder::new()
        .name("rig-shell".to_owned())
        .spawn(move || {
            for stream in listener.incoming() {
                if stopped.load(Ordering::Relaxed) {
                    return;
                }
                let Ok(stream) = stream else { continue };
                let sender = sender.clone();
                std::thread::Builder::new()
                    .name("rig-shell-client".to_owned())
                    .spawn(move || serve(stream, &sender))
                    .ok();
            }
        })?;
    let listening = Listening {
        path: path.to_path_buf(),
        stop,
    };
    Ok((listening, arrivals))
}

/// Reads the request, hands it over, and writes back each event line.
fn serve(stream: UnixStream, arrivals: &Sender<Arrived>) {
    let Ok(reader) = stream.try_clone() else {
        return;
    };
    let mut line = String::new();
    if BufReader::new(reader)
        .take(MAX_LINE)
        .read_line(&mut line)
        .is_err()
    {
        return;
    }
    let Ok(ask) = serde_json::from_str::<ShellAsk>(&line) else {
        let mut stream = stream;
        writeln!(stream, r#"{{"type":"failed","why":"not a shell request"}}"#).ok();
        return;
    };
    let (events, lines) = mpsc::channel();
    if arrivals
        .send_blocking(Arrived(Mutex::new(Some((ask, events)))))
        .is_err()
    {
        return;
    }
    let mut stream = stream;
    for line in lines {
        if writeln!(stream, "{line}").is_err() {
            return;
        }
    }
}
