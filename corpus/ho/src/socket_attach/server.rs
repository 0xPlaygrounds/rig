//! The socket side: a listener thread that hands each connection to the
//! app, and per client a writer thread that sends what the app queued and a
//! reader thread that hands the app each line the client sends.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use async_channel::{Receiver, Sender};

/// How far behind a client may fall, in bytes queued for it and not yet
/// written to its socket, before it is dropped.
pub(crate) const LIMIT: usize = 1 << 20;

/// The longest line a client may send.
const MAX_LINE: u64 = 64 * 1024;

/// How long the session's exit waits, at most, for the clients to get
/// what was queued for them.
const FLUSH: Duration = Duration::from_secs(1);

/// What the socket hands the app.
pub(crate) enum Request {
    /// A client connected.
    Connect(UnixStream),
    /// Client `client` sent `line`.
    Line { client: u64, line: String },
    /// Client `client` went away.
    Gone(u64),
}

/// The socket `RIG_HOME/sockets/<session>.sock` and its listener thread.
/// Dropping it stops the listener and removes the socket.
pub(crate) struct Listening {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    /// For the clients' reader threads.
    pub(crate) requests: Sender<Request>,
}

impl Drop for Listening {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wakes the blocking accept, which then sees `stop`.
        UnixStream::connect(&self.path).ok();
        std::fs::remove_file(&self.path).ok();
    }
}

/// The directory of the sessions' sockets.
pub(crate) fn sockets(home: &Path) -> PathBuf {
    home.join("sockets")
}

/// Removes the sockets in `dir` no session listens on.
fn remove_stale(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
        let socket = path
            .extension()
            .is_some_and(|extension| extension == "sock");
        if socket && UnixStream::connect(&path).is_err() {
            std::fs::remove_file(&path).ok();
        }
    }
}

/// Removes the stale sockets under `home`, listens on the session's own,
/// readable by the user alone, and returns what its connections hand over.
pub(crate) fn listen(home: &Path, session: &str) -> io::Result<(Listening, Receiver<Request>)> {
    let dir = sockets(home);
    std::fs::create_dir_all(&dir)?;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    remove_stale(&dir);
    let path = dir.join(format!("{session}.sock"));
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!("{} is served by another process", path.display()),
        ));
    }
    let listener = UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    let (requests, received) = async_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let (stopped, sender) = (Arc::clone(&stop), requests.clone());
    let accept = move || {
        for stream in listener.incoming() {
            if stopped.load(Ordering::Relaxed) {
                return;
            }
            let Ok(stream) = stream else {
                continue;
            };
            if sender.send_blocking(Request::Connect(stream)).is_err() {
                return;
            }
        }
    };
    let listening = Listening {
        path,
        stop,
        requests,
    };
    std::thread::Builder::new()
        .name("rig-attach".to_owned())
        .spawn(accept)?;
    Ok((listening, received))
}

/// Whether a line reached a client's queue.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Sent {
    /// It is queued.
    Queued,
    /// The client is more than [`LIMIT`] behind.
    Behind,
    /// The client went away.
    Gone,
}

/// A connected client. Dropping it closes the connection.
pub(crate) struct Client {
    /// The client's number, in the order they connected.
    pub(crate) id: u64,
    lines: Option<mpsc::Sender<Arc<str>>>,
    /// The bytes queued and not yet written.
    pending: Arc<AtomicUsize>,
    stream: UnixStream,
    /// Closed once the writer wrote all it was given.
    written: Mutex<mpsc::Receiver<()>>,
}

impl Client {
    /// Starts the threads that write to and read from `stream`.
    pub(crate) fn open(id: u64, stream: UnixStream, requests: Sender<Request>) -> io::Result<Self> {
        let (lines, queued) = mpsc::channel::<Arc<str>>();
        let (done, written) = mpsc::channel();
        let pending = Arc::new(AtomicUsize::new(0));
        let mut writer = stream.try_clone()?;
        let left = Arc::clone(&pending);
        std::thread::Builder::new()
            .name("rig-attach-write".to_owned())
            .spawn(move || {
                for line in queued {
                    if writer.write_all(line.as_bytes()).is_err() {
                        break;
                    }
                    left.fetch_sub(line.len(), Ordering::Relaxed);
                }
                drop(done);
            })?;
        let reader = BufReader::new(stream.try_clone()?);
        std::thread::Builder::new()
            .name("rig-attach-read".to_owned())
            .spawn(move || read_lines(id, reader, &requests))?;
        Ok(Self {
            id,
            lines: Some(lines),
            pending,
            stream,
            written: Mutex::new(written),
        })
    }

    /// Queues `line`, unless the client is already [`LIMIT`] behind or
    /// gone; never waits on it.
    pub(crate) fn send(&self, line: &Arc<str>) -> Sent {
        let Some(lines) = &self.lines else {
            return Sent::Gone;
        };
        if self.pending.load(Ordering::Relaxed) > LIMIT {
            return Sent::Behind;
        }
        self.pending.fetch_add(line.len(), Ordering::Relaxed);
        match lines.send(Arc::clone(line)) {
            Ok(()) => Sent::Queued,
            Err(_) => Sent::Gone,
        }
    }

    /// Lets the writer write what is queued, until `by`, then closes.
    pub(crate) fn finish(mut self, by: Instant) {
        self.lines = None;
        let left = by.saturating_duration_since(Instant::now());
        let written = self
            .written
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        written.recv_timeout(left).ok();
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // Ends a writer blocked on a client that does not read, and the
        // reader.
        self.stream.shutdown(std::net::Shutdown::Both).ok();
    }
}

/// Closes every client once each got what was queued for it, or
/// [`FLUSH`] passed.
pub(crate) fn finish_all(clients: Vec<Client>) {
    let by = Instant::now() + FLUSH;
    for client in clients {
        client.finish(by);
    }
}

/// Hands the app each line of client `id`, then says it went away.
fn read_lines(id: u64, mut reader: BufReader<UnixStream>, requests: &Sender<Request>) {
    loop {
        let mut line = String::new();
        match reader.by_ref().take(MAX_LINE).read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let line = line.trim_end_matches(['\r', '\n']).to_owned();
                if requests
                    .send_blocking(Request::Line { client: id, line })
                    .is_err()
                {
                    return;
                }
            }
        }
    }
    requests.send_blocking(Request::Gone(id)).ok();
}
