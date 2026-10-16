//! The scrape side: a listener thread and a thread per connection, which
//! render the shared registry without the app's loop.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use rig_harness::prelude::Resource;

use crate::metrics_endpoint::registry::SharedRegistry;

/// How long a connection may take to send its request.
const WAIT: Duration = Duration::from_secs(5);

/// The longest request line or header line read.
const MAX_LINE: u64 = 8 * 1024;

/// The running listener. Dropping it stops the listener thread.
#[derive(Resource)]
pub(crate) struct Listener {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
}

impl Listener {
    /// The bound address.
    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wakes the blocking accept, which then sees `stop`.
        let mut addr = self.addr;
        if addr.ip().is_unspecified() {
            addr.set_ip([127, 0, 0, 1].into());
        }
        TcpStream::connect_timeout(&addr, Duration::from_secs(1)).ok();
    }
}

/// Binds `addr`, which must be a loopback address unless `public`, and
/// serves `registry` at `/metrics` from threads of its own.
pub(crate) fn start(
    addr: &str,
    public: bool,
    registry: SharedRegistry,
) -> Result<Listener, String> {
    let cannot =
        |failure: &dyn std::fmt::Display| format!("Cannot serve metrics on {addr}: {failure}.");
    let wanted = addr
        .to_socket_addrs()
        .map_err(|failure| cannot(&failure))?
        .next()
        .ok_or_else(|| cannot(&"it names no address"))?;
    if !wanted.ip().is_loopback() && !public {
        return Err(format!(
            "Refusing to serve metrics on {wanted}, which is not a loopback address: \
             set `metrics_public = true` in [metrics] to expose them to the network."
        ));
    }
    let listener = TcpListener::bind(wanted).map_err(|failure| cannot(&failure))?;
    let bound = listener.local_addr().map_err(|failure| cannot(&failure))?;
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    std::thread::Builder::new()
        .name("rig-metrics".to_owned())
        .spawn(move || {
            for stream in listener.incoming() {
                if stopped.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let registry = registry.clone();
                let serve = move || serve(stream, &registry).ok();
                std::thread::Builder::new()
                    .name("rig-metrics-scrape".to_owned())
                    .spawn(serve)
                    .ok();
            }
        })
        .map_err(|failure| cannot(&failure))?;
    Ok(Listener { addr: bound, stop })
}

/// Answers one request: the metrics for `GET /metrics`, else 404.
fn serve(stream: TcpStream, registry: &SharedRegistry) -> std::io::Result<()> {
    stream.set_read_timeout(Some(WAIT))?;
    stream.set_write_timeout(Some(WAIT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.by_ref().take(MAX_LINE).read_line(&mut line)?;
    // The headers say nothing a scrape needs; read up to the blank line.
    let mut header = String::new();
    while reader.by_ref().take(MAX_LINE).read_line(&mut header)? > 2 {
        header.clear();
    }
    let mut words = line.split_whitespace();
    let (method, path) = (words.next(), words.next());
    let path = path.map(|path| path.split('?').next().unwrap_or(path));
    let (status, kind, body) = match (method, path) {
        (Some("GET"), Some("/metrics")) => {
            let body = registry.lock().render(SystemTime::now());
            ("200 OK", "text/plain; version=0.0.4; charset=utf-8", body)
        }
        _ => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            "Not found: try /metrics\n".to_owned(),
        ),
    };
    let mut stream = stream;
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}
