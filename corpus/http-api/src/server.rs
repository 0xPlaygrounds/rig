//! The HTTP side: a listener thread, and a thread per connection that
//! parses one request, hands it to the app through a channel (waking its
//! loop, as the terminal view's input thread does) and writes the reply,
//! or streams the app's events as Server-Sent Events.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use rig_harness::prelude::{DeliveryMode, Resource, Wake};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::Serve;

/// How long a connection waits for the app's answer, or for a request.
const WAIT: Duration = Duration::from_secs(10);

/// How often an idle event stream says it is alive, which also finds
/// clients that went away.
const KEEPALIVE: Duration = Duration::from_secs(15);

/// The events a client may lag behind before it is dropped.
const EVENT_BUFFER: usize = 256;

/// The longest request line or header line.
const MAX_LINE: u64 = 8 * 1024;

/// The most headers a request may have.
const MAX_HEADERS: usize = 64;

/// The largest request body.
const MAX_BODY: usize = 1 << 20;

/// What a client asks the app.
pub(crate) enum Call {
    /// `GET /agents`.
    Agents,
    /// `GET /agents/{id}/conversation`.
    Conversation(String),
    /// `POST /agents/{id}/messages`.
    Deliver {
        agent: String,
        text: String,
        mode: DeliveryMode,
    },
    /// `POST /agents/{id}/interrupt`.
    Interrupt(String),
    /// `POST /approvals/{id}`.
    Answer { id: u64, choice: Value },
}

/// The app's answer to a [`Call`]: an HTTP status and a JSON body.
pub(crate) struct Reply {
    pub(crate) status: u16,
    pub(crate) body: Value,
}

impl Reply {
    pub(crate) fn new(status: u16, body: Value) -> Self {
        Self { status, body }
    }

    pub(crate) fn error(status: u16, why: impl Into<String>) -> Self {
        Self::new(status, json!({ "error": why.into() }))
    }
}

/// What a connection hands the app.
pub(crate) enum Request {
    /// A call, answered on the sender.
    Call(Call, Sender<Reply>),
    /// An event stream: the app sends each event, formatted, and drops the
    /// sender of a client that lags [`EVENT_BUFFER`] events behind.
    Subscribe(Sender<String>),
}

/// The running server, as the app holds it: the requests of its
/// connections. Dropping it stops the listener.
#[derive(Resource)]
pub(crate) struct Server {
    pub(crate) requests: Receiver<Request>,
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wakes the listener's blocking accept, which then sees `stop`.
        let mut addr = self.addr;
        if addr.ip().is_unspecified() {
            addr.set_ip([127, 0, 0, 1].into());
        }
        TcpStream::connect_timeout(&addr, Duration::from_secs(1)).ok();
    }
}

/// Binds `serve.addr`, writes the token file, and starts the listener.
/// Returns the bound address and the token with the server.
pub(crate) fn start(serve: &Serve, wake: Wake) -> Result<(SocketAddr, String, Server), String> {
    let addr = serve
        .addr
        .to_socket_addrs()
        .map_err(|failure| format!("cannot serve on {}: {failure}", serve.addr))?
        .next()
        .ok_or_else(|| format!("cannot serve on {}: no address", serve.addr))?;
    if !addr.ip().is_loopback() && !serve.allow_remote {
        return Err(format!(
            "refusing to serve on {addr}, which is not localhost, without allow_remote"
        ));
    }
    let listener =
        TcpListener::bind(addr).map_err(|failure| format!("cannot serve on {addr}: {failure}"))?;
    let addr = listener
        .local_addr()
        .map_err(|failure| format!("cannot serve on {addr}: {failure}"))?;
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    if let Some(path) = &serve.token_file {
        write_token(path, &token)
            .map_err(|failure| format!("cannot write {}: {failure}", path.display()))?;
    }
    let (sender, requests) = crossbeam_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    let shared = token.clone();
    std::thread::Builder::new()
        .name("rig-http-api".to_owned())
        .spawn(move || listen(&listener, &stopped, &shared, &sender, &wake))
        .map_err(|failure| format!("cannot start the HTTP API: {failure}"))?;
    Ok((
        addr,
        token,
        Server {
            requests,
            addr,
            stop,
        },
    ))
}

/// Writes the token, readable by the user alone.
fn write_token(path: &Path, token: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)?.write_all(token.as_bytes())
}

/// Accepts connections until `stop`, each on a thread of its own.
fn listen(
    listener: &TcpListener,
    stop: &AtomicBool,
    token: &str,
    requests: &Sender<Request>,
    wake: &Wake,
) {
    for stream in listener.incoming() {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Ok(stream) = stream else {
            continue;
        };
        let (token, requests, wake) = (token.to_owned(), requests.clone(), wake.clone());
        std::thread::Builder::new()
            .name("rig-http-api-connection".to_owned())
            .spawn(move || connection(stream, &token, &requests, &wake).ok())
            .ok();
    }
}

/// A parsed request.
struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

/// The body of `POST /agents/{id}/messages`.
#[derive(Deserialize)]
struct MessageBody {
    text: String,
    #[serde(default)]
    mode: DeliveryMode,
}

/// The body of `POST /approvals/{id}`: the choice's position, label or
/// command.
#[derive(Deserialize)]
struct AnswerBody {
    choice: Value,
}

/// Serves one request of `stream`.
fn connection(
    mut stream: TcpStream,
    token: &str,
    requests: &Sender<Request>,
    wake: &Wake,
) -> io::Result<()> {
    stream.set_read_timeout(Some(WAIT))?;
    stream.set_write_timeout(Some(WAIT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let Some(request) = read_request(&mut reader)? else {
        return respond(&mut stream, &Reply::error(400, "bad request"));
    };
    let bearer = format!("Bearer {token}");
    if request.authorization.as_deref() != Some(bearer.as_str()) {
        return respond(
            &mut stream,
            &Reply::error(401, "missing or wrong bearer token"),
        );
    }
    let path = request.path.split('?').next().unwrap_or_default();
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();
    let call = match (request.method.as_str(), segments.as_slice()) {
        ("GET", ["events"]) => return events(stream, requests, wake),
        ("GET", ["agents"]) => Ok(Call::Agents),
        ("GET", ["agents", agent, "conversation"]) => Ok(Call::Conversation((*agent).to_owned())),
        ("POST", ["agents", agent, "interrupt"]) => Ok(Call::Interrupt((*agent).to_owned())),
        ("POST", ["agents", agent, "messages"]) => serde_json::from_slice(&request.body)
            .map(|body: MessageBody| Call::Deliver {
                agent: (*agent).to_owned(),
                text: body.text,
                mode: body.mode,
            })
            .map_err(|failure| Reply::error(400, failure.to_string())),
        ("POST", ["approvals", id]) => match (
            id.parse(),
            serde_json::from_slice::<AnswerBody>(&request.body),
        ) {
            (Ok(id), Ok(body)) => Ok(Call::Answer {
                id,
                choice: body.choice,
            }),
            _ => Err(Reply::error(
                400,
                "expected /approvals/<number> {\"choice\": …}",
            )),
        },
        _ => Err(Reply::error(404, "no such endpoint")),
    };
    let reply = match call {
        Ok(call) => ask(call, requests, wake),
        Err(reply) => reply,
    };
    respond(&mut stream, &reply)
}

/// Hands `call` to the app and waits for its reply.
fn ask(call: Call, requests: &Sender<Request>, wake: &Wake) -> Reply {
    let (sender, replies) = crossbeam_channel::bounded(1);
    if requests.send(Request::Call(call, sender)).is_err() {
        return Reply::error(503, "the agent is shutting down");
    }
    wake.wake();
    replies
        .recv_timeout(WAIT)
        .unwrap_or_else(|_| Reply::error(503, "the agent did not answer"))
}

/// Streams the app's events to the client until it goes away or lags.
fn events(mut stream: TcpStream, requests: &Sender<Request>, wake: &Wake) -> io::Result<()> {
    let (sender, events) = crossbeam_channel::bounded(EVENT_BUFFER);
    if requests.send(Request::Subscribe(sender)).is_err() {
        return respond(
            &mut stream,
            &Reply::error(503, "the agent is shutting down"),
        );
    }
    wake.wake();
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()?;
    loop {
        match events.recv_timeout(KEEPALIVE) {
            Ok(event) => stream.write_all(event.as_bytes())?,
            Err(RecvTimeoutError::Timeout) => stream.write_all(b": keepalive\n\n")?,
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
        stream.flush()?;
    }
}

/// Reads one request; `None` when it is malformed or too large.
fn read_request(reader: &mut impl BufRead) -> io::Result<Option<HttpRequest>> {
    let Some(line) = read_line(reader)? else {
        return Ok(None);
    };
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        return Ok(None);
    };
    let (method, path) = (method.to_owned(), path.to_owned());
    let mut authorization = None;
    let mut length = 0;
    for _ in 0..MAX_HEADERS {
        let Some(line) = read_line(reader)? else {
            return Ok(None);
        };
        if line.is_empty() {
            if length > MAX_BODY {
                return Ok(None);
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body)?;
            return Ok(Some(HttpRequest {
                method,
                path,
                authorization,
                body,
            }));
        }
        let Some((name, value)) = line.split_once(':') else {
            return Ok(None);
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "authorization" => authorization = Some(value.to_owned()),
            "content-length" => match value.parse() {
                Ok(parsed) => length = parsed,
                Err(_) => return Ok(None),
            },
            _ => {}
        }
    }
    Ok(None)
}

/// One line without its line ending; `None` at the end of the stream or
/// past [`MAX_LINE`].
fn read_line(reader: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut line = String::new();
    let read = reader.by_ref().take(MAX_LINE).read_line(&mut line)?;
    if read == 0 || !line.ends_with('\n') {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\r', '\n']).to_owned()))
}

/// Writes `reply` as a JSON response and closes.
fn respond(stream: &mut TcpStream, reply: &Reply) -> io::Result<()> {
    let body = reply.body.to_string();
    let reason = match reply.status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Service Unavailable",
    };
    write!(
        stream,
        "HTTP/1.1 {} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        reply.status,
        body.len()
    )?;
    stream.flush()
}
