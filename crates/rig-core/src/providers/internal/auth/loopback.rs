//! A one-shot loopback listener for an OAuth authorization-code sign-in in
//! the browser. It binds a local port, reports the redirect address to send
//! to the provider, and hands back the code the browser returns, checked
//! against the sign-in's state. The page the browser lands on names the
//! provider and the app to return to. Native only.
//!
//! ```no_run
//! use rig_core::providers::internal::loopback::{Loopback, LoopbackHost};
//! use std::time::{Duration, Instant};
//!
//! # async fn sign_in() -> Result<(), rig_core::providers::internal::loopback::LoopbackError> {
//! let listener = Loopback::new("Example", LoopbackHost::Localhost, [0], "/callback")
//!     .listen("state".into(), Instant::now() + Duration::from_secs(300))?;
//! // Send the user to the provider with `listener.redirect_uri()`, then:
//! let received = listener.code().await?;
//! // Exchange `received.code()` for tokens, then tell the browser:
//! received.succeeded();
//! # Ok(())
//! # }
//! ```

use futures::channel::oneshot;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How often the listener checks whether its waiter is gone.
const ACCEPT_POLL: Duration = Duration::from_millis(100);
/// Reading one request head, or writing a page, gives up after this.
const STREAM_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the browser waits for the token exchange before its page.
const EXCHANGE_WAIT: Duration = Duration::from_secs(120);
const MAX_REQUEST_BYTES: usize = 16 * 1024;
/// How many times a port the system picks is picked again when its IPv6
/// side is taken.
const ANY_PORT_ATTEMPTS: usize = 6;

/// The host the redirect address names. The listener binds loopback only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopbackHost {
    /// `127.0.0.1`, listening on IPv4.
    Ipv4,
    /// `localhost`, listening on IPv4 and, where the system has it, on the
    /// same port on IPv6, since a browser may resolve `localhost` to either.
    Localhost,
}

/// Why a loopback sign-in ended without a code.
#[derive(Debug, thiserror::Error)]
pub enum LoopbackError {
    /// None of the ports could be bound. Of kind
    /// [`io::ErrorKind::AddrInUse`] when every port was taken.
    #[error(transparent)]
    Bind(io::Error),
    /// The listener failed after binding.
    #[error(transparent)]
    Io(io::Error),
    /// The deadline passed before the browser came back.
    #[error("timed out waiting for the sign-in in the browser")]
    TimedOut,
    /// The provider refused or failed the sign-in; the message says why.
    #[error("{0}")]
    Refused(String),
    /// The listener's thread ended without an answer.
    #[error("the sign-in listener stopped unexpectedly")]
    Stopped,
}

/// Where a browser sign-in returns: the provider and app its pages name,
/// the host, the ports to try in order (`0` lets the system pick one), and
/// the callback path.
#[derive(Clone, Debug)]
pub struct Loopback {
    provider: String,
    app: String,
    host: LoopbackHost,
    ports: Vec<u16>,
    path: String,
}

impl Loopback {
    /// A listener for `provider`'s sign-in on `host` at the first free port
    /// of `ports`, taking callbacks on `path`, such as `/callback`. Its
    /// pages tell the user to return to `rig` until
    /// [`Self::with_app_name`] says otherwise.
    pub fn new(
        provider: impl Into<String>,
        host: LoopbackHost,
        ports: impl IntoIterator<Item = u16>,
        path: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            app: "rig".into(),
            host,
            ports: ports.into_iter().collect(),
            path: path.into(),
        }
    }

    /// This listener's pages telling the user to return to `app`.
    #[must_use]
    pub fn with_app_name(mut self, app: impl Into<String>) -> Self {
        self.app = app.into();
        self
    }

    /// Binds the first free port and serves it on a thread of its own until
    /// a callback with `state` arrives, `deadline` passes, or the returned
    /// [`Listener`] is dropped, which the thread notices within a fraction
    /// of a second. The port is freed when the thread ends.
    ///
    /// # Errors
    ///
    /// [`LoopbackError::Bind`] when no port could be bound, of kind
    /// [`io::ErrorKind::AddrInUse`] when every one was taken, and
    /// [`LoopbackError::Io`] when the thread cannot start.
    pub fn listen(&self, state: String, deadline: Instant) -> Result<Listener, LoopbackError> {
        let (listeners, port) = self.bind()?;
        for listener in &listeners {
            listener.set_nonblocking(true).map_err(LoopbackError::Io)?;
        }
        let host = match self.host {
            LoopbackHost::Ipv4 => "127.0.0.1",
            LoopbackHost::Localhost => "localhost",
        };
        let redirect_uri = format!("http://{host}:{port}{}", self.path);
        let (sender, callback) = oneshot::channel();
        let serving = Serving {
            pages: Pages {
                provider: escape_html(&self.provider),
                app: escape_html(&self.app),
            },
            path: self.path.clone(),
            state,
            deadline,
        };
        std::thread::Builder::new()
            .name("sign-in-listener".into())
            .spawn(move || serving.serve(&listeners, sender))
            .map_err(LoopbackError::Io)?;
        Ok(Listener {
            redirect_uri,
            callback,
        })
    }

    /// The listeners on the first port of [`Self::ports`] free on every
    /// address the host needs, and that port.
    fn bind(&self) -> Result<(Vec<TcpListener>, u16), LoopbackError> {
        for &port in &self.ports {
            let attempts = if port == 0 { ANY_PORT_ATTEMPTS } else { 1 };
            for _ in 0..attempts {
                if let Some(bound) = self.bind_port(port)? {
                    return Ok(bound);
                }
            }
        }
        let ports: Vec<String> = self.ports.iter().map(u16::to_string).collect();
        Err(LoopbackError::Bind(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!(
                "the sign-in callback ports {} on the loopback address are in use",
                ports.join(", ")
            ),
        )))
    }

    /// The listeners on `port`, or `None` when it is taken.
    fn bind_port(&self, port: u16) -> Result<Option<(Vec<TcpListener>, u16)>, LoopbackError> {
        let ipv4 = match TcpListener::bind((Ipv4Addr::LOCALHOST, port)) {
            Ok(listener) => listener,
            Err(error) if taken(&error) => return Ok(None),
            Err(error) => return Err(LoopbackError::Bind(error)),
        };
        let port = ipv4.local_addr().map_err(LoopbackError::Bind)?.port();
        if self.host == LoopbackHost::Ipv4 {
            return Ok(Some((vec![ipv4], port)));
        }
        match TcpListener::bind((Ipv6Addr::LOCALHOST, port)) {
            Ok(ipv6) => Ok(Some((vec![ipv4, ipv6], port))),
            Err(error) if taken(&error) => Ok(None),
            // No IPv6 loopback here, so `localhost` resolves to IPv4.
            Err(_) => Ok(Some((vec![ipv4], port))),
        }
    }
}

/// Windows reports a reserved or taken port as access denied.
fn taken(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::AddrInUse | io::ErrorKind::PermissionDenied
    )
}

/// A listening sign-in: the redirect address to send to the provider, and
/// the code to wait for. Dropping it stops the listener.
pub struct Listener {
    redirect_uri: String,
    callback: oneshot::Receiver<Result<Received, LoopbackError>>,
}

impl Listener {
    /// The address the provider sends the browser back to, such as
    /// `http://localhost:54545/callback`.
    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }

    /// Waits for the browser to come back with the code.
    ///
    /// # Errors
    ///
    /// [`LoopbackError::Refused`] when the provider refused the sign-in,
    /// [`LoopbackError::TimedOut`] when the deadline passed,
    /// [`LoopbackError::Io`] when the listener failed, and
    /// [`LoopbackError::Stopped`] when its thread ended without an answer.
    pub async fn code(self) -> Result<Received, LoopbackError> {
        self.callback.await.map_err(|_| LoopbackError::Stopped)?
    }
}

/// The code the browser brought back, its state checked. The browser waits
/// for [`Self::succeeded`] or [`Self::failed`] before showing its page;
/// dropping this tells it the sign-in was cancelled.
pub struct Received {
    code: String,
    outcome: mpsc::Sender<Result<(), String>>,
}

impl Received {
    /// The authorization code.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Tells the browser the sign-in is done.
    pub fn succeeded(self) {
        self.outcome.send(Ok(())).ok();
    }

    /// Tells the browser the sign-in failed, for `reason`.
    pub fn failed(self, reason: &str) {
        self.outcome.send(Err(reason.to_owned())).ok();
    }
}

/// What one request to the listener asked for.
#[derive(Debug, PartialEq, Eq)]
enum Callback {
    /// The authorization code, its state checked.
    Code(String),
    /// The sign-in failed or was refused; the message says why.
    Failed(String),
    /// A callback whose state is not this sign-in's, such as a stale tab.
    StateMismatch,
    /// This sign-in's callback without exactly one non-empty code.
    NoCode,
    /// Anything else, such as `/favicon.ico`.
    Other,
}

/// Reads the request line of a request to the listener, whose callback is
/// on `path`. A repeated state is a mismatch and a repeated code no code,
/// so no copy is picked over another.
fn parse_callback(request_line: &str, path: &str, expected_state: &str) -> Callback {
    let mut parts = request_line.split_whitespace();
    let (Some("GET"), Some(target)) = (parts.next(), parts.next()) else {
        return Callback::Other;
    };
    let (target_path, query) = target.split_once('?').unwrap_or((target, ""));
    if target_path != path {
        return Callback::Other;
    }
    let (mut code, mut state, mut error, mut description) = (Vec::new(), Vec::new(), None, None);
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        match key.as_ref() {
            "code" => code.push(value.into_owned()),
            "state" => state.push(value.into_owned()),
            "error" => error = Some(value.into_owned()),
            "error_description" => description = Some(value.into_owned()),
            _ => {}
        }
    }
    if state.as_slice() != [expected_state] {
        return Callback::StateMismatch;
    }
    if let Some(error) = error {
        return Callback::Failed(match description.filter(|text| !text.trim().is_empty()) {
            Some(description) => format!("the browser sign-in returned {error} ({description})"),
            None => format!("the browser sign-in returned {error}"),
        });
    }
    match code.as_slice() {
        [code] if !code.is_empty() => Callback::Code(code.clone()),
        _ => Callback::NoCode,
    }
}

/// What the listener's thread needs.
struct Serving {
    pages: Pages,
    path: String,
    state: String,
    deadline: Instant,
}

impl Serving {
    fn serve(
        &self,
        listeners: &[TcpListener],
        sender: oneshot::Sender<Result<Received, LoopbackError>>,
    ) {
        let pages = &self.pages;
        let result = 'serve: loop {
            if sender.is_canceled() {
                return;
            }
            if Instant::now() >= self.deadline {
                break Err(LoopbackError::TimedOut);
            }
            let mut accepted = None;
            for listener in listeners {
                match listener.accept() {
                    Ok((stream, _)) => {
                        accepted = Some(stream);
                        break;
                    }
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                        ) => {}
                    Err(error) => break 'serve Err(LoopbackError::Io(error)),
                }
            }
            let Some(mut stream) = accepted else {
                std::thread::sleep(ACCEPT_POLL);
                continue;
            };
            // A browser that stalls or sends garbage only loses its own request.
            let until = (Instant::now() + STREAM_TIMEOUT).min(self.deadline);
            let Ok(line) = read_request_line(&mut stream, until, || sender.is_canceled()) else {
                continue;
            };
            match parse_callback(&line, &self.path, &self.state) {
                Callback::Other => respond(&mut stream, "404 Not Found", "Not found."),
                Callback::StateMismatch => {
                    respond(&mut stream, "400 Bad Request", &pages.state_mismatch())
                }
                Callback::NoCode => respond(&mut stream, "400 Bad Request", &pages.no_code()),
                Callback::Failed(message) => {
                    respond(&mut stream, "200 OK", &pages.failed(&message));
                    break Err(LoopbackError::Refused(message));
                }
                Callback::Code(code) => {
                    let (outcome, wait) = mpsc::channel();
                    if sender.send(Ok(Received { code, outcome })).is_err() {
                        respond(&mut stream, "200 OK", &pages.cancelled());
                        return;
                    }
                    let body = match wait.recv_timeout(EXCHANGE_WAIT) {
                        Ok(Ok(())) => pages.signed_in(),
                        Ok(Err(message)) => pages.failed(&message),
                        Err(_) => pages.cancelled(),
                    };
                    respond(&mut stream, "200 OK", &body);
                    return;
                }
            }
        };
        sender.send(result).ok();
    }
}

/// Reads the request head and returns its first line, giving up at `until`
/// or once `stopped` says the waiter is gone. The whole head is read so that
/// closing the connection does not reset it before the page.
fn read_request_line(
    stream: &mut TcpStream,
    until: Instant,
    stopped: impl Fn() -> bool,
) -> io::Result<String> {
    // Accepted sockets inherit non-blocking mode on some platforms.
    stream.set_nonblocking(false)?;
    stream.set_write_timeout(Some(STREAM_TIMEOUT))?;
    let mut head = Vec::new();
    let mut chunk = [0u8; 1024];
    while !head.windows(4).any(|window| window == b"\r\n\r\n") {
        if head.len() >= MAX_REQUEST_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request head too large",
            ));
        }
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() || stopped() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        // Short reads, so a client trickling bytes cannot outlast `until`.
        stream.set_read_timeout(Some(left.min(ACCEPT_POLL)))?;
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => head.extend(chunk.iter().take(read)),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            Err(error) => return Err(error),
        }
    }
    let head = String::from_utf8_lossy(&head);
    Ok(head.lines().next().unwrap_or_default().to_owned())
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).ok();
    stream.flush().ok();
}

/// The listener's pages, with the provider and app names HTML-escaped.
struct Pages {
    provider: String,
    app: String,
}

impl Pages {
    fn signed_in(&self) -> String {
        let Self { provider, app } = self;
        page(
            &format!("Signed in to {provider}"),
            &format!("You can close this tab and return to {app}."),
        )
    }

    fn failed(&self, message: &str) -> String {
        let message = escape_html(message);
        page(
            "Sign-in failed",
            &format!("Sign-in failed: {message}. Return to {}.", self.app),
        )
    }

    fn state_mismatch(&self) -> String {
        let app = &self.app;
        page(
            "Sign-in not recognised",
            &format!(
                "This sign-in link is not the one {app} is waiting for. Start it again from {app}."
            ),
        )
    }

    fn no_code(&self) -> String {
        let app = &self.app;
        page(
            "No sign-in code",
            &format!(
                "This page brought back no sign-in code. {app} is still waiting: finish signing in from the provider's page."
            ),
        )
    }

    fn cancelled(&self) -> String {
        let app = &self.app;
        page(
            "Sign-in cancelled",
            &format!("{app} stopped waiting for this sign-in. Start it again from {app}."),
        )
    }
}

/// A page from an escaped `title` and `message`.
fn page(title: &str, message: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title></head>\
         <body style=\"font-family: system-ui, sans-serif; margin: 4rem auto; max-width: 32rem\">\
         <h1>{title}</h1><p>{message}</p></body></html>"
    )
}

fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            other => escaped.push(other),
        }
    }
    escaped
}

#[cfg(test)]
mod tests;
