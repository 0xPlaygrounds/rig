//! Browser sign-in pieces: PKCE codes, the authorize URL, a one-shot
//! callback listener on `127.0.0.1`, and launching the default browser.

use super::{AuthError, CHATGPT_AUTH_BASE, CHATGPT_CLIENT_ID};
use base64::Engine;
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
use futures::channel::oneshot;
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The registered callback ports, tried in order.
pub(super) const CALLBACK_PORTS: [u16; 2] = [1455, 1457];
pub(super) const CALLBACK_PATH: &str = "/auth/callback";
/// Enough for the token to work and to refresh it.
pub(super) const SCOPE: &str = "openid profile email offline_access";
/// How often the listener checks whether its waiter is gone.
const ACCEPT_POLL: Duration = Duration::from_millis(100);
/// Reading a request or writing a page gives up after this.
const STREAM_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the browser waits for the token exchange before its page.
const EXCHANGE_WAIT: Duration = Duration::from_secs(120);
const MAX_REQUEST_BYTES: usize = 16 * 1024;

/// A PKCE verifier and its S256 challenge.
pub(super) struct Pkce {
    pub(super) verifier: String,
    pub(super) challenge: String,
}

impl Pkce {
    pub(super) fn generate() -> Self {
        // Four v4 UUIDs: 64 bytes from the OS generator.
        let verifier = random_token(4);
        Self {
            challenge: pkce_challenge(&verifier),
            verifier,
        }
    }
}

/// The S256 challenge of `verifier`.
pub(super) fn pkce_challenge(verifier: &str) -> String {
    BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// `uuids` × 16 random bytes, URL-safe base64.
pub(super) fn random_token(uuids: usize) -> String {
    let bytes: Vec<u8> = (0..uuids)
        .flat_map(|_| uuid::Uuid::new_v4().into_bytes())
        .collect();
    BASE64_URL_SAFE_NO_PAD.encode(bytes)
}

pub(super) fn redirect_uri(port: u16) -> String {
    format!("http://127.0.0.1:{port}{CALLBACK_PATH}")
}

/// The page that asks the user to sign in and sends them back to
/// `redirect_uri` with a code.
pub(super) fn authorize_url(
    redirect_uri: &str,
    code_challenge: &str,
    state: &str,
    originator: &str,
) -> String {
    let query = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs([
            ("response_type", "code"),
            ("client_id", CHATGPT_CLIENT_ID),
            ("redirect_uri", redirect_uri),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("scope", SCOPE),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", originator),
        ])
        .finish();
    format!("{CHATGPT_AUTH_BASE}/oauth/authorize?{query}")
}

/// What one request to the listener asked for.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Callback {
    /// The authorization code, its state checked.
    Code(String),
    /// The sign-in failed or was refused; the message says why.
    Failed(String),
    /// A callback whose state is not this sign-in's, such as a stale tab.
    StateMismatch,
    /// Anything else, such as `/favicon.ico`.
    Other,
}

/// Reads the request line of a request to the listener.
pub(super) fn parse_callback(request_line: &str, expected_state: &str) -> Callback {
    let mut parts = request_line.split_whitespace();
    let (Some("GET"), Some(target)) = (parts.next(), parts.next()) else {
        return Callback::Other;
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    if path != CALLBACK_PATH {
        return Callback::Other;
    }
    let (mut code, mut state, mut error, mut description) = (None, None, None, None);
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let slot = match key.as_ref() {
            "code" => &mut code,
            "state" => &mut state,
            "error" => &mut error,
            "error_description" => &mut description,
            _ => continue,
        };
        *slot = Some(value.into_owned());
    }
    if state.as_deref() != Some(expected_state) {
        return Callback::StateMismatch;
    }
    if let Some(error) = error {
        return Callback::Failed(match description.filter(|text| !text.trim().is_empty()) {
            Some(description) => format!("the browser sign-in returned {error} ({description})"),
            None => format!("the browser sign-in returned {error}"),
        });
    }
    match code.filter(|code| !code.is_empty()) {
        Some(code) => Callback::Code(code),
        None => Callback::Failed("the browser sign-in returned no authorization code".into()),
    }
}

/// Binds the first free callback port on `127.0.0.1`. When every port is
/// taken, returns [`AuthError::Io`] of kind [`io::ErrorKind::AddrInUse`].
pub(super) fn bind() -> Result<(TcpListener, u16), AuthError> {
    for port in CALLBACK_PORTS {
        match TcpListener::bind((Ipv4Addr::LOCALHOST, port)) {
            Ok(listener) => return Ok((listener, port)),
            // Windows reports a reserved or taken port as access denied.
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::AddrInUse | io::ErrorKind::PermissionDenied
                ) => {}
            Err(error) => return Err(error.into()),
        }
    }
    let [port, fallback] = CALLBACK_PORTS;
    Err(io::Error::new(
        io::ErrorKind::AddrInUse,
        format!("the sign-in callback ports {port} and {fallback} on 127.0.0.1 are in use"),
    )
    .into())
}

/// The code the browser brought back. [`Received::finish`] tells the
/// waiting browser how the sign-in ended; dropping it says it was cancelled.
pub(super) struct Received {
    pub(super) code: String,
    outcome: mpsc::Sender<Result<(), String>>,
}

impl Received {
    pub(super) fn finish(self, outcome: Result<(), String>) {
        self.outcome.send(outcome).ok();
    }
}

/// Serves `listener` on a thread of its own until a callback with `state`
/// arrives, `deadline` passes, or the returned receiver is dropped, which
/// the thread notices within [`ACCEPT_POLL`]. The listener closes when the
/// thread ends.
pub(super) fn spawn_listener(
    listener: TcpListener,
    state: String,
    deadline: Instant,
) -> Result<oneshot::Receiver<Result<Received, AuthError>>, AuthError> {
    listener.set_nonblocking(true)?;
    let (sender, receiver) = oneshot::channel();
    std::thread::Builder::new()
        .name("chatgpt-sign-in".into())
        .spawn(move || serve(listener, &state, deadline, sender))?;
    Ok(receiver)
}

fn serve(
    listener: TcpListener,
    state: &str,
    deadline: Instant,
    sender: oneshot::Sender<Result<Received, AuthError>>,
) {
    let result = loop {
        if sender.is_canceled() {
            return;
        }
        if Instant::now() >= deadline {
            break Err(AuthError::Message(
                "timed out waiting for the sign-in in the browser".into(),
            ));
        }
        let mut stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL);
                continue;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => break Err(error.into()),
        };
        // A browser that stalls or sends garbage only loses its own request.
        let Ok(line) = read_request_line(&mut stream) else {
            continue;
        };
        match parse_callback(&line, state) {
            Callback::Other => respond(&mut stream, "404 Not Found", "Not found."),
            Callback::StateMismatch => respond(
                &mut stream,
                "400 Bad Request",
                &page(
                    "Sign-in not recognised",
                    "This sign-in link is not the one rig is waiting for. Start it again from rig.",
                ),
            ),
            Callback::Failed(message) => {
                respond(
                    &mut stream,
                    "200 OK",
                    &page(
                        "Sign-in failed",
                        &format!("Sign-in failed: {message}. Return to rig."),
                    ),
                );
                break Err(AuthError::Message(message));
            }
            Callback::Code(code) => {
                let (outcome, wait) = mpsc::channel();
                if sender.send(Ok(Received { code, outcome })).is_err() {
                    respond(&mut stream, "200 OK", &cancelled_page());
                    return;
                }
                let body = match wait.recv_timeout(EXCHANGE_WAIT) {
                    Ok(Ok(())) => page(
                        "Signed in to ChatGPT",
                        "You can close this tab and return to rig.",
                    ),
                    Ok(Err(message)) => page(
                        "Sign-in failed",
                        &format!("Sign-in failed: {message}. Return to rig."),
                    ),
                    Err(_) => cancelled_page(),
                };
                respond(&mut stream, "200 OK", &body);
                return;
            }
        }
    };
    sender.send(result).ok();
}

/// Reads the request head and returns its first line. The whole head is
/// read so that closing the connection does not reset it before the page.
fn read_request_line(stream: &mut TcpStream) -> io::Result<String> {
    // Accepted sockets inherit non-blocking mode on some platforms.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(STREAM_TIMEOUT))?;
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
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        head.extend(chunk.iter().take(read));
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

fn cancelled_page() -> String {
    page(
        "Sign-in cancelled",
        "rig stopped waiting for this sign-in. Start it again from rig.",
    )
}

fn page(title: &str, message: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title></head>\
         <body style=\"font-family: system-ui, sans-serif; margin: 4rem auto; max-width: 32rem\">\
         <h1>{title}</h1><p>{}</p></body></html>",
        escape_html(message)
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

/// Asks the desktop to open `url` in the default browser, without waiting
/// and without writing to the terminal. Returns whether the opener started.
pub(super) fn open_browser(url: &str) -> bool {
    let mut command = opener(url);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let Ok(mut child) = command.spawn() else {
        return false;
    };
    // Reap the opener when it exits.
    std::thread::Builder::new()
        .name("browser-opener".into())
        .spawn(move || {
            child.wait().ok();
        })
        .ok();
    true
}

#[cfg(target_os = "macos")]
fn opener(url: &str) -> Command {
    let mut command = Command::new("open");
    command.arg(url);
    command
}

#[cfg(windows)]
fn opener(url: &str) -> Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut command = Command::new("cmd");
    // Quoted so that `cmd` keeps the query's `&`s; the URL holds no `"`.
    command
        .raw_arg(format!("/c start \"\" \"{url}\""))
        .creation_flags(CREATE_NO_WINDOW);
    command
}

#[cfg(not(any(target_os = "macos", windows)))]
fn opener(url: &str) -> Command {
    let mut command = Command::new("xdg-open");
    command.arg(url);
    command
}
