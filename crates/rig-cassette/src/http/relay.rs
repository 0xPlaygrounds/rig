//! A byte-exact TCP relay in front of the recording proxy that writes the
//! created-resource ledger as replies pass through.
//!
//! The relay copies bytes both ways unchanged, so the proxy records exactly
//! what it would without it, WebSocket upgrades included. A tap parses the
//! HTTP/1.1 exchanges it sees and, for each reply that created provider
//! state, appends a [`ledger`](super::ledger) line before forwarding the
//! bytes that complete it: the id is on disk before the test can read it.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::ledger::{self, CreatedResource, LedgerEntry};

/// Where the relay's ledger lines go and whose resources they are.
#[derive(Clone, Debug)]
pub(crate) struct LedgerTarget {
    pub(crate) path: PathBuf,
    pub(crate) provider: String,
    pub(crate) scenario: String,
    /// The real provider's scheme and host, for delete URLs.
    pub(crate) origin: String,
}

/// A running relay: accept on `base_url`, forward to the proxy.
pub(crate) struct Relay {
    pub(crate) base_url: String,
    task: JoinHandle<()>,
}

impl Relay {
    pub(crate) async fn start(proxy_addr: String, ledger: LedgerTarget) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}", listener.local_addr()?);
        let ledger = Arc::new(ledger);
        let task = tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                let proxy_addr = proxy_addr.clone();
                let ledger = ledger.clone();
                tokio::spawn(async move {
                    if let Ok(proxy) = TcpStream::connect(&proxy_addr).await {
                        relay_connection(client, proxy, ledger).await;
                    }
                });
            }
        });
        Ok(Self { base_url, task })
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn relay_connection(client: TcpStream, proxy: TcpStream, ledger: Arc<LedgerTarget>) {
    let (mut client_read, mut client_write) = client.into_split();
    let (mut proxy_read, mut proxy_write) = proxy.into_split();
    let tap = Arc::new(Mutex::new(Tap::default()));

    let upstream_tap = tap.clone();
    let upstream = async move {
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let read = match client_read.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            upstream_tap.lock().await.request_bytes(&buffer[..read]);
            if proxy_write.write_all(&buffer[..read]).await.is_err() {
                break;
            }
        }
        let _ = proxy_write.shutdown().await;
    };
    let downstream = async move {
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let read = match proxy_read.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            let created = tap.lock().await.response_bytes(&buffer[..read]);
            record(&ledger, created);
            if client_write.write_all(&buffer[..read]).await.is_err() {
                break;
            }
        }
        let created = tap.lock().await.end_of_responses();
        record(&ledger, created);
        let _ = client_write.shutdown().await;
    };
    tokio::join!(upstream, downstream);
}

fn record(ledger: &LedgerTarget, created: Vec<TappedCreation>) {
    let entries: Vec<LedgerEntry> = created
        .into_iter()
        .filter(|creation| (200..300).contains(&creation.status))
        .flat_map(|creation| {
            ledger::created_resources(
                &ledger.provider,
                &ledger.origin,
                &creation.method,
                &creation.path,
                &creation.request_body,
                &creation.response_body,
            )
        })
        .map(|resource| {
            LedgerEntry::Created(CreatedResource {
                scenario: ledger.scenario.clone(),
                ..resource
            })
        })
        .collect();
    ledger::append(&ledger.path, &entries);
}

/// A request and the reply bytes seen so far that may name created state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TappedCreation {
    pub(crate) status: u16,
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) request_body: Vec<u8>,
    pub(crate) response_body: Vec<u8>,
}

/// Incremental HTTP/1.1 parsing of one connection's two directions.
#[derive(Default)]
pub(crate) struct Tap {
    requests: Parser,
    responses: Parser,
    /// Requests whose replies have not completed, oldest first.
    pending: std::collections::VecDeque<(String, String, Vec<u8>)>,
    /// SSE events of the current reply already handed to the ledger.
    reported_events: usize,
    /// A `101` switched the connection to another protocol: stop parsing.
    upgraded: bool,
}

impl Tap {
    pub(crate) fn request_bytes(&mut self, bytes: &[u8]) {
        if self.upgraded {
            return;
        }
        for message in self.requests.feed(bytes, Direction::Request) {
            self.pending
                .push_back((message.method, message.path, message.body));
        }
    }

    /// Feed reply bytes; return the creations the ledger must see now: each
    /// completed reply, and each new complete SSE event of a streaming one.
    pub(crate) fn response_bytes(&mut self, bytes: &[u8]) -> Vec<TappedCreation> {
        let mut created = Vec::new();
        if self.upgraded {
            return created;
        }
        let completed = self.responses.feed(bytes, Direction::Response);
        for message in completed {
            // An interim reply (`100 Continue`) precedes the real one.
            if (100..200).contains(&message.status) && message.status != 101 {
                continue;
            }
            let (method, path, request_body) = self.pending.pop_front().unwrap_or_default();
            if message.status == 101 {
                self.upgraded = true;
                self.requests = Parser::default();
                self.responses = Parser::default();
                break;
            }
            // Events already reported while the stream was open are not
            // reported again.
            let response_body = if message.event_stream {
                let reported: usize = complete_events(&message.body)
                    .iter()
                    .take(self.reported_events)
                    .map(Vec::len)
                    .sum();
                message.body[reported..].to_vec()
            } else {
                message.body
            };
            created.push(TappedCreation {
                status: message.status,
                method,
                path,
                request_body,
                response_body,
            });
            self.reported_events = 0;
        }
        // A streaming reply names what it created on an early event; report
        // each complete event as it lands rather than when the stream ends.
        if let Some((status, partial)) = self.responses.partial_event_stream()
            && let Some((method, path, request_body)) = self.pending.front()
        {
            let events = complete_events(partial);
            if events.len() > self.reported_events {
                let fresh = events[self.reported_events..].concat();
                self.reported_events = events.len();
                created.push(TappedCreation {
                    status,
                    method: method.clone(),
                    path: path.clone(),
                    request_body: request_body.clone(),
                    response_body: fresh,
                });
            }
        }
        created
    }

    /// The proxy closed the connection: a reply delimited by close is complete.
    pub(crate) fn end_of_responses(&mut self) -> Vec<TappedCreation> {
        let Some((status, body)) = self.responses.close() else {
            return Vec::new();
        };
        let (method, path, request_body) = self.pending.pop_front().unwrap_or_default();
        vec![TappedCreation {
            status,
            method,
            path,
            request_body,
            response_body: body,
        }]
    }
}

/// Complete `\n\n`-terminated SSE events in `body`, each with its terminator.
fn complete_events(body: &[u8]) -> Vec<Vec<u8>> {
    let mut events = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index + 1 < body.len() {
        let end = if body[index] == b'\n' && body[index + 1] == b'\n' {
            Some(index + 2)
        } else if body[index..].starts_with(b"\r\n\r\n") {
            Some(index + 4)
        } else {
            None
        };
        if let Some(end) = end {
            events.push(body[start..end].to_vec());
            start = end;
            index = end;
        } else {
            index += 1;
        }
    }
    events
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Request,
    Response,
}

#[derive(Debug, PartialEq, Eq)]
struct Message {
    method: String,
    path: String,
    status: u16,
    event_stream: bool,
    body: Vec<u8>,
}

enum Framing {
    Length(usize),
    Chunked(ChunkState),
    UntilClose,
}

enum ChunkState {
    Size,
    Data(usize),
    DataEnd,
    Trailers,
}

#[derive(Default)]
struct Parser {
    buffer: Vec<u8>,
    current: Option<InProgress>,
}

struct InProgress {
    method: String,
    path: String,
    status: u16,
    framing: Framing,
    event_stream: bool,
    body: Vec<u8>,
}

impl Parser {
    fn feed(&mut self, bytes: &[u8], direction: Direction) -> Vec<Message> {
        self.buffer.extend_from_slice(bytes);
        let mut done = Vec::new();
        loop {
            if self.current.is_none() {
                let Some(head_end) = find(&self.buffer, b"\r\n\r\n") else {
                    break;
                };
                let head = String::from_utf8_lossy(&self.buffer[..head_end]).into_owned();
                self.buffer.drain(..head_end + 4);
                self.current = Some(start_message(&head, direction));
            }
            let Some(message) = self.current.as_mut() else {
                break;
            };
            if !advance_body(message, &mut self.buffer) {
                break;
            }
            if let Some(message) = self.current.take() {
                done.push(Message {
                    method: message.method,
                    path: message.path,
                    status: message.status,
                    event_stream: message.event_stream,
                    body: message.body,
                });
            }
        }
        done
    }

    fn partial_event_stream(&self) -> Option<(u16, &[u8])> {
        self.current
            .as_ref()
            .filter(|message| message.event_stream)
            .map(|message| (message.status, message.body.as_slice()))
    }

    fn close(&mut self) -> Option<(u16, Vec<u8>)> {
        let message = self.current.take()?;
        matches!(message.framing, Framing::UntilClose).then_some((message.status, message.body))
    }
}

fn start_message(head: &str, direction: Direction) -> InProgress {
    let mut lines = head.split("\r\n");
    let start = lines.next().unwrap_or_default();
    let mut parts = start.split(' ');
    let (method, path, status) = match direction {
        Direction::Request => (
            parts.next().unwrap_or_default().to_owned(),
            parts.next().unwrap_or_default().to_owned(),
            0,
        ),
        Direction::Response => {
            let _version = parts.next();
            let status = parts
                .next()
                .and_then(|status| status.parse::<u16>().ok())
                .unwrap_or(0);
            (String::new(), String::new(), status)
        }
    };
    let mut length = None;
    let mut chunked = false;
    let mut event_stream = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => length = value.parse::<usize>().ok(),
            "transfer-encoding" => chunked = value.to_ascii_lowercase().contains("chunked"),
            "content-type" => {
                event_stream = value.to_ascii_lowercase().starts_with("text/event-stream");
            }
            _ => {}
        }
    }
    let bodiless =
        direction == Direction::Response && (status / 100 == 1 || status == 204 || status == 304);
    let framing = if bodiless {
        Framing::Length(0)
    } else if chunked {
        Framing::Chunked(ChunkState::Size)
    } else if let Some(length) = length {
        Framing::Length(length)
    } else if direction == Direction::Request {
        Framing::Length(0)
    } else {
        Framing::UntilClose
    };
    InProgress {
        method,
        path,
        status,
        framing,
        event_stream,
        body: Vec::new(),
    }
}

/// Move body bytes from `buffer` into `message`; `true` once it is complete.
fn advance_body(message: &mut InProgress, buffer: &mut Vec<u8>) -> bool {
    loop {
        match &mut message.framing {
            Framing::Length(remaining) => {
                let take = (*remaining).min(buffer.len());
                message.body.extend(buffer.drain(..take));
                *remaining -= take;
                return *remaining == 0;
            }
            Framing::UntilClose => {
                message.body.append(buffer);
                return false;
            }
            Framing::Chunked(state) => match state {
                ChunkState::Size => {
                    let Some(end) = find(buffer, b"\r\n") else {
                        return false;
                    };
                    let line = String::from_utf8_lossy(&buffer[..end]).into_owned();
                    buffer.drain(..end + 2);
                    let size =
                        usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
                            .unwrap_or(0);
                    *state = if size == 0 {
                        ChunkState::Trailers
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(remaining) => {
                    let take = (*remaining).min(buffer.len());
                    message.body.extend(buffer.drain(..take));
                    *remaining -= take;
                    if *remaining > 0 {
                        return false;
                    }
                    *state = ChunkState::DataEnd;
                }
                ChunkState::DataEnd => {
                    if buffer.len() < 2 {
                        return false;
                    }
                    buffer.drain(..2);
                    *state = ChunkState::Size;
                }
                ChunkState::Trailers => {
                    let Some(end) = find(buffer, b"\r\n") else {
                        return false;
                    };
                    buffer.drain(..end + 2);
                    if end == 0 {
                        return true;
                    }
                }
            },
        }
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests;
