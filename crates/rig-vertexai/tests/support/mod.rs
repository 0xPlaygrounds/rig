//! Deterministic doubles for the *transport and credential* ends of the real
//! Vertex AI SDK: a local HTTP/1.1 endpoint the real
//! `PredictionService` talks to, and `google-cloud-auth` credential
//! implementations whose tokens are observable sentinels.
//!
//! Nothing here replaces a Rig type. The completion model, the SDK client and
//! the request/response conversion under test are the real ones; only the
//! socket on the far end and the identity presented to it are ours.

#![allow(
    dead_code,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use google_cloud_auth::credentials::{
    CacheableResource, Credentials, CredentialsProvider, EntityTag,
};
use google_cloud_auth::errors::CredentialsError;
use http::{Extensions, HeaderMap, HeaderValue};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// What the endpoint does with the next request it receives.
pub enum Reply {
    /// Answer with this status and JSON body, then close the connection.
    Json { status: u16, body: String },
    /// Never answer. The request is still captured, and the endpoint records
    /// the moment the client goes away — which is how a dropped completion
    /// future is observed as released work rather than merely as a missing
    /// result.
    Hang,
}

impl Reply {
    pub fn ok(body: impl Into<String>) -> Self {
        Self::Json {
            status: 200,
            body: body.into(),
        }
    }

    pub fn error(status: u16, body: impl Into<String>) -> Self {
        Self::Json {
            status,
            body: body.into(),
        }
    }
}

/// One request as it arrived on the wire.
#[derive(Clone, Debug)]
pub struct CapturedRequest {
    pub method: String,
    /// Request target, path and query together.
    pub target: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl CapturedRequest {
    pub fn path(&self) -> &str {
        self.target
            .split_once('?')
            .map(|(path, _)| path)
            .unwrap_or(&self.target)
    }

    pub fn query(&self) -> Option<&str> {
        self.target.split_once('?').map(|(_, query)| query)
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).expect("request body is JSON")
    }
}

#[derive(Default)]
struct EndpointState {
    replies: VecDeque<Reply>,
    requests: Vec<CapturedRequest>,
    disconnects: usize,
}

/// A local HTTP/1.1 endpoint for the real SDK client to call.
pub struct LocalEndpoint {
    addr: SocketAddr,
    state: Arc<Mutex<EndpointState>>,
    progress: Arc<Notify>,
    accept: JoinHandle<()>,
}

impl Drop for LocalEndpoint {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

impl LocalEndpoint {
    /// Bind on loopback and answer the queued replies in order. A request
    /// arriving after the queue is empty gets a 500 that says so, which shows
    /// up as an unexpected extra call rather than as a hang.
    pub async fn spawn(replies: impl IntoIterator<Item = Reply>) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let state = Arc::new(Mutex::new(EndpointState {
            replies: replies.into_iter().collect(),
            ..EndpointState::default()
        }));
        let progress = Arc::new(Notify::new());

        let accept = tokio::spawn({
            let state = Arc::clone(&state);
            let progress = Arc::clone(&progress);
            async move {
                // The accept task owns every socket task, including on panic
                // and endpoint drop. JoinSet also reaps completed connections.
                let mut connections = tokio::task::JoinSet::new();
                loop {
                    tokio::select! {
                        accepted = listener.accept() => {
                            let Ok((stream, _)) = accepted else { return; };
                            let state = Arc::clone(&state);
                            let progress = Arc::clone(&progress);
                            connections.spawn(async move {
                                let _ = tokio::time::timeout(
                                    std::time::Duration::from_secs(60),
                                    serve(stream, state, progress),
                                ).await;
                            });
                        }
                        _ = connections.join_next(), if !connections.is_empty() => {}
                    }
                }
            }
        });

        Self {
            addr,
            state,
            progress,
            accept,
        }
    }

    /// The `http://` URL to hand to `PredictionService::builder().with_endpoint()`.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<CapturedRequest> {
        self.state.lock().expect("endpoint state").requests.clone()
    }

    pub fn request_count(&self) -> usize {
        self.state.lock().expect("endpoint state").requests.len()
    }

    /// How many connections the client closed while the endpoint was still
    /// holding the request open.
    pub fn disconnects(&self) -> usize {
        self.state.lock().expect("endpoint state").disconnects
    }

    /// Wait until the endpoint has seen `count` requests, or time out.
    pub async fn wait_for_requests(&self, count: usize) {
        self.wait_until(|state| state.requests.len() >= count).await
    }

    /// Wait until the client has abandoned `count` held-open requests, or
    /// time out.
    pub async fn wait_for_disconnects(&self, count: usize) {
        self.wait_until(|state| state.disconnects >= count).await
    }

    async fn wait_until(&self, done: impl Fn(&EndpointState) -> bool) {
        let deadline = std::time::Duration::from_secs(10);
        tokio::time::timeout(deadline, async {
            loop {
                // Enable the waiter before checking the state: a notification
                // that lands between the check and the await must not be lost.
                let notified = self.progress.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if done(&self.state.lock().expect("endpoint state")) {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("endpoint reached the expected state within the deadline");
    }
}

async fn serve(mut stream: TcpStream, state: Arc<Mutex<EndpointState>>, progress: Arc<Notify>) {
    let Some(request) = read_request(&mut stream).await else {
        return;
    };

    let reply = {
        let mut state = state.lock().expect("endpoint state");
        state.requests.push(request);
        state.replies.pop_front()
    };
    progress.notify_waiters();

    match reply {
        Some(Reply::Json { status, body }) => {
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\n\
                 content-type: application/json\r\n\
                 content-length: {len}\r\n\
                 connection: close\r\n\r\n{body}",
                reason = reason(status),
                len = body.len(),
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
        }
        Some(Reply::Hang) => {
            // Hold the request open and wait for the peer to go away.
            let mut discard = [0_u8; 256];
            loop {
                match stream.read(&mut discard).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
            state.lock().expect("endpoint state").disconnects += 1;
            progress.notify_waiters();
        }
        None => {
            let body =
                r#"{"error":{"code":500,"message":"no canned reply left","status":"INTERNAL"}}"#;
            let response = format!(
                "HTTP/1.1 500 Internal Server Error\r\n\
                 content-type: application/json\r\n\
                 content-length: {}\r\n\
                 connection: close\r\n\r\n{body}",
                body.len(),
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
        }
    }
}

fn reason(status: u16) -> &'static str {
    http::StatusCode::from_u16(status)
        .ok()
        .and_then(|status| status.canonical_reason())
        .unwrap_or("Status")
}

/// Read one HTTP/1.1 request: head up to the blank line, then exactly
/// `content-length` bytes of body. The SDK never sends a chunked request body.
async fn read_request(stream: &mut TcpStream) -> Option<CapturedRequest> {
    let mut buffer = Vec::new();
    let head_end = loop {
        if let Some(index) = find_head_end(&buffer) {
            break index;
        }
        let mut chunk = [0_u8; 1024];
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return None,
            Ok(read) => buffer.extend_from_slice(chunk.get(..read)?),
        }
    };

    let head = String::from_utf8(buffer.get(..head_end)?.to_vec()).ok()?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next()?.split(' ');
    let method = request_line.next()?.to_owned();
    let target = request_line.next()?.to_owned();

    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
    }

    let length: usize = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse().ok())
        .unwrap_or(0);

    let mut body = buffer.get(head_end + 4..)?.to_vec();
    while body.len() < length {
        let mut chunk = [0_u8; 1024];
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return None,
            Ok(read) => body.extend_from_slice(chunk.get(..read)?),
        }
    }

    Some(CapturedRequest {
        method,
        target,
        headers,
        body: String::from_utf8(body).ok()?,
    })
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

/// Credentials that mint an observable sentinel token, so a test can tell one
/// presented identity from the next without any real credential material.
///
/// This is a real `google_cloud_auth::credentials::CredentialsProvider`: the
/// SDK calls it exactly as it calls ADC-derived credentials, and
/// [`Credentials::from`] erases it the same way.
#[derive(Debug, Clone)]
pub struct SentinelCredentials {
    prefix: String,
    issued: Arc<AtomicUsize>,
    /// When set, the credentials fail permanently from this issue number
    /// onward, standing in for a revoked grant. The refusal is deliberately
    /// *not* transient: the SDK retries a transient credential failure for up
    /// to a minute, which is its contract, not something to re-test here.
    fail_from: Option<usize>,
    universe_domain: Option<String>,
}

impl SentinelCredentials {
    /// Credentials that issue `{prefix}-1`, `{prefix}-2`, ... one sentinel per
    /// request, so rotation is visible on the wire.
    pub fn rotating(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            issued: Arc::new(AtomicUsize::new(0)),
            fail_from: None,
            universe_domain: None,
        }
    }

    /// Fail permanently from the `nth` (1-based) issue onward.
    pub fn failing_from(mut self, nth: usize) -> Self {
        self.fail_from = Some(nth);
        self
    }

    /// Claim a universe domain, which the SDK checks against the client's own
    /// during construction.
    pub fn with_universe_domain(mut self, domain: impl Into<String>) -> Self {
        self.universe_domain = Some(domain.into());
        self
    }

    /// How many tokens have been issued (successfully or not).
    pub fn issued(&self) -> usize {
        self.issued.load(Ordering::SeqCst)
    }

    /// The `nth` (1-based) sentinel token this provider would issue.
    pub fn token(&self, nth: usize) -> String {
        format!("{}-{nth}", self.prefix)
    }

    pub fn credentials(&self) -> Credentials {
        Credentials::from(self.clone())
    }
}

impl CredentialsProvider for SentinelCredentials {
    async fn headers(
        &self,
        _extensions: Extensions,
    ) -> Result<CacheableResource<HeaderMap>, CredentialsError> {
        let nth = self.issued.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_from.is_some_and(|first| nth >= first) {
            return Err(CredentialsError::from_msg(
                false,
                "sentinel credentials refused to issue a token",
            ));
        }
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", self.token(nth)))
            .map_err(|error| CredentialsError::new(false, "sentinel header", error))?;
        headers.insert(http::header::AUTHORIZATION, value);
        Ok(CacheableResource::New {
            entity_tag: EntityTag::new(),
            data: headers,
        })
    }

    async fn universe_domain(&self) -> Option<String> {
        self.universe_domain.clone()
    }
}

/// A successful `generateContent` body, in the enum-as-int JSON encoding the
/// SDK asks for with `$alt=json;enum-encoding=int`.
pub fn text_response(text: &str) -> String {
    serde_json::json!({
        "responseId": "local-endpoint-response",
        "modelVersion": "gemini-2.5-flash-001",
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": 1
        }],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}
    })
    .to_string()
}

/// A successful `generateContent` body whose candidate calls a tool.
pub fn tool_call_response(name: &str, args: serde_json::Value) -> String {
    serde_json::json!({
        "responseId": "local-endpoint-tool-response",
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": name, "args": args}}]
            },
            "finishReason": 1
        }]
    })
    .to_string()
}
