//! The port on which the run hears of the session's new prompts: Linear's
//! `AgentSessionEvent` webhooks with the action `prompted`, as whatever
//! received them forwards them, body and `Linear-Signature` header
//! unchanged. A request counts only when its signature is the HMAC-SHA256
//! of its body with the webhook's signing secret and it was sent within
//! [`FRESH`] of now.

use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_channel::Sender;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use hmac::{Hmac, KeyInit, Mac};
use serde_json::Value;
use sha2::Sha256;
use tokio::task::JoinHandle;

use crate::api::Seen;

#[cfg(test)]
mod tests;

/// The header holding the body's signature, hex encoded.
pub(crate) const SIGNATURE_HEADER: &str = "linear-signature";

/// The path the webhooks are posted to.
pub(crate) const PATH: &str = "/linear/webhook";

/// How far from now a webhook's `webhookTimestamp` may be, as Linear
/// advises against replays.
const FRESH: Duration = Duration::from_secs(60);

/// A person's prompt on an agent session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Prompted {
    /// The agent session's id.
    pub(crate) session: String,
    /// The prompt.
    pub(crate) seen: Seen,
}

/// The port, served until dropped.
pub(crate) struct Listener {
    /// Where it listens.
    pub(crate) addr: SocketAddr,
    serving: JoinHandle<()>,
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.serving.abort();
    }
}

/// What the handler needs.
#[derive(Clone)]
struct Door {
    secret: std::sync::Arc<str>,
    prompts: Sender<Prompted>,
}

/// Binds `addr` and serves it on the current tokio runtime, sending each
/// signed, fresh prompt to `prompts`.
pub(crate) async fn listen(
    addr: &str,
    secret: String,
    prompts: Sender<Prompted>,
) -> std::io::Result<Listener> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let addr = listener.local_addr()?;
    let door = Door {
        secret: secret.into(),
        prompts,
    };
    let router = axum::Router::new()
        .route(PATH, axum::routing::post(receive))
        .with_state(door);
    let serving = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Ok(Listener { addr, serving })
}

/// One webhook: 401 unless signed and fresh, 400 unless JSON, 202 for a
/// prompt, which goes on, 204 for any other event.
async fn receive(State(door): State<Door>, headers: HeaderMap, body: Bytes) -> StatusCode {
    let signature = headers
        .get(SIGNATURE_HEADER)
        .and_then(|value| value.to_str().ok());
    if !signature.is_some_and(|signature| signed(&door.secret, &body, signature)) {
        return StatusCode::UNAUTHORIZED;
    }
    let Ok(event) = serde_json::from_slice::<Value>(&body) else {
        return StatusCode::BAD_REQUEST;
    };
    if !fresh(&event, now_millis()) {
        return StatusCode::UNAUTHORIZED;
    }
    match prompted(&event) {
        Some(prompt) => {
            door.prompts.send(prompt).await.ok();
            StatusCode::ACCEPTED
        }
        None => StatusCode::NO_CONTENT,
    }
}

/// Whether `signature` (hex) is the HMAC-SHA256 of `body` with `secret`,
/// compared in constant time.
pub(crate) fn signed(secret: &str, body: &[u8], signature: &str) -> bool {
    let Some(tag) = unhex(signature.trim()) else {
        return false;
    };
    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    mac.update(body);
    mac.verify_slice(&tag).is_ok()
}

/// The bytes `hex` encodes, if it is hex.
fn unhex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(hex.get(at..at + 2)?, 16).ok())
        .collect()
}

/// Whether `event` was sent within [`FRESH`] of `now` (milliseconds since
/// the epoch).
pub(crate) fn fresh(event: &Value, now: u128) -> bool {
    let Some(sent) = event.get("webhookTimestamp").and_then(Value::as_f64) else {
        return false;
    };
    (now as f64 - sent).abs() <= FRESH.as_millis() as f64
}

/// Milliseconds since the epoch.
fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |now| now.as_millis())
}

/// The prompt `event` carries, when it is an `AgentSessionEvent` with the
/// action `prompted`. Its body is `agentActivity.content.body`, or
/// `agentActivity.body`, which Linear's guide names instead.
pub(crate) fn prompted(event: &Value) -> Option<Prompted> {
    let text =
        |value: &Value, at: &str| value.pointer(at).and_then(Value::as_str).map(str::to_owned);
    if text(event, "/type").as_deref() != Some("AgentSessionEvent")
        || text(event, "/action").as_deref() != Some("prompted")
    {
        return None;
    }
    let activity = event.get("agentActivity")?;
    let session = text(event, "/agentSession/id").or_else(|| text(activity, "/agentSessionId"))?;
    Some(Prompted {
        session,
        seen: Seen {
            id: text(activity, "/id")?,
            created_at: text(activity, "/createdAt").unwrap_or_default(),
            prompt: Some(
                text(activity, "/content/body")
                    .or_else(|| text(activity, "/body"))
                    .unwrap_or_default(),
            ),
            stop: text(activity, "/signal").as_deref() == Some("stop"),
            by: text(activity, "/user/name"),
            from_app: activity
                .pointer("/user/app")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
    })
}
