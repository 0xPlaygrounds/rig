//! Classify a provider reply as an account-level failure.
//!
//! A recording that captured a spent quota, a rate limit, an empty credit
//! balance or a refused credential holds no evidence about the behavior a
//! cell tests, yet it matches status-and-type assertions well enough to pass.
//! The recorder refuses such a reply unless the cell declared it as its
//! subject (see [`crate::http::ProviderCassette::expect_account_failure`]).
//!
//! ```
//! use rig_cassette::http::{AccountFailure, account_failure};
//! let body = r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota."}}"#;
//! assert_eq!(account_failure(429, body), Some(AccountFailure::Quota));
//! assert_eq!(account_failure(404, r#"{"error":{"type":"not_found_error"}}"#), None);
//! ```

use serde_json::Value;

/// Why a provider refused a request for account reasons rather than for
/// anything about the request itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AccountFailure {
    /// The credential was missing, invalid or not permitted.
    Auth,
    /// A per-minute or per-day request or token limit was hit.
    RateLimit,
    /// A usage or spending quota for the account or workspace is exhausted.
    Quota,
    /// The account has no credit or balance left.
    Credit,
}

impl AccountFailure {
    /// Every kind, for iterating declarations.
    pub const ALL: [Self; 4] = [Self::Auth, Self::RateLimit, Self::Quota, Self::Credit];

    pub(crate) const fn bit(self) -> u8 {
        match self {
            Self::Auth => 1,
            Self::RateLimit => 1 << 1,
            Self::Quota => 1 << 2,
            Self::Credit => 1 << 3,
        }
    }
}

const AUTH_IDS: &[&str] = &[
    "authentication_error",
    "invalid_api_key",
    "api_key_invalid",
    "unauthenticated",
    "unauthorized",
    "forbidden",
    "invalid_authentication",
];
const CREDIT_IDS: &[&str] = &[
    "insufficient_balance",
    "insufficient_credits",
    "payment_required",
    "billing_not_active",
];
const QUOTA_IDS: &[&str] = &[
    "insufficient_quota",
    "quota_exceeded",
    "billing_hard_limit_reached",
];
const RATE_IDS: &[&str] = &[
    "rate_limit_exceeded",
    "rate_limit_error",
    "rate_limited",
    "too_many_requests",
    "request_rate_limit_exceeded",
    "resource_exhausted",
];

const AUTH_MESSAGES: &[&str] = &[
    "api key not valid",
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
    "missing api key",
    "no api key",
    "invalid authentication",
    "invalid credentials",
];
const CREDIT_MESSAGES: &[&str] = &[
    "credit balance",
    "insufficient balance",
    "insufficient credits",
    "out of credits",
    "payment required",
    // xAI: "has either used all available credits or reached its monthly
    // spending limit".
    "available credits",
    "spending limit",
];
const QUOTA_MESSAGES: &[&str] = &[
    "usage limit",
    "exceeded your current quota",
    "quota exceeded",
    "exceeded the quota",
    "billing",
];
const RATE_MESSAGES: &[&str] = &["rate limit", "too many requests", "trial key"];

/// The account failure a reply with `status` and `body` represents, if any.
///
/// Only error statuses (400 and up) qualify. The error's own type, code,
/// status and reason fields decide first; message text is the fallback for
/// providers whose envelope carries none. A 403 counts as an auth failure
/// only when it names the credential or carries the bare `forbidden` code
/// Doubleword answers a rejected key with, so a permission error about a
/// resource (`PERMISSION_DENIED`) is not one.
pub fn account_failure(status: u16, body: &str) -> Option<AccountFailure> {
    if status < 400 {
        return None;
    }
    let fields = ErrorFields::parse(body);
    let has_id = |ids: &[&str]| fields.ids.iter().any(|id| ids.contains(&id.as_str()));
    let says = |phrases: &[&str]| phrases.iter().any(|phrase| fields.text.contains(phrase));

    if status == 401 {
        return Some(AccountFailure::Auth);
    }
    if status == 402 || has_id(CREDIT_IDS) || says(CREDIT_MESSAGES) {
        return Some(AccountFailure::Credit);
    }
    if has_id(QUOTA_IDS) || says(QUOTA_MESSAGES) {
        return Some(AccountFailure::Quota);
    }
    if status == 429 || has_id(RATE_IDS) || says(RATE_MESSAGES) {
        return Some(AccountFailure::RateLimit);
    }
    if has_id(AUTH_IDS) || says(AUTH_MESSAGES) {
        return Some(AccountFailure::Auth);
    }
    None
}

/// The account failure a reply carries, whatever its status: for an error
/// status, [`account_failure`]; for a successful one, an error its body
/// delivers instead of a result. That is a top-level `error` object, an
/// event of type `error`, or a Responses `response.failed` event, as a
/// stream that already answered 200 reports a limit it hit. The error's own
/// numeric `code` stands in for the status.
pub fn reply_account_failure(status: u16, body: &str) -> Option<AccountFailure> {
    if status >= 400 {
        return account_failure(status, body);
    }
    // A JSON-array reply (Gemini's non-SSE stream) delivers one document per
    // element.
    super::ledger::response_documents(body.as_bytes())
        .into_iter()
        .flat_map(|document| match document {
            Value::Array(items) => items,
            document => vec![document],
        })
        .filter_map(|document| delivered_error(&document))
        .find_map(|error| {
            let code = error
                .get("error")
                .unwrap_or(&error)
                .get("code")
                .and_then(Value::as_u64)
                .and_then(|code| u16::try_from(code).ok())
                .filter(|code| (400..600).contains(code))
                .unwrap_or(400);
            account_failure(code, &error.to_string())
        })
}

/// The error a successful reply's document delivers, if it is one.
fn delivered_error(document: &Value) -> Option<Value> {
    let kind = document.get("type").and_then(Value::as_str);
    if kind == Some("response.failed") {
        return document
            .get("response")
            .and_then(|response| response.get("error"))
            .filter(|error| !error.is_null())
            .cloned();
    }
    match document.get("error") {
        Some(Value::Null) | None if kind == Some("error") => Some(document.clone()),
        Some(Value::Null) | None => None,
        Some(error) => Some(serde_json::json!({ "error": error })),
    }
}

/// The identifying fields of an error body, lowercased: `ids` holds type,
/// code, status and reason values; `text` the messages, or the whole body
/// when it is not JSON.
struct ErrorFields {
    ids: Vec<String>,
    text: String,
}

impl ErrorFields {
    fn parse(body: &str) -> Self {
        let mut fields = Self {
            ids: Vec::new(),
            text: String::new(),
        };
        match serde_json::from_str::<Value>(body.trim()) {
            Ok(value) => fields.collect(&value, 0),
            Err(_) => fields.text = body.to_ascii_lowercase(),
        }
        fields
    }

    fn collect(&mut self, value: &Value, depth: usize) {
        // Envelopes nest the error a few levels down (a gateway wraps the
        // upstream's); deeper values are content, not the error.
        if depth > 4 {
            return;
        }
        match value {
            Value::Object(object) => {
                for (key, value) in object {
                    match (key.as_str(), value) {
                        ("type" | "code" | "status" | "reason", Value::String(id)) => {
                            self.ids.push(id.to_ascii_lowercase());
                        }
                        (
                            "message" | "detail" | "error_description" | "raw",
                            Value::String(text),
                        ) => {
                            self.text.push_str(&text.to_ascii_lowercase());
                            self.text.push('\n');
                            // A gateway can relay the upstream's body as a string.
                            if let Ok(inner) = serde_json::from_str::<Value>(text) {
                                self.collect(&inner, depth + 1);
                            }
                        }
                        ("error", Value::String(text)) => {
                            self.text.push_str(&text.to_ascii_lowercase());
                            self.text.push('\n');
                        }
                        _ => self.collect(value, depth + 1),
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.collect(item, depth + 1);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
