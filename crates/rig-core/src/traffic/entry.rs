//! One exchange as a JSON line, its credentials removed: the writer
//! thread's work.

use std::borrow::Cow;
use std::io::Read;
use std::ops::Range;
use std::time::Duration;

use http::{HeaderMap, header};
use serde_json::{Map, Value, json};

#[cfg(feature = "websocket")]
use super::EventData;
use super::{End, Exchange, RequestBody};
use crate::http_client::multipart::PartContent;
use crate::redact;
#[cfg(feature = "websocket")]
use crate::ws_client::Frame;

/// Headers whose values are credentials, or name the account they belong
/// to, besides those [`is_secret_header`] finds by their names' shape.
const SECRET_HEADERS: &[&str] = &[
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "cookie",
    "set-cookie",
    "chatgpt-account-id",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "x-amz-date",
    "key",
];

/// JSON fields that hold a credential, such as a token exchange's reply,
/// by their names lowercased without `_` and `-`: `access_token`,
/// `accessToken` and `access-token` are all `accesstoken`.
const SECRET_FIELDS: &[&str] = &[
    "apikey",
    "apitoken",
    "accesstoken",
    "authtoken",
    "authorizationcode",
    "refreshtoken",
    "idtoken",
    "sessiontoken",
    "token",
    "clientsecret",
    "secretkey",
    "privatekey",
    "subscriptionkey",
    "password",
    "devicecode",
    "codeverifier",
];

/// The shortest credential searched for elsewhere in an exchange: one or
/// two characters would match ordinary text. Below [`WHOLE_WORD`] bytes a
/// credential is replaced only where it stands alone, not inside a word.
const SHORTEST_SECRET: usize = 3;
const WHOLE_WORD: usize = 8;

/// Whether a header named `name` holds a credential: one of
/// [`SECRET_HEADERS`], one naming authentication (`authorization`,
/// `x-auth-token`, ...), or one ending in `-key`, `-token`, `-secret` or
/// `-signature`.
fn is_secret_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SECRET_HEADERS.contains(&name.as_str())
        || name.contains("auth")
        || ["-key", "-token", "-secret", "-signature"]
            .iter()
            .any(|end| name.ends_with(end))
}

fn is_secret_field(name: &str) -> bool {
    let name: String = name
        .chars()
        .filter(|c| !matches!(c, '_' | '-'))
        .map(|c| c.to_ascii_lowercase())
        .collect();
    SECRET_FIELDS.contains(&name.as_str())
}

/// Whether a query or form parameter named `name` holds a credential: a
/// secret field's name, one of a few more, or one ending in `key`,
/// `token`, `secret` or `signature` after `_` or `-`.
fn is_secret_param(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    is_secret_field(&lower)
        || matches!(
            lower.as_str(),
            "key" | "secret" | "signature" | "sig" | "code" | "assertion"
        )
        || lower.starts_with("x-amz-")
        || ["key", "token", "secret", "signature"].iter().any(|end| {
            ["_", "-"]
                .iter()
                .any(|sep| lower.ends_with(&format!("{sep}{end}")))
        })
}

/// A `name=value&...` text's pairs: each raw pair, with its name and value
/// decoded.
fn form_pairs(form: &str) -> impl Iterator<Item = (&str, String, String)> {
    form.split('&').map(|pair| {
        let decoded = url::form_urlencoded::parse(pair.as_bytes()).next();
        let (name, value) = decoded
            .map(|(name, value)| (name.into_owned(), value.into_owned()))
            .unwrap_or_default();
        (pair, name, value)
    })
}

/// Whether `text` is a `name=value&...` form, as a token exchange sends.
fn is_form(text: &str) -> bool {
    let text = text.trim_end();
    text.contains('=') && !text.contains(char::is_whitespace)
}

/// Whether `text` is an absolute HTTP or websocket URL.
fn is_url(text: &str) -> bool {
    ["https://", "http://", "wss://", "ws://"]
        .iter()
        .any(|scheme| text.starts_with(scheme))
        && !text.contains(char::is_whitespace)
}

/// Whether an object is a JSON Schema, such as a tool's parameter: its
/// `type` names JSON types, or it is made of `anyOf`, `oneOf`, `allOf` or
/// a `$ref`.
fn is_schema(fields: &Map<String, Value>) -> bool {
    let json_type = |kind: &Value| {
        matches!(
            kind.as_str(),
            Some("string" | "object" | "array" | "number" | "integer" | "boolean" | "null")
        )
    };
    let typed = match fields.get("type") {
        Some(Value::Array(kinds)) => !kinds.is_empty() && kinds.iter().all(json_type),
        Some(kind) => json_type(kind),
        None => false,
    };
    typed
        || ["anyOf", "oneOf", "allOf", "$ref"]
            .iter()
            .any(|key| fields.contains_key(*key))
}

/// The JSON a string holds, such as a tool call's arguments or a tool's
/// JSON result: an object or array.
fn json_in(text: &str) -> Option<Value> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return None;
    }
    serde_json::from_str::<Value>(text)
        .ok()
        .filter(|value| value.is_object() || value.is_array())
}

/// Where a JSON string starting at the quote at `open` ends: the index
/// after its closing quote, or the end of `text` when it has none.
fn string_end(text: &str, open: usize) -> usize {
    let mut escaped = false;
    text.get(open + 1..)
        .and_then(|rest| {
            rest.char_indices().find(|&(_, c)| {
                let end = c == '"' && !escaped;
                escaped = c == '\\' && !escaped;
                end
            })
        })
        .map_or(text.len(), |(at, _)| open + 1 + at + 1)
}

/// Where a JSON value starting at `start` ends: a string, an object or
/// array through its matching bracket, else up to the next `,`, `}` or
/// `]`; the end of `text` when it is cut short.
fn value_end(text: &str, start: usize) -> usize {
    match text.get(start..).and_then(|rest| rest.chars().next()) {
        Some('"') => string_end(text, start),
        Some('{' | '[') => {
            let mut depth = 0usize;
            let mut at = start;
            while let Some(c) = text.get(at..).and_then(|rest| rest.chars().next()) {
                match c {
                    '"' => {
                        at = string_end(text, at);
                        continue;
                    }
                    '{' | '[' => depth += 1,
                    '}' | ']' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            return at + 1;
                        }
                    }
                    _ => {}
                }
                at += c.len_utf8();
            }
            text.len()
        }
        _ => text
            .get(start..)
            .and_then(|rest| rest.find([',', '}', ']']))
            .map_or(text.len(), |len| start + len),
    }
}

/// The values of secret JSON fields in `text`, which need not parse, as a
/// stream cut short does not: each `"name": value` whose name, unescaped,
/// is a secret field, its value through its end or the end of `text`.
/// Strings are read whole, escapes included, so a quote inside one starts
/// nothing.
fn secret_values(text: &str) -> Vec<Range<usize>> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(open) = text.get(at..).and_then(|rest| rest.find('"')) {
        let open = at + open;
        let end = string_end(text, open);
        at = end;
        let raw = text.get(open..end).unwrap_or_default();
        let key = serde_json::from_str::<String>(raw)
            .unwrap_or_else(|_| raw.trim_matches('"').to_owned());
        let after = text.get(end..).unwrap_or_default();
        let Some(value) = after
            .trim_start()
            .strip_prefix(':')
            .map(str::trim_start)
            .filter(|_| is_secret_field(&key))
        else {
            continue;
        };
        let start = text.len() - value.len();
        let stop = value_end(text, start);
        if stop > start {
            found.push(start..stop);
        }
        at = stop.max(end);
    }
    found
}

/// A body as it is logged: decoded from its `content-encoding` when that
/// is gzip or deflate, else kept only by its size when it is not text.
enum Body<'a> {
    Bytes(Cow<'a, [u8]>),
    Opaque { len: usize, encoding: String },
}

fn decode<'a>(bytes: &'a [u8], headers: Option<&HeaderMap>) -> Body<'a> {
    let encoding = headers
        .and_then(|headers| headers.get(header::CONTENT_ENCODING))
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty() && value != "identity");
    let Some(encoding) = encoding else {
        return Body::Bytes(Cow::Borrowed(bytes));
    };
    let read = |mut reader: Box<dyn Read + '_>| {
        let mut out = Vec::new();
        reader.read_to_end(&mut out).ok().map(|_| out)
    };
    let decoded = match encoding.as_str() {
        "gzip" | "x-gzip" => read(Box::new(flate2::read::GzDecoder::new(bytes))),
        "deflate" => read(Box::new(flate2::read::ZlibDecoder::new(bytes))),
        _ => None,
    };
    match decoded {
        Some(decoded) => Body::Bytes(Cow::Owned(decoded)),
        // A transport that decoded the body and kept the header.
        None if std::str::from_utf8(bytes).is_ok() => Body::Bytes(Cow::Borrowed(bytes)),
        None => Body::Opaque {
            len: bytes.len(),
            encoding,
        },
    }
}

/// A streamed event or a websocket frame, before it is redacted.
struct Raw {
    at: Duration,
    /// A frame's direction (sent) and kind.
    frame: Option<(bool, &'static str)>,
    payload: Payload,
}

enum Payload {
    Text(String),
    /// Bytes that are not text, kept only by their size.
    Bytes(usize),
    #[cfg(feature = "websocket")]
    Close(Option<(u16, String)>),
}

/// The streamed body joined, so nothing is split between chunks, then cut
/// into SSE events (`sse`) or lines, each with the time of the chunk that
/// ended it. A body that is not UTF-8 stays in its chunks, by size; one
/// that ends inside a character, as a stopped stream may, loses that end.
fn stream_events(events: &[super::Event], sse: bool) -> Vec<Raw> {
    let chunks: Vec<(Duration, &[u8])> = events
        .iter()
        .filter_map(|event| Some((event.at, event.chunk()?.as_ref())))
        .collect();
    let joined: Vec<u8> = chunks
        .iter()
        .flat_map(|(_, bytes)| bytes.iter().copied())
        .collect();
    let text = match std::str::from_utf8(&joined) {
        Ok(text) => text,
        Err(cut) if cut.error_len().is_none() => joined
            .get(..cut.valid_up_to())
            .and_then(|valid| std::str::from_utf8(valid).ok())
            .unwrap_or_default(),
        Err(_) => {
            return chunks
                .iter()
                .map(|(at, bytes)| Raw {
                    at: *at,
                    frame: None,
                    payload: Payload::Bytes(bytes.len()),
                })
                .collect();
        }
    };
    let mut ends = Vec::with_capacity(chunks.len());
    let mut end = 0;
    for (at, bytes) in &chunks {
        end += bytes.len();
        ends.push((end, *at));
    }
    let came = |offset: usize| {
        let at = ends.partition_point(|(end, _)| *end < offset);
        ends.get(at)
            .or(ends.last())
            .map_or(Duration::ZERO, |(_, at)| *at)
    };
    let mut out = Vec::new();
    let mut lines: Vec<&str> = Vec::new();
    let mut offset = 0;
    let mut push = |lines: &mut Vec<&str>, offset: usize| {
        if !lines.is_empty() {
            out.push(Raw {
                at: came(offset),
                frame: None,
                payload: Payload::Text(lines.join("\n")),
            });
            lines.clear();
        }
    };
    for line in text.split_inclusive('\n') {
        offset += line.len();
        let bare = line.trim_end_matches(['\r', '\n']);
        if !bare.is_empty() {
            lines.push(bare);
        }
        if !sse || bare.is_empty() {
            push(&mut lines, offset);
        }
    }
    push(&mut lines, offset);
    out
}

#[cfg(feature = "websocket")]
fn frames(events: &[super::Event]) -> Vec<Raw> {
    let bytes = |bytes: &[u8]| match std::str::from_utf8(bytes) {
        Ok(text) => Payload::Text(text.to_owned()),
        Err(_) => Payload::Bytes(bytes.len()),
    };
    events
        .iter()
        .filter_map(|event| {
            let EventData::Frame { sent, frame } = &event.data else {
                return None;
            };
            let (kind, payload) = match frame {
                Frame::Text(text) => ("text", Payload::Text(text.clone())),
                Frame::Binary(data) => ("binary", bytes(data)),
                Frame::Ping(data) => ("ping", bytes(data)),
                Frame::Pong(data) => ("pong", bytes(data)),
                Frame::Close(close) => (
                    "close",
                    Payload::Close(
                        close
                            .as_ref()
                            .map(|close| (close.code, close.reason.clone())),
                    ),
                ),
            };
            Some(Raw {
                at: event.at,
                frame: Some((*sent, kind)),
                payload,
            })
        })
        .collect()
}

/// An event's payload: an SSE event's `data` lines joined, with its other
/// lines, or the whole text.
struct EventPayload<'a> {
    sse: bool,
    others: Vec<&'a str>,
    payload: Cow<'a, str>,
}

impl<'a> EventPayload<'a> {
    fn of(text: &'a str) -> Self {
        let data: Vec<&str> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|data| data.strip_prefix(' ').unwrap_or(data))
            .collect();
        if data.is_empty() {
            return Self {
                sse: false,
                others: Vec::new(),
                payload: Cow::Borrowed(text),
            };
        }
        Self {
            sse: true,
            others: text
                .lines()
                .filter(|line| !line.starts_with("data:"))
                .collect(),
            payload: Cow::Owned(data.join("\n")),
        }
    }

    /// The payload as JSON, when it is an object or array.
    fn json(&self) -> Option<Value> {
        serde_json::from_str::<Value>(&self.payload)
            .ok()
            .filter(|value| value.is_object() || value.is_array())
    }
}

/// The credentials one exchange carried, removed wherever they show up,
/// such as an error body that echoes the key; then secret JSON fields in
/// text that does not parse, and every secret [`redact`] finds by its
/// shape.
struct Scrubber {
    secrets: Vec<String>,
}

impl Scrubber {
    fn new(
        exchange: &Exchange,
        request: Option<&Body<'_>>,
        response: Option<&Body<'_>>,
        events: &[Raw],
    ) -> Self {
        let mut secrets = Vec::new();
        for (name, value) in &exchange.request_headers {
            if is_secret_header(name.as_str()) {
                let value = String::from_utf8_lossy(value.as_bytes());
                header_credentials(name.as_str(), &value, &mut secrets);
            }
        }
        for (name, value) in exchange.response_headers.iter().flatten() {
            let value = String::from_utf8_lossy(value.as_bytes());
            if name == header::SET_COOKIE {
                // The cookie's value, not its attributes.
                let cookie = value
                    .split(';')
                    .next()
                    .and_then(|cookie| cookie.split_once('='));
                secrets.extend(cookie.map(|(_, value)| value.trim().to_owned()));
            } else if is_secret_header(name.as_str()) {
                header_credentials(name.as_str(), &value, &mut secrets);
            } else if is_url(&value)
                && let Some((_, query)) = value.split_once('?')
            {
                form_secrets(query, &mut secrets);
            }
        }
        if let Some((_, query)) = exchange.uri.split_once('?') {
            form_secrets(query, &mut secrets);
        }
        // The user info is cut from the URL; it may show up again.
        if let Ok(url) = url::Url::parse(&exchange.uri) {
            for part in [url.username()].into_iter().chain(url.password()) {
                let decoded = url::form_urlencoded::parse(part.as_bytes())
                    .next()
                    .map(|(decoded, _)| decoded.into_owned());
                secrets.push(part.to_owned());
                secrets.extend(decoded);
            }
        }
        if let RequestBody::Multipart(form) = &exchange.request_body {
            for part in form.parts() {
                if let PartContent::Text(text) = part.content() {
                    if is_secret_param(part.name()) {
                        secrets.push(text.clone());
                    }
                    body_secrets(text.as_bytes(), &mut secrets);
                }
            }
        }
        for body in [request, response].into_iter().flatten() {
            if let Body::Bytes(bytes) = body {
                body_secrets(bytes, &mut secrets);
            }
        }
        for event in events {
            if let Payload::Text(text) = &event.payload {
                let event = EventPayload::of(text);
                match event.json() {
                    Some(value) => json_secrets(&value, &mut secrets),
                    None => text_secrets(&event.payload, &mut secrets),
                }
            }
        }
        secrets.retain(|secret| secret.len() >= SHORTEST_SECRET);
        secrets.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        secrets.dedup();
        Self { secrets }
    }

    fn text(&self, text: &str) -> String {
        let mut text = Cow::Borrowed(text);
        for secret in &self.secrets {
            if text.contains(secret.as_str()) {
                text = Cow::Owned(replace(&text, secret, secret.len() < WHOLE_WORD));
            }
        }
        let text = redact_values(&text);
        redact::text(&text).into_owned()
    }

    /// A string that holds JSON, such as a tool call's arguments, redacted
    /// as JSON, and kept as it was when nothing in it changed; a URL as a
    /// URL; else as text.
    fn string(&self, text: &str) -> String {
        if let Some(parsed) = json_in(text) {
            let mut redacted = parsed.clone();
            self.json(&mut redacted);
            return if redacted == parsed {
                text.to_owned()
            } else {
                redacted.to_string()
            };
        }
        if is_url(text) {
            return self.url(text);
        }
        self.text(text)
    }

    /// Every string redacted, keys too, and a secret field's string or
    /// number; an object or array under one keeps only what its own
    /// redaction leaves. Keys that redact alike are told apart by a `#n`
    /// suffix.
    fn json(&self, value: &mut Value) {
        match value {
            Value::String(text) => *text = self.string(text),
            Value::Array(items) => items.iter_mut().for_each(|item| self.json(item)),
            Value::Object(fields) => {
                for (name, mut field) in std::mem::take(fields) {
                    if is_secret_field(&name) && (field.is_string() || field.is_number()) {
                        field = Value::String(redact::REDACTED.to_owned());
                    } else {
                        self.json(&mut field);
                    }
                    let name = self.text(&name);
                    let mut key = name.clone();
                    let mut n = 1;
                    while fields.contains_key(&key) {
                        n += 1;
                        key = format!("{name}#{n}");
                    }
                    fields.insert(key, field);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    /// `form` with its secret parameters' values replaced.
    fn form(&self, form: &str) -> String {
        let pairs: Vec<String> = form_pairs(form)
            .map(|(pair, name, _)| match pair.split_once('=') {
                Some((raw, _)) if is_secret_param(&name) => {
                    format!("{raw}={}", redact::REDACTED)
                }
                _ => pair.to_owned(),
            })
            .collect();
        self.text(&pairs.join("&"))
    }

    /// A body as JSON when it is, a form's text with its secret fields
    /// replaced, other text, or its size when it is not text.
    fn body(&self, body: &Body<'_>) -> Value {
        let bytes = match body {
            Body::Bytes(bytes) => bytes,
            Body::Opaque { len, encoding } => {
                return json!({ "bytes": len, "content_encoding": encoding });
            }
        };
        if let Ok(mut value) = serde_json::from_slice::<Value>(bytes) {
            self.json(&mut value);
            return value;
        }
        match std::str::from_utf8(bytes) {
            Ok(text) if is_form(text) => Value::String(self.form(text)),
            Ok(text) => Value::String(self.text(text)),
            Err(_) => json!({ "bytes": bytes.len() }),
        }
    }

    /// An event's text: an SSE event's other lines, then its data, JSON
    /// redacted field by field on one `data:` line, or the data redacted on
    /// its own lines.
    fn event(&self, text: &str) -> String {
        let event = EventPayload::of(text);
        let data = match event.json() {
            Some(mut value) => {
                self.json(&mut value);
                value.to_string()
            }
            None => self.text(&event.payload),
        };
        if !event.sse {
            return data;
        }
        let others = event.others.iter().map(|line| self.text(line));
        let data = data.split('\n').map(|line| format!("data: {line}"));
        others.chain(data).collect::<Vec<_>>().join("\n")
    }

    fn raw(&self, event: &Raw) -> Value {
        let data = match &event.payload {
            Payload::Text(text) if event.frame.is_some() => match EventPayload::of(text).json() {
                Some(mut value) => {
                    self.json(&mut value);
                    value
                }
                None => Value::String(self.text(text)),
            },
            Payload::Text(text) => Value::String(self.event(text)),
            Payload::Bytes(len) => json!({ "bytes": len }),
            #[cfg(feature = "websocket")]
            Payload::Close(close) => close.as_ref().map_or(
                Value::Null,
                |(code, reason)| json!({ "code": code, "reason": self.text(reason) }),
            ),
        };
        let mut out = Map::new();
        out.insert("ms".into(), millis(event.at));
        if let Some((sent, kind)) = event.frame {
            out.insert("dir".into(), json!(if sent { "sent" } else { "received" }));
            out.insert("frame".into(), json!(kind));
        }
        out.insert("data".into(), data);
        Value::Object(out)
    }

    /// `uri` without its user info, its secret query parameters' values
    /// replaced.
    fn url(&self, uri: &str) -> String {
        let uri = without_userinfo(uri);
        match uri.split_once('?') {
            Some((path, query)) => format!("{}?{}", self.text(path), self.form(query)),
            None => self.text(&uri),
        }
    }

    /// Headers as an object, duplicate names joined with `, `.
    fn headers(&self, headers: &HeaderMap) -> Value {
        let mut object = Map::new();
        for name in headers.keys() {
            let value = if is_secret_header(name.as_str()) {
                redact::REDACTED.to_owned()
            } else {
                let values: Vec<_> = headers
                    .get_all(name)
                    .iter()
                    .map(|value| self.string(&String::from_utf8_lossy(value.as_bytes())))
                    .collect();
                values.join(", ")
            };
            object.insert(name.to_string(), Value::String(value));
        }
        Value::Object(object)
    }
}

/// `text` with `secret` replaced; with `whole`, only where it is not part
/// of a longer word.
fn replace(text: &str, secret: &str, whole: bool) -> String {
    if !whole {
        return text.replace(secret, redact::REDACTED);
    }
    let word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    let mut out = String::with_capacity(text.len());
    let mut kept = 0;
    for (at, _) in text.match_indices(secret) {
        let before = text.get(..at).and_then(|before| before.chars().next_back());
        let after = text
            .get(at + secret.len()..)
            .and_then(|after| after.chars().next());
        if at < kept || word(before) || word(after) {
            continue;
        }
        out.push_str(text.get(kept..at).unwrap_or_default());
        out.push_str(redact::REDACTED);
        kept = at + secret.len();
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    out
}

/// `text` with the string value of each secret JSON field replaced
/// ([`secret_values`]).
fn redact_values(text: &str) -> Cow<'_, str> {
    if !text.contains('"') {
        return Cow::Borrowed(text);
    }
    let found = secret_values(text);
    if found.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut kept = 0;
    for range in found {
        out.push_str(text.get(kept..range.start).unwrap_or_default());
        out.push('"');
        out.push_str(redact::REDACTED);
        out.push('"');
        kept = range.end;
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    Cow::Owned(out)
}

/// `uri` with the user info of its authority replaced.
fn without_userinfo(uri: &str) -> Cow<'_, str> {
    let Some((scheme, rest)) = uri.split_once("://") else {
        return Cow::Borrowed(uri);
    };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, path) = rest.split_at_checked(end).unwrap_or((rest, ""));
    match authority.rsplit_once('@') {
        Some((_, host)) => Cow::Owned(format!("{scheme}://{}@{host}{path}", redact::REDACTED)),
        None => Cow::Borrowed(uri),
    }
}

/// The credential in a header's value: the value, the part after its
/// scheme (`Bearer ...`), and, for `cookie`, each cookie's value.
fn header_credentials(name: &str, value: &str, secrets: &mut Vec<String>) {
    secrets.push(value.to_owned());
    if let Some((_, credential)) = value.split_once(' ') {
        secrets.push(credential.trim().to_owned());
    }
    if name.eq_ignore_ascii_case("cookie") {
        for cookie in value.split(';') {
            secrets.extend(
                cookie
                    .split_once('=')
                    .map(|(_, value)| value.trim().to_owned()),
            );
        }
    }
}

/// The values of a form's or query's secret parameters, decoded and as
/// sent.
fn form_secrets(form: &str, secrets: &mut Vec<String>) {
    for (pair, name, value) in form_pairs(form) {
        if is_secret_param(&name) {
            secrets.extend(pair.split_once('=').map(|(_, raw)| raw.to_owned()));
            secrets.push(value);
        }
    }
}

/// The values of secret fields of `value` of [`WHOLE_WORD`] bytes or
/// more, and the strings that long in an object or array under one, such
/// as `{"client_secret": {"value": ...}}`, but not in a JSON Schema, such
/// as a tool's `token` parameter.
fn json_secrets(value: &Value, secrets: &mut Vec<String>) {
    fn strings(value: &Value, secrets: &mut Vec<String>) {
        match value {
            Value::String(text) if text.len() >= WHOLE_WORD => secrets.push(text.clone()),
            Value::Array(items) => items.iter().for_each(|item| strings(item, secrets)),
            Value::Object(fields) if !is_schema(fields) => {
                fields.values().for_each(|field| strings(field, secrets));
            }
            _ => {}
        }
    }
    match value {
        Value::String(text) => match json_in(text) {
            Some(value) => json_secrets(&value, secrets),
            // JSON with text around it, such as a command's output.
            None => text_secrets(text, secrets),
        },
        Value::Array(items) => items.iter().for_each(|item| json_secrets(item, secrets)),
        Value::Object(fields) => {
            for (name, field) in fields {
                if is_secret_field(name) {
                    strings(field, secrets);
                } else {
                    json_secrets(field, secrets);
                }
            }
        }
        _ => {}
    }
}

/// The strings in the values of secret JSON fields in text that does not
/// parse, of [`WHOLE_WORD`] bytes or more.
fn text_secrets(text: &str, secrets: &mut Vec<String>) {
    for range in secret_values(text) {
        let value = text.get(range).unwrap_or_default();
        let mut at = 0;
        while let Some(open) = value.get(at..).and_then(|rest| rest.find('"')) {
            let open = at + open;
            let end = string_end(value, open);
            let raw = value.get(open..end).unwrap_or_default();
            let string = serde_json::from_str::<String>(raw)
                .unwrap_or_else(|_| raw.trim_matches('"').to_owned());
            if string.len() >= WHOLE_WORD {
                secrets.push(string);
            }
            at = end;
        }
    }
}

/// The values of a body's secret JSON fields or form parameters.
fn body_secrets(body: &[u8], secrets: &mut Vec<String>) {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        json_secrets(&value, secrets);
    } else if let Ok(text) = std::str::from_utf8(body) {
        if is_form(text) {
            form_secrets(text, secrets);
        }
        text_secrets(text, secrets);
    }
}

fn millis(duration: Duration) -> Value {
    json!((duration.as_secs_f64() * 1_000_000.0).round() / 1000.0)
}

/// `exchange` as its log line.
pub(super) fn line(exchange: &Exchange) -> Value {
    let request_body = match &exchange.request_body {
        RequestBody::Bytes(bytes) if !bytes.is_empty() => {
            Some(decode(bytes, Some(&exchange.request_headers)))
        }
        _ => None,
    };
    let response_body = exchange
        .response_body
        .as_ref()
        .map(|bytes| decode(bytes, exchange.response_headers.as_ref()));
    #[cfg(feature = "websocket")]
    // Only a websocket session has no request body.
    let events = if matches!(exchange.request_body, RequestBody::None) {
        frames(&exchange.events)
    } else {
        stream_events(&exchange.events, is_sse(exchange))
    };
    #[cfg(not(feature = "websocket"))]
    let events = stream_events(&exchange.events, is_sse(exchange));
    let scrub = Scrubber::new(
        exchange,
        request_body.as_ref(),
        response_body.as_ref(),
        &events,
    );

    let mut entry = Map::new();
    entry.insert("exchange".into(), json!(exchange.number));
    entry.insert("pid".into(), json!(std::process::id()));
    entry.insert("started_ms".into(), json!(exchange.started_ms));
    if let Some(serving) = &exchange.serving {
        entry.insert("effect".into(), json!(serving.effect));
        entry.insert("handler".into(), json!(serving.handler.as_str()));
        if let Some(provider) = &serving.provider {
            entry.insert("provider".into(), json!(provider.to_string()));
        }
    }

    let mut request = Map::new();
    request.insert("method".into(), json!(exchange.method));
    request.insert("url".into(), json!(scrub.url(&exchange.uri)));
    request.insert("headers".into(), scrub.headers(&exchange.request_headers));
    if let Some(body) = &request_body {
        request.insert("body".into(), scrub.body(body));
    }
    if let RequestBody::Multipart(form) = &exchange.request_body {
        let parts = form.parts().iter().map(|part| {
            let mut out = Map::new();
            out.insert("name".into(), json!(part.name()));
            if let Some(filename) = part.get_filename() {
                out.insert("filename".into(), json!(scrub.text(filename)));
            }
            if let Some(content_type) = part.get_content_type() {
                out.insert("content_type".into(), json!(content_type.to_string()));
            }
            match part.content() {
                PartContent::Text(_) if is_secret_param(part.name()) => {
                    out.insert("text".into(), json!(redact::REDACTED));
                }
                PartContent::Text(text) => {
                    out.insert("text".into(), json!(scrub.text(text)));
                }
                PartContent::Binary(bytes) => {
                    out.insert("bytes".into(), json!(bytes.len()));
                }
            }
            Value::Object(out)
        });
        request.insert("parts".into(), Value::Array(parts.collect()));
    }
    entry.insert("request".into(), Value::Object(request));

    let mut response = Map::new();
    if let Some(status) = exchange.status {
        response.insert("status".into(), json!(status.as_u16()));
    }
    if let Some(headers) = &exchange.response_headers {
        response.insert("headers".into(), scrub.headers(headers));
    }
    if let Some(body) = &response_body {
        response.insert("body".into(), scrub.body(body));
    }
    if !events.is_empty() {
        let events = events.iter().map(|event| scrub.raw(event)).collect();
        response.insert("events".into(), Value::Array(events));
    }
    if !response.is_empty() {
        entry.insert("response".into(), Value::Object(response));
    }

    let mut timings = Map::new();
    for (name, at) in [
        ("headers_ms", exchange.headers_at),
        ("first_byte_ms", exchange.first_byte_at),
        ("last_byte_ms", exchange.last_byte_at),
        ("end_ms", exchange.ended_at),
    ] {
        if let Some(at) = at {
            timings.insert(name.into(), millis(at));
        }
    }
    entry.insert("timings".into(), Value::Object(timings));
    if let Some(error) = &exchange.error {
        entry.insert("error".into(), json!(scrub.text(error)));
    }
    let end = match exchange.end {
        End::Complete => "complete",
        End::Error => "error",
        End::Stopped => "stopped",
    };
    entry.insert("end".into(), json!(end));
    Value::Object(entry)
}

fn is_sse(exchange: &Exchange) -> bool {
    exchange
        .response_headers
        .as_ref()
        .and_then(|headers| headers.get(header::CONTENT_TYPE))
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
}
