//! Shape coverage: the request facts and reply shapes the cassette corpus
//! records, per provider and encoder.
//!
//! An encoder is a request method and path template: model names and
//! resource ids in the path are erased. A request fact is one object of the
//! body: its path with array indices collapsed, its keys, and each key's
//! scalar type or container kind, keeping the values of discriminator keys
//! such as `role` and `type`. What the caller chose is not read: JSON Schema
//! property names, tool arguments and tool results. A body that is not JSON
//! is one fact, its kind. A reply shape is the status, the framing, and the
//! skeleton of the whole body, or for a stream the set of its event
//! skeletons, with the same discriminators kept; it is stored as a stable
//! hash. Each one is stored with its count of recordings and the first
//! fixture that holds it.
//!
//! A reply shape is also held by a reply bank entry that the `runtime`
//! target decodes: a verbatim provider reply with its source, so it counts
//! as a recording. A request fact needs a cassette, since its acceptance
//! needs a live request.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Error, Result, invalid};
use crate::support::files_under;

/// The baseline file's column header.
pub(crate) const HEADER: &str = "provider\tencoder\tkind\tshape\trecordings\texample";

/// Keys whose string values request facts and reply shapes keep: bounded
/// enums that select an encoder or decoder branch.
pub(crate) const DISCRIMINATORS: &[&str] = &[
    "finishReason",
    "finish_reason",
    "object",
    "reason",
    "role",
    "status",
    "stop_reason",
    "type",
];

/// JSON Schema keywords whose value maps names the caller chose to schemas.
/// No fact holds such a map; its members share the path `.<keyword>.*`.
const NAME_MAPS: &[&str] = &["$defs", "definitions", "patternProperties", "properties"];

/// Keys whose object value is a tool call's arguments or a tool's result:
/// values the caller chose, so a fact stops at them. An array `input` is a
/// Responses conversation and is read.
const CALLER_VALUES: &[&str] = &["args", "arguments", "input", "result"];

/// Whether a shape is a request fact or a reply shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    Request,
    Reply,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Reply => "reply",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "request" => Some(Self::Request),
            "reply" => Some(Self::Reply),
            _ => None,
        }
    }
}

/// One shape of one provider's encoder.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ShapeKey {
    pub(crate) provider: String,
    pub(crate) encoder: String,
    pub(crate) kind: Kind,
    /// A request [`Fact`] as text, or the hash of a reply shape.
    pub(crate) shape: String,
}

/// One object of a request body, as [`facts`] reads it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Fact {
    /// Where the object sits: `$` for the root, `.key` per member and `[]`
    /// per array element, so every element of an array shares one path.
    pub(crate) path: String,
    /// The object's sorted keys, each with its scalar type, a discriminator's
    /// value, `{}` for an object, or the set of element kinds of an array.
    pub(crate) object: String,
}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.path, self.object)
    }
}

/// The distinct facts of one body.
pub(crate) type Facts = BTreeSet<Fact>;

/// How often a shape is recorded, and where it was first seen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Recorded {
    pub(crate) recordings: usize,
    pub(crate) example: String,
}

/// Every shape the corpus records.
pub(crate) type Shapes = BTreeMap<ShapeKey, Recorded>;

/// One recorded exchange: its request (`when`) and reply (`then`).
#[derive(Debug, Deserialize)]
pub(crate) struct Interaction {
    pub(crate) when: Exchange,
    pub(crate) then: Exchange,
}

/// A recorded request or reply, as much of it as shapes read.
#[derive(Debug, Deserialize)]
pub(crate) struct Exchange {
    #[serde(default)]
    pub(crate) path: String,
    #[serde(default)]
    pub(crate) method: String,
    #[serde(default)]
    pub(crate) status: u16,
    #[serde(default)]
    pub(crate) header: Vec<NameValue>,
    #[serde(default)]
    pub(crate) body: Option<String>,
    #[serde(default)]
    pub(crate) body_encoding: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct NameValue {
    pub(crate) name: String,
    pub(crate) value: String,
}

/// One cassette fixture of the corpus.
#[derive(Debug)]
pub(crate) struct Fixture {
    /// The path below the corpus root, `<provider>/...yaml`.
    pub(crate) relative: String,
    pub(crate) provider: String,
    /// [`hash`] of the file's text: it changes when the fixture is re-recorded.
    pub(crate) hash: String,
    /// The file's length in bytes.
    pub(crate) size: usize,
    pub(crate) interactions: Vec<Interaction>,
}

impl Exchange {
    /// The encoder: the method and the path template.
    pub(crate) fn encoder(&self) -> String {
        format!("{} {}", self.method, path_template(&self.path))
    }

    pub(crate) fn content_type(&self) -> &str {
        self.header
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case("content-type"))
            .map_or("", |header| header.value.as_str())
    }

    pub(crate) fn is_base64(&self) -> bool {
        self.body_encoding.as_deref() == Some("base64")
    }
}

/// Every fixture under `cassettes` (`<provider>/**/*.yaml`), in path order.
pub(crate) fn corpus(cassettes: &Path) -> Result<Vec<Fixture>> {
    let mut fixtures = Vec::new();
    for file in files_under(cassettes, Some("yaml"))? {
        let relative = file
            .strip_prefix(cassettes)
            .map_err(|_| invalid(format!("{} is outside the corpus", file.display())))?
            .to_string_lossy()
            .replace('\\', "/");
        let Some((provider, _)) = relative.split_once('/') else {
            continue;
        };
        let provider = provider.to_owned();
        let text = std::fs::read_to_string(&file)?;
        let interactions =
            interactions(&text).map_err(|error| invalid(format!("{relative}: {error}")))?;
        fixtures.push(Fixture {
            relative,
            provider,
            hash: hash(&text),
            size: text.len(),
            interactions,
        });
    }
    Ok(fixtures)
}

/// Collect every shape under `cassettes` (`<provider>/**/*.yaml`).
pub(crate) fn collect(cassettes: &Path) -> Result<Shapes> {
    let mut shapes = Shapes::new();
    for fixture in corpus(cassettes)? {
        for (key, example) in fixture_shapes(&fixture) {
            shapes
                .entry(key)
                .and_modify(|seen| seen.recordings += 1)
                .or_insert_with(|| Recorded {
                    recordings: 1,
                    example,
                });
        }
    }
    Ok(shapes)
}

/// Every shape one fixture records, once per interaction that records it,
/// with that interaction (`<fixture>#<n>`).
pub(crate) fn fixture_shapes(fixture: &Fixture) -> Vec<(ShapeKey, String)> {
    let mut found = Vec::new();
    for (index, interaction) in fixture.interactions.iter().enumerate() {
        let encoder = interaction.when.encoder();
        let example = format!("{}#{index}", fixture.relative);
        let requests = request_facts(&interaction.when)
            .into_iter()
            .map(|fact| (Kind::Request, fact.to_string()));
        let reply = (Kind::Reply, hash(&reply_shape(&interaction.then)));
        for (kind, shape) in requests.chain([reply]) {
            let key = ShapeKey {
                provider: fixture.provider.clone(),
                encoder: encoder.clone(),
                kind,
                shape,
            };
            found.push((key, example.clone()));
        }
    }
    found
}

fn interactions(text: &str) -> std::result::Result<Vec<Interaction>, serde_yaml::Error> {
    serde_yaml::Deserializer::from_str(text)
        .map(Interaction::deserialize)
        .collect()
}

/// `path` with model names and resource ids replaced by placeholders.
pub(crate) fn path_template(path: &str) -> String {
    let mut previous = "";
    let mut segments = Vec::new();
    for segment in path.split('/') {
        let template = if matches!(previous, "models" | "model") && !segment.is_empty() {
            match segment.split_once(':') {
                Some((_, verb)) => format!("{{model}}:{verb}"),
                None => "{model}".to_owned(),
            }
        } else if segment.len() >= 20 && segment.chars().any(|ch| ch.is_ascii_digit()) {
            "{id}".to_owned()
        } else {
            segment.to_owned()
        };
        segments.push(template);
        previous = segment;
    }
    segments.join("/")
}

/// The facts of a request body: one per object of a JSON body, or the one
/// fact `$ <kind>` of a body that is empty, binary, multipart (its sorted
/// field names) or text.
pub(crate) fn request_facts(request: &Exchange) -> Facts {
    let root = |object: String| {
        Facts::from([Fact {
            path: "$".to_owned(),
            object,
        }])
    };
    let body = request.body.as_deref().unwrap_or("");
    if body.is_empty() {
        return root("empty".to_owned());
    }
    if request.is_base64() {
        return root("binary".to_owned());
    }
    if request.content_type().starts_with("multipart/form-data") {
        let names: BTreeSet<&str> = body
            .split("; name=\"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .collect();
        return root(format!(
            "multipart[{}]",
            names.into_iter().collect::<Vec<_>>().join(",")
        ));
    }
    match serde_json::from_str::<Value>(body) {
        Ok(value) => facts(&value, DISCRIMINATORS),
        Err(_) => root("text".to_owned()),
    }
}

/// Every object of `value` as a [`Fact`], and the root as one when it is not
/// an object. The string value of a key in `keep` survives.
pub(crate) fn facts(value: &Value, keep: &[&str]) -> Facts {
    let mut out = Facts::new();
    if !value.is_object() {
        out.insert(Fact {
            path: "$".to_owned(),
            object: kind(value),
        });
    }
    walk_facts(&mut out, "$".to_owned(), value, keep);
    out
}

fn walk_facts(out: &mut Facts, path: String, value: &Value, keep: &[&str]) {
    match value {
        Value::Array(items) => {
            let path = format!("{path}[]");
            for item in items {
                walk_facts(out, path.clone(), item, keep);
            }
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut object = String::from("{");
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    object.push(',');
                }
                object.push_str(key);
                object.push(':');
                match map.get(*key) {
                    Some(Value::String(text)) if keep.contains(&key.as_str()) => {
                        let _ = write!(object, "{text:?}");
                    }
                    Some(child) => object.push_str(&kind(child)),
                    None => {}
                }
            }
            object.push('}');
            out.insert(Fact {
                path: path.clone(),
                object,
            });
            for key in keys {
                match map.get(key) {
                    Some(Value::Object(names)) if NAME_MAPS.contains(&key.as_str()) => {
                        let path = format!("{path}.{key}.*");
                        for child in names.values() {
                            walk_facts(out, path.clone(), child, keep);
                        }
                    }
                    Some(Value::Object(_)) if CALLER_VALUES.contains(&key.as_str()) => {}
                    Some(child) => walk_facts(out, format!("{path}.{key}"), child, keep),
                    None => {}
                }
            }
        }
        _ => {}
    }
}

/// The kind of a value inside a fact: its scalar type, `{}` for an object,
/// or the sorted set of its elements' kinds for an array.
fn kind(value: &Value) -> String {
    match value {
        Value::Object(_) => "{}".to_owned(),
        Value::Array(items) => set(items.iter().map(kind)),
        scalar => skeleton(scalar, &[]),
    }
}

/// The shape of a reply: status, framing and body skeleton.
pub(crate) fn reply_shape(reply: &Exchange) -> String {
    let body = reply.body.as_deref().unwrap_or("");
    let content_type = reply.content_type();
    let framed = if body.is_empty() {
        "empty".to_owned()
    } else if content_type.starts_with("application/vnd.amazon.eventstream") {
        match decode_base64(body).and_then(|bytes| event_stream(&bytes)) {
            Some(events) => format!("eventstream{}", set(events)),
            None => "binary".to_owned(),
        }
    } else if reply.is_base64() {
        "binary".to_owned()
    } else if content_type.starts_with("text/event-stream") {
        format!("sse{}", set(sse_events(body)))
    } else {
        match serde_json::from_str::<Value>(body) {
            Ok(value) => format!("json{}", skeleton(&value, DISCRIMINATORS)),
            Err(_) => "text".to_owned(),
        }
    };
    format!("{} {framed}", reply.status)
}

/// The skeleton of every server-sent event in `body`, named by its `event:`.
fn sse_events(body: &str) -> Vec<String> {
    sse_payloads(body)
        .into_iter()
        .map(|(name, data)| {
            let data = if data == "[DONE]" {
                data
            } else {
                serde_json::from_str::<Value>(&data)
                    .map_or_else(|_| "text".to_owned(), |v| skeleton(&v, DISCRIMINATORS))
            };
            format!("{name}={data}")
        })
        .collect()
}

/// Every server-sent event in `body`: its `event:` name and its joined
/// `data:` lines.
fn sse_payloads(body: &str) -> Vec<(String, String)> {
    body.split("\n\n")
        .filter_map(|event| {
            let mut name = "";
            let mut data = Vec::new();
            for line in event.lines() {
                if let Some(value) = line.strip_prefix("event:") {
                    name = value.trim();
                } else if let Some(value) = line.strip_prefix("data:") {
                    data.push(value.trim_start());
                }
            }
            if name.is_empty() && data.is_empty() {
                return None;
            }
            Some((name.to_owned(), data.join("\n")))
        })
        .collect()
}

/// The skeleton of every message in an AWS event stream, named by its
/// `:event-type` header (or `:exception-type`). `None` when the frames do not
/// parse.
fn event_stream(bytes: &[u8]) -> Option<Vec<String>> {
    Some(
        event_messages(bytes)?
            .into_iter()
            .map(|(name, payload)| {
                let data = serde_json::from_slice::<Value>(payload)
                    .map_or_else(|_| "binary".to_owned(), |v| skeleton(&v, DISCRIMINATORS));
                format!("{name}={data}")
            })
            .collect(),
    )
}

/// Every message of an AWS event stream: its `:event-type` (or
/// `:exception-type`) header and its payload. `None` when the frames do not
/// parse.
fn event_messages(bytes: &[u8]) -> Option<Vec<(String, &[u8])>> {
    let mut messages = Vec::new();
    let mut rest = bytes;
    while !rest.is_empty() {
        let total = usize::try_from(u32::from_be_bytes(rest.get(0..4)?.try_into().ok()?)).ok()?;
        let headers_len =
            usize::try_from(u32::from_be_bytes(rest.get(4..8)?.try_into().ok()?)).ok()?;
        let message = rest.get(..total)?;
        let headers = message.get(12..12 + headers_len)?;
        let payload = message.get(12 + headers_len..total.checked_sub(4)?)?;
        let name = event_headers(headers)?
            .into_iter()
            .find(|(key, _)| key == ":event-type" || key == ":exception-type")
            .map_or_else(String::new, |(_, value)| value);
        messages.push((name, payload));
        rest = rest.get(total..)?;
    }
    Some(messages)
}

/// The JSON documents a reply carries: the whole body, every server-sent
/// event's data, or every event-stream payload. What does not parse as JSON
/// is left out.
pub(crate) fn reply_documents(reply: &Exchange) -> Vec<Value> {
    let body = reply.body.as_deref().unwrap_or("");
    let content_type = reply.content_type();
    if body.is_empty() {
        return Vec::new();
    }
    if content_type.starts_with("application/vnd.amazon.eventstream") {
        let bytes = decode_base64(body).unwrap_or_default();
        return event_messages(&bytes)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, payload)| serde_json::from_slice(payload).ok())
            .collect();
    }
    if reply.is_base64() {
        return Vec::new();
    }
    // A stream recorded without its content type is still a stream.
    let sniffed = body.starts_with("event:") || body.starts_with("data:");
    if content_type.starts_with("text/event-stream") || sniffed {
        return sse_payloads(&body.replace("\r\n", "\n"))
            .into_iter()
            .filter_map(|(_, data)| serde_json::from_str(&data).ok())
            .collect();
    }
    serde_json::from_str(body).into_iter().collect()
}

/// The string-valued headers of one event-stream message.
fn event_headers(mut bytes: &[u8]) -> Option<Vec<(String, String)>> {
    let mut headers = Vec::new();
    while !bytes.is_empty() {
        let name_len = usize::from(*bytes.first()?);
        let name = std::str::from_utf8(bytes.get(1..1 + name_len)?).ok()?;
        let kind = *bytes.get(1 + name_len)?;
        bytes = bytes.get(2 + name_len..)?;
        // Value lengths by header type; 7 is a string and 6 a byte array,
        // each with a two-byte length prefix.
        let fixed = match kind {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            9 => 16,
            6 | 7 => {
                let len = usize::from(u16::from_be_bytes(bytes.get(0..2)?.try_into().ok()?));
                let value = bytes.get(2..2 + len)?;
                if kind == 7 {
                    headers.push((name.to_owned(), String::from_utf8_lossy(value).into_owned()));
                }
                bytes = bytes.get(2 + len..)?;
                continue;
            }
            _ => return None,
        };
        bytes = bytes.get(fixed..)?;
    }
    Some(headers)
}

/// Standard base64 with optional padding; `None` on any other byte.
pub(crate) fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut buffer = 0u32;
    let mut bits = 0;
    for byte in text.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        };
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(u8::try_from((buffer >> bits) & 0xff).ok()?);
        }
    }
    Some(out)
}

/// The skeleton of `value`: object keys sorted, scalars reduced to their
/// type, arrays to the sorted set of their element skeletons. The string
/// value of a key in `keep` survives.
pub(crate) fn skeleton(value: &Value, keep: &[&str]) -> String {
    let mut out = String::new();
    write_skeleton(&mut out, value, keep);
    out
}

fn write_skeleton(out: &mut String, value: &Value, keep: &[&str]) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(_) => out.push_str("bool"),
        Value::Number(_) => out.push_str("num"),
        Value::String(_) => out.push_str("str"),
        Value::Array(items) => {
            out.push_str(&set(items.iter().map(|item| skeleton(item, keep))));
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(key);
                out.push(':');
                match map.get(key) {
                    Some(Value::String(text)) if keep.contains(&key.as_str()) => {
                        let _ = write!(out, "{text:?}");
                    }
                    Some(child) => write_skeleton(out, child, keep),
                    None => {}
                }
            }
            out.push('}');
        }
    }
}

/// `[a|b|…]`: the sorted, deduplicated items.
fn set(items: impl IntoIterator<Item = String>) -> String {
    let items: BTreeSet<String> = items.into_iter().collect();
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join("|"))
}

/// 64-bit FNV-1a of `text`, as 16 hex digits: stable across platforms and
/// releases, unlike the standard library's hasher.
pub(crate) fn hash(text: &str) -> String {
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        state ^= u64::from(byte);
        state = state.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{state:016x}")
}

/// The baseline text for `shapes`, one shape per line in key order.
pub(crate) fn render(shapes: &Shapes) -> String {
    let mut out = format!("{HEADER}\n");
    for (key, seen) in shapes {
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}",
            key.provider,
            key.encoder,
            key.kind.as_str(),
            key.shape,
            seen.recordings,
            seen.example
        );
    }
    out
}

/// Parse a baseline written by [`render`].
pub(crate) fn parse(text: &str) -> Result<Shapes> {
    let mut shapes = Shapes::new();
    for (number, line) in text.lines().enumerate().skip(1) {
        let columns: Vec<&str> = line.split('\t').collect();
        let [provider, encoder, kind, shape, recordings, example] = columns.as_slice() else {
            return Err(invalid(format!("shapes line {}: {line:?}", number + 1)));
        };
        let kind = Kind::parse(kind)
            .ok_or_else(|| invalid(format!("shapes line {}: kind {kind:?}", number + 1)))?;
        let recordings = recordings
            .parse()
            .map_err(|_| invalid(format!("shapes line {}: count {recordings:?}", number + 1)))?;
        shapes.insert(
            ShapeKey {
                provider: (*provider).to_owned(),
                encoder: (*encoder).to_owned(),
                kind,
                shape: (*shape).to_owned(),
            },
            Recorded {
                recordings,
                example: (*example).to_owned(),
            },
        );
    }
    Ok(shapes)
}

/// The example of a shape only the reply bank holds: `bank:` and the
/// entry's source interaction.
pub(crate) const BANKED: &str = "bank:";

/// Add the reply shapes `banked` holds (each with its entry's source) that
/// no cassette records, as one recording each.
pub(crate) fn add_banked(shapes: &mut Shapes, banked: BTreeMap<ShapeKey, String>) {
    for (key, source) in banked {
        if key.kind == Kind::Reply {
            shapes.entry(key).or_insert_with(|| Recorded {
                recordings: 1,
                example: format!("{BANKED}{source}"),
            });
        }
    }
}

/// Every baseline shape that no recording holds any more.
pub(crate) fn lost(baseline: &Shapes, current: &Shapes) -> Vec<String> {
    baseline
        .iter()
        .filter(|(key, _)| !current.contains_key(*key))
        .map(|(key, seen)| {
            let shape = match key.kind {
                Kind::Request => "fact",
                Kind::Reply => "reply shape",
            };
            format!(
                "{} `{}` {shape} `{}` lost its last recording (was {})",
                key.provider, key.encoder, key.shape, seen.example
            )
        })
        .collect()
}

/// Per provider: distinct request facts and reply shapes.
pub(crate) fn summary(shapes: &Shapes) -> BTreeMap<&str, (usize, usize)> {
    let mut counts: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for key in shapes.keys() {
        let entry = counts.entry(key.provider.as_str()).or_default();
        match key.kind {
            Kind::Request => entry.0 += 1,
            Kind::Reply => entry.1 += 1,
        }
    }
    counts
}

impl From<serde_yaml::Error> for Error {
    fn from(error: serde_yaml::Error) -> Self {
        invalid(error.to_string())
    }
}
