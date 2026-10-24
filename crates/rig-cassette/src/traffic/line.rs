//! One exchange as a scrubbed JSON line: the writer thread's work.

use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use rig_core::http_client::{Bytes, HeaderMap, MultipartForm, multipart::PartContent};
use serde_json::{Map, Value, json};

use super::{End, Exchange, RequestBody};
use crate::scrub::{Purpose, REDACTED, Scrubber, is_secret_header};

/// The version of the line format, the `v` of every line. A change that
/// renames, removes or retypes a field takes the next one.
pub const LINE_VERSION: u64 = 1;

/// The latest request bodies of each handler and URL a log wrote, newest
/// first, with their exchanges: the bases a next body can be written
/// against. Several agents on one model take turns, so each keeps its own
/// conversation's last body here, up to [`HEADS`].
#[derive(Default)]
pub(super) struct Bases(HashMap<String, VecDeque<(u64, Map<String, Value>)>>);

/// The most bases kept per handler and URL.
const HEADS: usize = 8;

/// `exchange` as one line, written by process `pid`, its request body
/// against `bases`.
pub(super) fn line(exchange: &Exchange, pid: u32, bases: &mut Bases) -> Value {
    let mut scrubber = Scrubber::new(Purpose::Traffic);
    let response_headers = exchange.response_headers.as_ref();
    // Learn every credential first, so its echoes anywhere are scrubbed.
    scrubber.learn_url(&exchange.uri);
    for headers in std::iter::once(&exchange.request_headers).chain(response_headers) {
        for (name, value) in headers {
            if let Ok(value) = value.to_str() {
                scrubber.learn_header(name.as_str(), value);
            }
        }
    }

    if let RequestBody::Bytes(bytes) = &exchange.request_body
        && let Ok(text) = std::str::from_utf8(bytes)
    {
        scrubber.learn_body(text);
    }
    if let Some(bytes) = &exchange.response_body
        && let Ok(text) = std::str::from_utf8(bytes)
    {
        scrubber.learn_body(text);
    }
    if let Some(chunks) = &exchange.chunks {
        let joined: Vec<u8> = chunks
            .iter()
            .flat_map(|(_, bytes)| bytes.iter().copied())
            .collect();
        if let Ok(text) = std::str::from_utf8(&joined) {
            scrubber.learn_body(text);
        }
    }

    let mut line = Map::new();
    line.insert("v".into(), json!(LINE_VERSION));
    line.insert("exchange".into(), json!(exchange.number));
    line.insert("pid".into(), json!(pid));
    line.insert("started_ms".into(), json!(exchange.started_ms));
    if let Some(serving) = &exchange.serving {
        line.insert("effect".into(), json!(serving.effect.as_u64()));
        line.insert("handler".into(), json!(serving.handler.as_str()));
        if let Some(provider) = &serving.provider {
            line.insert("provider".into(), json!(provider.to_string()));
        }
    }

    let mut request = Map::new();
    request.insert("method".into(), json!(exchange.method));
    request.insert("url".into(), json!(scrubber.scrub_url(&exchange.uri)));
    request.insert(
        "headers".into(),
        headers(&mut scrubber, &exchange.request_headers),
    );
    match &exchange.request_body {
        RequestBody::Bytes(bytes) if bytes.is_empty() => {}
        RequestBody::Bytes(bytes) => {
            let body = body(&mut scrubber, bytes);
            match (&exchange.serving, body) {
                (Some(serving), Value::Object(full)) => {
                    let chain = format!("{}\n{}", serving.handler.as_str(), exchange.uri);
                    let heads = bases.0.entry(chain).or_default();
                    // The base it shares the most with, the newest of equals.
                    let best = heads
                        .iter()
                        .enumerate()
                        .filter_map(|(at, (after, base))| {
                            delta(*after, base, &full).map(|delta| (at, delta))
                        })
                        .max_by_key(|(at, delta)| (kept(delta), Reverse(*at)));
                    match best {
                        Some((at, delta)) => {
                            // A body that extends its base whole continues
                            // that conversation, so it takes the base's place.
                            if heads.get(at).is_some_and(|(_, base)| extends(base, &delta)) {
                                heads.remove(at);
                            }
                            request.extend(delta);
                        }
                        None => {
                            request.insert("body".into(), Value::Object(full.clone()));
                        }
                    }
                    heads.push_front((exchange.number, full));
                    heads.truncate(HEADS);
                }
                (_, body) => {
                    request.insert("body".into(), body);
                }
            }
        }
        RequestBody::Multipart(form) => {
            request.insert("parts".into(), parts(&mut scrubber, form));
        }
    }
    line.insert("request".into(), Value::Object(request));

    if let Some(status) = exchange.status {
        let mut response = Map::new();
        response.insert("status".into(), json!(status.as_u16()));
        if let Some(headers_map) = response_headers {
            response.insert("headers".into(), headers(&mut scrubber, headers_map));
        }
        if let Some(bytes) = &exchange.response_body {
            response.insert("body".into(), body(&mut scrubber, bytes));
        } else if let Some(chunks) = &exchange.chunks {
            let sse = response_headers
                .and_then(|headers| headers.get("content-type"))
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"));
            response.insert("events".into(), events(&mut scrubber, chunks, sse));
        }
        line.insert("response".into(), Value::Object(response));
    }

    let mut timings = Map::new();
    for (name, at) in [
        ("headers_ms", exchange.headers_at),
        ("first_byte_ms", exchange.first_byte_at),
        ("last_byte_ms", exchange.last_byte_at),
        ("end_ms", exchange.ended_at),
    ] {
        if let Some(at) = at {
            timings.insert(name.into(), json!(ms(at)));
        }
    }
    line.insert("timings".into(), Value::Object(timings));
    if let Some(error) = &exchange.error {
        line.insert("error".into(), json!(scrubber.scrub_text(error)));
    }
    let end = match exchange.end {
        End::Complete => "complete",
        End::Error => "error",
        End::Stopped => "stopped",
    };
    line.insert("end".into(), json!(end));
    Value::Object(line)
}

/// `full` written against `base`, the body of exchange `after`: `after`;
/// `keep`, each field equal to the base's as `true` and each array that
/// starts with the base's first `n` items as `n`; and `body`, every other
/// field and those arrays' tails. A reader rebuilds `full` from the base's
/// fields in their order, those in `keep` or `body`, then `body`'s new ones.
/// `None` when nothing is kept, or when that rebuild would not give back
/// `full` exactly, key order included.
fn delta(
    after: u64,
    base: &Map<String, Value>,
    full: &Map<String, Value>,
) -> Option<Map<String, Value>> {
    let mut keep = Map::new();
    let mut body = Map::new();
    for (key, value) in full {
        match (base.get(key), value) {
            (Some(was), _) if was == value => {
                keep.insert(key.clone(), json!(true));
            }
            (Some(Value::Array(was)), Value::Array(items)) => {
                let kept = was
                    .iter()
                    .zip(items)
                    .take_while(|(was, item)| was == item)
                    .count();
                if kept > 0 {
                    keep.insert(key.clone(), json!(kept));
                    body.insert(
                        key.clone(),
                        Value::Array(items.iter().skip(kept).cloned().collect()),
                    );
                } else {
                    body.insert(key.clone(), value.clone());
                }
            }
            _ => {
                body.insert(key.clone(), value.clone());
            }
        }
    }
    // Compared as text: maps compare equal whatever their key order.
    let exact = rebuilt(base, &keep, &body).is_some_and(|rebuilt| same_text(&rebuilt, full));
    if keep.is_empty() || !exact {
        return None;
    }
    let mut delta = Map::new();
    delta.insert("after".into(), json!(after));
    delta.insert("keep".into(), Value::Object(keep));
    delta.insert("body".into(), Value::Object(body));
    Some(delta)
}

/// How much of its base `delta` keeps: each field kept whole counts one,
/// each array the items it keeps.
fn kept(delta: &Map<String, Value>) -> u64 {
    keep_of(delta)
        .values()
        .map(|kept| kept.as_u64().unwrap_or(1))
        .sum()
}

/// Whether `delta` keeps all of `base`: every field whole or, for an
/// array, every item.
fn extends(base: &Map<String, Value>, delta: &Map<String, Value>) -> bool {
    let keep = keep_of(delta);
    base.iter()
        .all(|(key, value)| match (keep.get(key), value) {
            (Some(Value::Bool(true)), _) => true,
            (Some(kept), Value::Array(items)) => kept.as_u64() == u64::try_from(items.len()).ok(),
            _ => false,
        })
}

fn keep_of(delta: &Map<String, Value>) -> Map<String, Value> {
    delta
        .get("keep")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// The body a reader rebuilds from `base`, `keep` and `body` (see
/// [`delta`]); `None` when they do not fit together.
fn rebuilt(
    base: &Map<String, Value>,
    keep: &Map<String, Value>,
    body: &Map<String, Value>,
) -> Option<Map<String, Value>> {
    let mut full = Map::new();
    let keys = base
        .keys()
        .filter(|key| keep.contains_key(*key) || body.contains_key(*key))
        .chain(body.keys().filter(|key| !base.contains_key(*key)));
    for key in keys {
        let value = match (keep.get(key), base.get(key), body.get(key)) {
            (Some(Value::Bool(true)), Some(was), _) => was.clone(),
            (Some(Value::Number(kept)), Some(Value::Array(was)), Some(Value::Array(tail))) => {
                let kept = usize::try_from(kept.as_u64()?).ok()?;
                Value::Array(was.iter().take(kept).chain(tail).cloned().collect())
            }
            (None, _, Some(value)) => value.clone(),
            _ => return None,
        };
        full.insert(key.clone(), value);
    }
    Some(full)
}

fn same_text(left: &Map<String, Value>, right: &Map<String, Value>) -> bool {
    serde_json::to_string(left).ok() == serde_json::to_string(right).ok()
}

fn ms(at: Duration) -> u64 {
    u64::try_from(at.as_millis()).unwrap_or(u64::MAX)
}

/// Headers as an object of lowercase names; a repeated header's values as
/// an array. A credential header's value is [`REDACTED`].
fn headers(scrubber: &mut Scrubber, headers: &HeaderMap) -> Value {
    let mut out = Map::new();
    for name in headers.keys() {
        let values: Vec<Value> = headers
            .get_all(name)
            .iter()
            .map(|value| match value.to_str() {
                _ if is_secret_header(name.as_str()) => json!(REDACTED),
                Ok(value) => json!(scrubber.scrub_text(value)),
                Err(_) => json!({ "bytes": value.len() }),
            })
            .collect();
        let value = match <[Value; 1]>::try_from(values) {
            Ok([value]) => value,
            Err(values) => Value::Array(values),
        };
        out.insert(name.as_str().to_owned(), value);
    }
    Value::Object(out)
}

/// A body as JSON, as text, or by its size when it is not UTF-8.
fn body(scrubber: &mut Scrubber, bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => text_value(scrubber, text),
        Err(_) => json!({ "bytes": bytes.len() }),
    }
}

/// `text` as the JSON it holds, else as a string, scrubbed.
fn text_value(scrubber: &mut Scrubber, text: &str) -> Value {
    match serde_json::from_str::<Value>(text) {
        Ok(mut json) => {
            scrubber.scrub_json_value(None, &mut json);
            json
        }
        Err(_) => json!(scrubber.scrub_body(text)),
    }
}

fn parts(scrubber: &mut Scrubber, form: &MultipartForm) -> Value {
    form.parts()
        .iter()
        .map(|part| {
            let mut out = Map::new();
            out.insert("name".into(), json!(part.name()));
            if let Some(filename) = part.get_filename() {
                out.insert("filename".into(), json!(scrubber.scrub_text(filename)));
            }
            if let Some(content_type) = part.get_content_type() {
                out.insert("content_type".into(), json!(content_type.to_string()));
            }
            let value = match part.content() {
                PartContent::Text(text) => text_value(scrubber, text),
                PartContent::Binary(bytes) => json!({ "bytes": bytes.len() }),
            };
            out.insert("value".into(), value);
            Value::Object(out)
        })
        .collect()
}

/// A stream's chunks joined, then cut into SSE events (`sse`) or lines,
/// each timed by the chunk that ended it, so nothing is split between
/// chunks. A stream that is not UTF-8 is kept by its size; text after the
/// last complete event, as a stopped stream leaves, is the last event.
fn events(scrubber: &mut Scrubber, chunks: &[(Duration, Bytes)], sse: bool) -> Value {
    let separator: &[u8] = if sse { b"\n\n" } else { b"\n" };
    let mut buffer: Vec<u8> = Vec::new();
    let mut raw: Vec<(Duration, Vec<u8>)> = Vec::new();
    for (at, chunk) in chunks {
        buffer.extend(chunk.iter().filter(|byte| !sse || **byte != b'\r'));
        while let Some(end) = find(&buffer, separator) {
            let rest = buffer.split_off(end + separator.len());
            buffer.truncate(end);
            raw.push((*at, std::mem::replace(&mut buffer, rest)));
        }
    }
    if let Some((at, _)) = chunks.last()
        && !buffer.iter().all(u8::is_ascii_whitespace)
    {
        raw.push((*at, buffer));
    }

    let mut out = Vec::new();
    for (at, bytes) in raw {
        let Ok(text) = String::from_utf8(bytes) else {
            let size: usize = chunks.iter().map(|(_, chunk)| chunk.len()).sum();
            return json!([{ "ms": ms(chunks.last().map_or(Duration::ZERO, |(at, _)| *at)), "data": { "bytes": size } }]);
        };
        if text.trim().is_empty() {
            continue;
        }
        let mut event = Map::new();
        event.insert("ms".into(), json!(ms(at)));
        if sse {
            let mut data: Vec<&str> = Vec::new();
            for field in text.lines() {
                let (name, value) = field.split_once(':').unwrap_or((field, ""));
                let value = value.strip_prefix(' ').unwrap_or(value);
                match name {
                    "data" => data.push(value),
                    "event" | "id" => {
                        event.insert(name.into(), json!(scrubber.scrub_text(value)));
                    }
                    // A comment, or a field SSE does not define.
                    _ => {}
                }
            }
            if data.is_empty() {
                continue;
            }
            event.insert("data".into(), text_value(scrubber, &data.join("\n")));
        } else {
            event.insert("data".into(), text_value(scrubber, &text));
        }
        out.push(Value::Object(event));
    }
    Value::Array(out)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
