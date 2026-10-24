//! One exchange as a scrubbed JSON line: the writer thread's work.

use std::time::Duration;

use rig_core::http_client::{Bytes, HeaderMap, MultipartForm, multipart::PartContent};
use serde_json::{Map, Value, json};

use super::{End, Exchange, RequestBody};
use crate::scrub::{Purpose, REDACTED, Scrubber, is_secret_header};

/// The version of the line format, the `v` of every line. A change that
/// renames, removes or retypes a field takes the next one.
pub const LINE_VERSION: u64 = 1;

/// `exchange` as one line, written by process `pid`.
pub(super) fn line(exchange: &Exchange, pid: u32) -> Value {
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
            request.insert("body".into(), body(&mut scrubber, bytes));
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
        Err(_) => json!(scrubber.scrub_text(text)),
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
