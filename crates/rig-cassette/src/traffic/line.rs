//! One exchange as a scrubbed JSON line: the writer thread's work.

use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use rig_core::http_client::{Bytes, HeaderMap, MultipartForm, multipart::PartContent};
use serde_json::{Map, Value, json};

use super::{End, Exchange, RequestBody};
use crate::scrub::{OpaqueBody, PreparedBody, Purpose, ScrubbedEvent, Scrubber};

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
    let raw_response = exchange.response_bytes();
    let prepared = scrubber.prepare_exchange_with_headers(
        &exchange.uri,
        crate::scrub::header_pairs(&exchange.request_headers),
        match &exchange.request_body {
            RequestBody::Bytes(bytes) => crate::scrub::RequestBodyRef::Bytes(bytes),
            RequestBody::Multipart(form) => crate::scrub::RequestBodyRef::Multipart(form),
        },
        response_headers
            .into_iter()
            .flat_map(crate::scrub::header_pairs),
        raw_response.as_deref(),
    );
    let request_withheld = prepared.request_withheld().cloned();
    let response_withheld = prepared.response_withheld().cloned();
    let request_metadata = prepared
        .request_body
        .as_ref()
        .map(|body| body.summary().clone());
    let response_metadata = prepared
        .response_body
        .as_ref()
        .map(|body| body.summary().clone());
    let request_sized = request_withheld.is_some()
        || prepared
            .request_body
            .as_ref()
            .is_some_and(PreparedBody::is_binary);
    let request_uninspected = request_withheld.is_some();
    let response_uninspected = response_withheld.is_some()
        || prepared
            .response_body
            .as_ref()
            .is_some_and(PreparedBody::is_binary)
        || exchange.incomplete().is_some();
    let response_headers = response_headers.map(|headers| {
        let mut scrubbed = headers.clone();
        for (name, value) in &mut scrubbed {
            if let Ok(text) = std::str::from_utf8(value.as_bytes()) {
                let rendered = prepared.scrub_response_header(&mut scrubber, name.as_str(), text);
                if let Ok(rendered) = rendered.parse() {
                    *value = rendered;
                }
            }
        }
        scrubbed
    });
    if request_uninspected {
        bases.0.clear();
    }
    let request_body = if request_sized {
        None
    } else {
        prepared
            .request_body
            .map(|body| scrubber.scrub_prepared(body))
    };
    let response_body = prepared.response_body;

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
    match (&exchange.request_body, &request_body) {
        _ if request_sized => {
            if let Some(metadata) = request_metadata {
                request.insert("body".into(), metadata.value());
            }
        }
        (RequestBody::Bytes(_), Some(bytes)) if bytes.is_empty() => {}
        (RequestBody::Bytes(_), Some(bytes)) => {
            let body = if std::str::from_utf8(bytes).is_err() {
                request_metadata
                    .as_ref()
                    .map_or_else(|| body(bytes), OpaqueBody::value)
            } else {
                body(bytes)
            };
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
        (RequestBody::Multipart(form), _) => {
            request.insert("parts".into(), parts(&mut scrubber, form));
        }
        (RequestBody::Bytes(_), None) => {}
    }
    line.insert("request".into(), Value::Object(request));

    if let Some(status) = exchange.status {
        let mut response = Map::new();
        response.insert("status".into(), json!(status.as_u16()));
        if let Some(headers_map) = &response_headers {
            response.insert("headers".into(), headers(&mut scrubber, headers_map));
        }
        if let (Some(prepared), Some(response_metadata)) = (response_body, response_metadata) {
            if let Some(chunks) = &exchange.chunks {
                response.insert(
                    "events".into(),
                    events(
                        &mut scrubber,
                        chunks,
                        prepared,
                        !response_uninspected,
                        &response_metadata,
                    ),
                );
            } else if response_uninspected {
                response.insert("body".into(), response_metadata.value());
            } else {
                let bytes = scrubber.scrub_prepared(prepared);
                response.insert(
                    "body".into(),
                    if std::str::from_utf8(&bytes).is_err() {
                        response_metadata.value()
                    } else {
                        body(&bytes)
                    },
                );
            }
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
        line.insert(
            "error".into(),
            json!(if request_uninspected {
                format!("[{} bytes]", error.len())
            } else {
                scrubber.scrub_text(error)
            }),
        );
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
            .map(|value| match std::str::from_utf8(value.as_bytes()) {
                Ok(value) => json!(scrubber.scrub_header_value(name.as_str(), value)),
                Err(_) => OpaqueBody::new(value.len(), None).value(),
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

/// A prepared body as JSON, text, or its size when it is opaque.
fn body(bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => serde_json::from_str(text).unwrap_or_else(|_| json!(text)),
        Err(_) => OpaqueBody::new(bytes.len(), None).value(),
    }
}

fn parts(scrubber: &mut Scrubber, form: &MultipartForm) -> Value {
    form.parts()
        .iter()
        .map(|part| {
            let mut out = Map::new();
            let metadata = scrubber.scrub_part_metadata(&crate::scrub::part_metadata(part));
            for (name, value) in metadata {
                let name = if name == "content-type" {
                    "content_type".to_owned()
                } else {
                    name
                };
                out.insert(name, json!(value));
            }
            let value = match part.content() {
                PartContent::Text(text) => {
                    body(scrubber.scrub_part_value(part.name(), text).as_bytes())
                }
                PartContent::Binary(bytes) => OpaqueBody::new(bytes.len(), None).value(),
            };
            out.insert("value".into(), value);
            Value::Object(out)
        })
        .collect()
}

/// Scrub complete events and retain the time of each event's last chunk.
fn events(
    scrubber: &mut Scrubber,
    chunks: &[(Duration, Bytes)],
    body: PreparedBody,
    inspected: bool,
    metadata: &OpaqueBody,
) -> Value {
    let mut ends = Vec::new();
    let mut end = 0;
    for (at, chunk) in chunks {
        end += chunk.len();
        ends.push((end, *at));
    }
    let came = |offset| {
        ends.get(ends.partition_point(|(end, _)| *end < offset))
            .or_else(|| ends.last())
            .map_or(Duration::ZERO, |(_, at)| *at)
    };
    let events = if inspected {
        scrubber.scrub_stream(body, ends.iter().map(|(end, _)| *end))
    } else {
        vec![ScrubbedEvent {
            end,
            event: None,
            id: None,
            data: metadata.value(),
        }]
    };
    Value::Array(
        events
            .into_iter()
            .map(|event| {
                let mut value = Map::new();
                value.insert("ms".into(), json!(ms(came(event.end))));
                if let Some(name) = event.event {
                    value.insert("event".into(), json!(name));
                }
                if let Some(id) = event.id {
                    value.insert("id".into(), json!(id));
                }
                value.insert("data".into(), event.data);
                Value::Object(value)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests;
