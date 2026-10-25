//! The one scrubber of recorded provider traffic, shared by the cassette
//! engine and the traffic log. It removes credentials (sensitive headers and
//! query parameters, and every secret [`rig_core::redact`] finds), AWS account
//! numbers and home-directory paths, and keeps everything else the provider
//! sent verbatim. A [`Purpose`] selects the few behaviours a replayable
//! fixture needs and a log of live traffic must not have. New captures retain
//! only size and content type when a complete body cannot be inspected.

use rig_core::http_client::framing::{SseField, SseFramer};
use rig_core::http_client::{
    Bytes, HeaderMap, MultipartForm,
    multipart::{Part, PartContent},
};
use rig_core::redact;
use serde_json::{Map, Value};
use std::ops::Range;

#[cfg(test)]
mod tests;
use std::collections::BTreeMap;

pub(crate) const REDACTED: &str = redact::REDACTED;

const AUTHORITATIVE_MINIMUM: usize = 3;
const PROSE_MINIMUM: usize = 8;

pub(crate) const SENSITIVE_HEADER_NAMES: &[&str] = &[
    // A session cookie, and the ChatGPT account a sign-in belongs to.
    "cookie",
    "chatgpt-account-id",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "x-amz-content-sha256",
    "x-amz-date",
    "key",
];

pub(crate) const VOLATILE_JSON_KEYS: &[&str] = &[
    "completed_at",
    "created",
    "created_at",
    "updated",
    "updated_at",
];

/// Credential material a token exchange can return in a body.
const OAUTH_TOKEN_KEYS: &[&str] = &["access_token", "id_token", "refresh_token"];

/// What a scrubbed exchange is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// A replayable fixture: volatile timestamps are normalized and an OAuth
    /// token becomes a numbered placeholder, so a replay can tell two apart.
    #[default]
    Cassette,
    /// A log of live traffic: timestamps stay as sent and an OAuth token is
    /// [`REDACTED`] like every other secret.
    Traffic,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SecretSource {
    Text,
    Field,
    SoftNumber,
    Wire,
    Authorization,
    Number,
}

impl SecretSource {
    fn minimum(self) -> usize {
        match self {
            Self::Wire | Self::Authorization | Self::Number => AUTHORITATIVE_MINIMUM,
            _ => PROSE_MINIMUM,
        }
    }
}

#[derive(PartialEq, Eq)]
struct LearnedSecret {
    value: String,
    source: SecretSource,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JsonContext {
    Value,
    Headers,
    Credential,
    ApplicationCredential,
    Definitions,
    Definition,
    Schema,
    Properties,
}

struct JsonObjectContext {
    kind: JsonContext,
    logprobs: bool,
}

impl JsonContext {
    fn object(self, fields: &Map<String, Value>) -> JsonObjectContext {
        JsonObjectContext {
            kind: match self {
                Self::Value if is_schema(fields) => Self::Schema,
                Self::Definitions => Self::Definition,
                context => context,
            },
            logprobs: is_logprobs(fields),
        }
    }
}

impl JsonObjectContext {
    fn field(&self, name: &str, purpose: Purpose) -> JsonContext {
        match self.kind {
            JsonContext::Credential | JsonContext::ApplicationCredential => return self.kind,
            JsonContext::Properties => return JsonContext::Schema,
            JsonContext::Definition => match name {
                "function" => return JsonContext::Definition,
                "functionDeclarations" | "function_declarations" => {
                    return JsonContext::Definitions;
                }
                "parameters"
                | "input_schema"
                | "inputSchema"
                | "parametersJsonSchema"
                | "schema"
                | "output_schema" => return JsonContext::Schema,
                _ => {}
            },
            JsonContext::Definitions => {}
            JsonContext::Schema => match name {
                "properties" | "patternProperties" | "definitions" | "$defs"
                | "dependentSchemas" => return JsonContext::Properties,
                "items"
                | "prefixItems"
                | "contains"
                | "additionalProperties"
                | "propertyNames"
                | "not"
                | "if"
                | "then"
                | "else"
                | "allOf"
                | "oneOf"
                | "anyOf" => return JsonContext::Schema,
                _ => {}
            },
            JsonContext::Headers if is_secret_header(name) => {
                return if is_cookie(name) {
                    JsonContext::ApplicationCredential
                } else {
                    JsonContext::Credential
                };
            }
            JsonContext::Headers | JsonContext::Value => {}
        }
        if is_secret_field(name, purpose) && !(name.eq_ignore_ascii_case("token") && self.logprobs)
        {
            credential_context(name)
        } else if name.eq_ignore_ascii_case("headers") {
            JsonContext::Headers
        } else if matches!(name, "tools" | "functions") {
            JsonContext::Definitions
        } else if name == "json_schema" {
            JsonContext::Definition
        } else {
            JsonContext::Value
        }
    }
}

/// Scrubs the text, JSON and SSE bodies of one recording. Placeholders are
/// numbered per scrubber, so one scrubber serves one recording.
#[derive(Default)]
pub(crate) struct Scrubber {
    purpose: Purpose,
    placeholders: BTreeMap<String, String>,
    counters: BTreeMap<&'static str, usize>,
    /// Credentials the recording carried, scrubbed wherever they show up
    /// again, such as an error body that echoes the key; longest first.
    learned: Vec<LearnedSecret>,
}

pub(crate) enum RequestBodyRef<'a> {
    Bytes(&'a [u8]),
    Multipart(&'a MultipartForm),
}

pub(crate) struct PreparedExchange {
    pub(crate) request_body: Option<PreparedBody>,
    pub(crate) response_body: Option<PreparedBody>,
}

/// Metadata retained when a recording cannot inspect a complete body.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct OpaqueBody {
    pub(crate) bytes: usize,
    pub(crate) content_type: Option<String>,
}

impl std::fmt::Display for OpaqueBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "uninspectable body ({} bytes, {:?})",
            self.bytes, self.content_type
        )
    }
}

impl std::error::Error for OpaqueBody {}

impl OpaqueBody {
    pub(crate) fn value(&self) -> Value {
        serde_json::json!({"bytes": self.bytes, "content_type": self.content_type})
    }
}

enum BodyFormat {
    Json,
    JsonLines,
    Multipart(MultipartLayout),
    Form,
    Sse,
    Text,
    #[cfg(feature = "bedrock")]
    EventStream,
    Binary,
    Opaque,
}

pub(crate) struct PreparedBody {
    bytes: Bytes,
    format: BodyFormat,
    metadata: OpaqueBody,
}

impl PreparedBody {
    fn new(raw: &[u8], headers: &[(&str, &str)]) -> Self {
        let format = capture_format(raw, headers);
        Self {
            bytes: if matches!(format, BodyFormat::Opaque) {
                Bytes::new()
            } else {
                Bytes::copy_from_slice(raw)
            },
            format,
            metadata: OpaqueBody {
                bytes: raw.len(),
                content_type: header_value(headers, "content-type").map(str::to_owned),
            },
        }
    }

    pub(crate) fn opaque(&self) -> Option<&OpaqueBody> {
        (matches!(self.format, BodyFormat::Opaque)).then_some(&self.metadata)
    }

    pub(crate) fn is_binary(&self) -> bool {
        matches!(self.format, BodyFormat::Binary)
    }

    pub(crate) fn summary(&self) -> &OpaqueBody {
        &self.metadata
    }
}

/// Whether a body's declared format must retain its exact binary bytes.
#[cfg(feature = "http")]
pub(crate) fn is_binary_body<'a>(
    raw: &[u8],
    headers: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> bool {
    matches!(
        capture_format(raw, &headers.into_iter().collect::<Vec<_>>()),
        BodyFormat::Binary
    )
}

/// Classify new captures once for both sinks; legacy fixture normalization bypasses it.
fn capture_format(raw: &[u8], headers: &[(&str, &str)]) -> BodyFormat {
    if headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-encoding")
            && !value.trim().eq_ignore_ascii_case("identity")
    }) {
        return BodyFormat::Opaque;
    }
    let content_type = header_value(headers, "content-type");
    let media = content_type
        .unwrap_or_default()
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let valid_media = media.split_once('/').is_some_and(|(kind, subtype)| {
        [kind, subtype].iter().all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
        })
    });
    if (content_type.is_some() && !valid_media)
        || headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("content-type"))
            .count()
            > 1
        || (content_type.is_some() && media.is_empty())
    {
        return BodyFormat::Opaque;
    }
    if media == "multipart/form-data" {
        return content_type
            .and_then(multipart_boundary)
            .and_then(|boundary| multipart_slices(raw, &boundary, true))
            .filter(|layout| {
                layout.parts.iter().all(|slice| {
                    let headers: Vec<_> = slice
                        .part
                        .headers
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str()))
                        .collect();
                    let nested = header_value(&headers, "content-type").is_some_and(|value| {
                        value
                            .trim_start()
                            .to_ascii_lowercase()
                            .starts_with("multipart/")
                    });
                    !nested
                        && !matches!(
                            capture_format(&slice.part.body, &headers),
                            BodyFormat::Opaque
                        )
                })
            })
            .map_or(BodyFormat::Opaque, BodyFormat::Multipart);
    }
    #[cfg(feature = "bedrock")]
    if media == "application/vnd.amazon.eventstream" {
        return if event_stream_messages(raw).is_some_and(|messages| {
            messages.iter().all(|message| {
                std::str::from_utf8(message.payload())
                    .is_ok_and(|text| serde_json::from_str::<Value>(text).is_ok())
            })
        }) {
            BodyFormat::EventStream
        } else {
            BodyFormat::Opaque
        };
    }
    if media.starts_with("audio/")
        || media.starts_with("image/")
        || media.starts_with("video/")
        || media.starts_with("font/")
        || matches!(
            media.as_str(),
            "application/octet-stream"
                | "application/pdf"
                | "application/zip"
                | "application/gzip"
                | "application/x-protobuf"
                | "application/protobuf"
                | "application/cbor"
        )
    {
        return BodyFormat::Binary;
    }
    let text_claim = media.starts_with("text/")
        || media.ends_with("+json")
        || media.ends_with("+xml")
        || matches!(
            media.as_str(),
            "application/json"
                | "application/x-ndjson"
                | "application/ndjson"
                | "application/jsonl"
                | "application/x-www-form-urlencoded"
                | "application/xml"
                | "application/yaml"
        );
    let Ok(text) = std::str::from_utf8(raw) else {
        return if text_claim || media.is_empty() {
            BodyFormat::Opaque
        } else {
            BodyFormat::Binary
        };
    };
    if raw.is_empty() {
        return BodyFormat::Text;
    }
    if media == "application/json" || media.ends_with("+json") {
        return if serde_json::from_str::<Value>(text).is_ok() {
            BodyFormat::Json
        } else {
            BodyFormat::Opaque
        };
    }
    if matches!(
        media.as_str(),
        "application/x-ndjson" | "application/ndjson" | "application/jsonl"
    ) {
        return if text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .all(|line| serde_json::from_str::<Value>(line).is_ok())
        {
            BodyFormat::JsonLines
        } else {
            BodyFormat::Opaque
        };
    }
    if media == "text/event-stream" || (media.is_empty() && parse_sse(text, false).is_some()) {
        return if ends_between_events(raw)
            && parse_sse(text, true).is_some_and(|events| {
                events
                    .iter()
                    .all(|event| event.data.as_deref().is_none_or(inspectable_text))
            }) {
            BodyFormat::Sse
        } else {
            BodyFormat::Opaque
        };
    }
    if media == "application/x-www-form-urlencoded" {
        return if inspectable_text(text) {
            BodyFormat::Form
        } else {
            BodyFormat::Opaque
        };
    }
    if text_claim && !matches!(media.as_str(), "text/plain") {
        return BodyFormat::Opaque;
    }
    if inspectable_text(text) {
        BodyFormat::Text
    } else {
        BodyFormat::Opaque
    }
}

fn inspectable_text(text: &str) -> bool {
    let text = text.trim();
    !text
        .chars()
        .any(|ch| ch.is_control() && !matches!(ch, '\t' | '\r' | '\n'))
        && (!text.starts_with(['{', '['])
            || text == "[DONE]"
            || serde_json::from_str::<Value>(text).is_ok())
}

#[cfg(feature = "bedrock")]
use aws_smithy_eventstream::frame::{read_message_from, write_message_to};
#[cfg(feature = "bedrock")]
use aws_smithy_types::event_stream::Message as EventStreamMessage;

impl Scrubber {
    /// A scrubber for `purpose`.
    pub(crate) fn new(purpose: Purpose) -> Self {
        Self {
            purpose,
            ..Self::default()
        }
    }

    /// Inspect and learn both sides before either sink renders a body.
    pub(crate) fn prepare_exchange<'a, 'b>(
        &mut self,
        uri: &str,
        request_headers: impl IntoIterator<Item = (&'a str, &'a str)>,
        request_body: RequestBodyRef<'_>,
        response_headers: impl IntoIterator<Item = (&'b str, &'b str)>,
        response_body: Option<&[u8]>,
    ) -> PreparedExchange {
        let request_headers: Vec<_> = request_headers.into_iter().collect();
        let response_headers: Vec<_> = response_headers.into_iter().collect();
        self.learn_url(uri);
        for (name, value) in request_headers.iter().chain(&response_headers) {
            self.learn_header(name, value);
        }
        let mut request_body = match request_body {
            RequestBodyRef::Bytes(bytes) => Some(PreparedBody::new(bytes, &request_headers)),
            RequestBodyRef::Multipart(form) => {
                let opaque = request_headers.iter().any(|(name, value)| {
                    name.eq_ignore_ascii_case("content-encoding")
                        && !value.trim().eq_ignore_ascii_case("identity")
                }) || form.parts().iter().any(|part| {
                    let metadata = part_metadata(part);
                    let headers: Vec<_> = metadata
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str()))
                        .collect();
                    match part.content() {
                        PartContent::Text(text) => PreparedBody::new(text.as_bytes(), &headers)
                            .opaque()
                            .is_some(),
                        PartContent::Binary(bytes) => {
                            PreparedBody::new(bytes, &headers).opaque().is_some()
                        }
                    }
                });
                if opaque {
                    let bytes = form
                        .parts()
                        .iter()
                        .map(|part| match part.content() {
                            PartContent::Text(text) => text.len(),
                            PartContent::Binary(bytes) => bytes.len(),
                        })
                        .sum();
                    Some(PreparedBody {
                        bytes: Bytes::new(),
                        format: BodyFormat::Opaque,
                        metadata: OpaqueBody {
                            bytes,
                            content_type: Some("multipart/form-data".into()),
                        },
                    })
                } else {
                    for part in form.parts() {
                        let metadata = part_metadata(part);
                        self.learn_part_metadata(&metadata);
                        let bytes = match part.content() {
                            PartContent::Text(text) => text.as_bytes(),
                            PartContent::Binary(bytes) => bytes.as_ref(),
                        };
                        if let Ok(text) = std::str::from_utf8(bytes) {
                            self.learn_param(part.name(), text);
                        }
                        let headers: Vec<_> = metadata
                            .iter()
                            .map(|(name, value)| (name.as_str(), value.as_str()))
                            .collect();
                        self.learn_body_parts(&PreparedBody::new(bytes, &headers));
                    }
                    None
                }
            }
        };
        let mut response_body =
            response_body.map(|bytes| PreparedBody::new(bytes, &response_headers));
        for body in request_body.iter().chain(&response_body) {
            self.learn_body_parts(body);
        }
        for body in request_body.iter_mut().chain(&mut response_body) {
            body.metadata.content_type = body
                .metadata
                .content_type
                .as_deref()
                .map(|text| self.scrub_text(text));
        }
        PreparedExchange {
            request_body,
            response_body,
        }
    }

    fn learn_body_parts(&mut self, body: &PreparedBody) {
        match &body.format {
            BodyFormat::Multipart(layout) => {
                for slice in &layout.parts {
                    self.learn_part_metadata(&slice.part.headers);
                    if let Ok(text) = std::str::from_utf8(&slice.part.body) {
                        self.learn_param(slice.part.name().unwrap_or_default(), text);
                    }
                    let headers: Vec<_> = slice
                        .part
                        .headers
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str()))
                        .collect();
                    self.learn_body_parts(&PreparedBody::new(&slice.part.body, &headers));
                }
                return;
            }
            BodyFormat::Opaque | BodyFormat::Binary => return,
            BodyFormat::JsonLines => {
                if let Ok(text) = std::str::from_utf8(&body.bytes) {
                    for line in text.lines() {
                        if let Ok(json) = serde_json::from_str::<Value>(line) {
                            self.learn_json(&json);
                        }
                    }
                }
                return;
            }
            #[cfg(feature = "bedrock")]
            BodyFormat::EventStream => {
                self.learn_event_stream(&body.bytes);
                return;
            }
            _ => {}
        }
        if let Ok(text) = std::str::from_utf8(&body.bytes) {
            if matches!(body.format, BodyFormat::Form) {
                self.learn_form(text, SecretSource::Wire);
            } else {
                self.learn_body(text);
            }
        } else {
            #[cfg(feature = "bedrock")]
            self.learn_event_stream(&body.bytes);
        }
    }

    #[cfg(feature = "bedrock")]
    fn learn_event_stream(&mut self, bytes: &[u8]) {
        if let Some(messages) = event_stream_messages(bytes) {
            for message in messages {
                if let Ok(text) = std::str::from_utf8(message.payload()) {
                    self.learn_body(text);
                }
            }
        }
    }

    pub(crate) fn scrub_prepared(&mut self, body: PreparedBody) -> Bytes {
        match body.format {
            BodyFormat::Multipart(layout) => self.scrub_multipart_body(body.bytes, layout),
            BodyFormat::Opaque => Bytes::from(body.metadata.value().to_string()),
            BodyFormat::Binary => body.bytes,
            BodyFormat::Form => Bytes::from(self.scrub_text(&mask_form(
                std::str::from_utf8(&body.bytes).unwrap_or_default(),
            ))),
            BodyFormat::JsonLines => {
                let mut output = String::new();
                for line in std::str::from_utf8(&body.bytes)
                    .unwrap_or_default()
                    .split_inclusive('\n')
                {
                    if let Ok(mut json) = serde_json::from_str::<Value>(line) {
                        self.scrub_json_in(None, &mut json, JsonContext::Value);
                        output.push_str(&json.to_string());
                    }
                    if line.ends_with('\n') {
                        output.push('\n');
                    }
                }
                Bytes::from(output)
            }
            _ => self.scrub_bytes(&body.bytes),
        }
    }

    fn scrub_multipart_body(&mut self, bytes: Bytes, layout: MultipartLayout) -> Bytes {
        let mut out = Vec::new();
        if !layout.preamble.is_empty() {
            out.extend_from_slice(b"[REDACTED]\r\n");
        }
        let mut kept = layout.preamble.end;
        for slice in layout.parts {
            out.extend_from_slice(bytes.get(kept..slice.header.start).unwrap_or_default());
            let headers = self.scrub_part_metadata(&slice.part.headers);
            if headers == slice.part.headers {
                out.extend_from_slice(bytes.get(slice.header.clone()).unwrap_or_default());
            } else {
                out.extend_from_slice(
                    headers
                        .iter()
                        .map(|(name, value)| format!("{name}: {value}"))
                        .collect::<Vec<_>>()
                        .join("\r\n")
                        .as_bytes(),
                );
            }
            out.extend_from_slice(
                bytes
                    .get(slice.header.end..slice.body.start)
                    .unwrap_or_default(),
            );
            let name = slice.part.name().unwrap_or_default();
            if is_secret_param(&decode_form_component(name)) {
                out.extend_from_slice(REDACTED.as_bytes());
            } else {
                let part_headers: Vec<_> = slice
                    .part
                    .headers
                    .iter()
                    .map(|(name, value)| (name.as_str(), value.as_str()))
                    .collect();
                out.extend_from_slice(
                    &self.scrub_prepared(PreparedBody::new(&slice.part.body, &part_headers)),
                );
            }
            kept = slice.body.end;
        }
        out.extend_from_slice(bytes.get(kept..layout.epilogue.start).unwrap_or_default());
        if !layout.epilogue.is_empty() {
            out.extend_from_slice(REDACTED.as_bytes());
        }
        out.into()
    }

    fn learn_part_metadata(&mut self, metadata: &[(String, String)]) {
        for (name, value) in metadata {
            self.learn_header(name, value);
        }
    }

    pub(crate) fn scrub_part_metadata(
        &mut self,
        metadata: &[(String, String)],
    ) -> Vec<(String, String)> {
        metadata
            .iter()
            .map(|(name, value)| (name.clone(), self.scrub_header_value(name, value)))
            .collect()
    }

    #[cfg(feature = "bedrock")]
    fn scrub_event_stream_body(&mut self, bytes: Vec<u8>) -> Option<Vec<u8>> {
        let mut output = Vec::new();
        for message in event_stream_messages(&bytes)? {
            let payload = std::str::from_utf8(message.payload()).ok()?;
            let scrubbed_payload = self.render_body(payload);
            let scrubbed = EventStreamMessage::new_from_parts(
                message.headers().to_vec(),
                scrubbed_payload.into_bytes(),
            );
            write_message_to(&scrubbed, &mut output).ok()?;
        }

        Some(output)
    }

    /// Normalize legacy fixture bytes, preserving their opaque representation.
    pub(crate) fn scrub_bytes(&mut self, bytes: &[u8]) -> Bytes {
        if let Some((text, tail)) = text_prefix(bytes) {
            let mut out = self.render_body(text).into_bytes();
            out.extend_from_slice(bytes.get(bytes.len() - tail..).unwrap_or_default());
            out.into()
        } else if let Some(bytes) = self.scrub_binary_frames(bytes) {
            bytes.into()
        } else if self.learned.iter().any(|secret| {
            secret.source != SecretSource::Text && secret_occurs(bytes, &secret.value)
        }) {
            Bytes::from_static(REDACTED.as_bytes())
        } else {
            Bytes::copy_from_slice(bytes)
        }
    }

    fn scrub_binary_frames(&mut self, bytes: &[u8]) -> Option<Vec<u8>> {
        #[cfg(feature = "bedrock")]
        {
            self.scrub_event_stream_body(bytes.to_vec())
        }
        #[cfg(not(feature = "bedrock"))]
        {
            let _ = bytes;
            None
        }
    }

    pub(crate) fn scrub_header_value(&mut self, name: &str, value: &str) -> String {
        if is_secret_header(name) {
            REDACTED.to_owned()
        } else {
            self.scrub_text(value)
        }
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn scrub_param_value(&mut self, name: &str, value: &str) -> String {
        if is_secret_param(&decode_form_component(name)) {
            REDACTED.to_owned()
        } else {
            self.scrub_text(value)
        }
    }

    pub(crate) fn scrub_part_value(&mut self, name: &str, value: &str) -> String {
        if is_secret_param(&decode_form_component(name)) {
            REDACTED.to_owned()
        } else {
            self.render_body(value)
        }
    }

    /// A cassette scrubber with the live request's authoritative context.
    #[cfg(any(feature = "http", test))]
    pub(crate) fn for_request(uri: &str, headers: &HeaderMap) -> Self {
        let mut scrubber = Self::default();
        scrubber.learn_url(uri);
        for (name, value) in header_pairs(headers) {
            scrubber.learn_header(name, value);
        }
        scrubber
    }

    /// Whether two request scrubbers learned the same authoritative credentials.
    #[cfg(any(feature = "http", test))]
    pub(crate) fn same_credentials(&self, other: &Self) -> bool {
        self.learned == other.learned
    }

    /// Learn legacy fixture bytes without applying new-capture format restrictions.
    #[cfg(feature = "http")]
    pub(crate) fn learn_fixture_body(&mut self, bytes: &[u8]) {
        self.learn_body_parts(&PreparedBody {
            bytes: Bytes::copy_from_slice(bytes),
            format: BodyFormat::Text,
            metadata: OpaqueBody {
                bytes: bytes.len(),
                content_type: None,
            },
        });
    }

    /// Normalize a request body using the caller's request context.
    #[cfg(any(feature = "http", test))]
    pub(crate) fn request_body(
        &mut self,
        content_type: Option<&str>,
        content_encoding: Option<&str>,
        bytes: &[u8],
    ) -> Result<Bytes, OpaqueBody> {
        let headers = content_type
            .map(|value| ("content-type", value))
            .into_iter()
            .chain(content_encoding.map(|value| ("content-encoding", value)));
        let prepared = self.prepare_exchange("", headers, RequestBodyRef::Bytes(bytes), [], None);
        let body = prepared
            .request_body
            .unwrap_or_else(|| PreparedBody::new(bytes, &[]));
        if let Some(opaque) = body.opaque() {
            if content_encoding.is_some_and(|value| !value.trim().eq_ignore_ascii_case("identity"))
            {
                return Err(opaque.clone());
            }
            // Existing fixtures can contain opaque bytes recorded before fail-closed capture.
            return Ok(self.scrub_bytes(bytes));
        }
        Ok(self.scrub_prepared(body))
    }

    /// Remembers credentials in a header, including individual cookies.
    pub(crate) fn learn_header(&mut self, name: &str, value: &str) {
        if is_secret_header(name) {
            let source = if is_cookie(name) {
                SecretSource::Field
            } else if is_auth_name(name) {
                SecretSource::Authorization
            } else {
                SecretSource::Wire
            };
            self.learn_credential(value, source);
            if is_cookie(name) {
                for cookie in value.split(';') {
                    if let Some((_, value)) = cookie.split_once('=') {
                        self.learn(value.trim(), SecretSource::Field);
                    }
                    if name.eq_ignore_ascii_case("set-cookie") {
                        break;
                    }
                }
            }
        } else {
            self.learn_text(value);
        }
    }

    /// Remembers raw and decoded credentials from the exchange's URL.
    pub(crate) fn learn_url(&mut self, uri: &str) {
        self.learn_url_from(uri, SecretSource::Wire);
    }

    fn learn_url_from(&mut self, uri: &str, source: SecretSource) {
        if let Some((_, fragment)) = uri.split_once('#') {
            self.learn_form(fragment, source);
        }
        if let Some((_, query)) = uri.split_once('?') {
            self.learn_form(query.split('#').next().unwrap_or_default(), source);
        }
        if let Ok(url) = url::Url::parse(uri) {
            self.learn(url.username(), SecretSource::Text);
            self.learn(&decode_form_component(url.username()), SecretSource::Text);
            if let Some(password) = url.password() {
                let source = if source == SecretSource::Text {
                    source
                } else {
                    SecretSource::Field
                };
                self.learn(password, source);
                self.learn(&decode_form_component(password), source);
            }
        }
    }

    /// Remembers raw and decoded values of a sensitive wire query or form pair.
    pub(crate) fn learn_param(&mut self, name: &str, value: &str) {
        self.learn_param_from(name, value, SecretSource::Wire);
    }

    fn learn_param_from(&mut self, name: &str, value: &str, source: SecretSource) {
        if is_secret_param(&decode_form_component(name)) {
            self.learn(value, source);
            self.learn(&decode_form_component(value), source);
        }
    }

    fn learn_form(&mut self, form: &str, source: SecretSource) {
        for pair in form.split('&') {
            if let Some((name, value)) = pair.split_once('=') {
                self.learn_param_from(name, value, source);
            }
        }
    }

    /// Remembers credentials in JSON, form, SSE, or incomplete text bodies.
    pub(crate) fn learn_body(&mut self, body: &str) {
        match classify_body(body) {
            TextBody::Json(json) => self.learn_json(&json),
            TextBody::Sse(events) => self.learn_sse(&events),
            TextBody::Form => self.learn_form(body, SecretSource::Wire),
            TextBody::Text => self.learn_text(body),
        }
    }

    fn learn_sse(&mut self, events: &[SseEvent<'_>]) {
        for event in events {
            for line in event
                .lines
                .iter()
                .filter(|line| line.field != SseField::Data)
            {
                self.learn_text(line.text);
            }
            if let Some(data) = &event.data {
                match canonical_json(data) {
                    Some(json) => self.learn_json(&json),
                    None => self.learn_text(data),
                }
            }
        }
    }

    fn learn_text(&mut self, text: &str) {
        for (_, nested, _) in embedded_strings(text) {
            match canonical_json(&nested) {
                Some(json) => self.learn_json_in(&json, JsonContext::Value, SecretSource::Text),
                None => self.learn_text(&nested),
            }
        }
        for range in url_spans(text) {
            self.learn_url_from(text.get(range).unwrap_or_default(), SecretSource::Text);
        }
        for range in secret_values(text, self.purpose) {
            if let Some(raw) = text.get(range) {
                if let Ok(value) = serde_json::from_str::<Value>(raw) {
                    self.learn_json_in(&value, JsonContext::Credential, SecretSource::Text);
                } else {
                    self.learn(raw.trim_matches('"'), SecretSource::Text);
                }
            }
        }
    }

    fn learn_json(&mut self, value: &Value) {
        self.learn_json_in(value, JsonContext::Value, SecretSource::Wire);
    }

    fn learn_json_in(&mut self, value: &Value, context: JsonContext, source: SecretSource) {
        match value {
            Value::String(text)
                if matches!(
                    context,
                    JsonContext::Credential | JsonContext::ApplicationCredential
                ) =>
            {
                let source = if source == SecretSource::Text {
                    source
                } else if context == JsonContext::ApplicationCredential {
                    SecretSource::Field
                } else if source == SecretSource::Authorization {
                    source
                } else {
                    SecretSource::Wire
                };
                self.learn_credential(text, source);
            }
            Value::Number(number)
                if matches!(
                    context,
                    JsonContext::Credential | JsonContext::ApplicationCredential
                ) =>
            {
                let source = if source == SecretSource::Text {
                    source
                } else if context == JsonContext::ApplicationCredential {
                    SecretSource::SoftNumber
                } else {
                    SecretSource::Number
                };
                self.learn(&number.to_string(), source);
            }
            Value::String(text) => match canonical_json(text) {
                Some(json) if json.is_object() || json.is_array() => {
                    self.learn_json_in(&json, context, source);
                }
                _ => self.learn_text(text),
            },
            Value::Array(items) => {
                for item in items {
                    self.learn_json_in(item, context, source);
                }
            }
            Value::Object(fields) => {
                let context = context.object(fields);
                for (name, field) in fields {
                    for range in url_spans(name) {
                        self.learn_url_from(
                            name.get(range).unwrap_or_default(),
                            SecretSource::Wire,
                        );
                    }
                    let field_source = if source != SecretSource::Text
                        && field.is_string()
                        && is_auth_name(name)
                    {
                        SecretSource::Authorization
                    } else {
                        source
                    };
                    self.learn_json_in(field, context.field(name, self.purpose), field_source);
                }
            }
            _ => {}
        }
    }

    fn learn_credential(&mut self, value: &str, source: SecretSource) {
        self.learn(value, source);
        if source == SecretSource::Authorization
            && let Some((scheme, credential)) = value.split_once(char::is_whitespace)
            && !scheme.is_empty()
            && scheme
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
        {
            self.learn(credential.trim(), source);
        }
    }

    fn learn(&mut self, secret: &str, source: SecretSource) {
        if secret.len() < source.minimum()
            || secret == REDACTED
            || decode_form_component(secret)
                .split_whitespace()
                .any(|part| part == REDACTED || is_redacted_placeholder(part))
        {
            return;
        }
        if let Some(known) = self.learned.iter_mut().find(|known| known.value == secret) {
            known.source = match (known.source, source) {
                (SecretSource::SoftNumber, SecretSource::Wire | SecretSource::Authorization)
                | (SecretSource::Wire | SecretSource::Authorization, SecretSource::SoftNumber) => {
                    SecretSource::Number
                }
                _ => known.source.max(source),
            };
            return;
        }
        self.learned.push(LearnedSecret {
            value: secret.to_owned(),
            source,
        });
        self.learned
            .sort_by_key(|known| std::cmp::Reverse(known.value.len()));
    }

    /// `url` with its credentials scrubbed.
    pub(crate) fn scrub_url(&mut self, url: &str) -> String {
        self.scrub_text(&mask_url(url))
    }

    /// `body` scrubbed as JSON (written canonically), SSE, or text.
    #[cfg(any(feature = "http", test))]
    pub(crate) fn scrub_body(&mut self, body: &str) -> String {
        self.learn_body(body);
        self.render_body(body)
    }

    fn render_body(&mut self, body: &str) -> String {
        let classified = if self.purpose == Purpose::Traffic {
            serde_json::from_str(body)
                .map(TextBody::Json)
                .unwrap_or_else(|_| classify_body(body))
        } else {
            classify_body(body)
        };
        match classified {
            TextBody::Json(mut json) => {
                self.scrub_json_in(None, &mut json, JsonContext::Value);
                json.to_string()
            }
            TextBody::Sse(events) => self.scrub_sse_body(events),
            TextBody::Form => self.scrub_text(&mask_form(body)),
            TextBody::Text => self.scrub_text(body),
        }
    }

    fn scrub_sse_events<'a>(&mut self, events: Vec<SseEvent<'a>>) -> Vec<ScrubbedSseEvent<'a>> {
        events
            .into_iter()
            .map(|source| {
                let event = source.event.as_deref().map(|text| self.scrub_text(text));
                let id = source.id.as_deref().map(|text| self.scrub_text(text));
                let data = source.data.as_deref().map(|text| self.scrub_data(text));
                let lines = source
                    .lines
                    .iter()
                    .map(|line| {
                        if line.field == SseField::Data {
                            String::new()
                        } else {
                            self.scrub_text(line.text)
                        }
                    })
                    .collect();
                ScrubbedSseEvent {
                    source,
                    event,
                    id,
                    data,
                    lines,
                }
            })
            .collect()
    }

    fn scrub_data(&mut self, text: &str) -> Value {
        match serde_json::from_str::<Value>(text) {
            Ok(mut json) => {
                if self.purpose == Purpose::Cassette {
                    json = rig_core::json_utils::canonical(json);
                }
                self.scrub_json_in(None, &mut json, JsonContext::Value);
                json
            }
            Err(_) => Value::String(self.render_body(text)),
        }
    }

    fn scrub_sse_body(&mut self, events: Vec<SseEvent<'_>>) -> String {
        let mut output = String::new();
        for event in self.scrub_sse_events(events) {
            let data_lines = event
                .source
                .lines
                .iter()
                .filter(|line| line.field == SseField::Data)
                .count();
            let original = event.source.data.as_deref().and_then(canonical_json);
            let changed = match original.as_ref() {
                Some(original) => event.data.as_ref() != Some(original),
                None => event.data.as_ref().and_then(Value::as_str) != event.source.data.as_deref(),
            };
            let mut wrote_data = false;
            for (index, line) in event.source.lines.iter().enumerate() {
                output.push_str(line.prefix);
                match line.field {
                    SseField::Data if event.data.is_some() => {
                        if !changed && (data_lines > 1 || original.is_none()) {
                            output.push_str(line.text);
                        } else if !wrote_data {
                            let text = event
                                .data
                                .as_ref()
                                .map(|data| {
                                    if original.is_some() {
                                        data.to_string()
                                    } else {
                                        data_text(data)
                                    }
                                })
                                .unwrap_or_default();
                            for (index, data) in text.split('\n').enumerate() {
                                if index > 0 {
                                    output.push_str(if line.ending.is_empty() {
                                        "\n"
                                    } else {
                                        line.ending
                                    });
                                }
                                output.push_str("data: ");
                                output.push_str(data);
                            }
                            wrote_data = true;
                        } else {
                            continue;
                        }
                    }
                    _ => output.push_str(
                        event
                            .lines
                            .get(index)
                            .map(String::as_str)
                            .unwrap_or(line.text),
                    ),
                }
                output.push_str(line.ending);
            }
        }
        output
    }

    /// Scrub inspected stream events; offsets refer to the original bytes.
    pub(crate) fn scrub_stream(
        &mut self,
        body: PreparedBody,
        chunk_ends: impl IntoIterator<Item = usize>,
    ) -> Vec<ScrubbedEvent> {
        if body.opaque().is_some() || body.is_binary() {
            return vec![ScrubbedEvent {
                end: body.metadata.bytes,
                event: None,
                id: None,
                data: body.metadata.value(),
            }];
        }
        if matches!(body.format, BodyFormat::Multipart(_)) {
            let end = body.metadata.bytes;
            let bytes = self.scrub_prepared(body);
            return vec![ScrubbedEvent {
                end,
                event: None,
                id: None,
                data: std::str::from_utf8(&bytes).map_or_else(
                    |_| serde_json::json!({"bytes":bytes.len()}),
                    |text| Value::String(text.to_owned()),
                ),
            }];
        }
        let Some((text, tail)) = text_prefix(&body.bytes) else {
            return vec![ScrubbedEvent {
                end: body.bytes.len(),
                event: None,
                id: None,
                data: serde_json::json!({"bytes": body.bytes.len()}),
            }];
        };
        let mut out = if let Some(events) =
            parse_sse_chunks(text, matches!(body.format, BodyFormat::Sse), chunk_ends)
        {
            self.scrub_sse_events(events)
                .into_iter()
                .filter_map(|event| {
                    Some(ScrubbedEvent {
                        end: event.source.end,
                        event: event.event,
                        id: event.id,
                        data: event.data?,
                    })
                })
                .collect::<Vec<_>>()
        } else {
            let mut end = 0;
            text.split_inclusive('\n')
                .filter_map(|line| {
                    end += line.len();
                    (!line.trim().is_empty()).then(|| ScrubbedEvent {
                        end,
                        event: None,
                        id: None,
                        data: self.scrub_data(line.strip_suffix('\n').unwrap_or(line)),
                    })
                })
                .collect()
        };
        if tail > 0 {
            out.push(ScrubbedEvent {
                end: body.bytes.len(),
                event: None,
                id: None,
                data: serde_json::json!({"bytes": tail}),
            });
        }
        out
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn scrub_json_value(&mut self, key: Option<&str>, value: &mut Value) {
        let context = if let Some(key) = key.filter(|key| is_secret_field(key, self.purpose)) {
            credential_context(key)
        } else {
            JsonContext::Value
        };
        self.learn_json_in(value, context, SecretSource::Wire);
        self.scrub_json_in(key, value, context);
    }

    fn scrub_json_in(&mut self, key: Option<&str>, value: &mut Value, context: JsonContext) {
        let key_lower = key.map(str::to_ascii_lowercase);
        let cassette = self.purpose == Purpose::Cassette;
        match value {
            Value::Object(map) => {
                let context = context.object(map);
                for (name, mut field) in std::mem::take(map) {
                    self.scrub_json_in(Some(&name), &mut field, context.field(&name, self.purpose));
                    let mut name = self.scrub_shapes_and_paths(&mask_urls(&name));
                    for secret in &self.learned {
                        if secret.source != SecretSource::Text {
                            name = replace_secret(&name, &secret.value);
                        }
                    }
                    let mut unique = name.clone();
                    let mut n = 1;
                    while map.contains_key(&unique) {
                        n += 1;
                        unique = format!("{name}#{n}");
                    }
                    map.insert(unique, field);
                }
            }
            Value::Array(values) => {
                for value in values {
                    self.scrub_json_in(key, value, context);
                }
            }
            Value::String(text) => {
                if is_redacted_placeholder(text) {
                    return;
                }
                if matches!(
                    context,
                    JsonContext::Credential | JsonContext::ApplicationCredential
                ) {
                    *text = if cassette
                        && key_lower
                            .as_deref()
                            .is_some_and(|key| OAUTH_TOKEN_KEYS.contains(&key))
                    {
                        self.placeholder(text, "id_")
                    } else {
                        REDACTED.to_owned()
                    };
                } else if cassette
                    && key_lower
                        .as_deref()
                        .is_some_and(|key| VOLATILE_JSON_KEYS.contains(&key))
                {
                    *text = "1970-01-01T00:00:00Z".to_owned();
                } else if let Some(mut json) =
                    canonical_json(text).filter(|json| json.is_object() || json.is_array())
                {
                    let original = json.clone();
                    self.scrub_json_in(None, &mut json, context);
                    if json != original {
                        *text = json.to_string();
                    }
                } else {
                    *text = self.scrub_text(text);
                }
            }
            Value::Number(number) => {
                if matches!(
                    context,
                    JsonContext::Credential | JsonContext::ApplicationCredential
                ) || self.learned.iter().any(|secret| {
                    matches!(
                        secret.source,
                        SecretSource::Wire
                            | SecretSource::Authorization
                            | SecretSource::Number
                            | SecretSource::SoftNumber
                    ) && secret.value == number.to_string()
                }) {
                    *value = Value::String(REDACTED.to_owned());
                } else if cassette
                    && key_lower
                        .as_deref()
                        .is_some_and(|key| VOLATILE_JSON_KEYS.contains(&key))
                {
                    *value = Value::Number(0.into());
                }
            }
            Value::Bool(_) | Value::Null => {}
        }
    }

    pub(crate) fn scrub_text(&mut self, text: &str) -> String {
        let mut nested_output = String::new();
        let mut kept = 0;
        for (range, nested, closed) in embedded_strings(text) {
            let rendered = self.render_body(&nested);
            let same_json = rendered != nested
                && canonical_json(&nested)
                    .zip(canonical_json(&rendered))
                    .is_some_and(|(before, after)| before == after);
            if rendered != nested && !same_json {
                nested_output.push_str(text.get(kept..range.start).unwrap_or_default());
                if let Ok(mut encoded) = serde_json::to_string(&rendered) {
                    if !closed {
                        encoded.pop();
                    }
                    nested_output.push_str(&encoded);
                }
                kept = range.end;
            }
        }
        nested_output.push_str(text.get(kept..).unwrap_or_default());
        let mut scrubbed = mask_urls(&nested_output);
        for secret in &self.learned {
            if scrubbed.contains(secret.value.as_str()) {
                scrubbed = replace_secret(&scrubbed, &secret.value);
            }
        }
        scrubbed = scrub_query_params_in_text(&scrubbed, self.purpose);
        let mut kept = 0;
        let mut masked = String::new();
        for range in secret_values(&scrubbed, self.purpose) {
            masked.push_str(scrubbed.get(kept..range.start).unwrap_or_default());
            masked.push('"');
            masked.push_str(REDACTED);
            masked.push('"');
            kept = range.end;
        }
        if kept != 0 {
            masked.push_str(scrubbed.get(kept..).unwrap_or_default());
            scrubbed = masked;
        }
        self.scrub_shapes_and_paths(&scrubbed)
    }

    fn scrub_shapes_and_paths(&mut self, text: &str) -> String {
        let scrubbed = redact::text(text);
        match self.purpose {
            Purpose::Cassette => {
                let scrubbed = self.scrub_aws_account_ids(&scrubbed);
                scrub_local_filesystem_paths(&scrubbed)
            }
            Purpose::Traffic => scrubbed.into_owned(),
        }
    }

    /// Replace 12-digit account-ID segments in ARNs, preserving partition,
    /// service, region, and resource fields.
    fn scrub_aws_account_ids(&mut self, text: &str) -> String {
        const PREFIX: &str = "arn:";
        const ACCOUNT_FIELD: usize = 4;
        const ACCOUNT_LEN: usize = 12;

        let mut output = String::with_capacity(text.len());
        let mut rest = text;

        while let Some(start) = rest.find(PREFIX) {
            let (before, arn) = rest.split_at(start);
            output.push_str(before);
            // An ARN ends at the first character that cannot appear in one;
            // the surrounding JSON quote or comma is the usual terminator.
            let end = arn
                .find(['"', ',', ' ', '\n', '}', ']'])
                .unwrap_or(arn.len());
            let (arn, tail) = arn.split_at(end);

            let mut fields = arn.split(':').map(str::to_string).collect::<Vec<_>>();
            match fields.get(ACCOUNT_FIELD).cloned() {
                Some(account)
                    if account.len() == ACCOUNT_LEN
                        && account.chars().all(|ch| ch.is_ascii_digit()) =>
                {
                    let placeholder = self.placeholder(&account, "account_");
                    if let Some(field) = fields.get_mut(ACCOUNT_FIELD) {
                        *field = placeholder;
                    }
                    output.push_str(&fields.join(":"));
                }
                _ => output.push_str(arn),
            }

            rest = tail;
        }

        output.push_str(rest);
        output
    }

    pub(crate) fn placeholder(&mut self, original: &str, kind: &'static str) -> String {
        // Empty wire values must stay empty rather than acquire invented identifiers.
        if original.is_empty() {
            return String::new();
        }

        if let Some(existing) = self.placeholders.get(original) {
            return existing.clone();
        }

        let counter = self.counters.entry(kind).or_insert(0);
        *counter += 1;
        let placeholder = format!("{kind}REDACTED_{counter}");
        self.placeholders
            .insert(original.to_string(), placeholder.clone());
        placeholder
    }
}

/// Whether a header named `name` carries a credential, or names the
/// account one belongs to.
pub(crate) fn is_secret_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    contains_case_insensitive(SENSITIVE_HEADER_NAMES, name)
        || is_explicit_secret_field(name)
        || lower == "secret"
        || is_auth_name(name)
        || ["-key", "-token", "-secret", "-signature"]
            .iter()
            .any(|suffix| lower.ends_with(suffix))
}

fn is_auth_name(name: &str) -> bool {
    name.to_ascii_lowercase()
        .split(['-', '_'])
        .any(|component| matches!(component, "auth" | "authorization" | "authentication"))
}

pub(crate) fn contains_case_insensitive(values: &[&str], needle: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(needle))
}

pub(crate) fn canonical_json(body: &str) -> Option<Value> {
    serde_json::from_str::<Value>(body)
        .ok()
        .map(rig_core::json_utils::canonical)
}

pub(crate) fn is_redacted_placeholder(value: &str) -> bool {
    let Some((kind, counter)) = value.split_once("REDACTED_") else {
        return false;
    };

    !kind.is_empty()
        && !counter.is_empty()
        && kind
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        && counter.chars().all(|ch| ch.is_ascii_digit())
}

fn scrub_query_params_in_text(text: &str, purpose: Purpose) -> String {
    let mut out = String::new();
    let mut kept = 0;
    for (equal, _) in text.match_indices('=') {
        if equal < kept {
            continue;
        }
        let before = text.get(..equal).unwrap_or_default();
        let start = before
            .char_indices()
            .rev()
            .find(|(_, c)| !c.is_ascii_alphanumeric() && !matches!(c, '-' | '_' | '%'))
            .map_or(0, |(at, c)| at + c.len_utf8());
        let preceding = text.get(..start).and_then(|text| text.chars().next_back());
        let boundary = start == 0
            || preceding.is_some_and(|c| matches!(c, ' ' | '?' | '#' | '&' | '\'' | '"'));
        let name = decode_form_component(text.get(start..equal).unwrap_or_default());
        if !boundary || !is_secret_param(&name) {
            continue;
        }
        // Bare token assignments in cassette prose remain application data, as in JSON.
        if purpose == Purpose::Cassette
            && name.eq_ignore_ascii_case("token")
            && !preceding.is_some_and(|c| matches!(c, '?' | '#' | '&'))
        {
            continue;
        }
        let value = text.get(equal + 1..).unwrap_or_default();
        let (value_start, end) = match value.chars().next() {
            Some(quote @ ('"' | '\'')) => {
                let stop = string_end(text, equal + 1);
                let closed = text
                    .get(stop.saturating_sub(1)..stop)
                    .is_some_and(|last| last.starts_with(quote));
                (equal + 2, if closed { stop - 1 } else { stop })
            }
            _ => (
                equal + 1,
                equal
                    + 1
                    + value
                        .find(['&', '#', '"', '\'', ' ', '\n', '\r', '<'])
                        .unwrap_or(value.len()),
            ),
        };
        out.push_str(text.get(kept..value_start).unwrap_or_default());
        out.push_str(REDACTED);
        kept = end;
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    out
}

/// Redact directories of home paths beginning at the input start or after a
/// double quote, preserving the final component. Embedded path segments are
/// unchanged to avoid corrupting URLs.
pub(crate) fn scrub_local_filesystem_paths(text: &str) -> String {
    const HOME_PREFIXES: &[&str] = &["/Users/", "/home/", "/root/"];

    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    let mut consumed = 0usize;

    while let Some((prefix, at)) = HOME_PREFIXES
        .iter()
        .filter_map(|p| rest.find(*p).map(|i| (*p, i)))
        .min_by_key(|(_, i)| *i)
    {
        // Anchored: the value must begin here. Anything else is a path segment
        // inside a larger string (a URL, prose) and is not ours to rewrite.
        let absolute = consumed + at;
        let starts_value = absolute == 0 || text.as_bytes().get(absolute - 1) == Some(&b'"');
        let (before, path) = rest.split_at(at);
        if !starts_value {
            let (segment, after) = path.split_at(prefix.len());
            output.push_str(before);
            output.push_str(segment);
            rest = after;
            consumed = absolute + prefix.len();
            continue;
        }

        output.push_str(before);
        // Spaces may belong to account names; treating them as terminators would
        // leak the remainder of the home path.
        let end = path.find(['"', '\\', '\n', '\r']).unwrap_or(path.len());
        let (path, tail) = path.split_at(end);

        match path.rsplit_once('/') {
            Some((_, base)) if !base.is_empty() => {
                output.push_str("/REDACTED_PATH/");
                output.push_str(base);
            }
            _ => output.push_str("/REDACTED_PATH"),
        }
        consumed = absolute + path.len();
        rest = tail;
        if rest.is_empty() {
            break;
        }
    }

    output.push_str(rest);
    output
}

fn short_secret_spans<'a>(
    bytes: &'a [u8],
    secret: &'a str,
) -> impl Iterator<Item = Range<usize>> + 'a {
    bytes
        .windows(secret.len().max(1))
        .enumerate()
        .filter_map(move |(at, candidate)| {
            if candidate != secret.as_bytes() {
                return None;
            }
            let end = at + secret.len();
            // Decode only the adjacent characters; unrelated opaque bytes do not affect a word boundary.
            let before = (at.saturating_sub(4)..at).find_map(|start| {
                std::str::from_utf8(bytes.get(start..at)?)
                    .ok()?
                    .chars()
                    .next_back()
            });
            let after = (1..=4).find_map(|len| {
                std::str::from_utf8(bytes.get(end..end.saturating_add(len).min(bytes.len()))?)
                    .ok()?
                    .chars()
                    .next()
            });
            (!before.is_some_and(char::is_alphanumeric)
                && !after.is_some_and(char::is_alphanumeric))
            .then_some(at..end)
        })
}

fn secret_occurs(bytes: &[u8], secret: &str) -> bool {
    if secret.len() >= PROSE_MINIMUM {
        find_bytes(bytes, secret.as_bytes()).is_some()
    } else {
        short_secret_spans(bytes, secret).next().is_some()
    }
}

fn replace_secret(text: &str, secret: &str) -> String {
    if secret.len() >= PROSE_MINIMUM {
        return text.replace(secret, REDACTED);
    }
    let mut out = String::new();
    let mut kept = 0;
    for range in short_secret_spans(text.as_bytes(), secret) {
        if range.start < kept {
            continue;
        }
        out.push_str(text.get(kept..range.start).unwrap_or_default());
        out.push_str(REDACTED);
        kept = range.end;
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    out
}

fn is_cookie(name: &str) -> bool {
    name.eq_ignore_ascii_case("cookie") || name.eq_ignore_ascii_case("set-cookie")
}

fn credential_context(name: &str) -> JsonContext {
    if name.eq_ignore_ascii_case("token") || is_cookie(name) {
        JsonContext::ApplicationCredential
    } else {
        JsonContext::Credential
    }
}

fn string_end(text: &str, open: usize) -> usize {
    let quote = text
        .get(open..)
        .and_then(|text| text.chars().next())
        .unwrap_or('"');
    let mut escaped = false;
    text.get(open + 1..)
        .and_then(|rest| {
            rest.char_indices().find(|&(_, c)| {
                let end = c == quote && !escaped;
                escaped = c == '\\' && !escaped;
                end
            })
        })
        .map_or(text.len(), |(at, _)| open + 1 + at + 1)
}

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

fn embedded_strings(text: &str) -> Vec<(Range<usize>, String, bool)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(open) = text.get(at..).and_then(|rest| rest.find('"')) {
        let open = at + open;
        let end = string_end(text, open);
        at = end;
        let raw = text.get(open..end).unwrap_or_default();
        if !raw.contains("\\\"") {
            continue;
        }
        let closed = serde_json::from_str::<String>(raw).ok();
        let is_closed = closed.is_some();
        let decoded = closed.or_else(|| {
            let mut repaired = raw.to_owned();
            let slashes = repaired
                .bytes()
                .rev()
                .take_while(|byte| *byte == b'\\')
                .count();
            if slashes % 2 == 1 {
                repaired.pop();
            }
            repaired.push('"');
            serde_json::from_str::<String>(&repaired).ok()
        });
        if let Some(decoded) = decoded {
            out.push((open..end, decoded, is_closed));
        }
    }
    out
}

fn secret_values(text: &str, purpose: Purpose) -> Vec<Range<usize>> {
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
            .filter(|_| is_secret_field(&key, purpose))
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

fn is_secret_field(name: &str, purpose: Purpose) -> bool {
    if name.eq_ignore_ascii_case("token") {
        return purpose == Purpose::Traffic;
    }
    !name.eq_ignore_ascii_case("key") && is_secret_header(name)
}

fn is_explicit_secret_field(name: &str) -> bool {
    let normalized: String = name
        .chars()
        .filter(|c| !matches!(c, '_' | '-'))
        .map(|c| c.to_ascii_lowercase())
        .collect();
    matches!(
        normalized.as_str(),
        "token"
            | "apikey"
            | "apitoken"
            | "accesstoken"
            | "authtoken"
            | "authorizationcode"
            | "refreshtoken"
            | "idtoken"
            | "sessiontoken"
            | "clientsecret"
            | "secretkey"
            | "privatekey"
            | "subscriptionkey"
            | "password"
            | "devicecode"
            | "deviceauthid"
            | "usercode"
            | "codeverifier"
            | "authorizationtoken"
            | "fallbackcredittoken"
            | "xamzcredential"
    )
}

pub(crate) fn is_secret_param(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    is_secret_header(name) || matches!(lower.as_str(), "signature" | "sig" | "code" | "assertion")
}

fn is_schema(fields: &Map<String, Value>) -> bool {
    fields.get("$schema").is_some_and(Value::is_string)
        || fields.get("$ref").is_some_and(Value::is_string)
        || (fields.get("type").and_then(Value::as_str) == Some("object")
            && fields.get("properties").is_some_and(Value::is_object))
}

fn is_logprobs(fields: &Map<String, Value>) -> bool {
    fields.iter().any(|(name, value)| {
        value.is_number()
            && matches!(
                name.to_ascii_lowercase().as_str(),
                "logprob" | "logprobability"
            )
    })
}

fn decode_form_component(text: &str) -> String {
    url::form_urlencoded::parse(format!("value={text}").as_bytes())
        .next()
        .map(|(_, value)| value.into_owned())
        .unwrap_or_default()
}

fn is_form(text: &str) -> bool {
    text.contains('=') && !text.trim_end().contains(char::is_whitespace)
}

fn mask_form(form: &str) -> String {
    form.split('&')
        .map(|pair| match pair.split_once('=') {
            Some((name, _)) if is_secret_param(&decode_form_component(name)) => {
                format!("{name}={REDACTED}")
            }
            _ => pair.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn mask_url(uri: &str) -> String {
    let mut uri = uri.to_owned();
    if let Some((scheme, rest)) = uri.split_once("://") {
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        if let Some((_, host)) = rest
            .get(..end)
            .and_then(|authority| authority.rsplit_once('@'))
        {
            uri = format!(
                "{scheme}://{REDACTED}@{host}{}",
                rest.get(end..).unwrap_or_default()
            );
        }
    }
    let (head, fragment) = uri
        .split_once('#')
        .map_or((uri.as_str(), None), |(head, fragment)| {
            (head, Some(fragment))
        });
    let mut out = match head.split_once('?') {
        Some((path, query)) => format!("{path}?{}", mask_form(query)),
        None => head.to_owned(),
    };
    if let Some(fragment) = fragment {
        out.push('#');
        out.push_str(&mask_form(fragment));
    }
    out
}

fn url_spans(text: &str) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut after = 0;
    for (at, _) in text.match_indices("://") {
        let before = text.get(..at).unwrap_or_default();
        let start = before
            .char_indices()
            .rev()
            .find(|(_, c)| !c.is_ascii_alphanumeric() && !matches!(c, '+' | '-' | '.'))
            .map_or(0, |(at, c)| at + c.len_utf8());
        if !text
            .get(start..at)
            .and_then(|scheme| scheme.chars().next())
            .is_some_and(|first| first.is_ascii_alphabetic())
        {
            continue;
        }
        if start < after {
            continue;
        }
        let rest = text.get(start..).unwrap_or_default();
        let mut len = 0;
        while let Some(tail) = rest.get(len..).filter(|tail| !tail.is_empty()) {
            if tail.starts_with(REDACTED) {
                len += REDACTED.len();
                continue;
            }
            let Some(c) = tail.chars().next() else {
                break;
            };
            if c.is_whitespace() || "\"'<>()[]{}`".contains(c) {
                break;
            }
            len += c.len_utf8();
        }
        spans.push(start..start + len);
        after = start + len;
    }
    spans
}

fn mask_urls(text: &str) -> String {
    let mut out = String::new();
    let mut kept = 0;
    for range in url_spans(text) {
        out.push_str(text.get(kept..range.start).unwrap_or_default());
        out.push_str(&mask_url(text.get(range.clone()).unwrap_or_default()));
        kept = range.end;
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    out
}

enum TextBody<'a> {
    Json(Value),
    Sse(Vec<SseEvent<'a>>),
    Form,
    Text,
}

fn classify_body(text: &str) -> TextBody<'_> {
    if let Some(json) = canonical_json(text) {
        TextBody::Json(json)
    } else if let Some(events) = parse_sse(text, false) {
        TextBody::Sse(events)
    } else if is_form(text) {
        TextBody::Form
    } else {
        TextBody::Text
    }
}

pub(crate) struct ScrubbedEvent {
    pub(crate) end: usize,
    pub(crate) event: Option<String>,
    pub(crate) id: Option<String>,
    pub(crate) data: Value,
}

struct ScrubbedSseEvent<'a> {
    source: SseEvent<'a>,
    event: Option<String>,
    id: Option<String>,
    data: Option<Value>,
    lines: Vec<String>,
}

fn data_text(data: &Value) -> String {
    match data {
        Value::String(text) => text.clone(),
        _ => data.to_string(),
    }
}

#[cfg(feature = "bedrock")]
fn event_stream_messages(bytes: &[u8]) -> Option<Vec<EventStreamMessage>> {
    let mut input = Bytes::copy_from_slice(bytes);
    let mut messages = Vec::new();
    while !input.is_empty() {
        messages.push(read_message_from(&mut input).ok()?);
    }
    Some(messages)
}

/// UTF-8 header pairs used to prepare an exchange for either sink.
pub(crate) fn header_pairs(headers: &HeaderMap) -> impl Iterator<Item = (&str, &str)> {
    headers.iter().filter_map(|(name, value)| {
        value
            .to_str()
            .ok()
            .or_else(|| {
                matches!(name.as_str(), "content-type" | "content-encoding")
                    .then_some("<non-UTF-8>")
            })
            .map(|value| (name.as_str(), value))
    })
}

/// Whether the shared SSE framer has no unfinished bytes after this body.
pub(crate) fn ends_between_events(body: &[u8]) -> bool {
    let mut framer = rig_core::http_client::framing::Framing::Sse.framer();
    let Some(ref mut framer) = framer else {
        return false;
    };
    framer.push(body).for_each(drop);
    framer.pending() == 0
}

pub(crate) fn is_sse(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("text/event-stream"))
}

/// An SSE event with its ending byte offset, metadata and joined data.
#[derive(Default)]
pub(crate) struct SseEvent<'a> {
    pub(crate) end: usize,
    pub(crate) event: Option<String>,
    pub(crate) id: Option<String>,
    pub(crate) data: Option<String>,
    lines: Vec<SseLine<'a>>,
}

struct SseLine<'a> {
    prefix: &'a str,
    text: &'a str,
    ending: &'a str,
    field: SseField,
}

impl SseEvent<'_> {
    fn parsed(&mut self, event: rig_core::http_client::framing::SseEvent) {
        let explicit_type = self.event.as_deref().is_some_and(|name| !name.is_empty());
        self.event = (explicit_type || event.event != "message").then(|| event.event.into_owned());
        self.id = (!event.id.is_empty() || self.id.is_some()).then_some(event.id);
        self.data = Some(event.data);
    }
}

/// Parse with the same grammar as provider framing, retaining a final tail.
pub(crate) fn parse_sse(text: &str, content_type_hint: bool) -> Option<Vec<SseEvent<'_>>> {
    parse_sse_chunks(text, content_type_hint, std::iter::once(text.len()))
}

pub(crate) fn parse_sse_chunks(
    text: &str,
    content_type_hint: bool,
    chunk_ends: impl IntoIterator<Item = usize>,
) -> Option<Vec<SseEvent<'_>>> {
    let mut framer = SseFramer::new();
    let mut lines = Vec::new();
    let mut parsed = BTreeMap::new();
    let mut start = 0;
    for end in chunk_ends {
        let end = end.min(text.len());
        if end < start {
            continue;
        }
        let batch = framer.push_recording(text.as_bytes().get(start..end).unwrap_or_default());
        lines.extend(batch.lines);
        parsed.extend(
            batch
                .events
                .into_iter()
                .map(|frame| (frame.end, frame.event)),
        );
        start = end;
    }
    if start < text.len() {
        let batch = framer.push_recording(text.as_bytes().get(start..).unwrap_or_default());
        lines.extend(batch.lines);
        parsed.extend(
            batch
                .events
                .into_iter()
                .map(|frame| (frame.end, frame.event)),
        );
    }
    let batch = framer.finish_recording();
    lines.extend(batch.lines);
    parsed.extend(
        batch
            .events
            .into_iter()
            .map(|frame| (frame.end, frame.event)),
    );
    if !content_type_hint
        && !lines
            .iter()
            .any(|line| line.field == SseField::Data && line.value.is_some())
    {
        return None;
    }
    let mut events = Vec::new();
    let mut event = SseEvent::default();
    let mut kept = 0;
    for line in lines {
        let value = line
            .value
            .as_ref()
            .and_then(|range| text.get(range.clone()))
            .unwrap_or_default();
        match line.field {
            SseField::Event => event.event = Some(value.to_owned()),
            SseField::Id => event.id = Some(value.to_owned()),
            _ => {}
        }
        event.lines.push(SseLine {
            prefix: text.get(kept..line.content.start).unwrap_or_default(),
            text: text.get(line.content.clone()).unwrap_or_default(),
            ending: text
                .get(line.content.end..line.span.end)
                .unwrap_or_default(),
            field: line.field,
        });
        kept = line.span.end;
        event.end = kept;
        if line.field == SseField::Blank {
            if let Some(parsed) = parsed.remove(&event.end) {
                event.parsed(parsed);
            }
            events.push(std::mem::take(&mut event));
        }
    }
    if !event.lines.is_empty() {
        if let Some(parsed) = parsed.remove(&event.end) {
            event.parsed(parsed);
        }
        events.push(event);
    }
    Some(events)
}

/// Valid text and the number of bytes of an incomplete UTF-8 character at EOF.
pub(crate) fn text_prefix(bytes: &[u8]) -> Option<(&str, usize)> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Some((text, 0)),
        Err(error) if error.error_len().is_none() => {
            std::str::from_utf8(bytes.get(..error.valid_up_to())?)
                .ok()
                .map(|text| (text, bytes.len() - error.valid_up_to()))
        }
        Err(_) => None,
    }
}

fn header_value<'a>(headers: &[(&'a str, &'a str)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| *value)
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct MultipartPart {
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) body: Vec<u8>,
}

impl MultipartPart {
    fn name(&self) -> Option<&str> {
        self.headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("content-disposition"))?
            .1
            .split(';')
            .find_map(|field| {
                let (name, value) = field.trim().split_once('=')?;
                name.eq_ignore_ascii_case("name")
                    .then(|| value.trim_matches('"'))
            })
    }
}

pub(crate) fn part_metadata(part: &Part) -> Vec<(String, String)> {
    let mut metadata = vec![("name".to_owned(), part.name().to_owned())];
    if let Some(filename) = part.get_filename() {
        metadata.push(("filename".to_owned(), filename.to_owned()));
    }
    if let Some(content_type) = part.get_content_type() {
        metadata.push(("content-type".to_owned(), content_type.to_string()));
    }
    metadata
}

struct MultipartLayout {
    parts: Vec<MultipartSlice>,
    preamble: Range<usize>,
    epilogue: Range<usize>,
}

struct MultipartSlice {
    part: MultipartPart,
    header: Range<usize>,
    body: Range<usize>,
}

#[cfg(feature = "http")]
pub(crate) fn parse_multipart_parts(body: &[u8], boundary: &str) -> Option<Vec<MultipartPart>> {
    multipart_slices(body, boundary, false)
        .map(|layout| layout.parts.into_iter().map(|slice| slice.part).collect())
}

pub(crate) fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').find_map(|part| {
        let (name, value) = part.trim().split_once('=')?;
        name.eq_ignore_ascii_case("boundary")
            .then(|| value.trim_matches('"').to_string())
            .filter(|value| !value.is_empty())
    })
}

fn multipart_slices(body: &[u8], boundary: &str, strict: bool) -> Option<MultipartLayout> {
    if strict
        && (boundary.is_empty()
            || boundary.len() > 70
            || boundary.ends_with(' ')
            || !boundary
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"'()+_,-./:=? ".contains(&byte)))
    {
        return None;
    }
    let marker = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let slices = split_bytes(body, &marker, strict);
    let mut closed = false;
    let preamble = 0..slices.first().map_or(0, |(_, bytes)| bytes.len());
    let mut epilogue = body.len()..body.len();
    if strict
        && !body
            .get(preamble.clone())?
            .iter()
            .all(u8::is_ascii_whitespace)
    {
        return None;
    }
    for (mut offset, raw_part) in slices.into_iter().skip(1) {
        if strict && !raw_part.starts_with(b"--") && !raw_part.starts_with(b"\r\n") {
            return None;
        }
        let raw_part = if let Some(part) = raw_part.strip_prefix(b"\r\n") {
            offset += 2;
            part
        } else {
            raw_part
        };
        if let Some(after) = raw_part.strip_prefix(b"--") {
            if strict && !after.is_empty() && !after.starts_with(b"\r\n") {
                return None;
            }
            let ending = if after.starts_with(b"\r\n") { 2 } else { 0 };
            closed = true;
            epilogue = offset + 2 + ending..body.len();
            break;
        }

        if strict && !raw_part.ends_with(b"\r\n") {
            return None;
        }
        let raw_part = raw_part.strip_suffix(b"\r\n").unwrap_or(raw_part);
        if raw_part.iter().all(u8::is_ascii_whitespace) {
            continue;
        }

        let header_end = find_bytes(raw_part, b"\r\n\r\n")?;
        let raw_headers = raw_part.get(..header_end)?;
        let body_start = header_end + b"\r\n\r\n".len();
        let raw_body = raw_part.get(body_start..)?;
        let raw_headers = std::str::from_utf8(raw_headers).ok()?;
        if strict
            && !raw_headers.split("\r\n").all(|line| {
                line.split_once(':').is_some_and(|(name, value)| {
                    !name.is_empty()
                        && name.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
                        })
                        && !value.chars().any(|ch| ch.is_control() && ch != '\t')
                })
            })
        {
            return None;
        }
        let mut headers = raw_headers
            .lines()
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                Some((
                    name.trim().to_ascii_lowercase(),
                    normalize_multipart_header_value(value.trim()),
                ))
            })
            .collect::<Vec<_>>();
        headers.sort();

        if strict {
            let dispositions: Vec<_> = headers
                .iter()
                .filter(|(name, _)| name == "content-disposition")
                .collect();
            if dispositions.len() != 1
                || !dispositions.first().is_some_and(|(_, value)| {
                    value
                        .split(';')
                        .next()
                        .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("form-data"))
                })
            {
                return None;
            }
            if headers.iter().any(|(name, value)| {
                name == "content-transfer-encoding"
                    && !matches!(
                        value.to_ascii_lowercase().as_str(),
                        "binary" | "7bit" | "8bit" | "identity"
                    )
            }) {
                return None;
            }
        }
        let part = MultipartPart {
            headers,
            body: raw_body.to_vec(),
        };
        if strict && part.name().is_none() {
            return None;
        }
        parts.push(MultipartSlice {
            part,
            header: offset..offset + header_end,
            body: offset + body_start..offset + raw_part.len(),
        });
    }

    if strict
        && (!closed
            || !body
                .get(epilogue.clone())?
                .iter()
                .all(u8::is_ascii_whitespace))
    {
        return None;
    }
    Some(MultipartLayout {
        parts,
        preamble,
        epilogue,
    })
}

fn split_bytes<'a>(body: &'a [u8], marker: &[u8], strict: bool) -> Vec<(usize, &'a [u8])> {
    let mut parts = Vec::new();
    let mut kept = 0;
    for (at, candidate) in body.windows(marker.len().max(1)).enumerate() {
        if at < kept || candidate != marker {
            continue;
        }
        let suffix = body.get(at + marker.len()..).unwrap_or_default();
        if strict
            && ((at != 0 && body.get(at.saturating_sub(2)..at) != Some(b"\r\n".as_slice()))
                || !(suffix.starts_with(b"\r\n") || suffix.starts_with(b"--")))
        {
            continue;
        }
        if let Some(bytes) = body.get(kept..at) {
            parts.push((kept, bytes));
        }
        kept = at + marker.len();
    }
    if let Some(bytes) = body.get(kept..) {
        parts.push((kept, bytes));
    }
    parts
}

/// Where `needle`, which is not empty, first occurs in `body`.
pub(crate) fn find_bytes(body: &[u8], needle: &[u8]) -> Option<usize> {
    body.windows(needle.len())
        .position(|window| window == needle)
}

fn normalize_multipart_header_value(value: &str) -> String {
    value
        .split(';')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("; ")
}
