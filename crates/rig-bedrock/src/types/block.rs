//! The Converse content blocks a turn keeps as provider items, as the
//! Converse API's JSON states them: reasoning, cited text, and a hosted
//! tool's use and result. Each converts both ways, so a kept item goes back
//! to the model that sent it exactly as it came.
//!
//! A block holding a part this SDK version does not model has no JSON
//! form; its turn keeps the canonical block alone.

use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_smithy_types::Blob;
use base64::{Engine, prelude::BASE64_STANDARD};
use serde_json::{Map, Value, json};

use super::json;

/// The JSON of a block a turn keeps, or `None` for a block it does not
/// keep or that holds a part this SDK version does not model.
#[deny(clippy::wildcard_enum_match_arm)]
pub(crate) fn to_json(block: &aws_bedrock::ContentBlock) -> Option<Value> {
    use aws_bedrock::ContentBlock as Block;
    match block {
        Block::ReasoningContent(aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning)) => {
            Some(reasoning_json(
                &reasoning.text,
                reasoning.signature.as_deref(),
            ))
        }
        Block::ReasoningContent(aws_bedrock::ReasoningContentBlock::RedactedContent(blob)) => {
            Some(redacted_json(blob.as_ref()))
        }
        Block::CitationsContent(cited) => {
            let content = cited
                .content
                .iter()
                .flatten()
                .map(|part| match part {
                    aws_bedrock::CitationGeneratedContent::Text(text) => {
                        Some(json!({ "text": text }))
                    }
                    aws_bedrock::CitationGeneratedContent::Unknown { .. } | _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let citations = cited
                .citations
                .iter()
                .flatten()
                .map(citation_json)
                .collect::<Option<Vec<_>>>()?;
            let mut fields = Map::new();
            if cited.content.is_some() {
                fields.insert("content".to_owned(), Value::Array(content));
            }
            if cited.citations.is_some() {
                fields.insert("citations".to_owned(), Value::Array(citations));
            }
            Some(json!({ "citationsContent": fields }))
        }
        Block::ToolUse(call) => Some(tool_use_json(
            &call.tool_use_id,
            &call.name,
            json::to_value(call.input.clone()),
            call.r#type.as_ref().map(aws_bedrock::ToolUseType::as_str),
        )),
        Block::ToolResult(result) => {
            let content = result
                .content
                .iter()
                .map(|part| match part {
                    aws_bedrock::ToolResultContentBlock::Text(text) => {
                        Some(json!({ "text": text }))
                    }
                    aws_bedrock::ToolResultContentBlock::Json(value) => {
                        Some(json!({ "json": json::to_value(value.clone()) }))
                    }
                    aws_bedrock::ToolResultContentBlock::Document(_)
                    | aws_bedrock::ToolResultContentBlock::Image(_)
                    | aws_bedrock::ToolResultContentBlock::SearchResult(_)
                    | aws_bedrock::ToolResultContentBlock::Video(_)
                    | aws_bedrock::ToolResultContentBlock::Unknown { .. }
                    | _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(tool_result_json(
                &result.tool_use_id,
                content,
                result
                    .status
                    .as_ref()
                    .map(aws_bedrock::ToolResultStatus::as_str),
                result.r#type.as_deref(),
            ))
        }
        Block::Audio(_)
        | Block::CachePoint(_)
        | Block::Document(_)
        | Block::GuardContent(_)
        | Block::Image(_)
        | Block::ReasoningContent(_)
        | Block::SearchResult(_)
        | Block::Text(_)
        | Block::Video(_) => None,
        Block::Unknown { .. } | _ => None,
    }
}

/// `{"reasoningContent": {"reasoningText": ...}}`.
pub(crate) fn reasoning_json(text: &str, signature: Option<&str>) -> Value {
    let mut reasoning = Map::from_iter([("text".to_owned(), json!(text))]);
    if let Some(signature) = signature {
        reasoning.insert("signature".to_owned(), json!(signature));
    }
    json!({ "reasoningContent": { "reasoningText": reasoning } })
}

/// `{"reasoningContent": {"redactedContent": <base64>}}`.
pub(crate) fn redacted_json(bytes: &[u8]) -> Value {
    json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(bytes) } })
}

/// `{"toolUse": ...}`.
pub(crate) fn tool_use_json(id: &str, name: &str, input: Value, kind: Option<&str>) -> Value {
    let mut call = Map::from_iter([
        ("toolUseId".to_owned(), json!(id)),
        ("name".to_owned(), json!(name)),
        ("input".to_owned(), input),
    ]);
    if let Some(kind) = kind {
        call.insert("type".to_owned(), json!(kind));
    }
    json!({ "toolUse": call })
}

/// `{"toolResult": ...}`.
pub(crate) fn tool_result_json(
    id: &str,
    content: Vec<Value>,
    status: Option<&str>,
    kind: Option<&str>,
) -> Value {
    let mut result = Map::from_iter([
        ("toolUseId".to_owned(), json!(id)),
        ("content".to_owned(), Value::Array(content)),
    ]);
    if let Some(status) = status {
        result.insert("status".to_owned(), json!(status));
    }
    if let Some(kind) = kind {
        result.insert("type".to_owned(), json!(kind));
    }
    json!({ "toolResult": result })
}

/// One citation of a whole cited block.
fn citation_json(citation: &aws_bedrock::Citation) -> Option<Value> {
    let source = citation.source_content.as_ref().map(|parts| {
        parts
            .iter()
            .map(|part| match part {
                aws_bedrock::CitationSourceContent::Text(text) => Some(json!({ "text": text })),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
    });
    let source = present(source)?;
    citation_fields(
        citation.title.as_deref(),
        citation.source.as_deref(),
        source,
        citation.location.as_ref(),
    )
}

/// One citation a stream delivers, in the form a whole block lists it.
pub(crate) fn citation_delta_json(citation: &aws_bedrock::CitationsDelta) -> Option<Value> {
    let source = citation.source_content.as_ref().map(|parts| {
        parts
            .iter()
            .map(|part| match &part.text {
                Some(text) => json!({ "text": text }),
                None => json!({}),
            })
            .collect()
    });
    citation_fields(
        citation.title.as_deref(),
        citation.source.as_deref(),
        source,
        citation.location.as_ref(),
    )
}

fn citation_fields(
    title: Option<&str>,
    source: Option<&str>,
    source_content: Option<Vec<Value>>,
    location: Option<&aws_bedrock::CitationLocation>,
) -> Option<Value> {
    let mut fields = Map::new();
    if let Some(title) = title {
        fields.insert("title".to_owned(), json!(title));
    }
    if let Some(source) = source {
        fields.insert("source".to_owned(), json!(source));
    }
    if let Some(parts) = source_content {
        fields.insert("sourceContent".to_owned(), Value::Array(parts));
    }
    if let Some(location) = location {
        fields.insert("location".to_owned(), location_json(location)?);
    }
    Some(Value::Object(fields))
}

#[deny(clippy::wildcard_enum_match_arm)]
fn location_json(location: &aws_bedrock::CitationLocation) -> Option<Value> {
    use aws_bedrock::CitationLocation as Location;
    let span = |index_key: &str, index: Option<i32>, start: Option<i32>, end: Option<i32>| {
        let mut fields = Map::new();
        for (key, value) in [(index_key, index), ("start", start), ("end", end)] {
            if let Some(value) = value {
                fields.insert(key.to_owned(), json!(value));
            }
        }
        Value::Object(fields)
    };
    let (kind, fields) = match location {
        Location::DocumentChar(at) => (
            "documentChar",
            span("documentIndex", at.document_index, at.start, at.end),
        ),
        Location::DocumentChunk(at) => (
            "documentChunk",
            span("documentIndex", at.document_index, at.start, at.end),
        ),
        Location::DocumentPage(at) => (
            "documentPage",
            span("documentIndex", at.document_index, at.start, at.end),
        ),
        Location::SearchResultLocation(at) => (
            "searchResultLocation",
            span(
                "searchResultIndex",
                at.search_result_index,
                at.start,
                at.end,
            ),
        ),
        Location::Web(web) => {
            let mut fields = Map::new();
            if let Some(url) = &web.url {
                fields.insert("url".to_owned(), json!(url));
            }
            if let Some(domain) = &web.domain {
                fields.insert("domain".to_owned(), json!(domain));
            }
            ("web", Value::Object(fields))
        }
        Location::Unknown { .. } | _ => return None,
    };
    Some(json!({ kind: fields }))
}

/// The block a kept item states, or `None` when `item` is not one this
/// module wrote.
pub(crate) fn from_json(item: &Value) -> Option<aws_bedrock::ContentBlock> {
    use aws_bedrock::ContentBlock as Block;
    let (kind, body) = item.as_object()?.iter().next()?;
    match kind.as_str() {
        "reasoningContent" => {
            if let Some(data) = body.get("redactedContent") {
                let bytes = BASE64_STANDARD.decode(data.as_str()?).ok()?;
                return Some(Block::ReasoningContent(
                    aws_bedrock::ReasoningContentBlock::RedactedContent(Blob::new(bytes)),
                ));
            }
            let reasoning = body.get("reasoningText")?;
            aws_bedrock::ReasoningTextBlock::builder()
                .text(text(reasoning, "text")?)
                .set_signature(text(reasoning, "signature"))
                .build()
                .ok()
                .map(|block| {
                    Block::ReasoningContent(aws_bedrock::ReasoningContentBlock::ReasoningText(
                        block,
                    ))
                })
        }
        "citationsContent" => {
            let content = list(body, "content").map(|parts| {
                parts
                    .iter()
                    .map(|part| text(part, "text").map(aws_bedrock::CitationGeneratedContent::Text))
                    .collect::<Option<Vec<_>>>()
            });
            let content = present(content)?;
            let citations = list(body, "citations")
                .map(|citations| citations.iter().map(citation).collect::<Option<Vec<_>>>());
            let citations = present(citations)?;
            Some(Block::CitationsContent(
                aws_bedrock::CitationsContentBlock::builder()
                    .set_content(content)
                    .set_citations(citations)
                    .build(),
            ))
        }
        "toolUse" => aws_bedrock::ToolUseBlock::builder()
            .tool_use_id(text(body, "toolUseId")?)
            .name(text(body, "name")?)
            .input(json::to_document(body.get("input")?.clone()))
            .set_type(text(body, "type").map(|kind| aws_bedrock::ToolUseType::from(kind.as_str())))
            .build()
            .ok()
            .map(Block::ToolUse),
        "toolResult" => {
            let content = list(body, "content")?
                .iter()
                .map(|part| {
                    if let Some(value) = part.get("json") {
                        return Some(aws_bedrock::ToolResultContentBlock::Json(
                            json::to_document(value.clone()),
                        ));
                    }
                    text(part, "text").map(aws_bedrock::ToolResultContentBlock::Text)
                })
                .collect::<Option<Vec<_>>>()?;
            aws_bedrock::ToolResultBlock::builder()
                .tool_use_id(text(body, "toolUseId")?)
                .set_content(Some(content))
                .set_status(
                    text(body, "status")
                        .map(|status| aws_bedrock::ToolResultStatus::from(status.as_str())),
                )
                .set_type(text(body, "type"))
                .build()
                .ok()
                .map(Block::ToolResult)
        }
        _ => None,
    }
}

fn citation(value: &Value) -> Option<aws_bedrock::Citation> {
    let source = list(value, "sourceContent").map(|parts| {
        parts
            .iter()
            .map(|part| text(part, "text").map(aws_bedrock::CitationSourceContent::Text))
            .collect::<Option<Vec<_>>>()
    });
    let source = present(source)?;
    let location = present(value.get("location").map(location))?;
    Some(
        aws_bedrock::Citation::builder()
            .set_title(text(value, "title"))
            .set_source(text(value, "source"))
            .set_source_content(source)
            .set_location(location)
            .build(),
    )
}

fn location(value: &Value) -> Option<aws_bedrock::CitationLocation> {
    use aws_bedrock::CitationLocation as Location;
    let (kind, at) = value.as_object()?.iter().next()?;
    let int = |key: &str| at.get(key)?.as_i64().and_then(|n| i32::try_from(n).ok());
    Some(match kind.as_str() {
        "documentChar" => Location::DocumentChar(
            aws_bedrock::DocumentCharLocation::builder()
                .set_document_index(int("documentIndex"))
                .set_start(int("start"))
                .set_end(int("end"))
                .build(),
        ),
        "documentChunk" => Location::DocumentChunk(
            aws_bedrock::DocumentChunkLocation::builder()
                .set_document_index(int("documentIndex"))
                .set_start(int("start"))
                .set_end(int("end"))
                .build(),
        ),
        "documentPage" => Location::DocumentPage(
            aws_bedrock::DocumentPageLocation::builder()
                .set_document_index(int("documentIndex"))
                .set_start(int("start"))
                .set_end(int("end"))
                .build(),
        ),
        "searchResultLocation" => Location::SearchResultLocation(
            aws_bedrock::SearchResultLocation::builder()
                .set_search_result_index(int("searchResultIndex"))
                .set_start(int("start"))
                .set_end(int("end"))
                .build(),
        ),
        "web" => Location::Web(
            aws_bedrock::WebLocation::builder()
                .set_url(text(at, "url"))
                .set_domain(text(at, "domain"))
                .build(),
        ),
        _ => return None,
    })
}

/// An optional field that converted: `Some(None)` when it is absent, and
/// `None` when it is present but did not convert.
fn present<T>(field: Option<Option<T>>) -> Option<Option<T>> {
    match field {
        None => Some(None),
        Some(converted) => converted.map(Some),
    }
}

fn text(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_owned)
}

fn list<'a>(value: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    value.get(key)?.as_array()
}

#[cfg(test)]
mod tests;
