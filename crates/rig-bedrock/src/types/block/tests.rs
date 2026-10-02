use super::*;
use crate::streaming::tests::{reasoning_text, tool_use};

fn cited(location: aws_bedrock::CitationLocation) -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::CitationsContent(
        aws_bedrock::CitationsContentBlock::builder()
            .content(aws_bedrock::CitationGeneratedContent::Text(
                "cited".to_owned(),
            ))
            .citations(
                aws_bedrock::Citation::builder()
                    .title("note")
                    .source("s3://bucket/note")
                    .source_content(aws_bedrock::CitationSourceContent::Text("quote".to_owned()))
                    .location(location)
                    .build(),
            )
            .build(),
    )
}

/// Every block a turn keeps converts to its JSON and back unchanged.
#[test]
fn kept_blocks_round_trip() {
    let span = (Some(1), Some(2), Some(3));
    let blocks = vec![
        reasoning_text("thought", Some("sig")),
        reasoning_text("unsigned", None),
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(Blob::new(b"\x00\xff".to_vec())),
        ),
        cited(aws_bedrock::CitationLocation::DocumentChar(
            aws_bedrock::DocumentCharLocation::builder()
                .set_document_index(span.0)
                .set_start(span.1)
                .set_end(span.2)
                .build(),
        )),
        cited(aws_bedrock::CitationLocation::DocumentChunk(
            aws_bedrock::DocumentChunkLocation::builder()
                .set_document_index(span.0)
                .build(),
        )),
        cited(aws_bedrock::CitationLocation::DocumentPage(
            aws_bedrock::DocumentPageLocation::builder()
                .set_start(span.1)
                .build(),
        )),
        cited(aws_bedrock::CitationLocation::SearchResultLocation(
            aws_bedrock::SearchResultLocation::builder()
                .search_result_index(0)
                .end(9)
                .build(),
        )),
        cited(aws_bedrock::CitationLocation::Web(
            aws_bedrock::WebLocation::builder()
                .url("https://example.com")
                .domain("example.com")
                .build(),
        )),
        tool_use(
            "srv_1",
            "nova_grounding",
            json!({ "q": [1, 2.5, "x", null, true] }),
            Some(aws_bedrock::ToolUseType::ServerToolUse),
        ),
        aws_bedrock::ContentBlock::ToolResult(
            aws_bedrock::ToolResultBlock::builder()
                .tool_use_id("srv_1")
                .content(aws_bedrock::ToolResultContentBlock::Text(
                    "found".to_owned(),
                ))
                .content(aws_bedrock::ToolResultContentBlock::Json(
                    json::to_document(json!({ "hits": 2 })),
                ))
                .status(aws_bedrock::ToolResultStatus::Error)
                .r#type("web_search_result")
                .build()
                .expect("builds"),
        ),
    ];
    for block in blocks {
        let item = to_json(&block).unwrap_or_else(|| panic!("{block:?} is kept"));
        assert_eq!(from_json(&item), Some(block), "{item}");
    }
}

/// Text is canonical, a block holding a part this SDK version does not
/// model has no JSON form, and an item this module did not write is no
/// block.
#[test]
fn other_blocks_have_no_json_form() {
    assert_eq!(
        to_json(&aws_bedrock::ContentBlock::Text("x".to_owned())),
        None
    );
    let with_image = aws_bedrock::ContentBlock::ToolResult(
        aws_bedrock::ToolResultBlock::builder()
            .tool_use_id("srv_1")
            .content(aws_bedrock::ToolResultContentBlock::Image(
                aws_bedrock::ImageBlock::builder()
                    .format(aws_bedrock::ImageFormat::Png)
                    .build()
                    .expect("builds"),
            ))
            .build()
            .expect("builds"),
    );
    assert_eq!(to_json(&with_image), None);
    for item in [
        json!({ "signature": "sig" }),
        json!({ "type": "document" }),
        json!({ "reasoningContent": { "redactedContent": "not base64!" } }),
        json!("text"),
    ] {
        assert_eq!(from_json(&item), None, "{item}");
    }
}
