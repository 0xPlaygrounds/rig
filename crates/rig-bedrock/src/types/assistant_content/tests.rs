use super::{reads_signatures, to_aws};
use crate::completion::{
    AMAZON_NOVA_LITE, AMAZON_NOVA_MICRO, ANTHROPIC_CLAUDE_SONNET_4_6, Converse,
};
use crate::streaming::tests::{assert_agrees, reply_of, restated, unary_as};
use crate::types::converse_output::{
    Blob, ContentBlock, ImageBlock, ImageFormat, ImageSource, ReasoningContentBlock,
    ReasoningTextBlock, StopReason, ToolUseBlock,
};
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine as _, prelude::BASE64_STANDARD};
use rig_core::completion::{CompletionRequest, ReplayTarget};
use rig_core::message::{
    AssistantContent, AssistantMessage, Message, Opaque, Origin, ToolName, ToolResultContent,
};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::json;

/// The Converse messages `history` becomes on `model`, through the core
/// adapter and this wire's encoder.
fn sent(model: &str, history: Vec<Message>) -> Vec<aws_bedrock::Message> {
    let wire = Converse::new(model);
    let request = CompletionRequest::new("again").messages(history);
    let request = Completion::prepare(request, &wire.describe()).expect("prepares");
    wire.encode(request, Mode::Unary)
        .expect("encodes")
        .request
        .messages()
        .expect("converts")
}

fn reply() -> crate::types::converse_output::InternalConverseOutput {
    reply_of(
        vec![
            ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(
                ReasoningTextBlock {
                    text: "let me think".to_owned(),
                    signature: Some("sig-abc".to_owned()),
                },
            )),
            ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(Blob {
                inner: b"\x00ciphertext\xff".to_vec(),
            })),
            ContentBlock::Text("Calling.".to_owned()),
            ContentBlock::Image(ImageBlock {
                format: ImageFormat::Png,
                source: Some(ImageSource::Bytes(Blob {
                    inner: b"png".to_vec(),
                })),
            }),
            ContentBlock::ToolUse(ToolUseBlock {
                tool_use_id: "tooluse_1".to_owned(),
                name: "lookup".to_owned(),
                input: json!({ "q": "harbor" }),
            }),
        ],
        StopReason::ToolUse,
    )
}

/// What Converse sent for [`reply`], as the SDK blocks.
fn reply_as_sent() -> Vec<aws_bedrock::ContentBlock> {
    vec![
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(
                aws_bedrock::ReasoningTextBlock::builder()
                    .text("let me think")
                    .signature("sig-abc")
                    .build()
                    .unwrap(),
            ),
        ),
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(
                b"\x00ciphertext\xff".to_vec(),
            )),
        ),
        aws_bedrock::ContentBlock::Text("Calling.".to_owned()),
        aws_bedrock::ContentBlock::Image(
            aws_bedrock::ImageBlock::builder()
                .format(aws_bedrock::ImageFormat::Png)
                .source(aws_bedrock::ImageSource::Bytes(
                    aws_smithy_types::Blob::new(b"png".to_vec()),
                ))
                .build()
                .unwrap(),
        ),
        aws_bedrock::ContentBlock::ToolUse(
            aws_bedrock::ToolUseBlock::builder()
                .tool_use_id("tooluse_1")
                .name("lookup")
                .input(crate::types::json::to_document(json!({ "q": "harbor" })))
                .build()
                .unwrap(),
        ),
    ]
}

/// The turns a reply decodes to, whole and streamed.
fn decoded(model: &str) -> [Message; 2] {
    let whole = unary_as(model, reply()).expect("decodes whole");
    let opened = crate::completion::ConverseFrame::Opened { request_id: None };
    let frames = std::iter::once(opened).chain(
        restated(&reply())
            .into_iter()
            .map(crate::completion::ConverseFrame::Event),
    );
    let streamed =
        rig_core::test_utils::history::decode(&Converse::new(model), Mode::Streaming, frames)
            .expect("decodes streamed");
    [whole, streamed].map(|response| response.message().expect("a turn"))
}

/// The same model gets every item back as it came, in both modes: signed
/// and redacted reasoning included.
#[test]
fn the_same_model_gets_its_items_back_verbatim() {
    assert_agrees(ANTHROPIC_CLAUDE_SONNET_4_6, &reply());
    for turn in decoded(ANTHROPIC_CLAUDE_SONNET_4_6) {
        let messages = sent(ANTHROPIC_CLAUDE_SONNET_4_6, vec![Message::user("hi"), turn]);
        assert_eq!(messages[1].content, reply_as_sent());
        // The adapter answered the call nothing answered.
        assert!(matches!(
            messages[2].content.first(),
            Some(aws_bedrock::ContentBlock::ToolResult(_))
        ));
    }
}

/// Another model gets canonical fields only: reasoning text becomes text,
/// redacted reasoning is dropped, and an image, which Converse reads only
/// from the user, becomes a placeholder.
#[test]
fn another_model_gets_canonical_fields() {
    for turn in decoded(ANTHROPIC_CLAUDE_SONNET_4_6) {
        let messages = sent(AMAZON_NOVA_LITE, vec![Message::user("hi"), turn]);
        let mut expected = reply_as_sent();
        expected.remove(1);
        expected[0] = aws_bedrock::ContentBlock::Text("let me think".to_owned());
        for block in &mut expected {
            if matches!(block, aws_bedrock::ContentBlock::Image(_)) {
                *block = aws_bedrock::ContentBlock::Text(
                    rig_core::completion::history::ASSISTANT_IMAGE_OMITTED.to_owned(),
                );
            }
        }
        assert_eq!(messages[1].content, expected);
    }
}

/// An edited block's item is stale: it is rebuilt from its canonical
/// fields, which for unsigned reasoning on Claude is text.
#[test]
fn an_edited_block_is_rebuilt() {
    let [Message::Assistant(mut turn), _] = decoded(ANTHROPIC_CLAUDE_SONNET_4_6) else {
        panic!("an assistant turn");
    };
    if let AssistantContent::Reasoning(reasoning) = &mut turn.content[0] {
        reasoning.text = "edited".to_owned();
    }
    let messages = sent(
        ANTHROPIC_CLAUDE_SONNET_4_6,
        vec![Message::user("hi"), turn.into()],
    );
    assert_eq!(
        messages[1].content[0],
        aws_bedrock::ContentBlock::Text("edited".to_owned())
    );
}

/// A call another provider made gets an id Converse accepts, and its result
/// follows it.
#[test]
fn a_foreign_call_id_is_normalized_with_its_result() {
    let id = format!("call_{}|fc.{}", "a".repeat(40), "b".repeat(40));
    let call = AssistantContent::tool_call(&id, ToolName::new("lookup").unwrap(), json!({}));
    let AssistantContent::ToolCall(held) = &call else {
        unreachable!()
    };
    let result = Message::tool_results(vec![held.result(vec![ToolResultContent::text("ok")])]);
    let turn = AssistantMessage {
        origin: Some(Origin::new("openai.responses", "openai", "gpt-5")),
        ..AssistantMessage::new(vec![call])
    };
    let messages = sent(
        AMAZON_NOVA_LITE,
        vec![Message::user("hi"), turn.into(), result],
    );
    let aws_bedrock::ContentBlock::ToolUse(call) = &messages[1].content[0] else {
        panic!("{:?}", messages[1]);
    };
    let aws_bedrock::ContentBlock::ToolResult(result) = &messages[2].content[0] else {
        panic!("{:?}", messages[2]);
    };
    let expected: String = format!("call_{}_fc_{}", "a".repeat(40), "b".repeat(40))
        .chars()
        .take(64)
        .collect();
    assert_eq!(call.tool_use_id, expected);
    assert_eq!(result.tool_use_id, expected);
}

#[test]
fn text_only_models_take_no_images() {
    assert!(
        !Converse::new(AMAZON_NOVA_MICRO)
            .accepts(AMAZON_NOVA_MICRO)
            .user_images
    );
    assert!(
        !Converse::new("us.deepseek.r1-v1:0")
            .accepts("us.deepseek.r1-v1:0")
            .user_images
    );
    assert!(
        Converse::new(AMAZON_NOVA_LITE)
            .accepts(AMAZON_NOVA_LITE)
            .user_images
    );
    assert!(
        Converse::new(ANTHROPIC_CLAUDE_SONNET_4_6)
            .accepts(ANTHROPIC_CLAUDE_SONNET_4_6)
            .user_images
    );
}

#[test]
fn only_claude_reads_signatures() {
    assert!(reads_signatures(ANTHROPIC_CLAUDE_SONNET_4_6));
    assert!(reads_signatures(
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-7-v1:0"
    ));
    assert!(!reads_signatures(AMAZON_NOVA_LITE));
}

fn signed(text: &str) -> AssistantContent {
    AssistantContent::reasoning(text).with_native(json!({ "signature": "sig" }))
}

fn reasoning_block(text: &str, signature: Option<&str>) -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::ReasoningContent(aws_bedrock::ReasoningContentBlock::ReasoningText(
        aws_bedrock::ReasoningTextBlock::builder()
            .text(text)
            .set_signature(signature.map(str::to_owned))
            .build()
            .unwrap(),
    ))
}

/// pi's reasoning rules: a signature goes only to Claude, signed reasoning
/// goes back even with no text, Claude gets unsigned reasoning as text, and
/// blank unsigned reasoning is not sent.
#[test]
fn reasoning_follows_the_targets_signature_support() {
    assert_eq!(
        to_aws(signed("thought"), true).unwrap(),
        Some(reasoning_block("thought", Some("sig")))
    );
    assert_eq!(
        to_aws(signed(""), true).unwrap(),
        Some(reasoning_block("", Some("sig")))
    );
    assert_eq!(
        to_aws(signed("thought"), false).unwrap(),
        Some(reasoning_block("thought", None))
    );
    assert_eq!(
        to_aws(AssistantContent::reasoning("thought"), true).unwrap(),
        Some(aws_bedrock::ContentBlock::Text("thought".to_owned()))
    );
    assert_eq!(
        to_aws(AssistantContent::reasoning(" "), true).unwrap(),
        None
    );
    assert_eq!(to_aws(signed(""), false).unwrap(), None);
}

/// Redacted reasoning goes back as its bytes; a payload that is not base64
/// is dropped rather than sent corrupt.
#[test]
fn redacted_reasoning_goes_back_as_its_bytes() {
    let redacted = |data: &str| {
        AssistantContent::Reasoning(rig_core::message::Reasoning {
            redacted: true,
            ..Default::default()
        })
        .with_native(json!({ "redacted": data }))
    };
    assert_eq!(
        to_aws(redacted(&BASE64_STANDARD.encode(b"bytes")), false).unwrap(),
        Some(aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(
                b"bytes".to_vec()
            ))
        ))
    );
    assert_eq!(to_aws(redacted("not base64!"), false).unwrap(), None);
}

/// Converse rejects blank text, and no opaque item this wire decodes is
/// sent back.
#[test]
fn blank_text_and_opaque_items_are_not_sent() {
    assert_eq!(to_aws(AssistantContent::text(" \n"), true).unwrap(), None);
    let opaque = AssistantContent::Opaque(Opaque {
        item: json!({ "Document": {} }),
        replay: true,
    });
    assert_eq!(to_aws(opaque, true).unwrap(), None);
}
