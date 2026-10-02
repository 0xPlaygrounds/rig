use super::to_aws;
use crate::completion::{
    AMAZON_NOVA_LITE, AMAZON_NOVA_MICRO, ANTHROPIC_CLAUDE_SONNET_4_6, Converse, Family,
};
use crate::streaming::tests::{
    assert_agrees, reasoning_text, reply_of, restated, tool_use, unary_as,
};
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine as _, prelude::BASE64_STANDARD};
use rig_core::completion::{CompletionRequest, ReplayTarget};
use rig_core::message::{
    AssistantContent, AssistantMessage, Message, Opaque, Origin, ToolFunction, ToolName,
    ToolResultContent,
};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::json;

const PROFILE: &str =
    "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4";

/// The Converse messages `history` becomes on `wire`, through the core
/// adapter and this wire's encoder.
fn sent_on(wire: &Converse, history: Vec<Message>) -> Vec<aws_bedrock::Message> {
    let request = CompletionRequest::new("again").messages(history);
    let request = Completion::prepare(request, &wire.describe()).expect("prepares");
    wire.encode(request, Mode::Unary)
        .expect("encodes")
        .request
        .messages()
        .expect("converts")
}

fn sent(model: &str, history: Vec<Message>) -> Vec<aws_bedrock::Message> {
    sent_on(&Converse::new(model), history)
}

fn image() -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::Image(
        aws_bedrock::ImageBlock::builder()
            .format(aws_bedrock::ImageFormat::Png)
            .source(aws_bedrock::ImageSource::Bytes(
                aws_smithy_types::Blob::new(b"png".to_vec()),
            ))
            .build()
            .unwrap(),
    )
}

/// A reply, as Converse sent it.
fn reply() -> Vec<aws_bedrock::ContentBlock> {
    vec![
        reasoning_text("let me think", Some("sig-abc")),
        aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(
                b"\x00ciphertext\xff".to_vec(),
            )),
        ),
        aws_bedrock::ContentBlock::Text("Calling.".to_owned()),
        image(),
        tool_use(
            "srv_1",
            "nova_grounding",
            json!({ "q": "harbor" }),
            Some(aws_bedrock::ToolUseType::ServerToolUse),
        ),
        tool_use("tooluse_1", "lookup", json!({ "q": "harbor" }), None),
    ]
}

/// The turns a reply decodes to from `wire`, whole and streamed.
fn decoded_on(wire: &Converse) -> [Message; 2] {
    let whole = rig_core::test_utils::history::decode(
        wire,
        Mode::Unary,
        [
            crate::completion::ConverseFrame::Opened { request_id: None },
            crate::completion::ConverseFrame::Whole(Box::new(reply_of(reply(), "tool_use"))),
        ],
    )
    .expect("decodes whole");
    let opened = crate::completion::ConverseFrame::Opened { request_id: None };
    let frames = std::iter::once(opened).chain(
        restated(&reply(), "tool_use")
            .into_iter()
            .map(crate::completion::ConverseFrame::Event),
    );
    let streamed = rig_core::test_utils::history::decode(wire, Mode::Streaming, frames)
        .expect("decodes streamed");
    [whole, streamed].map(|response| response.message().expect("a turn"))
}

fn decoded(model: &str) -> [Message; 2] {
    decoded_on(&Converse::new(model))
}

/// The same model gets every item back as it came, in both modes: signed
/// and redacted reasoning and the hosted tool's use included.
#[test]
fn the_same_model_gets_its_items_back_verbatim() {
    assert_agrees(ANTHROPIC_CLAUDE_SONNET_4_6, &reply(), "tool_use");
    for turn in decoded(ANTHROPIC_CLAUDE_SONNET_4_6) {
        let messages = sent(ANTHROPIC_CLAUDE_SONNET_4_6, vec![Message::user("hi"), turn]);
        assert_eq!(messages[1].content, reply());
        // The adapter answered the client call nothing answered, as an error.
        let Some(aws_bedrock::ContentBlock::ToolResult(result)) = messages[2].content.first()
        else {
            panic!("{:?}", messages[2]);
        };
        assert_eq!(result.tool_use_id, "tooluse_1");
        assert_eq!(result.status, Some(aws_bedrock::ToolResultStatus::Error));
    }
}

/// Another model gets canonical fields only: reasoning text becomes text,
/// redacted reasoning and the hosted tool's use are dropped, and an image,
/// which Converse reads only from the user, becomes a placeholder.
#[test]
fn another_model_gets_canonical_fields() {
    for turn in decoded(ANTHROPIC_CLAUDE_SONNET_4_6) {
        let messages = sent(AMAZON_NOVA_LITE, vec![Message::user("hi"), turn]);
        let expected = vec![
            aws_bedrock::ContentBlock::Text("let me think".to_owned()),
            aws_bedrock::ContentBlock::Text("Calling.".to_owned()),
            aws_bedrock::ContentBlock::Text(
                rig_core::completion::history::ASSISTANT_IMAGE_OMITTED.to_owned(),
            ),
            tool_use("tooluse_1", "lookup", json!({ "q": "harbor" }), None),
        ];
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

/// Claude behind an application inference profile, stated as Claude, gets
/// its signed reasoning back signed, and an edited reasoning block as text,
/// since Claude rejects unsigned reasoning.
#[test]
fn claude_behind_an_application_profile_keeps_its_signatures() {
    let wire = Converse::new(PROFILE).with_family(Family::Claude);
    for turn in decoded_on(&wire) {
        let messages = sent_on(&wire, vec![Message::user("hi"), turn.clone()]);
        assert_eq!(
            messages[1].content[0],
            reasoning_text("let me think", Some("sig-abc"))
        );
        let Message::Assistant(mut edited) = turn else {
            panic!("an assistant turn");
        };
        if let AssistantContent::Reasoning(reasoning) = &mut edited.content[0] {
            reasoning.text = "edited".to_owned();
        }
        let messages = sent_on(&wire, vec![Message::user("hi"), edited.into()]);
        assert_eq!(
            messages[1].content[0],
            aws_bedrock::ContentBlock::Text("edited".to_owned())
        );
    }
    assert!(wire.accepts(PROFILE).user_images);
}

/// The family comes from the provider a model id names, or from the
/// caller for an id that names none.
#[test]
fn the_family_is_the_provider_the_id_names() {
    assert_eq!(Family::of(ANTHROPIC_CLAUDE_SONNET_4_6), Family::Claude);
    assert_eq!(
        Family::of("anthropic.claude-3-5-haiku-20241022-v1:0"),
        Family::Claude
    );
    assert_eq!(
        Family::of("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-7-v1:0"),
        Family::Claude
    );
    assert_eq!(Family::of(AMAZON_NOVA_LITE), Family::Other);
    assert_eq!(Family::of(PROFILE), Family::Other);
    // A model that only mentions Claude in its name is not Claude.
    assert_eq!(
        Family::of("acme.anthropic-claude-clone-v1:0"),
        Family::Other
    );
    let wire = Converse::new(PROFILE).with_family(Family::Claude);
    assert_eq!(wire.family(PROFILE), Family::Claude);
    assert_eq!(wire.family(AMAZON_NOVA_LITE), Family::Other);
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
    let accepts = |model: &str| Converse::new(model).accepts(model);
    assert!(!accepts(AMAZON_NOVA_MICRO).user_images);
    assert!(!accepts("us.deepseek.r1-v1:0").user_images);
    assert!(accepts(AMAZON_NOVA_LITE).user_images);
    assert!(accepts(ANTHROPIC_CLAUDE_SONNET_4_6).user_images);
    let claude = accepts(ANTHROPIC_CLAUDE_SONNET_4_6);
    assert!(claude.tool_result_images && !claude.assistant_images && claude.tools);
}

fn signed(text: &str) -> AssistantContent {
    AssistantContent::reasoning(text)
        .with_native(crate::types::block::reasoning_json(text, Some("sig")))
}

/// A current reasoning item goes back as it came to either family, even
/// with no text; reasoning with none is canonical: text on Claude, which
/// rejects unsigned reasoning, and unsigned reasoning elsewhere, as pi
/// sends it.
#[test]
fn reasoning_follows_its_item_and_the_family() {
    for family in [Family::Claude, Family::Other] {
        assert_eq!(
            to_aws(signed("thought"), family).unwrap(),
            Some(reasoning_text("thought", Some("sig")))
        );
        assert_eq!(
            to_aws(signed(""), family).unwrap(),
            Some(reasoning_text("", Some("sig")))
        );
        assert_eq!(
            to_aws(AssistantContent::reasoning(" "), family).unwrap(),
            None
        );
    }
    assert_eq!(
        to_aws(AssistantContent::reasoning("thought"), Family::Claude).unwrap(),
        Some(aws_bedrock::ContentBlock::Text("thought".to_owned()))
    );
    assert_eq!(
        to_aws(AssistantContent::reasoning("thought"), Family::Other).unwrap(),
        Some(reasoning_text("thought", None))
    );
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
        .with_native(json!({ "reasoningContent": { "redactedContent": data } }))
    };
    assert_eq!(
        to_aws(redacted(&BASE64_STANDARD.encode(b"bytes")), Family::Other).unwrap(),
        Some(aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(
                b"bytes".to_vec()
            ))
        ))
    );
    assert_eq!(
        to_aws(redacted("not base64!"), Family::Other).unwrap(),
        None
    );
}

/// Converse rejects blank text; a marker is never sent, and a hosted
/// tool's item is.
#[test]
fn blank_text_and_markers_are_not_sent() {
    assert_eq!(
        to_aws(AssistantContent::text(" \n"), Family::Claude).unwrap(),
        None
    );
    let marker = AssistantContent::Opaque(Opaque {
        item: json!({ "type": "document" }),
        replay: false,
    });
    assert_eq!(to_aws(marker, Family::Claude).unwrap(), None);
    let hosted = AssistantContent::Opaque(Opaque {
        item: json!({ "toolUse": {
            "toolUseId": "srv_1", "name": "nova_grounding", "input": {}, "type": "server_tool_use",
        } }),
        replay: true,
    });
    assert_eq!(
        to_aws(hosted, Family::Other).unwrap(),
        Some(tool_use(
            "srv_1",
            "nova_grounding",
            json!({}),
            Some(aws_bedrock::ToolUseType::ServerToolUse)
        ))
    );
}

/// Call arguments are an object by type, so a call made with `null`
/// arguments sends `{}`, never a `null` input.
#[test]
fn null_arguments_are_sent_as_an_object() {
    let call = AssistantContent::ToolCall(rig_core::message::ToolCall::new(
        rig_core::message::CallId::from_wire("tooluse_1"),
        ToolFunction::new(ToolName::new("lookup").unwrap(), serde_json::Value::Null),
    ));
    let Some(aws_bedrock::ContentBlock::ToolUse(encoded)) = to_aws(call, Family::Other).unwrap()
    else {
        panic!("a tool use");
    };
    assert_eq!(
        encoded.input,
        aws_smithy_types::Document::Object(Default::default())
    );
    // The same holds through a whole reply that stated `null` input.
    let response = unary_as(
        AMAZON_NOVA_LITE,
        reply_of(
            vec![tool_use(
                "tooluse_2",
                "lookup",
                serde_json::Value::Null,
                None,
            )],
            "tool_use",
        ),
    )
    .expect("decodes");
    let messages = sent(
        AMAZON_NOVA_LITE,
        vec![Message::user("hi"), response.message().expect("a turn")],
    );
    let aws_bedrock::ContentBlock::ToolUse(replayed) = &messages[1].content[0] else {
        panic!("{:?}", messages[1]);
    };
    assert_eq!(
        replayed.input,
        aws_smithy_types::Document::Object(Default::default())
    );
}
