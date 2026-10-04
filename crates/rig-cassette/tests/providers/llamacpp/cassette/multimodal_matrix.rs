//! Part 5: multimodal.
//!
//! **Servers**: two vision configurations, and each cell says which and why.
//!
//! | Server | Model | Why |
//! | --- | --- | --- |
//! | vision (8082) | `ggml-org/Qwen3-VL-2B-Instruct-GGUF` Q8_0 + `mmproj` | the smoke vision tier; its template *does* support tool calls |
//! | large vision (8093) | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` Q4_K_M + `mmproj` | for cells that need the model to be *right*; its template does **not** support tool calls |
//!
//! The split is measured rather than assumed, and the two models fail in
//! opposite directions. Asked which of two images is the photograph,
//! Qwen3-VL-2B answers "FIRST" whichever order they arrive in — it is
//! responsive but not tracking order — while Qwen2.5-VL-7B answers FIRST and
//! SECOND correctly. Asked to call a tool about an image, Qwen3-VL-2B calls it
//! and Qwen2.5-VL-7B writes prose instead, because its chat template reports
//! `chat_template_caps.supports_tool_calls: false` (`GET /props`) — with or
//! without an image, so it is the template, not the modality.
//!
//! | Cell | Dimension | Server | Pinned |
//! | --- | --- | --- | --- |
//! | `image_tool_result::a_tool_result_image_is_read_by_the_model` | image in a tool result | vision | the #2380 capability |
//! | `image_tool_result::the_same_image_in_a_user_message_is_read_too` | image in a user message | vision | the control for the above |
//! | [`an_image_and_a_tool_reach_the_model_together`] | image + tools | vision | the call carries what the image showed |
//! | [`a_video_part_is_refused_even_though_props_advertises_video`] | video | vision | 400 `unsupported content[].type` |
//!
//! # `modalities.video: true` does not mean the chat endpoint takes a video
//!
//! `GET /props` reports `{"vision": true, "video": true, "audio": false}` for
//! both vision models, so "what does rig do with a video part" is a real
//! question rather than a hypothetical. The answer: rig maps
//! [`UserContent::Video`](rig::message::UserContent::Video) to OpenAI's
//! `{"type": "video_url", ...}` part, and llama.cpp answers
//! `400 unsupported content[].type`. The flag describes what the *model* can
//! encode, not what the chat-completions content vocabulary accepts; llama.cpp
//! takes video by other means. Rig's conversion is not wrong — there is no
//! other OpenAI-shaped part to send — and the refusal is clean, so this is a
//! recorded boundary rather than a defect.

use rig::message::{AssistantContent, ImageMediaType, Message, UserContent, VideoMediaType};
use serde_json::Value;

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};
use crate::support::{IMAGE_FIXTURE_PATH, VIDEO_FIXTURE_PATH};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

fn ant_photo() -> UserContent {
    let bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("image fixture should be readable");
    UserContent::image_base64(base64_encode(&bytes), Some(ImageMediaType::JPEG), None)
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// An image and a tool in the same request.
///
/// On the smaller vision model, whose template supports tool calls. The
/// assertion is on the *argument*, not on the fact of a call: a call carrying
/// nothing about the picture would mean the tools reached the model and the
/// image did not.
#[tokio::test]
async fn an_image_and_a_tool_reach_the_model_together() {
    with_llamacpp_vision_cassette("multimodal_matrix/image_plus_tools", |client| async move {
        let model = client.completion(CASSETTE_VISION_MODEL);
        let response = model
            .call(
                CompletionRequest::new(Message::User {
                    content: vec![
                        UserContent::text(
                            "Look at the image and call record_subject with what it shows.",
                        ),
                        ant_photo(),
                    ],
                })
                .tool(rig::completion::ToolDefinition {
                    name: rig_core::message::ToolName::new("record_subject").expect("tool name"),
                    description: "Record what the image shows.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": { "subject": { "type": "string" } },
                        "required": ["subject"],
                    }),
                })
                .tool_choice(rig::message::ToolChoice::Required)
                .max_tokens(256)
                .temperature(0.0),
            )
            .await
            .expect("an image alongside tools should be accepted");

        let call = response
            .choice
            .iter()
            .find_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .expect("tool_choice: required must produce a call");
        assert_eq!(call.function.name, "record_subject");
        let subject = call.function.arguments["subject"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        assert!(
            subject.contains("ant") || subject.contains("insect"),
            "the argument must describe the image, not the prompt: {subject:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "multimodal_matrix/image_plus_tools");
    assert_eq!(
        request["tools"].as_array().map(Vec::len),
        Some(1),
        "the tool definition reached the wire beside the image"
    );
    assert!(
        request["messages"][0]["content"]
            .as_array()
            .is_some_and(|parts| parts
                .iter()
                .any(|part| part["type"] == serde_json::json!("image_url"))),
        "and so did the image"
    );
}

/// A video part is refused, although `/props` advertises `video: true`.
#[tokio::test]
async fn a_video_part_is_refused_even_though_props_advertises_video() {
    with_llamacpp_vision_cassette(
        "multimodal_matrix/video_part_is_refused",
        |client| async move {
            let bytes =
                std::fs::read(VIDEO_FIXTURE_PATH).expect("video fixture should be readable");
            let model = client.completion(CASSETTE_VISION_MODEL);
            let error = model
                .call(
                    CompletionRequest::new(Message::User {
                        content: vec![
                            UserContent::text("Describe this video in one sentence."),
                            UserContent::video_base64(
                                base64_encode(&bytes),
                                Some(VideoMediaType::MP4),
                            ),
                        ],
                    })
                    .max_tokens(64),
                )
                .await
                .expect_err("the chat-completions content vocabulary has no video part here");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the status must reach the caller")
                    .as_u16(),
                400,
                "{error}"
            );
        },
    )
    .await;

    // The premise: rig really did send a `video_url` part.
    let request = recorded_json_request("llamacpp", "multimodal_matrix/video_part_is_refused");
    assert!(
        request["messages"][0]["content"]
            .as_array()
            .is_some_and(|parts| parts
                .iter()
                .any(|part| part["type"] == serde_json::json!("video_url"))),
        "the request must carry the video part this cell is about: {request}"
    );

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "multimodal_matrix/video_part_is_refused");
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 400);
    let json: Value = serde_json::from_str(body).expect("error body should be JSON");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("content[].type")),
        "llama.cpp names the offending part: {json}"
    );

    // And `/props` really does advertise video, which is what makes the
    // refusal worth recording rather than obvious.
    let props = recorded_statuses_and_bodies("llamacpp", "unmapped_surface/props");
    let props: Value = serde_json::from_str(&props[0].1).expect("props should be JSON");
    assert_eq!(props["modalities"]["video"], serde_json::json!(true));
}
