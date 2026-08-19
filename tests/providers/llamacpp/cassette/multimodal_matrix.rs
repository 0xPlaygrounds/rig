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
//! | [`two_images_in_one_turn_keep_their_order`] | multiple images | large vision | the answer flips when the images swap |
//! | [`an_image_and_a_tool_reach_the_model_together`] | image + tools | vision | the call carries what the image showed |
//! | [`a_malformed_data_uri_is_a_400`] | bad base64 | vision | 400 `invalid_request_error` |
//! | [`a_url_the_server_cannot_fetch_is_a_500`] | unreachable URL | vision | 500 — a caller error reported as a server error |
//! | [`an_image_to_a_text_only_server_names_the_missing_mmproj`] | no `--mmproj` | default | 500, and the message says which flag is missing |
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

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, ImageMediaType, Message, UserContent, VideoMediaType};
use serde_json::Value;

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};
use crate::support::{IMAGE_FIXTURE_PATH, VIDEO_FIXTURE_PATH, assistant_text_response};

use super::super::cassette_support::*;

/// A 256x256 solid magenta PNG.
///
/// 256 rather than the 16x16 the tool-result cells use: a vision encoder pads
/// an image below its patch size, and both models then read the padding rather
/// than the colour — measured, and the reason this constant is not shared with
/// `image_tool_result.rs`.
const MAGENTA_PNG_256: &str = "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAACvElEQVR4nO3TMQ0AAAjAMPybBhkca1IBezY7C1nzXgCPDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmANAOQZgDSDECaAUgzAGkGIM0ApBmAtAOTiR3j+3RuqwAAAABJRU5ErkJggg==";

fn ant_photo() -> UserContent {
    let bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("image fixture should be readable");
    UserContent::image_base64(base64_encode(&bytes), Some(ImageMediaType::JPEG), None)
}

fn magenta_square() -> UserContent {
    UserContent::image_base64(MAGENTA_PNG_256, Some(ImageMediaType::PNG), None)
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

const WHICH_IS_THE_PHOTOGRAPH: &str = "Two images follow. One is a photograph of an insect and \
     one is a plain coloured square. Answer with exactly one word, FIRST or SECOND: which image \
     is the photograph of an insect?";

/// Two images in one turn, and the answer tracks which came first.
///
/// The cell records **both orders** in one scenario. A single order proves
/// nothing: a model that ignored the images entirely and always said "FIRST"
/// would pass — and that is exactly what the smaller vision model does, which
/// is why this cell runs on the larger one.
#[tokio::test]
async fn two_images_in_one_turn_keep_their_order() {
    with_llamacpp_large_vision_cassette(
        "multimodal_matrix/two_images_keep_their_order",
        |client| async move {
            let model = client.completion_model(CASSETTE_LARGE_VISION_MODEL);

            let ask = |content: Vec<UserContent>| {
                let model = model.clone();
                async move {
                    let response = model
                        .completion(
                            model
                                .completion_request(Message::User { content })
                                .max_tokens(64)
                                .temperature(0.0)
                                .build(),
                        )
                        .await
                        .expect("a two-image turn should be accepted");
                    assistant_text_response(&response.choice)
                        .unwrap_or_default()
                        .to_ascii_uppercase()
                }
            };

            let photograph_first = ask(vec![
                UserContent::text(WHICH_IS_THE_PHOTOGRAPH),
                ant_photo(),
                magenta_square(),
            ])
            .await;
            let photograph_second = ask(vec![
                UserContent::text(WHICH_IS_THE_PHOTOGRAPH),
                magenta_square(),
                ant_photo(),
            ])
            .await;

            assert!(
                photograph_first.contains("FIRST"),
                "photograph first: {photograph_first:?}"
            );
            assert!(
                photograph_second.contains("SECOND"),
                "photograph second — the answer must move with the images, not \
                 stay put: {photograph_second:?}"
            );
        },
    )
    .await;

    // The premise: both images really did reach the wire, in order, in one
    // message.
    for (turn, request) in crate::cassettes::recorded_interaction_bodies(
        "llamacpp",
        "multimodal_matrix/two_images_keep_their_order",
    )
    .into_iter()
    .map(|(request, _)| request)
    .enumerate()
    {
        let request: Value = serde_json::from_str(&request).expect("request should be JSON");
        let parts = request["messages"][0]["content"]
            .as_array()
            .unwrap_or_else(|| panic!("turn {turn}: content parts: {request}"));
        let images = parts
            .iter()
            .filter(|part| part["type"] == serde_json::json!("image_url"))
            .count();
        assert_eq!(images, 2, "turn {turn}: two image parts in one message");
    }
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
        let model = client.completion_model(CASSETTE_VISION_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(Message::User {
                        content: vec![
                            UserContent::text(
                                "Look at the image and call record_subject with what it shows.",
                            ),
                            ant_photo(),
                        ],
                    })
                    .tool(rig::completion::ToolDefinition {
                        name: "record_subject".to_string(),
                        description: "Record what the image shows.".to_string(),
                        parameters: serde_json::json!({
                            "type": "object",
                            "properties": { "subject": { "type": "string" } },
                            "required": ["subject"],
                        }),
                    })
                    .tool_choice(rig::message::ToolChoice::Required)
                    .max_tokens(256)
                    .temperature(0.0)
                    .build(),
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

/// A data URI whose base64 does not decode is a 400.
#[tokio::test]
async fn a_malformed_data_uri_is_a_400() {
    with_llamacpp_vision_cassette(
        "multimodal_matrix/malformed_data_uri",
        |client| async move {
            let model = client.completion_model(CASSETTE_VISION_MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![
                                UserContent::text("What colour is this?"),
                                UserContent::image_base64(
                                    "!!!!not-base64!!!!",
                                    Some(ImageMediaType::PNG),
                                    None,
                                ),
                            ],
                        })
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect_err("undecodable image bytes must fail");

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

    let recorded = recorded_statuses_and_bodies("llamacpp", "multimodal_matrix/malformed_data_uri");
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 400);
    let json: Value = serde_json::from_str(body).expect("error body should be JSON");
    assert_eq!(
        json["error"]["type"],
        serde_json::json!("invalid_request_error")
    );
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("load image")),
        "{json}"
    );
}

/// An image URL the server cannot fetch is a **500**.
///
/// llama.cpp fetches remote images itself, and reports a failed fetch as
/// `server_error` — the third row in this corpus where a caller error arrives
/// as a 5xx, after `--no-jinja` with tools and the schema/grammar conflict.
#[tokio::test]
async fn a_url_the_server_cannot_fetch_is_a_500() {
    with_llamacpp_vision_cassette(
        "multimodal_matrix/unfetchable_image_url",
        |client| async move {
            let model = client.completion_model(CASSETTE_VISION_MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![
                                UserContent::text("What colour is this?"),
                                // Port 1 on the loopback interface: reserved,
                                // and nothing binds it.
                                UserContent::image_url(
                                    "http://127.0.0.1:1/nope.png",
                                    Some(ImageMediaType::PNG),
                                    None,
                                ),
                            ],
                        })
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect_err("an unfetchable image URL must fail");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the status must reach the caller")
                    .as_u16(),
                500,
                "{error}"
            );
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "multimodal_matrix/unfetchable_image_url");
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 500);
    let json: Value = serde_json::from_str(body).expect("error body should be JSON");
    assert_eq!(json["error"]["type"], serde_json::json!("server_error"));
}

/// An image sent to a server started without `--mmproj`.
#[tokio::test]
async fn an_image_to_a_text_only_server_names_the_missing_mmproj() {
    with_llamacpp_cassette(
        "multimodal_matrix/image_without_mmproj",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![
                                UserContent::text("What colour is this?"),
                                magenta_square(),
                            ],
                        })
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect_err("a text-only server cannot see an image");

            let body = error
                .provider_response_body()
                .expect("the body must be preserved");
            assert!(
                body.contains("mmproj"),
                "the flag the operator is missing is the actionable half: {body}"
            );
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "multimodal_matrix/image_without_mmproj");
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(
        *status, 500,
        "reported as a server error, though it is a configuration mismatch: {body}"
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
            let model = client.completion_model(CASSETTE_VISION_MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![
                                UserContent::text("Describe this video in one sentence."),
                                UserContent::video(
                                    base64_encode(&bytes),
                                    Some(VideoMediaType::MP4),
                                ),
                            ],
                        })
                        .max_tokens(64)
                        .build(),
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
