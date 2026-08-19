//! Handing an image *back* to the model through a tool result.
//!
//! Official OpenAI cannot do this on Chat Completions: `gpt-4o` answers 400
//! (*"Image URLs are only allowed for messages with role 'user'"*) and the
//! GPT-5 family answers 200 with the image discarded, the model then describing
//! what it never received. llama.cpp does honour it, which is why
//! `LlamafileExt::SUPPORTS_IMAGE_TOOL_RESULTS` is `true` and the shared
//! conversion is gated rather than hard-coded.
//!
//! Recorded against `llama-server` built from source at commit `6d05498`
//! (`build_info` `b1-6d05498`) running `ggml-org/Qwen3-VL-2B-Instruct-GGUF`
//! Q8_0 with its `mmproj`, `--jinja --seed 42 --temp 0 -c 4096`. The subject
//! and its control were recorded in one run so the negative half cannot be
//! explained away by a bad image or a blind model.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against a local llama.cpp server.

use rig::client::CompletionClient as _;
use rig::completion::CompletionModel as _;
use rig::message::{ImageMediaType, ProviderCallId, ToolCallId, ToolResult, ToolResultContent};

use super::super::cassette_support::with_llamafile_cassette;

/// 16x16 solid magenta. Distinctive on purpose: "red" and "blue" are plausible
/// blind guesses for "what colour is this image", so a cell using them could
/// pass without the model ever receiving the bytes.
const MAGENTA_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4nGP4z/CfJMQwqmFUw/DVAAAg0/4QF5nKuAAAAABJRU5ErkJggg==";

/// The vision model the fixture was recorded against.
const VISION_MODEL: &str = "Qwen3-VL-2B-Instruct-Q8_0";

fn image_tool_result() -> ToolResult {
    ToolResult {
        call: ToolCallId::new_or_mint("call_1"),
        provider: ProviderCallId::new("call_1"),
        name: "view_file".to_string(),
        content: vec![ToolResultContent::image_base64(
            MAGENTA_PNG_BASE64,
            Some(ImageMediaType::PNG),
            None,
        )],
    }
}

fn tool_call_turn() -> rig::message::Message {
    rig::message::Message::Assistant {
        id: None,
        content: vec![rig::message::AssistantContent::tool_call(
            "call_1",
            "view_file",
            serde_json::json!({}),
        )],
    }
}

fn assistant_text(response: &rig::completion::CompletionResponse) -> String {
    response
        .choice
        .iter()
        .filter_map(|c| match c {
            rig::message::AssistantContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The image reaches the model through a `role:"tool"` message.
///
/// The assertion is the colour, not a 200. A cell that only checked the request
/// was accepted would pass against a server that took the image and dropped it
/// — which is exactly what the GPT-5 family does, and exactly the failure worth
/// catching.
#[tokio::test]
async fn a_tool_result_image_is_read_by_the_model() {
    with_llamafile_cassette(
        "image_tool_result/a_tool_result_image_is_read_by_the_model",
        |client| async move {
            let model = client.completion_model(VISION_MODEL);
            let request = model
                .completion_request(
                    "Call view_file, then reply with ONLY the dominant colour name.",
                )
                .max_tokens(30)
                .temperature(0.0)
                .messages(vec![
                    tool_call_turn(),
                    rig::message::Message::User {
                        content: vec![rig::message::UserContent::ToolResult(image_tool_result())],
                    },
                ])
                .build();

            let response = model
                .completion(request)
                .await
                .expect("llama.cpp accepts an image in a tool result");

            let said = assistant_text(&response);
            assert!(
                said.contains("pink") || said.contains("magenta"),
                "the model has to have seen the image to name its colour; said: {said:?}"
            );
        },
    )
    .await;
}

/// The control: identical bytes in a `user` message.
///
/// Without this the cell above proves only that *something* produced a colour
/// word. With it, the image, the model and the run are all held constant and
/// the only variable is the message role.
#[tokio::test]
async fn the_same_image_in_a_user_message_is_read_too() {
    with_llamafile_cassette(
        "image_tool_result/the_same_image_in_a_user_message_is_read_too",
        |client| async move {
            let model = client.completion_model(VISION_MODEL);
            let request = model
                .completion_request(rig::message::Message::User {
                    content: vec![
                        rig::message::UserContent::text(
                            "Reply with ONLY the dominant colour name.",
                        ),
                        rig::message::UserContent::image_base64(
                            MAGENTA_PNG_BASE64,
                            Some(ImageMediaType::PNG),
                            None,
                        ),
                    ],
                })
                .max_tokens(30)
                .temperature(0.0)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("an image in a user message is ordinary and must work");

            let said = assistant_text(&response);
            assert!(
                said.contains("pink") || said.contains("magenta"),
                "the control must succeed or the subject cell proves nothing: {said:?}"
            );
        },
    )
    .await;
}
