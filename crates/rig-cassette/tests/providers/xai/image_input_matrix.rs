//! A generated image fed back as input on xAI: see `common/image_inputs.rs`.

use rig::providers::{openai, xai};
use rig::wire::Wire as _;

use super::support::with_xai_cassette;
use crate::image_inputs;

/// Grok reads the image Grok generated, as user content.
#[tokio::test]
async fn generated_image_as_user_content() {
    const SCENARIO: &str = "image_input_matrix/generated_image_as_user_content";
    with_xai_cassette(
        "image_input_matrix/generated_image_as_user_content",
        |client| async move {
            // The images route is OpenAI-shaped, rebuilt from the cassette's
            // credential and base URL as the image smoke test does.
            let responses = client.clone();
            let generator = openai::wire::OpenAI::with_key(&xai::DIALECT, responses.api_key)
                .with_base_url(responses.base_url)
                .image_generation(xai::image_generation::GROK_IMAGINE_IMAGE)
                .on(rig::transport());
            let bytes = image_inputs::generate(&generator, None, None).await;
            image_inputs::as_user_content(
                &client.completion(xai::GROK_4).on(rig::transport()),
                &bytes,
                None,
            )
            .await;
        },
    )
    .await;
    image_inputs::assert_recorded("xai", SCENARIO, image_inputs::Slot::UserContent);
}

/// Grok reads the image Grok generated, as a tool result.
#[tokio::test]
async fn generated_image_as_tool_result() {
    const SCENARIO: &str = "image_input_matrix/generated_image_as_tool_result";
    with_xai_cassette(
        "image_input_matrix/generated_image_as_tool_result",
        |client| async move {
            let responses = client.clone();
            let generator = openai::wire::OpenAI::with_key(&xai::DIALECT, responses.api_key)
                .with_base_url(responses.base_url)
                .image_generation(xai::image_generation::GROK_IMAGINE_IMAGE)
                .on(rig::transport());
            let bytes = image_inputs::generate(&generator, None, None).await;
            image_inputs::as_tool_result(
                &client.completion(xai::GROK_4).on(rig::transport()),
                &bytes,
                Some(serde_json::json!({ "store": false })),
            )
            .await;
        },
    )
    .await;
    image_inputs::assert_recorded("xai", SCENARIO, image_inputs::Slot::ToolResult);
}
