//! OpenRouter live coverage for image-generation responses.

use base64::Engine;
use futures::StreamExt;
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{AssistantArtifact, AssistantContent, CompletionModel, Message};
use rig::message::{Image, ImageMediaType, UserContent};
use rig::providers::openrouter;
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use crate::support::{IMAGE_FIXTURE_PATH, assert_nonempty_response};

const IMAGE_GENERATION_MODEL: &str = "google/gemini-2.5-flash-image";
const VISION_MODEL: &str = "google/gemini-2.5-flash";

fn generated_image_params() -> serde_json::Value {
    json!({
        "modalities": ["image", "text"]
    })
}

fn generated_images(artifacts: &[AssistantArtifact]) -> Vec<&Image> {
    artifacts
        .iter()
        .map(|artifact| match artifact {
            AssistantArtifact::Image(image) => image,
        })
        .collect()
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY and an OpenRouter image-generation model"]
async fn generated_image_response_surfaces_image_artifact() {
    let client = openrouter::Client::from_env().expect("client should build");
    let model = client.completion_model(IMAGE_GENERATION_MODEL);

    let response = model
        .completion_request(
            "Generate a simple square icon of a red lighthouse on a white background.",
        )
        .additional_params(generated_image_params())
        .send()
        .await
        .expect("image-generation completion should succeed");

    let images = generated_images(&response.artifacts);
    assert!(
        !images.is_empty(),
        "expected generated images in normalized assistant content, saw {:?}",
        response.choice
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY and an OpenRouter image-generation model"]
async fn generated_image_history_can_be_replayed_on_followup() {
    let client = openrouter::Client::from_env().expect("client should build");
    let image_model = client.completion_model(IMAGE_GENERATION_MODEL);

    let first = image_model
        .completion_request(
            "Generate a simple square icon of a blue lighthouse on a white background.",
        )
        .additional_params(generated_image_params())
        .send()
        .await
        .expect("image-generation completion should succeed");

    assert!(
        !generated_images(&first.artifacts).is_empty(),
        "test requires an image-generation response, saw {:?}",
        first.choice
    );

    let followup = image_model
        .completion_request("Reply with exactly: followup ok")
        .message(Message::Assistant {
            id: None,
            content: first.choice,
        })
        .send()
        .await
        .expect("generated image artifacts should not break follow-up history conversion");

    let text = followup
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_nonempty_response(&text);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY and an OpenRouter image-generation model"]
async fn streaming_generated_image_history_can_be_replayed_on_followup() {
    let client = openrouter::Client::from_env().expect("client should build");
    let image_model = client.completion_model(IMAGE_GENERATION_MODEL);

    let request = image_model
        .completion_request(
            "Generate a simple square icon of a green lighthouse on a white background.",
        )
        .additional_params(generated_image_params())
        .build();
    let mut stream = image_model
        .stream(request)
        .await
        .expect("image-generation stream should start");
    let mut streamed_images = Vec::new();

    while let Some(item) = stream.next().await {
        match item.expect("image-generation stream should not error") {
            StreamedAssistantContent::Artifact(AssistantArtifact::Image(image)) => {
                streamed_images.push(image)
            }
            StreamedAssistantContent::Final(_) => {}
            _ => {}
        }
    }

    assert!(
        !streamed_images.is_empty(),
        "expected generated image events in streaming response"
    );
    assert!(
        !generated_images(&stream.artifacts).is_empty(),
        "expected generated images in final aggregated streaming choice, saw {:?}",
        stream.choice
    );

    let followup = image_model
        .completion_request("Reply with exactly: streaming followup ok")
        .message(Message::Assistant {
            id: None,
            content: stream.choice.clone(),
        })
        .send()
        .await
        .expect("streamed generated image artifacts should not break follow-up history conversion");

    let text = followup
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_nonempty_response(&text);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn user_image_input_still_serializes_for_vision_models() {
    let client = openrouter::Client::from_env().expect("client should build");
    let model = client.completion_model(VISION_MODEL);
    let bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");

    let response = model
        .completion_request(Message::User {
            content: OneOrMany::many(vec![
                UserContent::text("Describe this image in one concise sentence."),
                UserContent::image_base64(
                    base64::prelude::BASE64_STANDARD.encode(bytes),
                    Some(ImageMediaType::JPEG),
                    None,
                ),
            ])
            .expect("user content should be non-empty"),
        })
        .send()
        .await
        .expect("user image input should succeed");

    let text = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_nonempty_response(&text);
}
