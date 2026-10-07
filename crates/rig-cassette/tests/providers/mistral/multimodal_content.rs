//! Cassette-backed matrix for Mistral multimodal request content (#2290).
//!
//! `Mistral::finalize_request_body` used to flatten every message `content`
//! array with the text-only helper, which keeps only parts carrying a
//! `text`/`refusal` key. Every image, audio clip and document attached to a
//! Mistral prompt was therefore removed from the request before it was sent,
//! and the caller got an ordinary completion answering a prompt it never made.
//!
//! What makes a regressed cell fail is the cassette harness itself: it matches
//! each outbound request against the recorded one, so a request that stopped
//! carrying its chunk misses the fixture and 404s. The `assert_recorded_*`
//! helpers below read the committed YAML, which in replay is a constant — they
//! do not police the code under test. Their job is at *record* time, where they
//! fail a cell whose freshly recorded request does not carry the shape the cell
//! is named for, so a cassette can never be committed that covers nothing.

use anyhow::Result;
use base64::Engine as _;
use rig::completion::Message;
use rig::message::{Document, DocumentMediaType, DocumentSourceKind, UserContent};
use rig::providers::mistral;

use crate::support::{AUDIO_FIXTURE_PATH, collect_stream_final_response};

use super::support::with_mistral_multimodal_cassette;
use rig::completion::CompletionRequest;

/// Vision-capable and the model every other Mistral fixture already uses.
const VISION_MODEL: &str = mistral::MISTRAL_SMALL;
/// The only Mistral family that reports `capabilities.audio`. Every other
/// model silently substitutes a server-side transcript, so an audio cell run
/// against one would pass without proving the audio chunk was understood.
const AUDIO_MODEL: &str = "voxtral-small-latest";

/// A 64×64 PNG filled with one flat colour (#DC1414). Small enough to embed in
/// a fixture, unambiguous enough that "what colour is this" has one answer.
const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsAtx/ZMZxgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGBywKqxUDxqh7TUQAAAABJRU5ErkJggg==";

/// A 585-byte single-page PDF whose only text is the token below. A made-up
/// token, so an answer containing it can only have come from the attachment
/// rather than from the model's own knowledge.
const PDF_BASE64: &str = "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCAyMDAgMTAwXSAvQ29udGVudHMgNCAwIFIgL1Jlc291cmNlcyA8PCAvRm9udCA8PCAvRjEgNSAwIFIgPj4gPj4gPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA0MiA+PgpzdHJlYW0KQlQgL0YxIDE4IFRmIDIwIDUwIFRkIChCQU5BTkEtNzM5MSkgVGogRVQKZW5kc3RyZWFtCmVuZG9iago1IDAgb2JqCjw8IC9UeXBlIC9Gb250IC9TdWJ0eXBlIC9UeXBlMSAvQmFzZUZvbnQgL0hlbHZldGljYSA+PgplbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDExNyAwMDAwMCBuIAowMDAwMDAwMjU0IDAwMDAwIG4gCjAwMDAwMDAzNDYgMDAwMDAgbiAKdHJhaWxlcgo8PCAvU2l6ZSA2IC9Sb290IDEgMCBSID4+CnN0YXJ0eHJlZgo0MjkKJSVFT0YK";

/// The token embedded in [`PDF_BASE64`].
const PDF_TOKEN: &str = "BANANA-7391";

const COLOUR_PROMPT: &str = "What colour fills this image? Answer with one word.";
const AUDIO_PROMPT: &str = "Transcribe the audio verbatim.";
/// A distinctive word from the sentence spoken in
/// `tests/data/en-us-natural-speech.mp3`: "The sun was setting slowly, casting
/// long shadows across the empty field." Derived from the recorded
/// transcription, not assumed.
const AUDIO_KEYWORD: &str = "shadows";

fn red_png() -> UserContent {
    UserContent::image_base64(
        RED_PNG_BASE64,
        Some(rig::message::ImageMediaType::PNG),
        None,
    )
}

fn pdf_document() -> UserContent {
    UserContent::Document(Document {
        data: DocumentSourceKind::Base64(PDF_BASE64.to_string()).into(),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    })
}

fn speech_audio() -> UserContent {
    let bytes = std::fs::read(AUDIO_FIXTURE_PATH).expect("audio fixture should be readable");
    UserContent::audio_base64(
        base64::engine::general_purpose::STANDARD.encode(bytes),
        Some(rig::message::AudioMediaType::MP3),
    )
}

fn user_message(content: Vec<UserContent>) -> Message {
    Message::User { content }
}

fn recorded(scenario: &str) -> String {
    let path = crate::cassettes::cassette_path("mistral", scenario);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()))
}

/// Assert the recorded request carries the chunk this cell is about. Bodies are
/// stored as canonical JSON, so the discriminator survives key sorting.
fn assert_recorded_chunk(scenario: &str, chunk_type: &str) {
    let recorded = recorded(scenario);
    let needle = format!("\"type\":\"{chunk_type}\"");
    assert!(
        recorded.contains(&needle),
        "the recorded request for {scenario} must carry a {needle} chunk; without it this cell \
         asserts nothing about the content it claims to send"
    );
}

/// The turn's own arithmetic: input + output must be the total the provider
/// reported. An audio turn is where that stops holding if audio tokens are
/// dropped from the input count.
fn assert_usage_adds_up(usage: &rig::completion::Usage) {
    let total_tokens = usage.total_tokens.unwrap_or(0);
    assert!(total_tokens > 0, "the turn must report usage");
    assert_eq!(
        usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0),
        total_tokens,
        "input + output must equal the total Mistral reported: {usage:?}"
    );
}

fn assert_mentions(response: &str, expected: &str) {
    assert!(
        response.to_lowercase().contains(&expected.to_lowercase()),
        "response should mention {expected:?}, got {response:?}"
    );
}

/// Assert the recorded request contains `needle`. `why` states what the
/// presence proves, so a failure reads as the broken guarantee rather than a
/// missing substring.
fn assert_recorded_contains(scenario: &str, needle: &str, why: &str) {
    assert!(
        recorded(scenario).contains(needle),
        "the recorded request for {scenario} must contain {needle:?}: {why}"
    );
}

// =====================================================================
// Images
// =====================================================================

#[tokio::test]
#[ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
async fn blocking_image_survives_a_replayed_history() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_survives_a_replayed_history",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let mut history = vec![
                user_message(vec![UserContent::text(COLOUR_PROMPT), red_png()]),
                Message::assistant("Red."),
            ];
            let response = agent
                .chat("Repeat the colour you just named.", &mut history)
                .await?;
            assert_mentions(&response.output(), "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // History messages go through the same finalization, so an image replayed
    // as context must survive it too.
    assert_recorded_chunk(
        "multimodal_content/blocking_image_survives_a_replayed_history",
        "image_url",
    );
    Ok(())
}

#[tokio::test]
#[ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
async fn streaming_image_survives_a_replayed_history() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_image_survives_a_replayed_history",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let history = vec![
                user_message(vec![UserContent::text(COLOUR_PROMPT), red_png()]),
                Message::assistant("Red."),
            ];
            let mut stream = agent
                .prompt("Repeat the colour you just named.")
                .history(history)
                .stream();
            let response = collect_stream_final_response(&mut stream).await?;
            assert_mentions(&response, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/streaming_image_survives_a_replayed_history",
        "image_url",
    );
    Ok(())
}

// =====================================================================
// Documents
// =====================================================================

#[tokio::test]
async fn blocking_document_and_image_in_one_message() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_document_and_image_in_one_message",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text(
                        "Give the document's code word and the image's colour, in one sentence.",
                    ),
                    pdf_document(),
                    red_png(),
                ]))
                .await?;
            assert_mentions(&response.output(), PDF_TOKEN);
            assert_mentions(&response.output(), "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // Two different chunk kinds in one array: each is converted independently,
    // so a mapping that only handled the first would fail here.
    let scenario = "multimodal_content/blocking_document_and_image_in_one_message";
    assert_recorded_chunk(scenario, "document_url");
    assert_recorded_chunk(scenario, "image_url");
    Ok(())
}

// =====================================================================
// Audio
// =====================================================================

#[tokio::test]
async fn blocking_raw_model_sends_audio() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_raw_model_sends_audio",
        |client| async move {
            let model = client.completion(AUDIO_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(user_message(vec![
                        UserContent::text(AUDIO_PROMPT),
                        speech_audio(),
                    ]))
                    .temperature(0.0)
                    .max_tokens(48),
                )
                .await?;

            let text = crate::support::assistant_text_response(&response.choice)
                .expect("the turn should carry text");
            assert_mentions(&text, AUDIO_KEYWORD);
            // Mistral bills audio outside `prompt_tokens`, so counting only
            // that field leaves the parts short of the total it reported.
            assert_usage_adds_up(&response.usage);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/blocking_raw_model_sends_audio",
        "input_audio",
    );
    Ok(())
}

#[tokio::test]
async fn streaming_agent_sends_audio() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_agent_sends_audio",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(AUDIO_MODEL))
                .temperature(0.0)
                .build();
            let mut stream = agent
                .prompt(user_message(vec![
                    UserContent::text(AUDIO_PROMPT),
                    speech_audio(),
                ]))
                .stream();
            let response = collect_stream_final_response(&mut stream).await?;
            assert_mentions(&response, AUDIO_KEYWORD);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/streaming_agent_sends_audio",
        "input_audio",
    );
    Ok(())
}

// =====================================================================
// Controls — the text-only shape must not move
// =====================================================================

// =====================================================================
// Tools alongside multimodal content
// =====================================================================

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct RecordColourArgs {
    colour: String,
}

#[derive(Debug, thiserror::Error)]
#[error("multimodal tool error")]
struct MultimodalToolError;

/// A tool whose only job is to give the turn something to call, so a cell can
/// prove content chunks and tool wiring survive the same finalization.
#[derive(Clone)]
struct RecordColour;

impl rig::tool::Tool for RecordColour {
    const NAME: &'static str = "record_colour";
    type Error = MultimodalToolError;
    type Args = RecordColourArgs;
    type Output = String;

    fn description(&self) -> String {
        "Record the dominant colour of an image the user attached.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"colour": {"type": "string"}},
            "required": ["colour"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!("recorded {}", args.colour))
    }
}

#[tokio::test]
#[ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
async fn blocking_image_with_a_tool_configured() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_with_a_tool_configured",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
                .preamble("Call record_colour with the dominant colour of the attached image.")
                .tool(RecordColour)
                .temperature(0.0)
                .default_max_turns(3)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text("Record this image's colour."),
                    red_png(),
                ]))
                .await?;
            assert_mentions(&response.output(), "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // `finalize_request_body` rewrites tool_choice in the same pass that now
    // renders content chunks; this pins that the two do not interfere.
    let scenario = "multimodal_content/blocking_image_with_a_tool_configured";
    assert_recorded_chunk(scenario, "image_url");
    assert_recorded_contains(
        scenario,
        "record_colour",
        "the tool must still reach the wire beside the image",
    );
    Ok(())
}

#[tokio::test]
#[ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
async fn streaming_image_with_a_tool_configured() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_image_with_a_tool_configured",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(VISION_MODEL))
                .preamble("Call record_colour with the dominant colour of the attached image.")
                .tool(RecordColour)
                .temperature(0.0)
                .default_max_turns(3)
                .build();
            let mut stream = agent
                .prompt(user_message(vec![
                    UserContent::text("Record this image's colour."),
                    red_png(),
                ]))
                .stream();
            let response = collect_stream_final_response(&mut stream).await?;
            assert_mentions(&response, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    let scenario = "multimodal_content/streaming_image_with_a_tool_configured";
    assert_recorded_chunk(scenario, "image_url");
    assert_recorded_contains(
        scenario,
        "record_colour",
        "the tool must still reach the wire beside the image on the streaming path",
    );
    Ok(())
}
