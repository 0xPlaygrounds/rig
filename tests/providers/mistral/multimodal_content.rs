//! Cassette-backed matrix for Mistral multimodal request content (#2290).
//!
//! `MistralExt::finalize_request_body` used to flatten every message `content`
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
use rig::completion::{CompletionModel, Message};
use rig::message::{Document, DocumentMediaType, DocumentSourceKind, UserContent};
use rig::prelude::*;
use rig::providers::mistral;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response, collect_stream_final_response};

use super::support::with_mistral_multimodal_cassette;

/// Vision-capable and the model every other Mistral fixture already uses.
const VISION_MODEL: &str = mistral::MISTRAL_SMALL;
/// A second, cheaper vision-capable family, so the matrix does not pin the
/// behaviour to one model's request handling.
const SECOND_VISION_MODEL: &str = mistral::MINISTRAL_3B;
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
/// The distinctive half of [`PDF_TOKEN`], for cells run against a smaller
/// model that reliably reads the word but not always the digits after it.
const PDF_TOKEN_PREFIX: &str = "BANANA";

const COLOUR_PROMPT: &str = "What colour fills this image? Answer with one word.";
const DOCUMENT_PROMPT: &str = "What code word appears in the document? Answer with just the word.";
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
        data: DocumentSourceKind::Base64(PDF_BASE64.to_string()),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    })
}

fn speech_audio() -> UserContent {
    let bytes = std::fs::read(AUDIO_FIXTURE_PATH).expect("audio fixture should be readable");
    UserContent::audio(
        base64::engine::general_purpose::STANDARD.encode(bytes),
        Some(rig::message::AudioMediaType::MP3),
    )
}

fn user_message(content: Vec<UserContent>) -> Message {
    Message::User { content }
}

/// Read a recorded cassette back. Called *after* the wrapper returns, because
/// record mode writes the fixture when the cassette finishes — so the same
/// assertion holds whether the cell was just recorded or replayed.
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

/// Assert the recorded request never flattened its content to a plain string.
fn assert_recorded_content_is_an_array(scenario: &str) {
    let recorded = recorded(scenario);
    assert!(
        recorded.contains("\"content\":["),
        "the recorded request for {scenario} must keep its content as a chunk array; a plain \
         string is exactly the flattening this matrix exists to catch"
    );
}

/// The turn's own arithmetic: input + output must be the total the provider
/// reported. An audio turn is where that stops holding if audio tokens are
/// dropped from the input count.
fn assert_usage_adds_up(usage: &rig::completion::Usage) {
    assert!(usage.total_tokens > 0, "the turn must report usage");
    assert_eq!(
        usage.input_tokens + usage.output_tokens,
        usage.total_tokens,
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

/// Assert the recorded request does *not* contain `needle`.
fn assert_recorded_lacks(scenario: &str, needle: &str, why: &str) {
    assert!(
        !recorded(scenario).contains(needle),
        "the recorded request for {scenario} must not contain {needle:?}: {why}"
    );
}

/// Assert exactly `expected` chunks of one type reached the wire.
fn assert_recorded_chunk_count(scenario: &str, chunk_type: &str, expected: usize) {
    let needle = format!("\"type\":\"{chunk_type}\"");
    let found = recorded(scenario).matches(&needle).count();
    assert_eq!(
        found, expected,
        "the recorded request for {scenario} should carry {expected} {chunk_type} chunk(s), \
         found {found}"
    );
}

// =====================================================================
// Images
// =====================================================================

#[tokio::test]
async fn blocking_raw_model_sends_a_base64_image() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_raw_model_sends_a_base64_image",
        |client| async move {
            let model = client.completion_model(VISION_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(user_message(vec![
                            UserContent::text(COLOUR_PROMPT),
                            red_png(),
                        ]))
                        .temperature(0.0)
                        .max_tokens(12)
                        .build(),
                )
                .await?;

            let text = crate::support::assistant_text_response(&response.choice)
                .expect("the turn should carry text");
            assert_mentions(&text, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/blocking_raw_model_sends_a_base64_image",
        "image_url",
    );
    assert_recorded_content_is_an_array(
        "multimodal_content/blocking_raw_model_sends_a_base64_image",
    );
    Ok(())
}

#[tokio::test]
async fn streaming_raw_model_sends_a_base64_image() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_raw_model_sends_a_base64_image",
        |client| async move {
            let agent = client.agent(VISION_MODEL).temperature(0.0).build();
            let mut stream = agent
                .stream_prompt(user_message(vec![
                    UserContent::text(COLOUR_PROMPT),
                    red_png(),
                ]))
                .stream()
                .await;
            let response = collect_stream_final_response(&mut stream).await?;
            assert_mentions(&response, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/streaming_raw_model_sends_a_base64_image",
        "image_url",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_agent_prompt_sends_a_base64_image() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_agent_prompt_sends_a_base64_image",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer with a single word.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text(COLOUR_PROMPT),
                    red_png(),
                ]))
                .await?;
            assert_mentions(&response.output, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/blocking_agent_prompt_sends_a_base64_image",
        "image_url",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_image_only_message_carries_no_text_part() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_only_message_carries_no_text_part",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Name the dominant colour of any image you are shown, in one word.")
                .temperature(0.0)
                .build();
            let response = agent.prompt(user_message(vec![red_png()])).await?;
            assert_mentions(&response.output, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // The pre-fix flattening turned an image-only message into `content: ""`,
    // so this is the cell where the request was most obviously not the
    // caller's.
    assert_recorded_chunk(
        "multimodal_content/blocking_image_only_message_carries_no_text_part",
        "image_url",
    );
    assert_recorded_content_is_an_array(
        "multimodal_content/blocking_image_only_message_carries_no_text_part",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_two_images_in_one_message() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_two_images_in_one_message",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text("How many images did I attach, and what colour are they?"),
                    red_png(),
                    red_png(),
                ]))
                .await?;
            assert_mentions(&response.output, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // Both images must reach the wire; dropping one is the same data loss in
    // miniature.
    assert_recorded_chunk_count(
        "multimodal_content/blocking_two_images_in_one_message",
        "image_url",
        2,
    );
    Ok(())
}

#[tokio::test]
async fn blocking_image_url_reference_is_sent() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_url_reference_is_sent",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text("Describe this image in one short sentence."),
                    UserContent::image_url(
                        "https://docs.mistral.ai/img/eiffel-tower-paris.jpg",
                        None,
                        None,
                    ),
                ]))
                .await?;
            assert_nonempty_response(&response.output);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // The URL form takes a different conversion branch from base64, so it is
    // pinned separately rather than assumed to follow.
    assert_recorded_contains(
        "multimodal_content/blocking_image_url_reference_is_sent",
        "eiffel-tower-paris.jpg",
        "the caller's image URL must reach the wire",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_image_on_a_second_model_family() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_on_a_second_model_family",
        |client| async move {
            let agent = client
                .agent(SECOND_VISION_MODEL)
                .preamble("Answer with a single word.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text(COLOUR_PROMPT),
                    red_png(),
                ]))
                .await?;
            assert_mentions(&response.output, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/blocking_image_on_a_second_model_family",
        "image_url",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_image_survives_a_replayed_history() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_survives_a_replayed_history",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
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
            assert_mentions(&response.output, "red");
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
async fn streaming_image_survives_a_replayed_history() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_image_survives_a_replayed_history",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let history = vec![
                user_message(vec![UserContent::text(COLOUR_PROMPT), red_png()]),
                Message::assistant("Red."),
            ];
            let mut stream = agent
                .stream_chat("Repeat the colour you just named.", history)
                .stream()
                .await;
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

#[tokio::test]
async fn blocking_unicode_text_beside_an_image() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_unicode_text_beside_an_image",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer with a single word.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text("☃ — 何色ですか? What colour fills this image? One word."),
                    red_png(),
                ]))
                .await?;
            assert_mentions(&response.output, "red");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // Text chunks beside an image are re-tagged rather than concatenated, so
    // multi-byte text must survive that path intact.
    assert_recorded_contains(
        "multimodal_content/blocking_unicode_text_beside_an_image",
        "何色",
        "multi-byte text beside an image must reach the wire unmangled",
    );
    Ok(())
}

// =====================================================================
// Documents
// =====================================================================

#[tokio::test]
async fn blocking_raw_model_reads_an_attached_pdf() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_raw_model_reads_an_attached_pdf",
        |client| async move {
            let model = client.completion_model(VISION_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(user_message(vec![
                            UserContent::text(DOCUMENT_PROMPT),
                            pdf_document(),
                        ]))
                        .temperature(0.0)
                        .max_tokens(24)
                        .build(),
                )
                .await?;

            let text = crate::support::assistant_text_response(&response.choice)
                .expect("the turn should carry text");
            assert_mentions(&text, PDF_TOKEN);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // The strongest cell in the matrix: the answer contains a token that only
    // exists inside the attachment, so the document provably reached Mistral.
    assert_recorded_chunk(
        "multimodal_content/blocking_raw_model_reads_an_attached_pdf",
        "document_url",
    );
    Ok(())
}

#[tokio::test]
async fn streaming_agent_reads_an_attached_pdf() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_agent_reads_an_attached_pdf",
        |client| async move {
            let agent = client.agent(VISION_MODEL).temperature(0.0).build();
            let mut stream = agent
                .stream_prompt(user_message(vec![
                    UserContent::text(DOCUMENT_PROMPT),
                    pdf_document(),
                ]))
                .stream()
                .await;
            let response = collect_stream_final_response(&mut stream).await?;
            assert_mentions(&response, PDF_TOKEN);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/streaming_agent_reads_an_attached_pdf",
        "document_url",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_agent_reads_an_attached_pdf() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_agent_reads_an_attached_pdf",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer with just the word.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text(DOCUMENT_PROMPT),
                    pdf_document(),
                ]))
                .await?;
            assert_mentions(&response.output, PDF_TOKEN);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    let scenario = "multimodal_content/blocking_agent_reads_an_attached_pdf";
    assert_recorded_chunk(scenario, "document_url");
    // Mistral's document chunk has its own optional `document_name`; the
    // filename the shared conversion attaches is carried there, not dropped.
    assert_recorded_contains(
        scenario,
        "\"document_name\":\"document.pdf\"",
        "the document's filename must reach Mistral's own `document_name` field",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_document_only_message_carries_no_text_part() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_document_only_message_carries_no_text_part",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Reply with the single code word found in any attached document.")
                .temperature(0.0)
                .build();
            let response = agent.prompt(user_message(vec![pdf_document()])).await?;
            assert_mentions(&response.output, PDF_TOKEN);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    let scenario = "multimodal_content/blocking_document_only_message_carries_no_text_part";
    assert_recorded_content_is_an_array(scenario);
    assert_recorded_chunk(scenario, "document_url");
    assert_recorded_chunk_count(scenario, "text", 0);
    Ok(())
}

#[tokio::test]
async fn blocking_document_and_image_in_one_message() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_document_and_image_in_one_message",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
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
            assert_mentions(&response.output, PDF_TOKEN);
            assert_mentions(&response.output, "red");
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

#[tokio::test]
async fn blocking_document_on_a_second_model_family() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_document_on_a_second_model_family",
        |client| async move {
            let agent = client
                .agent(SECOND_VISION_MODEL)
                .preamble("Answer with just the word.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text(DOCUMENT_PROMPT),
                    pdf_document(),
                ]))
                .await?;
            assert_mentions(&response.output, PDF_TOKEN_PREFIX);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_chunk(
        "multimodal_content/blocking_document_on_a_second_model_family",
        "document_url",
    );
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
            let model = client.completion_model(AUDIO_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(user_message(vec![
                            UserContent::text(AUDIO_PROMPT),
                            speech_audio(),
                        ]))
                        .temperature(0.0)
                        .max_tokens(48)
                        .build(),
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
            let agent = client.agent(AUDIO_MODEL).temperature(0.0).build();
            let mut stream = agent
                .stream_prompt(user_message(vec![
                    UserContent::text(AUDIO_PROMPT),
                    speech_audio(),
                ]))
                .stream()
                .await;
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

#[tokio::test]
async fn blocking_agent_sends_audio() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_agent_sends_audio",
        |client| async move {
            let agent = client.agent(AUDIO_MODEL).temperature(0.0).build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text(AUDIO_PROMPT),
                    speech_audio(),
                ]))
                .await?;
            assert_mentions(&response.output, AUDIO_KEYWORD);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    let scenario = "multimodal_content/blocking_agent_sends_audio";
    assert_recorded_chunk(scenario, "input_audio");
    // Mistral's audio chunk documents a bare base64 string. The OpenAI-shaped
    // `{data, format}` object is accepted today (the server flattens it and
    // ignores `format`), but the documented form is what rig sends — and a
    // `format` placed beside `input_audio` is rejected outright.
    assert_recorded_lacks(
        scenario,
        "\"format\":\"mp3\"",
        "the audio chunk must not carry OpenAI's sibling `format`",
    );
    Ok(())
}

// =====================================================================
// Controls — the text-only shape must not move
// =====================================================================

#[tokio::test]
async fn blocking_text_only_content_still_flattens_to_a_string() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_text_only_content_still_flattens_to_a_string",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text("Name the capital of France."),
                    UserContent::text(" Answer with one word."),
                ]))
                .await?;
            assert_mentions(&response.output, "paris");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // The control for every cell above: text-only content keeps Mistral's
    // plain-string form, so the fix cannot have changed the shape of the
    // requests every other Mistral fixture already pins.
    assert_recorded_contains(
        "multimodal_content/blocking_text_only_content_still_flattens_to_a_string",
        "\"content\":\"Name the capital of France. Answer with one word.\"",
        "text-only content must still be joined into one plain string",
    );
    Ok(())
}

#[tokio::test]
async fn streaming_text_only_content_still_flattens_to_a_string() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_text_only_content_still_flattens_to_a_string",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer in one short sentence.")
                .temperature(0.0)
                .build();
            let mut stream = agent
                .stream_prompt(user_message(vec![
                    UserContent::text("Name the capital of France."),
                    UserContent::text(" Answer with one word."),
                ]))
                .stream()
                .await;
            let response = collect_stream_final_response(&mut stream).await?;
            assert_mentions(&response, "paris");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    assert_recorded_contains(
        "multimodal_content/streaming_text_only_content_still_flattens_to_a_string",
        "\"content\":\"Name the capital of France. Answer with one word.\"",
        "text-only content must still be joined into one plain string on the streaming path too",
    );
    Ok(())
}

#[tokio::test]
async fn blocking_text_document_still_flattens_into_the_prompt() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_text_document_still_flattens_into_the_prompt",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Answer with just the word.")
                .temperature(0.0)
                .build();
            let response = agent
                .prompt(user_message(vec![
                    UserContent::text("What code word is in these notes?"),
                    UserContent::document("# Notes\nThe code word is WALRUS-4412.", None),
                ]))
                .await?;
            assert_mentions(&response.output, "WALRUS-4412");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await?;

    // A document with no media type converts to a *text* part upstream, so it
    // is still flattened — the fix must not promote it to a chunk array.
    assert_recorded_contains(
        "multimodal_content/blocking_text_document_still_flattens_into_the_prompt",
        "WALRUS-4412",
        "a text document must still reach the prompt",
    );
    assert_recorded_lacks(
        "multimodal_content/blocking_text_document_still_flattens_into_the_prompt",
        "\"type\":\"document_url\"",
        "a text document is not a Mistral document chunk; it flattens into the prompt text",
    );
    Ok(())
}

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
async fn blocking_image_with_a_tool_configured() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/blocking_image_with_a_tool_configured",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
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
            assert_mentions(&response.output, "red");
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
async fn streaming_image_with_a_tool_configured() -> Result<()> {
    with_mistral_multimodal_cassette(
        "multimodal_content/streaming_image_with_a_tool_configured",
        |client| async move {
            let agent = client
                .agent(VISION_MODEL)
                .preamble("Call record_colour with the dominant colour of the attached image.")
                .tool(RecordColour)
                .temperature(0.0)
                .default_max_turns(3)
                .build();
            let mut stream = agent
                .stream_prompt(user_message(vec![
                    UserContent::text("Record this image's colour."),
                    red_png(),
                ]))
                .stream()
                .await;
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
