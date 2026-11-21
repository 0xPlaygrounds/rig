//! OpenAI Chat video input against BigModel's compatible endpoint.
//!
//! Replay needs no credentials. Recording uses `ZHIPU_API_KEY` and
//! `glm-4.1v-thinking-flash`.

use std::panic::AssertUnwindSafe;

use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::FutureExt;
use rig::{
    Model,
    catalog::{ModelFacts, ModelSpec},
    completion::{CompletionRequest, Message},
    message::{UserContent, VideoMediaType},
    providers::{openai::OpenAIConfig, registry::ProviderId},
};

use crate::cassettes::{CassetteSpec, ProviderCassette, RecordVia};

const MODEL: &str = "glm-4.1v-thinking-flash";
const BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";

#[tokio::test]
async fn bound_video_facts_send_a_url() -> Result<()> {
    with_openai_video_cassette(
        "openai_compatible/video_url",
        "Describe the visible action in this video briefly in English.",
        UserContent::video_url("https://cdn.bigmodel.cn/agent-demos/lark/113123.mov", None),
    )
    .await
}

#[tokio::test]
async fn bound_video_facts_send_typed_base64() -> Result<()> {
    let video = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/media/color-sequence.mp4"
    ))?;
    with_openai_video_cassette(
        "openai_compatible/video_base64",
        "Name the background colors in this video in temporal order. Answer briefly in English.",
        UserContent::video_base64(STANDARD.encode(video), Some(VideoMediaType::MP4)),
    )
    .await
}

async fn with_openai_video_cassette(
    scenario: &'static str,
    prompt: &str,
    video: UserContent,
) -> Result<()> {
    let provider = ProviderId::catalog("openai").context("OpenAI dialect is registered")?;
    let mut spec = ModelSpec::new(provider, MODEL);
    spec.input.video = true;
    let cassette = ProviderCassette::start_via(
        RecordVia::Direct,
        &crate::cassettes::cassette_root(),
        "openai",
        CassetteSpec::new(scenario).exact_matched(),
        BASE_URL,
    )
    .await;
    let wire = OpenAIConfig::new(cassette.api_key("ZHIPU_API_KEY"))
        .with_base_url(cassette.base_url())
        .chat(MODEL)
        .with_facts(ModelFacts::new(spec));
    let model = Model::new(wire, cassette.direct_http());
    let request = CompletionRequest::new(Message::User {
        content: vec![UserContent::text(prompt), video],
    })
    .max_tokens(1024);
    let mut text = String::new();
    let result = AssertUnwindSafe(async {
        text = model.call(request).await?.text();
        ensure!(!text.trim().is_empty(), "video response is empty");
        Ok::<(), anyhow::Error>(())
    })
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await?;

    let recorded = crate::cassettes::recorded_json_response("openai", scenario);
    let expected = recorded
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .context("recorded video response contains text")?;
    ensure!(
        text == expected,
        "decoded video response differs from the recording"
    );
    Ok(())
}
