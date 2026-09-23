//! Groq model-constant matrix (rig#2424).
//!
//! **Bug.** Every chat constant in `rig::providers::groq` named a model Groq
//! had retired (`mixtral-8x7b-32768`, `llama3-*-8192`, the `llama-3.2-*`
//! previews, `gemma2-9b-it`, `deepseek-r1-distill-llama-70b`, …), so each one
//! 404'd with `model_not_found`. The fix replaces them with the catalog at
//! <https://console.groq.com/docs/models>.
//!
//! These cells pin the constants to what Groq actually served at record time
//! rather than to the docs page: the listing cell reads `GET /models` back and
//! requires every non-enterprise constant to appear, and one completion cell
//! per production model proves the id is accepted on `/chat/completions`.
//!
//! | # | cell | surface | constant |
//! |---|------|---------|----------|
//! | 1 | `catalog_lists_current_constants` | `GET /models` | every non-enterprise chat + whisper constant |
//! | 2 | `gpt_oss_120b_completion_smoke` | chat | `GPT_OSS_120B` |
//! | 3 | `gpt_oss_20b_completion_smoke` | chat | `GPT_OSS_20B` |
//! | 4 | `qwen3_8_27b_completion_smoke` | chat | `QWEN3_8_27B` |
//!
//! `LLAMA_3_1_8B_INSTANT`, `LLAMA_3_3_70B_VERSATILE` and `MINIMAX_M2_7` are
//! enterprise-contract models after 2026-08-16 and are not required in the
//! listing, which is recorded from a developer-tier key. The compound systems
//! (`groq/compound`, `groq/compound-mini`) left the catalog entirely and their
//! constants were removed.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use anyhow::Result;
use rig::completion::CompletionModel;
use rig::model::ModelLister;
use rig::providers::groq;

use super::support::{BoundGroq, with_groq_cassette_result};
use crate::support::{assert_nonempty_response, assistant_text_response};

const PROMPT: &str = "Reply with the single word OK.";

/// Constants a developer-tier key must see in the catalog.
const PUBLIC_CONSTANTS: &[&str] = &[
    groq::GPT_OSS_120B,
    groq::GPT_OSS_20B,
    groq::GPT_OSS_SAFEGUARD_20B,
    groq::QWEN3_8_27B,
    groq::WHISPER_LARGE_V3,
    groq::WHISPER_LARGE_V3_TURBO,
];

#[tokio::test]
async fn catalog_lists_current_constants() -> Result<()> {
    with_groq_cassette_result(
        "constants_matrix/catalog_lists_current_constants",
        |client| async move {
            let models = client.models().list_all().await?;
            let served: Vec<&str> = models.data.iter().map(|model| model.id.as_str()).collect();
            let missing: Vec<&str> = PUBLIC_CONSTANTS
                .iter()
                .copied()
                .filter(|constant| !served.contains(constant))
                .collect();
            anyhow::ensure!(
                missing.is_empty(),
                "every public groq constant must be in the served catalog; missing {missing:?}, served {served:?}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

async fn assert_completion_smoke(client: BoundGroq, model_id: &str) -> Result<()> {
    let model = client.completion(model_id);
    let request = model.completion_request(PROMPT).max_tokens(64).build();
    let response = model.completion(request).await?;
    let text = assistant_text_response(&response.choice)
        .ok_or_else(|| anyhow::anyhow!("{model_id} should answer with text"))?;
    assert_nonempty_response(&text);
    Ok(())
}

#[tokio::test]
async fn gpt_oss_120b_completion_smoke() -> Result<()> {
    with_groq_cassette_result(
        "constants_matrix/gpt_oss_120b_completion_smoke",
        |client| async move { assert_completion_smoke(client, groq::GPT_OSS_120B).await },
    )
    .await
}

#[tokio::test]
async fn gpt_oss_20b_completion_smoke() -> Result<()> {
    with_groq_cassette_result(
        "constants_matrix/gpt_oss_20b_completion_smoke",
        |client| async move { assert_completion_smoke(client, groq::GPT_OSS_20B).await },
    )
    .await
}

#[tokio::test]
async fn qwen3_8_27b_completion_smoke() -> Result<()> {
    with_groq_cassette_result(
        "constants_matrix/qwen3_8_27b_completion_smoke",
        |client| async move { assert_completion_smoke(client, groq::QWEN3_8_27B).await },
    )
    .await
}
