//! Contract-vs-reality (rig#2265 / PR #2313 follow-up): Doubleword is bound to
//! `OpenAICompatibleProvider` (`doubleword/client.rs`), whose
//! `REQUEST_ID_HEADER` default is `None` — it does *not* inherit Anthropic's
//! `Some("request-id")` (`anthropic/completion.rs`), and it declares no
//! contract of its own. `provider_request_id` is therefore `None` on every
//! Doubleword turn, which is what these two cells pin.
//!
//! Keeping `None` is a decision, not an oversight, and the reason is worth
//! writing down because the obvious counter-evidence is real: Doubleword *does*
//! stamp `x-request-id` on some responses — but only on some. Measured live,
//! three live requests per model: `openai/gpt-oss-20b`, `gemma-4-31B-it` and
//! `DeepSeek-V4-Flash` sent it on 3 of 3; `gpt-oss-120b`, `Qwen3.5-9B`,
//! `Qwen3.5-397B-A17B-FP8` and `GLM-5.2-FP8` sent it on 0 of 3. The embeddings
//! endpoint sent it on 0 of 10 (though a few recorded embedding fixtures in
//! `embedding_dimensions/` did capture one, so it is intermittent there too).
//!
//! The header is a property of whichever backend serves the request, not of
//! the provider, so declaring the contract would make `provider_request_id`
//! Some-or-None depending on the model — and would reclassify every non-success
//! status from `HttpError` to `ProviderResponse` provider-wide for a header
//! that is absent more often than present. `DEFAULT_MODEL` here is
//! `Qwen/Qwen3.5-9B`, one of the backends that sends none, which is why both
//! fixtures below carry only `content-type`.

use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn blocking_identity_contract_vs_reality() {
    with_doubleword_cassette(
        "response_identity_edge/blocking_identity_contract_vs_reality",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .max_tokens(128)
                .send()
                .await
                .expect("completion should succeed");
            // Derived from this recording's own response headers, which carry
            // only `content-type`: no contract is declared, and the backend
            // serving `DEFAULT_MODEL` stamps no id either, so `None` is what
            // both halves agree on.
            assert_eq!(response.provider_request_id, None);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_identity_contract_vs_reality() {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    with_doubleword_cassette(
        "response_identity_edge/streaming_identity_contract_vs_reality",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .max_tokens(128)
                .stream()
                .await
                .expect("stream should open");
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) =
                    item.expect("stream item should succeed")
                {
                    terminal = Some(final_record);
                }
            }
            let terminal = terminal.expect("terminal record");
            // Derived from the recording, matching the blocking surface.
            assert_eq!(terminal.provider_request_id, None);
        },
    )
    .await;
}
