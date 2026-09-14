//! Matrix O's mock cells: a rerank through a mock `RerankModel` behind a
//! `RerankAdapter` on a host's bus (no keyed provider in the tree has a
//! rerank cassette suite), and a `Prompted` answer the run returns
//! unvalidated.

use rig::agent::AgentBuilder;
use rig::bus::Bus;
use rig::effect::{EffectFamily, HandlerKey};
use rig::run::OutputMode;
use rig::test_utils::MockCompletionModel;

use crate::goldens::{MockRerank, RERANK_KEY, RerankDocs, event_schema, families};

const PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const PROMPT: &str = "Reply with the single word: ready.";

/// The prompt reranks two documents through the host's reranker before
/// the completion: `[Rerank, Completion]`.
#[tokio::test]
async fn oracle_rerank_effect_log_is_the_golden_fixture() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            rig::serve::ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                MockCompletionModel::text("ready"),
            )),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(RERANK_KEY),
            rig::serve::ErasedHandler::new(rig::serve::adapters::RerankAdapter::new(
                "host", MockRerank,
            )),
        )
        .expect("a fresh key");
    let recorder = rig::effect_log::EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(PREAMBLE)
        .add_hook(RerankDocs)
        .build();
    let output = agent
        .prompt(PROMPT)
        .await
        .expect("the agent answers")
        .output;
    assert_eq!(output, "ready");
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(
        families(&log),
        [EffectFamily::Rerank, EffectFamily::Completion]
    );
    assert!(
        !log.header
            .required
            .contains_key(&HandlerKey::from(RERANK_KEY)),
        "the host's reranker is not in the agent's row"
    );
    crate::goldens::golden_effects("mock_oracle_rerank", &log);
}

/// A `Prompted` answer the model gives as prose: the run returns it as it
/// is — the consumer's deserialization validates a prompted answer, the
/// run does not.
#[tokio::test]
async fn oracle_prompted_unvalidated_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::text("not an object"))
        .name("golden")
        .preamble(PREAMBLE)
        .output_schema_raw(event_schema())
        .output_mode(OutputMode::Prompted)
        .record_effects()
        .build();
    let output = agent
        .prompt("Return a concise event object for a local Rust meetup in Seattle.")
        .await
        .expect("the run does not validate a prompted answer")
        .output;
    assert_eq!(output, "not an object");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_oracle_prompted_unvalidated", &log);
}
