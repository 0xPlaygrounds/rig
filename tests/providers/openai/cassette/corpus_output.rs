//! Matrix H of the effect corpus, the OpenAI cells: `Tool` and `Prompted`
//! output modes on the Responses wire (`gpt-4o`, temperature 0), where the
//! output tool's call carries a dual id. Every cell is a new recording
//! under `tests/cassettes/openai/corpus_output/`.

use rig::effect::{EffectFamily, EffectKind};
use rig::prelude::*;
use rig::providers::openai;
use rig::run::OutputMode;

use super::super::support::with_openai_corpus_output_cassette;
use crate::goldens::{event_schema, families};
use crate::support::{BASIC_PREAMBLE, STRUCTURED_OUTPUT_PROMPT};

fn assert_event(output: &str) {
    let object: serde_json::Value =
        serde_json::from_str(output).unwrap_or_else(|_| panic!("the schema's object: {output}"));
    assert!(
        object["title"].is_string() && object["summary"].is_string(),
        "{object}"
    );
}

fn tool_names(log: &rig::effect_log::EffectLog) -> Vec<String> {
    match &log.records[0].kind {
        EffectKind::Completion { request, .. } => {
            request.tools.iter().map(|tool| tool.name.clone()).collect()
        }
        other => panic!("a completion, not {other:?}"),
    }
}

#[tokio::test]
async fn tool_unary_effect_log_is_the_golden_fixture() {
    with_openai_corpus_output_cassette("corpus_output/tool_unary", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Tool)
            .record_effects()
            .build();
        let response = agent
            .prompt(STRUCTURED_OUTPUT_PROMPT)
            .await
            .expect("the agent answers");
        assert_event(&response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert_eq!(tool_names(&log), ["final_result"]);
        crate::goldens::golden_effects("openai_output_tool_unary", &log);
    })
    .await;
}

#[tokio::test]
async fn prompted_unary_effect_log_is_the_golden_fixture() {
    with_openai_corpus_output_cassette("corpus_output/prompted_unary", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Prompted)
            .record_effects()
            .build();
        let response = agent
            .prompt(STRUCTURED_OUTPUT_PROMPT)
            .await
            .expect("the agent answers");
        assert_event(&response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(tool_names(&log).is_empty());
        crate::goldens::golden_effects("openai_output_prompted_unary", &log);
    })
    .await;
}
