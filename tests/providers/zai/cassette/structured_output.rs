//! Structured-output coverage for Z.AI's general endpoint.
//!
//! Z.AI documents `response_format` as an object whose `type` is
//! `text | json_object`; there is no `json_schema` member, so rig does not map
//! `output_schema` onto `response_format` for Z.AI (`ZAiExt`'s
//! `SUPPORTS_RESPONSE_FORMAT = false`) — it drops the schema with a warning.
//!
//! Be precise about what that leaves. `prompt_typed` pins `OutputMode::Native`
//! unconditionally (`rig-agent/src/agent/prompt_request/mod.rs`), and Native is
//! the mode that would have carried the schema to the provider — so a typed
//! prompt on Z.AI is now *unconstrained*, exactly as on every other provider
//! with this flag false, and whether GLM returns conforming JSON anyway is a
//! model behavior these cells observe rather than a guarantee rig enforces.
//! Tool-mode enforcement covers only the untyped `output_schema`-plus-tools
//! path. The load-bearing assertion here is therefore the request boundary.

use rig::completion::TypedPrompt;
use rig::prelude::*;
use rig_agent::test_utils::validate_extraction_fields;

use super::super::CHEAP_GENERAL_MODEL;
use super::super::support::{recorded_request_body, with_zai_general_cassette};
use crate::support::{
    EXTRACTOR_TEXT, STRUCTURED_OUTPUT_PROMPT, SmokePerson, SmokeStructuredOutput,
    assert_smoke_structured_output,
};

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_structured_output_native_blocking() {
    with_zai_general_cassette(
        "general/structured_output_native_blocking",
        |client| async move {
            let response: SmokeStructuredOutput = client
                .agent(CHEAP_GENERAL_MODEL)
                .build()
                .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("Z.AI structured output prompt should succeed");

            // An observation, not a guarantee: nothing constrained this turn
            // (see the module doc). If a recording ever shows GLM answering
            // with prose here, that is the finding, and the remedy is
            // `OutputMode::Prompted`, not re-enabling `response_format`.
            assert_smoke_structured_output(&response);
        },
    )
    .await;

    // A schema with no tools in play is the only shape that would put
    // `response_format` on turn 1 (the shared layer defers it whenever tools
    // are present without a tool result), so this is where a regression back
    // to the OpenAI `json_schema` block would show.
    let request = recorded_request_body("general/structured_output_native_blocking");
    assert!(
        request.get("response_format").is_none(),
        "Z.AI accepts only text/json_object response formats; request was {request}"
    );
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_extractor_blocking() {
    with_zai_general_cassette("general/extractor_blocking", |client| async move {
        let response = client
            .extractor::<SmokePerson>(CHEAP_GENERAL_MODEL)
            .build()
            .extract_with_usage(EXTRACTOR_TEXT)
            .await
            .expect("Z.AI extractor request should succeed");

        validate_extraction_fields(
            "zai_extractor_blocking",
            response.data.first_name.as_deref(),
            response.data.last_name.as_deref(),
            response.data.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");
    })
    .await;

    // Every rig extractor forces `tool_choice: "required"`, and Z.AI's API
    // reference documents `tool_choice` as a one-member enum (`auto`). The
    // recording is what settles whether the reference is merely terse or Z.AI
    // really rejects the value; pin what rig sends so the recorded answer is
    // attributable to it.
    let request = recorded_request_body("general/extractor_blocking");
    assert_eq!(
        request["tool_choice"], "required",
        "the extractor path forces a required tool choice; request was {request}"
    );
}
