//! Wire-sequence conformance suite over the provider streaming pipelines.
//!
//! Scenarios live in `rig_core::test_utils::streaming_conformance`; this file
//! instantiates each sequence family per provider fixture. Every family pins a
//! shipped bug from the #2257 review rounds — the scenario doc comments cite
//! the `rig-2257-code-review-findings-*.md` entry.
//!
//! Run the suite with:
//! `cargo test -p rig --test core core::streaming_conformance`

use rig_core::test_utils::streaming_conformance::{
    self as conformance,
    fixtures::{cohere, gemini_rest, ollama, openai_chat, openai_responses},
};

macro_rules! conformance_test {
    ($name:ident, $fixture:path, $scenario:path) => {
        #[tokio::test]
        async fn $name() {
            let fixture = $fixture();
            $scenario(&fixture).await.expect("scenario should hold");
        }
    };
}

macro_rules! conformance_suite {
    ($module:ident, $fixture:path) => {
        mod $module {
            use super::*;

            conformance_test!(
                truncation_preserves_content_without_terminal,
                $fixture,
                conformance::truncation_preserves_content_without_terminal
            );
            conformance_test!(
                transport_error_after_tool_call_yields_err_then_end,
                $fixture,
                conformance::transport_error_after_tool_call_yields_err_then_end
            );
            conformance_test!(
                malformed_frame_surfaces_err_and_terminal_still_completes,
                $fixture,
                conformance::malformed_frame_surfaces_err_and_terminal_still_completes
            );
            conformance_test!(
                unknown_event_is_skipped,
                $fixture,
                conformance::unknown_event_is_skipped
            );
            conformance_test!(
                delta_less_choice_prelude_is_a_noop,
                $fixture,
                conformance::delta_less_choice_prelude_is_a_noop
            );
            conformance_test!(
                refusal_frames_deliver_text_without_error,
                $fixture,
                conformance::refusal_frames_deliver_text_without_error
            );
            conformance_test!(
                bare_terminal_after_only_unparseable_frames_fabricates_nothing,
                $fixture,
                conformance::bare_terminal_after_only_unparseable_frames_fabricates_nothing
            );
            conformance_test!(
                usage_variants_are_reported_or_zero_sentinel,
                $fixture,
                conformance::usage_variants_are_reported_or_zero_sentinel
            );
        }
    };
}

conformance_suite!(openai_chat_suite, openai_chat::fixture);
conformance_suite!(openai_responses_suite, openai_responses::fixture);
conformance_suite!(gemini_rest_suite, gemini_rest::fixture);
conformance_suite!(cohere_suite, cohere::fixture);
conformance_suite!(ollama_suite, ollama::fixture);

// The defective-known-payload family. For the OpenAI Responses fixture it
// pins the P2 in `rig-2257-code-review-findings-34ee8ba5.md` ("Round-5
// known-type strictness silently reverted for content parts"), fixed by the
// hand-written tag dispatch on `ContentPartChunkPart`: a known part tag with
// a malformed payload classifies as `Corrupt` and surfaces an `Err` item.
mod defective_known_event {
    use super::*;

    conformance_test!(
        openai_chat,
        openai_chat::fixture,
        conformance::defective_known_event_surfaces_err
    );
    conformance_test!(
        openai_responses,
        openai_responses::fixture,
        conformance::defective_known_event_surfaces_err
    );
    conformance_test!(
        gemini_rest,
        gemini_rest::fixture,
        conformance::defective_known_event_surfaces_err
    );
    conformance_test!(
        cohere,
        cohere::fixture,
        conformance::defective_known_event_surfaces_err
    );
    conformance_test!(
        ollama,
        ollama::fixture,
        conformance::defective_known_event_surfaces_err
    );
}

// The terminal-body/delta per-kind merge lives on the buffered-body pipeline
// (the ChatGPT backend re-parses the SSE body after the fact); the live
// streaming path delivers message text only through deltas.
#[tokio::test]
async fn terminal_body_content_merges_per_kind() {
    let driver = openai_responses::buffered_driver();
    conformance::terminal_body_content_merges_per_kind(
        &driver,
        vec![
            (
                "body-only",
                openai_responses::terminal_body_only_sse_body("from body"),
            ),
            (
                "body+delta",
                openai_responses::terminal_body_and_delta_sse_body("from body"),
            ),
            (
                "delta-only",
                openai_responses::delta_only_sse_body("from body"),
            ),
        ],
        "from body",
    )
    .await
    .expect("scenario should hold");
}

// Reasoning sequence families. Only the OpenAI Responses wire spells
// multi-part reasoning items, so these run against that fixture alone.
mod reasoning {
    use super::*;

    // Pins P1-1 of `rig-2257-code-review-findings-34ee8ba5.md`: the
    // `reasoning_summary_text.delta` handler drops `item_id`, so the strict
    // same-item table appends the full block beside the delta-built item and
    // every o-series reasoning-summary stream aggregates the summary twice.
    #[tokio::test]
    async fn summary_deltas_are_superseded_without_duplication() {
        let driver = openai_responses::driver();
        let (frames, summary) = openai_responses::reasoning_summary_supersede_frames();
        conformance::reasoning_summary_deltas_are_superseded_without_duplication(
            &driver, frames, summary,
        )
        .await
        .expect("scenario should hold");
    }

    // Pins P1-2 of `rig-2257-code-review-findings-34ee8ba5.md`: the
    // `rposition` by-id fallback replaces the just-appended same-id sibling,
    // so a multi-part reasoning item survives only as its last part.
    #[tokio::test]
    async fn multi_part_same_id_reasoning_keeps_every_part() {
        let driver = openai_responses::driver();
        let (frames, expected) = openai_responses::multi_part_reasoning_frames();
        conformance::multi_part_same_id_reasoning_keeps_every_part(&driver, frames, &expected)
            .await
            .expect("scenario should hold");
    }

    #[tokio::test]
    async fn interleaved_reasoning_aggregates_to_one_item() {
        let driver = openai_responses::driver();
        let (frames, expected) = openai_responses::interleaved_reasoning_frames();
        conformance::interleaved_reasoning_aggregates_to_one_item(&driver, frames, expected)
            .await
            .expect("scenario should hold");
    }
}
