use super::*;

/// Vertex reports `thoughts_token_count` and rig has a field for it, but the
/// mapping hardcoded `reasoning_tokens: 0` — so the thinking spend, which on
/// the sibling Gemini surface is routinely the largest component of the
/// bill, was discarded on every Vertex response.
///
/// Drives the real conversion. An earlier version of this test built a
/// `UsageMetadata` and then re-implemented the mapping inline in its own
/// body, so reverting the production fix left it green — it guarded nothing.
#[test]
fn thinking_tokens_survive_the_real_conversion() {
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(14)
        .set_candidates_token_count(34)
        .set_thoughts_token_count(222)
        .set_total_token_count(270)
        .set_cached_content_token_count(9);

    let response = vertexai::model::GenerateContentResponse::new()
        .set_usage_metadata(usage_metadata)
        .set_candidates(vec![
            vertexai::model::Candidate::new().set_content(
                vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts(vec![vertexai::model::Part::new().set_text("hi")]),
            ),
        ]);

    let converted = CompletionResponse::try_from(VertexGenerateContentOutput(response))
        .expect("a response with content should convert");

    assert_eq!(converted.usage.reasoning_tokens, 222);
    assert_eq!(converted.usage.cached_input_tokens, 9);
    assert_eq!(converted.usage.input_tokens, 14);
    assert_eq!(converted.usage.output_tokens, 34);
    assert_eq!(converted.usage.total_tokens, 270);
}
