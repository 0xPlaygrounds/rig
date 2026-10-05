use super::*;

/// The tool-use prompt is input, as Vertex's `totalTokenCount` counts it:
/// 100 + (60 + 15) + 30 + 34 = 239. Vertex reports it only per modality, so
/// its count is the breakdown's sum. No cassette harness exists for
/// rig-vertexai, so the mapping's arithmetic is pinned here.
#[test]
fn tool_use_prompt_tokens_are_input_in_the_real_conversion() {
    let modality = |modality, tokens| {
        vertexai::model::ModalityTokenCount::new()
            .set_modality(modality)
            .set_token_count(tokens)
    };
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(100)
        .set_tool_use_prompt_tokens_details([
            modality(vertexai::model::Modality::Text, 60),
            modality(vertexai::model::Modality::Image, 15),
        ])
        .set_candidates_token_count(30)
        .set_thoughts_token_count(34)
        .set_total_token_count(239)
        .set_cached_content_token_count(40);

    let response = vertexai::model::GenerateContentResponse::new()
        .set_usage_metadata(usage_metadata)
        .set_candidates(vec![
            vertexai::model::Candidate::new()
                .set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
                .set_content(
                    vertexai::model::Content::new()
                        .set_role("model")
                        .set_parts(vec![vertexai::model::Part::new().set_text("hi")]),
                ),
        ]);

    let converted = crate::types::completion_response::tests::complete(response)
        .expect("a response with content should convert");

    let usage = converted.usage;
    assert_eq!(usage.input_tokens, Some(175));
    assert_eq!(usage.tool_use_prompt_tokens, Some(75));
    assert_eq!(usage.cached_input_tokens, Some(40));
    assert_eq!(usage.output_tokens, Some(64));
    assert_eq!(usage.reasoning_tokens, Some(34));
    assert_eq!(usage.total_tokens, Some(239));
}
