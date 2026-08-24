//! PDF-backed citations, recorded from the real API.
//!
//! `Citation` has hand-written serde (`completion.rs`) with five locator
//! variants, and only two of them — `char_location` (plain-text documents) and
//! `web_search_result_location` — had ever been recorded. `page_location` is
//! the shape a PDF produces, and the recorded PDF scenarios never enabled
//! citations, so that variant's decoding had only unit coverage.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::CompletionModel;
use rig::message::{Document, DocumentSourceKind, Message, UserContent};
use rig::prelude::*;
use rig::providers::anthropic::completion::{
    self as anthropic_completion, CLAUDE_SONNET_4_6, Citation,
};
use serde_json::json;

use super::super::support::with_anthropic_cassette;

const PDF_URL: &str = "https://bitcoin.org/bitcoin.pdf";

fn cited_pdf() -> Document {
    Document {
        data: DocumentSourceKind::Url(PDF_URL.to_string()),
        media_type: None,
        additional_params: rig::message::AdditionalParams::try_from_value(json!({
            "title": "Bitcoin Whitepaper",
            "citations": { "enabled": true }
        }))
        .expect("object params"),
    }
}

fn collect_citations(choice: &[rig::message::AssistantContent]) -> Vec<Citation> {
    choice
        .iter()
        .filter_map(|content| match content {
            rig::message::AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|text| {
            anthropic_completion::anthropic_citations(text)
                .expect("citations should decode from Anthropic text metadata")
        })
        .collect()
}

/// A PDF-grounded answer cites back into the document by *page*, so the
/// response must decode as [`Citation::PageLocation`] rather than falling
/// through to the forward-compatible `Unknown` bucket.
#[tokio::test]
async fn pdf_document_citations_decode_as_page_locations() {
    with_anthropic_cassette(
        "pdf_citations/pdf_document_citations_decode_as_page_locations",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let response = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![
                                UserContent::Document(cited_pdf()),
                                UserContent::text(
                                    "Using citations, state in one sentence what problem this \
                                     paper says it solves.",
                                ),
                            ],
                        })
                        .temperature(0.0)
                        .max_tokens(512)
                        .build(),
                )
                .await
                .expect("cited PDF completion should succeed");

            let citations = collect_citations(&response.choice);
            assert!(
                !citations.is_empty(),
                "a citations-enabled PDF answer should carry citation metadata",
            );

            let page_locations: Vec<_> = citations
                .iter()
                .filter_map(|citation| match citation {
                    Citation::PageLocation(page) => Some(page),
                    _ => None,
                })
                .collect();
            assert!(
                !page_locations.is_empty(),
                "a PDF citation must decode as PageLocation, got {citations:?}",
            );
            for page in page_locations {
                assert!(
                    page.start_page_number >= 1,
                    "page numbers are 1-indexed, got {page:?}",
                );
                assert!(
                    page.end_page_number >= page.start_page_number,
                    "page range must not run backwards, got {page:?}",
                );
                assert!(
                    !page.cited_text.is_empty(),
                    "a citation must quote the text it points at, got {page:?}",
                );
            }
        },
    )
    .await;

    // Premise: the recorded response really did carry `page_location`
    // citations, so this cell cannot pass by decoding something else.
    let body = super::super::support::recorded_response_body(
        "pdf_citations/pdf_document_citations_decode_as_page_locations",
    );
    let recorded = body.to_string();
    assert!(
        recorded.contains(r#""type":"page_location""#),
        "the recorded response must contain page_location citations",
    );
}
