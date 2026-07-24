//! Cassette-backed Anthropic coverage for URL-backed PDF documents.
//!
//! Regression coverage for sending a `DocumentSourceKind::Url` PDF through
//! the request-side message conversion: the document must map to a
//! `"source": {"type": "url", ...}` content block.
//! See <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::message::{DocumentMediaType, Message, UserContent};
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_cassette;
use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

const PDF_URL: &str = "https://bitcoin.org/bitcoin.pdf";

#[tokio::test]
async fn url_pdf_document_prompt() {
    with_anthropic_cassette(
        "url_pdf_document/url_pdf_document_prompt",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a helpful assistant that analyzes documents.")
                .temperature(0.0)
                .build();

            let response = agent
                .prompt(Message::User {
                    content: OneOrMany::many(vec![
                        UserContent::document_url(PDF_URL, Some(DocumentMediaType::PDF)),
                        UserContent::text(
                            "What is the title of this paper? Answer in one short sentence.",
                        ),
                    ])
                    .expect("content should be non-empty"),
                })
                .await
                .expect("URL PDF document prompt should succeed");

            assert_nonempty_response(&response);
            assert_contains_any_case_insensitive(&response, &["bitcoin"]);
        },
    )
    .await;
}
