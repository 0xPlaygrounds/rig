//! Cassette-backed OpenAI Responses coverage for URL-backed PDF documents.
//!
//! Regression coverage for sending a `DocumentSourceKind::Url` PDF through
//! `Model::call`: the request must carry `file_url` without
//! the hardcoded `filename`, which the Responses API rejects alongside a URL
//! with 400 `mutually_exclusive_parameters`.
//! See <https://platform.openai.com/docs/guides/pdf-files>.
use rig::message::{DocumentMediaType, Message, UserContent};
use rig::providers::openai;
use rig::wire::Wire as _;

use super::super::support::with_openai_cassette;
use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

const PDF_URL: &str = "https://bitcoin.org/bitcoin.pdf";

#[tokio::test]
async fn url_pdf_document_prompt() {
    with_openai_cassette(
        "url_pdf_document/url_pdf_document_prompt",
        |client| async move {
            let agent = rig::AgentBuilder::new(
                client
                    .openai
                    .completion(openai::GPT_4O)
                    .on(rig::transport()),
            )
            .preamble("You are a helpful assistant that analyzes documents.")
            .temperature(0.0)
            .build();

            let response = agent
                .prompt(Message::User {
                    content: vec![
                        UserContent::document_url(PDF_URL, Some(DocumentMediaType::PDF)),
                        UserContent::text(
                            "What is the title of this paper? Answer in one short sentence.",
                        ),
                    ],
                })
                .await
                .expect("URL PDF document prompt should succeed")
                .output;

            assert_nonempty_response(&response);
            assert_contains_any_case_insensitive(&response, &["bitcoin"]);
        },
    )
    .await;
}
