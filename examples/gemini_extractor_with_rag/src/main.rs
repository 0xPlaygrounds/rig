//! RAG-backed structured extraction.
//!
//! Extraction is a free function now ([`extract_with_options`]) and carries no
//! hook stack, so the retrieval that used to live in a `BeforeModelCall` hook
//! runs up front and lands on [`AgentConfig::static_context`]. A one-call
//! extraction retrieves exactly once either way, so the request the model sees
//! is the same.
use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::completion::Document;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::prelude::*;
use rig::provider::Runtime;
use rig::providers::gemini;
use rig::providers::gemini::client::Client;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, vector_store::VectorSearchRequest,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::vec;

// Data to be RAGged.
// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
#[derive(Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct Question {
    #[embed]
    id: String,
    #[embed]
    text: String,
    #[embed]
    answer_options: String,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Answer {
    /// The id of the question you are answering
    id: String,
    /// The answer to the question
    text: String,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct QuestionnaireResponses {
    /// The list of responses to the questionnaire
    responses: Vec<Answer>,
}

/// Passive RAG for the extractor: on every extraction attempt, embed the
/// input text, search the questionnaire store, and inject the best-matching
/// questions as per-turn context documents.
///
/// Hooks are attach-and-forget records — a named `HookEntry` wrapping a
/// closure that receives an owned `HookEvent` and returns a `HookDecision`;
/// the embedding model, the store, and the sample count are captured behind an
/// `Arc` so the returned future stays `'static + Send + Sync`.
/// Retrieve the `samples` most relevant questionnaire entries for `query`.
async fn retrieve_questions(
    embedding_model: &gemini::embedding::EmbeddingModel,
    store: &InMemoryVectorStore,
    samples: u64,
    query: &str,
) -> Result<Vec<Document>, anyhow::Error> {
    let embedded = embedding_model.embed_text(query).await?;
    let request = VectorSearchRequest::builder()
        .query(embedded)
        .samples(samples)
        .build();
    Ok(store
        .top_n(request)
        .await?
        .into_iter()
        .map(|hit| Document {
            id: hit.id,
            text: hit.payload.to_string(),
            additional_props: Default::default(),
        })
        .collect())
}

const APPLICANT_INFO: &str = r#"
Subject: Application details / quick background

Hi Procurement Team,

Thanks for reaching out. Here are a few details about me so you can route my application to the right person.

My full name is John Doe. I’ve been working in and around manufacturing for about 6 years now (mostly in operations + automation support). Over the last couple of roles I’ve done a bit of everything: supporting production lines, troubleshooting recurring quality issues, and helping roll out small process improvements that reduce downtime.

On the technical side, I’m comfortable with Python for data cleanup/automation, SQL for reporting, and I’ve done some light work with PLC/HMI troubleshooting (Siemens/Allen-Bradley basics). I also use Excel heavily (Power Query, pivot tables) and I’m familiar with Git and basic CI setups from internal tooling projects.

Unrelated but possibly helpful: I’m based in Montreal, can travel a couple times per quarter, and I’m generally available for calls after 2pm ET. I’m also finishing a part-time course in project management this spring.

Also, if you need references, I can share them once you confirm which role this is being matched to.

Best regards,
John Doe
"#;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create Gemini client
    let gemini_client = Client::from_env()?;
    let embedding_model = gemini_client.embedding_model(gemini::EMBEDDING_001);

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(vec![
            Question {
                id: "question_1".to_string(),
                text: "Complete name".to_string(),
                answer_options: "Open question".to_string(),
            },
            Question {
                id: "question_2".to_string(),
                text: "Years of experience in the manufacturing industry".to_string(),
                answer_options:
                    "The answers should be one of the following: Less than 1 year, 1-2 years, 2-5 years, 5-10 years, More than 10 years"
                        .to_string(),
            },
            Question {
                id: "question_3".to_string(),
                text: "Which technical skills do you have related to the job offer?".to_string(),
                answer_options: "Open question. Examples are: Python, SQL, Excel, Git, CI, PLC/HMI troubleshooting (Siemens/Allen-Bradley basics)".to_string(),
            },
        ])?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;

    // Samples should match the number of questions.
    let context = retrieve_questions(&embedding_model, &vector_store, 3, APPLICANT_INFO).await?;

    const ROLE: &str = "
            You are a questionnaire assistant provided by the procurement department to assist the user in answering the questions.
            You are provided with the questions and based on the information available, you must answer the questions with the right format.
            Use the answer ID field to map the answer to the right question ID. Answer as much as possible without inventing information.
            ";
    let classic = ExtractOptions::classic_extractor();
    let extraction_preamble = classic.preamble.clone().unwrap_or_default();
    let options = classic.with_preamble(format!(
        "{extraction_preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{ROLE}"
    ));

    let mut config = AgentConfig::new();
    config.static_context = context;

    // Prompt the model and print the response
    let response = extract_with_options::<QuestionnaireResponses>(
        config,
        gemini_client.provider_config("gemini-2.5-flash"),
        Arc::new(Runtime::new()),
        APPLICANT_INFO,
        options,
    )
    .await?
    .value;

    println!("{response:#?}");

    Ok(())
}
