use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::client::Client;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, vector_store::in_memory_store::InMemoryVectorStore,
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
    let gemini_client = Client::from_env();
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
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    // Create vector store index
    let index = vector_store.index(embedding_model);
    let rag_extractor = gemini_client.extractor::<QuestionnaireResponses>("gemini-2.5-flash")
        .preamble("
            You are a questionnaire assistant provided by the procurement department to assist the user in answering the questions.
            You are provided with the questions and based on the information available, you must answer the questions with the right format.
            Use the answer ID field to map the answer to the right question ID. Answer as much as possible without inventing information.
            ")
        .with_agent_builder(|agent| agent.dynamic_context(3, index)) // Samples should match the number of questions
        .build();

    // Prompt the agent and print the response
    let response = rag_extractor.extract(APPLICANT_INFO).await?;

    println!("{response:#?}");

    Ok(())
}
