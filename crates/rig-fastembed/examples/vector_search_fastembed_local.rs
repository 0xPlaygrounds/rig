use anyhow::Context;
use fastembed::{
    EmbeddingModel as FastembedModel, Pooling, TextEmbedding as FastembedTextEmbedding,
    TokenizerFiles, UserDefinedEmbeddingModel, read_file_to_bytes,
};
use rig_core::OneOrMany;
use rig_core::{
    Embed,
    embeddings::embed_documents,
    vector_store::{in_memory_store::InMemoryVectorStore, request::VectorSearchRequest},
};
use rig_fastembed::{EmbeddingModel, functions};
use serde::{Deserialize, Serialize};
use std::path::Path;

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embed, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Get model info
    let test_model_info = FastembedTextEmbedding::get_model_info(&FastembedModel::AllMiniLML6V2)?;

    // Set up model directory
    let model_dir = Path::new("./models/Qdrant--all-MiniLM-L6-v2-onnx/snapshots");
    println!("Loading model from: {model_dir:?}");

    // Load model files
    let onnx_file = read_file_to_bytes(&model_dir.join("model.onnx"))
        .context("Could not read model.onnx file")?;

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read_file_to_bytes(&model_dir.join("tokenizer.json"))
            .context("Could not read tokenizer.json")?,
        config_file: read_file_to_bytes(&model_dir.join("config.json"))
            .context("Could not read config.json")?,
        special_tokens_map_file: read_file_to_bytes(&model_dir.join("special_tokens_map.json"))
            .context("Could not read special_tokens_map.json")?,
        tokenizer_config_file: read_file_to_bytes(&model_dir.join("tokenizer_config.json"))
            .context("Could not read tokenizer_config.json")?,
    };

    // Create embedding model
    let user_defined_model =
        UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

    let embedding_model =
        EmbeddingModel::new_from_user_defined(user_defined_model, 384, test_model_info)?;

    // Create documents
    let documents = vec![
        WordDefinition {
            id: "doc0".to_string(),
            word: "flurbo".to_string(),
            definitions: vec![
                "A green alien that lives on cold planets.".to_string(),
                "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
            ]
        },
        WordDefinition {
            id: "doc1".to_string(),
            word: "glarb-glarb".to_string(),
            definitions: vec![
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
            ]
        },
        WordDefinition {
            id: "doc2".to_string(),
            word: "linglingdong".to_string(),
            definitions: vec![
                "A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
                "A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
            ]
        },
    ];

    // Create embeddings. `embed_documents` replaces the old
    // `EmbeddingsBuilder`: it flattens each document's `#[embed]` fields,
    // embeds them in batches, and re-associates the vectors with their
    // document.
    let embeddings = embed_documents(documents, rig_fastembed::MAX_DOCUMENTS, 1, |texts| async {
        functions::embed(&embedding_model, texts)
    })
    .await?;

    // Create vector store. The store never embeds text itself: queries arrive
    // pre-embedded.
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone())?;

    let query =
        "I need to buy something in a fictional universe. What type of money can I use for this?";
    let query_embedding = functions::embed_text(&embedding_model, query)?;

    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    let results = vector_store
        .top_n_as::<WordDefinition>(req)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    Ok(())
}
