use rig_core::OneOrMany;
use rig_core::{
    Embed,
    embeddings::EmbeddingJob,
    vector_store::{in_memory_store::InMemoryVectorStore, request::VectorSearchRequest},
};
use rig_fastembed::{EmbeddingConfig, FastembedModel, functions};
use serde::{Deserialize, Serialize};

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
    // Load the local FastEmbed model. There is no client type: the config
    // says which weights to load, and `load()` hands back the runtime handle.
    let embedding_model = EmbeddingConfig::new(FastembedModel::AllMiniLML6V2Q).load()?;

    // `EmbeddingJob` replaces the old `EmbeddingsBuilder`: it flattens each
    // document's `#[embed]` fields, embeds them in batches, and re-associates
    // the vectors with their document.
    //
    // The batch size and concurrency are explicit rather than descriptor-derived:
    // fastembed runs the model in-process, so its limit is a crate constant and
    // the embedding calls are deliberately serialized.
    let embeddings = EmbeddingJob::new()
        .documents(vec![
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
        ])
        .max_documents(rig_fastembed::MAX_DOCUMENTS)
        .concurrency(1)
        .run(|texts| async { functions::embed(&embedding_model, texts) })
        .await?;

    // Create vector store with the embeddings. The store never embeds text
    // itself: queries arrive pre-embedded.
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone())?;

    let query =
        "I need to buy something in a fictional universe. What type of money can I use for this?";
    let query_embedding = functions::embed_text(&embedding_model, query)?;

    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    let results = vector_store
        .top_n_as::<WordDefinition>(req.clone())
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    let id_results = vector_store
        .top_n_ids(req)
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
