/// Builder for creating a collection of embeddings
pub struct EmbeddingsBuilder<M: EmbeddingModel, T: Embeddable> {
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

trait Embeddable {
    // Return list of strings that need to be embedded.
    // Instead of Vec<String>, should be Vec<T: Serialize>
    fn embeddable(&self) -> Vec<String>;
}

type EmbeddingVector = Vec<f64>;

impl<M: EmbeddingModel, T: Embeddable> EmbeddingsBuilder<M, T> {
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: vec![],
        }
    }

    pub fn add<T: Embeddable>(
        mut self,
        document: T,
    ) -> Self {
        let embed_documents: Vec<String> = document.embeddable();

        self.documents.push((
            document,
            embed_documents,
        ));
        self
    }

    pub fn build(&self) -> Result<Vec<(T, Vec<EmbeddingVector>)>, EmbeddingError> {
        self.documents.iter().map(|(doc, values_to_embed)| {
            values_to_embed.iter().map(|value| {
                let value_str = serde_json::to_string(value)?;
                generate_embedding(value_str)
            })
        })
    }

    pub fn build_simple(&self) -> Result<Vec<(T, EmbeddingVector)>, EmbeddingError> {
        self.documents.iter().map(|(doc, value_to_embed)| {
            let value_str = serde_json::to_string(value_to_embed)?;
            generate_embedding(value_str)
        })
    }
}


// Example
#[derive(Embeddable)]
struct DictionaryEntry {
    word: String,
    #[embed]
    definitions: String,
}

#[derive(Embeddable)]
struct MetadataEmbedding {
    pub id: String,
    #[embed(with = serde_json::to_value)]
    pub content: CategoryMetadata,
    pub created: Option<DateTime<Utc>>,
    pub modified: Option<DateTime<Utc>>,
    pub dataset_ids: Vec<String>,
}

#[derive(serde::Serialize)]
struct CategoryMetadata {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
}

// Inside macro:
impl Embeddable for DictionaryEntry {
    fn embeddable(&self) -> Vec<String> {
        // Find the field tagged with #[embed] and return its value
        // If there are no embedding tags, return the entire struct
    }
}

fn main() {
    let embeddings: Vec<(DictionaryEntry, Vec<EmbeddingVector>)> = EmbeddingsBuilder::new(model.clone())
        .add(DictionaryEntry::new("blah", vec!["definition of blah"]))
        .add(DictionaryEntry::new("foo", vec!["definition of foo"]))
        .build()?;

    // In relational vector store like LanceDB, need to flatten result (create row for each item in definitions vector):
    // Column: word (string)
    // Column: definition (vector)

    // In document vector store like MongoDB, might need to merge the vector results back with their corresponding definition string:
    // Field: word (string)
    // Field: definitions
    //  // Field: definition (string)
    //  // Field: vector

    Ok(())
}



// Iterations:
// 1 - Multiple fields to embed?
#[derive(Embedding)]
struct DictionaryEntry {
    word: String,
    #[embed]
    definitions: Vec<String>,
    #[embed]
    synonyms: Vec<String>
}

// 2 - Embed recursion? Ex:
#[derive(Embedding)]
struct DictionaryEntry {
    word: String,
    #[embed]
    definitions: Vec<Definition>,
}
struct Definition {
    definition: String,
    #[embed]
    links: Vec<String>
}

// {
//     word: "blah",
//     definitions: [
//         {
//             definition: "definition of blah",
//             links: ["link1", "link2"]
//         },
//         {
//             definition: "another definition for blah",
//             links: ["link3"]
//         }
//     ]
// }

// blah | definition of blah | link1 | embedding for link1
// blah | definition of blah | link2 | embedding for link2
// blah | another definition for blah | link3 | embedding for link3