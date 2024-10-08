use std::sync::Arc;

use arrow_array::{types::Float64Type, ArrayRef, FixedSizeListArray, RecordBatch, StringArray};
use lancedb::arrow::arrow_schema::{DataType, Field, Fields, Schema};
use rig::embeddings::Embedding;
use rig_derive::Embed;
use serde::Deserialize;

#[derive(Embed, Clone)]
pub struct FakeDefinition {
    id: String,
    #[embed]
    definition: String,
}

#[derive(Deserialize, Debug)]
pub struct VectorSearchResult {
    pub id: String,
    pub definition: String,
}

pub fn fake_definitions() -> Vec<FakeDefinition> {
    vec![
        FakeDefinition {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string()
        },
        FakeDefinition {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string()
        },
        FakeDefinition {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string()
        }
    ]
}

pub fn fake_definition(id: String) -> FakeDefinition {
    FakeDefinition {
        id,
        definition: "Definition of *flumbuzzle (verb)*: to bewilder or confuse someone completely, often by using nonsensical or overly complex explanations or instructions.".to_string()
    }
}

// Schema of table in LanceDB.
pub fn schema(dims: usize) -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("definition", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float64, true)),
                dims as i32,
            ),
            false,
        ),
    ]))
}

// Convert DocumentEmbeddings objects to a RecordBatch.
pub fn as_record_batch(
    records: Vec<(FakeDefinition, Embedding)>,
    dims: usize,
) -> Result<RecordBatch, lancedb::arrow::arrow_schema::ArrowError> {
    let id = StringArray::from_iter_values(
        records
            .iter()
            .map(|(FakeDefinition { id, .. }, _)| id)
            .collect::<Vec<_>>(),
    );

    let definition = StringArray::from_iter_values(
        records
            .iter()
            .map(|(_, Embedding { document, .. })| document)
            .collect::<Vec<_>>(),
    );

    let embedding = FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        records
            .into_iter()
            .map(|(_, Embedding { vec, .. })| Some(vec.into_iter().map(Some).collect::<Vec<_>>()))
            .collect::<Vec<_>>(),
        dims as i32,
    );

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(id) as ArrayRef),
        ("definition", Arc::new(definition) as ArrayRef),
        ("embedding", Arc::new(embedding) as ArrayRef),
    ])
}
