use std::sync::Arc;

use arrow_array::{types::Float64Type, ArrayRef, FixedSizeListArray, RecordBatch, StringArray};
use lancedb::arrow::arrow_schema::{DataType, Field, Fields, Schema};
use rig::embeddings::{
    embeddable::{EmbeddingGenerationError, SingleEmbedding},
    embedding::Embedding,
};
use rig::Embeddable;
use serde::Deserialize;

#[derive(Embeddable, Clone, Deserialize, Debug)]
pub struct FakeDefinition {
    id: String,
    #[embed]
    definition: String,
}

pub fn fake_definitions() -> Vec<FakeDefinition> {
    vec![
        FakeDefinition {
            id: "doc0".to_string(),
            definition: "Definition of *flumbrel (noun)*: a small, seemingly insignificant item that you constantly lose or misplace, such as a pen, hair tie, or remote control.".to_string()
        },
        FakeDefinition {
            id: "doc1".to_string(),
            definition: "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive.".to_string()
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
        definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string()
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

// Convert FakeDefinition objects and their embedding to a RecordBatch.
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
            .map(|(FakeDefinition { definition, .. }, _)| definition)
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
