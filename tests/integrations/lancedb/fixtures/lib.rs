use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, StringArray, types::Float64Type};
use rig::Embed;
use rig::embeddings::Embedding;
use serde::Deserialize;

#[derive(Embed, Clone, Deserialize, Debug)]
pub struct Word {
    pub id: String,
    #[embed]
    pub definition: String,
}

pub fn words() -> Vec<Word> {
    vec![
        Word {
            id: "doc0".to_string(),
            definition: "Definition of *flumbrel (noun)*: a small, seemingly insignificant item that you constantly lose or misplace, such as a pen, hair tie, or remote control.".to_string()
        },
        Word {
            id: "doc1".to_string(),
            definition: "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive.".to_string()
        },
        Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string()
        }
    ]
}

/// A definition repeated to pad the table: an IVF-PQ index needs at least 256 rows.
pub const FLUMBUZZLE: &str = "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.";

/// 256 rows of [`FLUMBUZZLE`], with ids `doc0`..`doc255`.
pub fn flumbuzzles() -> impl Iterator<Item = Word> {
    (0..256).map(|i| Word {
        id: format!("doc{i}"),
        definition: FLUMBUZZLE.to_string(),
    })
}

// Convert Word objects and their embedding to a RecordBatch.
pub fn as_record_batch(
    records: Vec<(Word, Vec<Embedding>)>,
    dims: usize,
) -> Result<RecordBatch, lancedb::arrow::arrow_schema::ArrowError> {
    let id = StringArray::from_iter_values(records.iter().map(|(Word { id, .. }, _)| id));

    let definition = StringArray::from_iter_values(
        records
            .iter()
            .map(|(Word { definition, .. }, _)| definition),
    );

    let embedding = FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        records.into_iter().map(|(_, embeddings)| {
            Some(
                embeddings
                    .into_iter()
                    .next()
                    .expect("expected at least one embedding")
                    .vec
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>(),
            )
        }),
        dims as i32,
    );

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(id) as ArrayRef),
        ("definition", Arc::new(definition) as ArrayRef),
        ("embedding", Arc::new(embedding) as ArrayRef),
    ])
}
