use std::sync::Arc;

use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use rig::Embed;
use serde::{Deserialize, Serialize};

#[derive(Embed, Clone, Deserialize, Serialize, Debug)]
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

/// The table [`Word`]s are inserted into, with `dims`-dimension embeddings.
pub fn schema(dims: usize) -> Arc<Schema> {
    let item = Arc::new(Field::new("item", DataType::Float64, true));
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("definition", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(item, dims as i32),
            false,
        ),
    ]))
}
