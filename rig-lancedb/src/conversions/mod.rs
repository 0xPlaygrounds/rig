use std::sync::Arc;

use lancedb::arrow::arrow_schema::{DataType, Field, Fields};

pub mod document_embeddings;
pub mod record_batch;

#[derive(Clone)]
pub struct DocumentEmbeddings(pub Vec<rig::embeddings::DocumentEmbeddings>);

impl DocumentEmbeddings {
    pub fn new(documents: Vec<rig::embeddings::DocumentEmbeddings>) -> Self {
        Self(documents)
    }

    pub fn as_iter(&self) -> impl Iterator<Item = &rig::embeddings::DocumentEmbeddings> {
        self.0.iter()
    }

    pub fn schema(&self) -> Vec<Field> {
        vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document", DataType::Utf8, false),
            Field::new(
                "embeddings",
                DataType::List(Arc::new(Field::new(
                    "embedding_item",
                    DataType::Struct(Fields::from(vec![
                        Arc::new(Field::new("document", DataType::Utf8, false)),
                        Arc::new(Field::new(
                            "vec",
                            DataType::List(Arc::new(Field::new("float", DataType::Float64, false))),
                            false,
                        )),
                    ])),
                    false,
                ))),
                false,
            ),
        ]
    }
}
