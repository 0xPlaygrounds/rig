// src/document_loaders/csv.rs

use async_trait::async_trait;
use csv::Reader;
use serde_json::Value;
use std::error::Error as StdError;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::embeddings::DocumentEmbeddings;
use super::DocumentLoader;

pub struct CsvLoader {
    path: String,
    id_column: Option<String>,
}

impl CsvLoader {
    pub fn new(path: &str, id_column: Option<&str>) -> Self {
        Self {
            path: path.to_string(),
            id_column: id_column.map(String::from),
        }
    }
}

#[async_trait]
impl DocumentLoader for CsvLoader {
    async fn load(&self) -> Result<Vec<DocumentEmbeddings>, Box<dyn StdError + Send + Sync>> {
        let mut file = File::open(&self.path).await?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).await?;

        let mut reader = Reader::from_reader(contents.as_bytes());
        let headers = reader.headers()?.clone();

        let mut documents = Vec::new();

        for result in reader.records() {
            let record = result?;
            let mut doc = serde_json::Map::new();

            for (i, field) in record.iter().enumerate() {
                doc.insert(headers[i].to_string(), Value::String(field.to_string()));
            }

            let id = if let Some(id_col) = &self.id_column {
                doc.get(id_col).and_then(|v| v.as_str()).unwrap_or_default().to_string()
            } else {
                format!("csv_row_{}", documents.len())
            };

            documents.push(DocumentEmbeddings {
                id,
                document: Value::Object(doc),
                embeddings: vec![],
            });
        }

        Ok(documents)
    }
}