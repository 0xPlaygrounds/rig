use async_trait::async_trait;
use csv::Reader;
use serde_json::json;
use std::error::Error as StdError;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use super::DocumentLoader;
use crate::embeddings::DocumentEmbeddings;

pub struct CsvLoader {
    path: String,
}

impl CsvLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
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
        let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

        let mut csv_content = String::new();

        for result in reader.records() {
            let record = result?;
            for (i, field) in record.iter().enumerate() {
                csv_content.push_str(&format!("{}: {}\n", headers[i], field));
            }
            csv_content.push_str("\n");
        }

        Ok(vec![DocumentEmbeddings {
            id: self.path.clone(),
            document: json!({"text": csv_content}),
            embeddings: vec![],
        }])
    }
}
