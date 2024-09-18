// src/document_loaders/pdf.rs

use async_trait::async_trait;
use lopdf::Document;
use std::error::Error as StdError;
use crate::embeddings::DocumentEmbeddings;
use super::DocumentLoader;

pub struct PdfLoader {
    path: String,
}

impl PdfLoader {
    pub fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    async fn load(&self) -> Result<Vec<DocumentEmbeddings>, Box<dyn StdError + Send + Sync>> {
        let doc = Document::load(&self.path)?;
        let mut text = String::new();
        for page in doc.get_pages() {
            if let Ok(content) = doc.extract_text(&[page.0]) {
                text.push_str(&content);
            }
        }

        Ok(vec![DocumentEmbeddings {
            id: self.path.clone(),
            document: serde_json::Value::String(text),
            embeddings: vec![],
        }])
    }
}