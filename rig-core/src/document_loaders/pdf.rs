use super::DocumentLoader;
use crate::embeddings::DocumentEmbeddings;
use async_trait::async_trait;
use lopdf::Document;
use serde_json::json;

pub struct PdfLoader {
    path: String,
}

impl PdfLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    async fn load(
        &self,
    ) -> Result<Vec<DocumentEmbeddings>, Box<dyn std::error::Error + Send + Sync>> {
        let doc = Document::load(&self.path)?;
        let mut text = String::new();
        for page in doc.get_pages() {
            if let Ok(content) = doc.extract_text(&[page.0]) {
                text.push_str(&content);
            }
        }
        println!("Extracted text from PDF: {}", text); // Debug print
        Ok(vec![DocumentEmbeddings {
            id: self.path.clone(),
            document: json!({"text": text}),
            embeddings: vec![], // Empty vector, embeddings will be generated later
        }])
    }
}
