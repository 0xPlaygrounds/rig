// Import necessary dependencies
use super::DocumentLoader;
use crate::embeddings::DocumentEmbeddings;
use async_trait::async_trait;
use lopdf::Document;
use serde_json::json;

// Define a struct for loading PDF documents
pub struct PdfLoader {
    path: String,
}

impl PdfLoader {
    // Implement a constructor for the PdfLoader struct
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    // Implement the load function for the DocumentLoader trait
    async fn load(
        &self,
    ) -> Result<Vec<DocumentEmbeddings>, Box<dyn std::error::Error + Send + Sync>> {
        // Load the PDF document from the specified path
        let doc = Document::load(&self.path)?;

        // Extract text from each page of the PDF document
        let mut text = String::new();
        for page in doc.get_pages() {
            if let Ok(content) = doc.extract_text(&[page.0]) {
                text.push_str(&content);
            }
        }

        // Print the extracted text for debugging purposes
        println!("Extracted text from PDF: {}", text);

        // Create a DocumentEmbeddings object with the extracted text
        Ok(vec![DocumentEmbeddings {
            id: self.path.clone(),
            document: json!({"text": text}),
            embeddings: vec![], // Empty vector, embeddings will be generated later
        }])
    }
}
