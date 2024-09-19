//! This module contains the implementation of document loaders for various file formats.
//! Currently, it includes loaders for CSV and PDF files.

mod csv;
// mod directory;
// mod html;
// mod json;
// mod markdown;
// mod office;
mod pdf;

use crate::embeddings::DocumentEmbeddings;
use async_trait::async_trait;
use std::error::Error as StdError;

#[async_trait]
pub trait DocumentLoader {
    /// Asynchronously loads the document and returns a vector of document embeddings.
    async fn load(&self) -> Result<Vec<DocumentEmbeddings>, Box<dyn StdError + Send + Sync>>;
}

pub use csv::CsvLoader;
// pub use directory::DirectoryLoader;
// pub use html::HtmlLoader;
// pub use json::JsonLoader;
// pub use markdown::MarkdownLoader;
// pub use office::OfficeLoader;
pub use pdf::PdfLoader;
