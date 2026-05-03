//! File loading utilities for preparing local documents as model or embedding input.
//!
//! [`FileLoader`] provides a common interface for reading files from disk, glob
//! matches, directories, or in-memory bytes. It can return content alone or pair
//! content with source paths, and it can optionally skip per-file errors.
//!
//! `PdfFileLoader` is available with the `pdf` feature. It loads PDFs and can
//! split extracted text by page while preserving page numbers.
//!
//! `EpubFileLoader` is available with the `epub` feature. It loads EPUB files
//! and can split extracted text by chapter while preserving chapter numbers.

pub mod file;

pub use file::FileLoader;

#[cfg(feature = "pdf")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdf")))]
pub mod pdf;

#[cfg(feature = "pdf")]
pub use pdf::PdfFileLoader;

#[cfg(feature = "epub")]
#[cfg_attr(docsrs, doc(cfg(feature = "epub")))]
pub mod epub;

#[cfg(feature = "epub")]
pub use epub::{EpubFileLoader, RawTextProcessor, StripXmlProcessor, TextProcessor};
