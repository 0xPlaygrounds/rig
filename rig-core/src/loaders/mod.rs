//! This module provides utility structs for loading and preprocessing files.
//!
//! The [FileLoader] struct can be used to define a common interface for loading any type of files from disk,
//! as well as performing minimal preprocessing on the files, such as reading their contents, ignoring errors
//! and keeping track of file paths along with their contents.
//!
//! The [PdfFileLoader] works similarly to the [FileLoader], but is specifically designed to load PDF
//! files. This loader also provides PDF-specific preprocessing methods for splitting the PDF into pages
//! and keeping track of the page numbers along with their contents.
//!
//! Note: The [PdfFileLoader] requires the `pdf` feature to be enabled in the `Cargo.toml` file.

pub mod file;

pub use file::FileLoader;

#[cfg(feature = "pdf")]
pub mod pdf;

#[cfg(feature = "pdf")]
pub use pdf::PdfFileLoader;
