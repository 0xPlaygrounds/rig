//! CSV file loading utilities for Rig.
//!
//! This module provides the [`CsvFileLoader`] struct for loading and processing CSV files.
//! It follows the same patterns as [`FileLoader`] and [`PdfFileLoader`], providing a consistent
//! API for loading structured data into Rig applications.
//!
//! # Example
//!
//! ```rust,ignore
//! use rig::loaders::CsvFileLoader;
//!
//! // Load all CSV files in a directory
//! let loader = CsvFileLoader::with_glob("data/*.csv")?;
//!
//! // Process CSV files into documents
//! for document in loader.load().ignore_errors() {
//!     println!("{}", document);
//! }
//! ```
//!
//! # Features
//!
//! - Glob pattern matching for loading multiple CSV files
//! - Directory traversal
//! - Configurable delimiters
//! - Header handling (use headers as field names or skip them)
//! - Row-by-row or whole-document loading
//! - Error handling with `ignore_errors()` support

use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use glob::{glob, GlobError, PatternError};
use thiserror::Error;

/// Errors that can occur when loading CSV files.
#[derive(Error, Debug)]
pub enum CsvLoaderError {
    /// Error when glob pattern is invalid.
    #[error("Invalid glob pattern: {0}")]
    PatternError(#[from] PatternError),

    /// Error when traversing glob results.
    #[error("Glob error: {0}")]
    GlobError(#[from] GlobError),

    /// Error when reading the CSV file.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Error when parsing CSV content.
    #[error("CSV parse error: {0}")]
    CsvError(#[from] csv::Error),

    /// Error when the CSV file has no headers.
    #[error("CSV file has no headers")]
    NoHeaders,

    /// Error when the CSV file is empty.
    #[error("CSV file is empty")]
    EmptyFile,
}

/// Configuration options for the CSV loader.
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// The delimiter character (default: ',')
    pub delimiter: u8,
    /// Whether the CSV file has headers (default: true)
    pub has_headers: bool,
    /// Whether to trim whitespace from fields (default: true)
    pub trim: bool,
    /// Whether to allow flexible field counts per row (default: false)
    pub flexible: bool,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            trim: true,
            flexible: false,
        }
    }
}

impl CsvConfig {
    /// Create a new CSV config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter character.
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether the CSV has headers.
    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    /// Set whether to trim whitespace.
    pub fn trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }

    /// Set whether to allow flexible field counts.
    pub fn flexible(mut self, flexible: bool) -> Self {
        self.flexible = flexible;
        self
    }

    /// Create a config for TSV (tab-separated values) files.
    pub fn tsv() -> Self {
        Self::default().delimiter(b'\t')
    }
}

/// A loader for CSV files that converts tabular data into text documents.
///
/// `CsvFileLoader` provides methods to load CSV files from glob patterns or directories,
/// and convert them into text format suitable for embedding and LLM processing.
///
/// # Example
///
/// ```rust,ignore
/// use rig::loaders::CsvFileLoader;
///
/// // Load CSV files with default settings
/// let documents: Vec<String> = CsvFileLoader::with_glob("data/*.csv")?
///     .load()
///     .ignore_errors()
///     .collect();
///
/// // Load with custom configuration
/// let documents: Vec<String> = CsvFileLoader::with_glob("data/*.tsv")?
///     .with_config(CsvConfig::tsv())
///     .load()
///     .ignore_errors()
///     .collect();
/// ```
#[derive(Debug)]
pub struct CsvFileLoader {
    paths: Vec<Result<PathBuf, CsvLoaderError>>,
    config: CsvConfig,
}

impl CsvFileLoader {
    /// Create a new `CsvFileLoader` from a glob pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - A glob pattern to match CSV files (e.g., "data/*.csv")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = CsvFileLoader::with_glob("data/**/*.csv")?;
    /// ```
    pub fn with_glob(pattern: &str) -> Result<Self, CsvLoaderError> {
        let paths = glob(pattern)?
            .map(|result| result.map_err(CsvLoaderError::from))
            .collect();

        Ok(Self {
            paths,
            config: CsvConfig::default(),
        })
    }

    /// Create a new `CsvFileLoader` from a directory path.
    ///
    /// This will load all `.csv` files in the specified directory (non-recursive).
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the directory containing CSV files
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = CsvFileLoader::with_dir("data/")?;
    /// ```
    pub fn with_dir(dir: impl AsRef<Path>) -> Result<Self, CsvLoaderError> {
        let pattern = dir.as_ref().join("*.csv");
        let pattern_str = pattern.to_string_lossy();
        Self::with_glob(&pattern_str)
    }

    /// Create a new `CsvFileLoader` from a single file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = CsvFileLoader::from_path("data/users.csv");
    /// ```
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self {
            paths: vec![Ok(path.as_ref().to_path_buf())],
            config: CsvConfig::default(),
        }
    }

    /// Create a new `CsvFileLoader` from bytes.
    ///
    /// This is useful when you have CSV data in memory (e.g., from a network request).
    ///
    /// # Arguments
    ///
    /// * `bytes` - The CSV data as bytes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let csv_data = b"name,age\nAlice,30\nBob,25";
    /// let document = CsvFileLoader::from_bytes(csv_data, CsvConfig::default())?;
    /// ```
    pub fn from_bytes(bytes: &[u8], config: CsvConfig) -> Result<String, CsvLoaderError> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(config.delimiter)
            .has_headers(config.has_headers)
            .trim(if config.trim {
                csv::Trim::All
            } else {
                csv::Trim::None
            })
            .flexible(config.flexible)
            .from_reader(bytes);

        Self::reader_to_document(&mut reader, config.has_headers)
    }

    /// Set the CSV configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = CsvFileLoader::with_glob("data/*.tsv")?
    ///     .with_config(CsvConfig::tsv());
    /// ```
    pub fn with_config(mut self, config: CsvConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the delimiter character.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = CsvFileLoader::with_glob("data/*.csv")?
    ///     .delimiter(b';');
    /// ```
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.config.delimiter = delimiter;
        self
    }

    /// Set whether the CSV files have headers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = CsvFileLoader::with_glob("data/*.csv")?
    ///     .has_headers(false);
    /// ```
    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.config.has_headers = has_headers;
        self
    }

    /// Load CSV files and convert them to text documents.
    ///
    /// Each CSV file is converted to a single text document where each row
    /// is formatted as "header: value" pairs, separated by newlines.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let documents: Vec<Result<String, CsvLoaderError>> = CsvFileLoader::with_glob("data/*.csv")?
    ///     .load()
    ///     .collect();
    /// ```
    pub fn load(self) -> CsvLoadedIterator {
        CsvLoadedIterator {
            inner: self.paths.into_iter(),
            config: self.config,
        }
    }

    /// Load CSV files and return them with their file paths.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for (path, content) in loader.load_with_path().ignore_errors() {
    ///     println!("File: {:?}\n{}", path, content);
    /// }
    /// ```
    pub fn load_with_path(self) -> CsvLoadedWithPathIterator {
        CsvLoadedWithPathIterator {
            inner: self.paths.into_iter(),
            config: self.config,
        }
    }

    /// Load CSV files and return each row as a separate document.
    ///
    /// This is useful when you want to embed each row separately for more
    /// granular retrieval.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let rows: Vec<String> = CsvFileLoader::with_glob("data/*.csv")?
    ///     .load_rows()
    ///     .ignore_errors()
    ///     .collect();
    /// ```
    pub fn load_rows(self) -> CsvRowIterator {
        CsvRowIterator {
            paths: self.paths.into_iter(),
            config: self.config,
            current_reader: None,
            current_headers: None,
        }
    }

    /// Convert a CSV reader to a document string.
    fn reader_to_document<R: std::io::Read>(
        reader: &mut csv::Reader<R>,
        has_headers: bool,
    ) -> Result<String, CsvLoaderError> {
        let headers: Vec<String> = if has_headers {
            reader
                .headers()?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // Generate column names like "column_0", "column_1", etc.
            vec![]
        };

        let mut document_parts: Vec<String> = Vec::new();

        for (row_idx, result) in reader.records().enumerate() {
            let record = result?;
            let row_parts: Vec<String> = record
                .iter()
                .enumerate()
                .map(|(col_idx, value)| {
                    let header = if has_headers && col_idx < headers.len() {
                        headers[col_idx].clone()
                    } else {
                        format!("column_{}", col_idx)
                    };
                    format!("{}: {}", header, value)
                })
                .collect();

            if !row_parts.is_empty() {
                document_parts.push(format!("Row {}:\n{}", row_idx + 1, row_parts.join("\n")));
            }
        }

        if document_parts.is_empty() {
            return Err(CsvLoaderError::EmptyFile);
        }

        Ok(document_parts.join("\n\n"))
    }

    /// Convert a CSV reader row to a document string.
    fn row_to_document(
        record: &csv::StringRecord,
        headers: &[String],
        has_headers: bool,
    ) -> String {
        record
            .iter()
            .enumerate()
            .map(|(col_idx, value)| {
                let header = if has_headers && col_idx < headers.len() {
                    headers[col_idx].clone()
                } else {
                    format!("column_{}", col_idx)
                };
                format!("{}: {}", header, value)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Iterator that yields loaded CSV documents.
pub struct CsvLoadedIterator {
    inner: std::vec::IntoIter<Result<PathBuf, CsvLoaderError>>,
    config: CsvConfig,
}

impl Iterator for CsvLoadedIterator {
    type Item = Result<String, CsvLoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.inner.next()?;

        Some(match path {
            Ok(path) => load_csv_file(&path, &self.config),
            Err(e) => Err(e),
        })
    }
}

impl CsvLoadedIterator {
    /// Filter out errors and return only successful results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let documents: Vec<String> = loader.load().ignore_errors().collect();
    /// ```
    pub fn ignore_errors(self) -> impl Iterator<Item = String> {
        self.filter_map(Result::ok)
    }
}

/// Iterator that yields loaded CSV documents with their file paths.
pub struct CsvLoadedWithPathIterator {
    inner: std::vec::IntoIter<Result<PathBuf, CsvLoaderError>>,
    config: CsvConfig,
}

impl Iterator for CsvLoadedWithPathIterator {
    type Item = Result<(PathBuf, String), CsvLoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.inner.next()?;

        Some(match path {
            Ok(path) => {
                let content = load_csv_file(&path, &self.config)?;
                Ok((path, content))
            }
            Err(e) => Err(e),
        })
    }
}

impl CsvLoadedWithPathIterator {
    /// Filter out errors and return only successful results.
    pub fn ignore_errors(self) -> impl Iterator<Item = (PathBuf, String)> {
        self.filter_map(Result::ok)
    }
}

/// Iterator that yields individual rows from CSV files.
pub struct CsvRowIterator {
    paths: std::vec::IntoIter<Result<PathBuf, CsvLoaderError>>,
    config: CsvConfig,
    current_reader: Option<csv::Reader<BufReader<File>>>,
    current_headers: Option<Vec<String>>,
}

impl Iterator for CsvRowIterator {
    type Item = Result<String, CsvLoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to get next record from current reader
            if let Some(ref mut reader) = self.current_reader {
                match reader.records().next() {
                    Some(Ok(record)) => {
                        let headers = self.current_headers.as_ref().map(|h| h.as_slice()).unwrap_or(&[]);
                        return Some(Ok(CsvFileLoader::row_to_document(
                            &record,
                            headers,
                            self.config.has_headers,
                        )));
                    }
                    Some(Err(e)) => return Some(Err(CsvLoaderError::from(e))),
                    None => {
                        // Current file exhausted, move to next
                        self.current_reader = None;
                        self.current_headers = None;
                    }
                }
            }

            // Get next file
            let path = match self.paths.next()? {
                Ok(p) => p,
                Err(e) => return Some(Err(e)),
            };

            // Open new reader
            match open_csv_reader(&path, &self.config) {
                Ok(mut reader) => {
                    // Store headers
                    if self.config.has_headers {
                        match reader.headers() {
                            Ok(headers) => {
                                self.current_headers = Some(
                                    headers.iter().map(|h| h.to_string()).collect()
                                );
                            }
                            Err(e) => return Some(Err(CsvLoaderError::from(e))),
                        }
                    }
                    self.current_reader = Some(reader);
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

impl CsvRowIterator {
    /// Filter out errors and return only successful results.
    pub fn ignore_errors(self) -> impl Iterator<Item = String> {
        self.filter_map(Result::ok)
    }
}

/// Open a CSV file and create a reader with the given configuration.
fn open_csv_reader(path: &Path, config: &CsvConfig) -> Result<csv::Reader<BufReader<File>>, CsvLoaderError> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);

    let reader = csv::ReaderBuilder::new()
        .delimiter(config.delimiter)
        .has_headers(config.has_headers)
        .trim(if config.trim {
            csv::Trim::All
        } else {
            csv::Trim::None
        })
        .flexible(config.flexible)
        .from_reader(buf_reader);

    Ok(reader)
}

/// Load a CSV file and convert it to a document string.
fn load_csv_file(path: &Path, config: &CsvConfig) -> Result<String, CsvLoaderError> {
    let mut reader = open_csv_reader(path, config)?;
    CsvFileLoader::reader_to_document(&mut reader, config.has_headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn test_load_simple_csv() {
        let csv_content = "name,age,city\nAlice,30,New York\nBob,25,Los Angeles";
        let file = create_test_csv(csv_content);

        let loader = CsvFileLoader::from_path(file.path());
        let documents: Vec<String> = loader.load().ignore_errors().collect();

        assert_eq!(documents.len(), 1);
        assert!(documents[0].contains("name: Alice"));
        assert!(documents[0].contains("age: 30"));
        assert!(documents[0].contains("city: New York"));
    }

    #[test]
    fn test_load_csv_without_headers() {
        let csv_content = "Alice,30,New York\nBob,25,Los Angeles";
        let file = create_test_csv(csv_content);

        let loader = CsvFileLoader::from_path(file.path()).has_headers(false);
        let documents: Vec<String> = loader.load().ignore_errors().collect();

        assert_eq!(documents.len(), 1);
        assert!(documents[0].contains("column_0: Alice"));
        assert!(documents[0].contains("column_1: 30"));
    }

    #[test]
    fn test_load_tsv() {
        let tsv_content = "name\tage\tcity\nAlice\t30\tNew York";
        let file = create_test_csv(tsv_content);

        let loader = CsvFileLoader::from_path(file.path())
            .with_config(CsvConfig::tsv());
        let documents: Vec<String> = loader.load().ignore_errors().collect();

        assert_eq!(documents.len(), 1);
        assert!(documents[0].contains("name: Alice"));
    }

    #[test]
    fn test_load_rows() {
        let csv_content = "name,age\nAlice,30\nBob,25";
        let file = create_test_csv(csv_content);

        let loader = CsvFileLoader::from_path(file.path());
        let rows: Vec<String> = loader.load_rows().ignore_errors().collect();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains("name: Alice"));
        assert!(rows[1].contains("name: Bob"));
    }

    #[test]
    fn test_load_with_path() {
        let csv_content = "name,age\nAlice,30";
        let file = create_test_csv(csv_content);

        let loader = CsvFileLoader::from_path(file.path());
        let results: Vec<(PathBuf, String)> = loader.load_with_path().ignore_errors().collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, file.path());
        assert!(results[0].1.contains("name: Alice"));
    }

    #[test]
    fn test_from_bytes() {
        let csv_bytes = b"name,age\nAlice,30\nBob,25";
        let document = CsvFileLoader::from_bytes(csv_bytes, CsvConfig::default()).unwrap();

        assert!(document.contains("name: Alice"));
        assert!(document.contains("name: Bob"));
    }

    #[test]
    fn test_empty_csv() {
        let csv_content = "name,age";
        let file = create_test_csv(csv_content);

        let loader = CsvFileLoader::from_path(file.path());
        let results: Vec<Result<String, CsvLoaderError>> = loader.load().collect();

        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], Err(CsvLoaderError::EmptyFile)));
    }

    #[test]
    fn test_custom_delimiter() {
        let csv_content = "name;age;city\nAlice;30;New York";
        let file = create_test_csv(csv_content);

        let loader = CsvFileLoader::from_path(file.path()).delimiter(b';');
        let documents: Vec<String> = loader.load().ignore_errors().collect();

        assert_eq!(documents.len(), 1);
        assert!(documents[0].contains("name: Alice"));
    }
}
