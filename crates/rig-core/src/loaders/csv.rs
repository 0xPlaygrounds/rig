//! Lazy CSV loading and row-level text extraction from paths or bytes.
//!
//! Each CSV file is converted into a single text document with rows formatted
//! as `header: value` pairs, suitable for embedding in RAG pipelines.
//!
//! ```
//! use rig_core::loaders::CsvFileLoader;
//!
//! let csv_data = b"name,age\nAlice,30\nBob,25".to_vec();
//! let documents: Vec<_> = CsvFileLoader::from_bytes(csv_data)
//!     .read()
//!     .ignore_errors()
//!     .into_iter()
//!     .collect();
//! assert_eq!(documents.len(), 1);
//! ```

use std::path::PathBuf;

use thiserror::Error;

use super::file::FileLoaderError;

#[derive(Error, Debug)]
pub enum CsvLoaderError {
    #[error("{0}")]
    FileLoaderError(#[from] FileLoaderError),

    #[error("CSV parse error: {0}")]
    CsvError(#[from] ::csv::Error),

    #[error("CSV has no headers")]
    MissingHeaders,
}

loadable_trait!(Readable, CsvLoaderError, String, read, read_with_path);

impl Readable for PathBuf {
    fn read(self) -> Result<String, CsvLoaderError> {
        let bytes = std::fs::read(&self).map_err(FileLoaderError::IoError)?;
        parse_csv(&bytes)
    }

    fn read_with_path(self) -> Result<(PathBuf, String), CsvLoaderError> {
        let bytes = std::fs::read(&self).map_err(FileLoaderError::IoError)?;
        let content = parse_csv(&bytes)?;
        Ok((self, content))
    }
}

impl Readable for Vec<u8> {
    fn read(self) -> Result<String, CsvLoaderError> {
        parse_csv(&self)
    }

    fn read_with_path(self) -> Result<(PathBuf, String), CsvLoaderError> {
        let content = parse_csv(&self)?;
        Ok((PathBuf::from("<memory>"), content))
    }
}

/// Parses CSV bytes into a text document. Each row becomes a block of
/// `header: value` lines separated by a blank line from the next row.
fn parse_csv(data: &[u8]) -> Result<String, CsvLoaderError> {
    let mut reader = ::csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(data);

    let headers = reader.headers().map_err(CsvLoaderError::CsvError)?.clone();

    if headers.is_empty() {
        return Err(CsvLoaderError::MissingHeaders);
    }

    let mut output = String::new();
    for record in reader.records() {
        let record = record?;
        if !output.is_empty() {
            output.push('\n');
        }
        for (header, value) in headers.iter().zip(record.iter()) {
            output.push_str(header);
            output.push_str(": ");
            output.push_str(value);
            output.push('\n');
        }
    }

    Ok(output)
}

/// Iterator pipeline for loading CSV files and extracting text. Each CSV file
/// becomes a single document with rows formatted as `header: value` pairs.
pub struct CsvFileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

#[allow(private_bounds)]
impl<'a, T: Readable + 'a> CsvFileLoader<'a, T> {
    /// Parses each input as CSV during iteration, formatting rows as
    /// `header: value` pairs. Yields parse errors per document.
    ///
    /// ```no_run
    /// # use rig_core::loaders::CsvFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = CsvFileLoader::with_glob("data/*.csv")?.read();
    /// for result in content {
    ///     match result {
    ///         Ok(text) => println!("{text}"),
    ///         Err(e) => eprintln!("Error reading CSV: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(self) -> CsvFileLoader<'a, Result<String, CsvLoaderError>> {
        CsvFileLoader {
            iterator: Box::new(self.iterator.map(Readable::read)),
        }
    }

    /// Parses each CSV and pairs it with its source path. In-memory inputs
    /// use the path `<memory>`.
    ///
    /// ```no_run
    /// # use rig_core::loaders::CsvFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = CsvFileLoader::with_glob("data/*.csv")?.read_with_path();
    /// for result in content {
    ///     match result {
    ///         Ok((path, text)) => println!("{path:?}: {text}"),
    ///         Err(e) => eprintln!("Error reading CSV: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_with_path(self) -> CsvFileLoader<'a, Result<(PathBuf, String), CsvLoaderError>> {
        CsvFileLoader {
            iterator: Box::new(self.iterator.map(Readable::read_with_path)),
        }
    }
}

loader_scaffold!(CsvFileLoader, CsvLoaderError, dir: files_only);
loader_from_bytes!(CsvFileLoader);

#[cfg(test)]
mod tests;
