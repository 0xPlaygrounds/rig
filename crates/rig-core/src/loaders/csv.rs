//! CSV loading from paths or bytes, as one document per file or per row.
//!
//! Each row renders as one `header: value` line per column, so a model or
//! embedding sees every value next to its column name. A file renders as its
//! rows separated by blank lines.
//!
//! ```
//! use rig_core::loaders::CsvFileLoader;
//!
//! let rows = CsvFileLoader::from_bytes(b"name,role\nAda,engineer\n".to_vec())
//!     .rows().into_iter().collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(rows, vec![vec!["name: Ada\nrole: engineer".to_string()]]);
//! # Ok::<(), rig_core::loaders::csv::CsvLoaderError>(())
//! ```

use std::path::PathBuf;

use ::csv::{ReaderBuilder, StringRecord};
use thiserror::Error;

use super::file::FileLoaderError;

#[derive(Error, Debug)]
pub enum CsvLoaderError {
    #[error("{0}")]
    FileLoaderError(#[from] FileLoaderError),

    #[error("CSV error: {0}")]
    CsvError(#[from] ::csv::Error),
}

loadable_trait!(Loadable, CsvLoaderError, Vec<u8>, load, load_with_path);

impl Loadable for PathBuf {
    fn load(self) -> Result<Vec<u8>, CsvLoaderError> {
        Ok(std::fs::read(self).map_err(FileLoaderError::IoError)?)
    }

    fn load_with_path(self) -> Result<(PathBuf, Vec<u8>), CsvLoaderError> {
        let bytes = std::fs::read(&self).map_err(FileLoaderError::IoError)?;
        Ok((self, bytes))
    }
}

impl Loadable for Vec<u8> {
    fn load(self) -> Result<Vec<u8>, CsvLoaderError> {
        Ok(self)
    }

    fn load_with_path(self) -> Result<(PathBuf, Vec<u8>), CsvLoaderError> {
        Ok((PathBuf::from("<memory>"), self))
    }
}

/// How the loader parses its input.
#[derive(Debug, Clone, Copy)]
struct CsvFormat {
    delimiter: u8,
    has_headers: bool,
    flexible: bool,
}

impl Default for CsvFormat {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            flexible: false,
        }
    }
}

impl CsvFormat {
    /// Parses `bytes` and renders each record, failing on the first malformed
    /// record.
    fn rows(self, bytes: &[u8]) -> Result<Vec<String>, CsvLoaderError> {
        let mut reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_headers)
            .flexible(self.flexible)
            .from_reader(bytes);
        let headers = if self.has_headers {
            Some(reader.headers()?.clone())
        } else {
            None
        };
        reader
            .records()
            .map(|record| Ok(render_row(headers.as_ref(), &record?)))
            .collect()
    }

    fn document(self, bytes: &[u8]) -> Result<String, CsvLoaderError> {
        Ok(self.rows(bytes)?.join("\n\n"))
    }
}

/// One line per field: `header: value` under a header, the bare value otherwise.
fn render_row(headers: Option<&StringRecord>, record: &StringRecord) -> String {
    let lines: Vec<String> = record
        .iter()
        .enumerate()
        .map(
            |(column, value)| match headers.and_then(|headers| headers.get(column)) {
                Some(header) => format!("{header}: {value}"),
                None => value.to_string(),
            },
        )
        .collect();
    lines.join("\n")
}

/// Iterator pipeline for loading CSV files as text documents. Reads and parsing
/// happen synchronously during iteration; per-item I/O and parse errors are
/// yielded unless filtered.
///
/// Input is comma-separated, with a header row and the same number of fields in
/// every row, unless [`CsvFileLoader::delimiter`], [`CsvFileLoader::has_headers`]
/// or [`CsvFileLoader::flexible`] say otherwise.
pub struct CsvFileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
    format: CsvFormat,
}

#[allow(private_bounds)] // `Loadable` deliberately seals which states expose these methods
impl<'a, T: Loadable + 'a> CsvFileLoader<'a, T> {
    /// Sets the field delimiter, such as `b'\t'` for tab-separated values.
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.format.delimiter = delimiter;
        self
    }

    /// Sets whether the first row names the columns. Without headers, each
    /// row renders as its bare values, one per line.
    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.format.has_headers = has_headers;
        self
    }

    /// Sets whether rows may have different numbers of fields. Otherwise such
    /// a row fails its file. Fields beyond the header row render as bare values.
    pub fn flexible(mut self, flexible: bool) -> Self {
        self.format.flexible = flexible;
        self
    }

    /// Renders each file as one document, yielding I/O or parse errors.
    ///
    /// ```no_run
    /// # use rig_core::loaders::CsvFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// for result in CsvFileLoader::with_glob("data/*.csv")?.read() {
    ///     match result {
    ///         Ok(document) => println!("{document}"),
    ///         Err(e) => eprintln!("Error reading csv: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(self) -> CsvFileLoader<'a, Result<String, CsvLoaderError>> {
        let format = self.format;
        CsvFileLoader {
            iterator: Box::new(
                self.iterator
                    .map(move |item| format.document(&item.load()?)),
            ),
            format,
        }
    }

    /// Renders each file as one document paired with its path. In-memory inputs
    /// use the path `<memory>`.
    pub fn read_with_path(self) -> CsvFileLoader<'a, Result<(PathBuf, String), CsvLoaderError>> {
        let format = self.format;
        CsvFileLoader {
            iterator: Box::new(self.iterator.map(move |item| {
                let (path, bytes) = item.load_with_path()?;
                Ok((path, format.document(&bytes)?))
            })),
            format,
        }
    }

    /// Renders each file as one document per row, yielding I/O or parse errors
    /// for the whole file.
    ///
    /// ```no_run
    /// # use rig_core::loaders::CsvFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let rows: Vec<String> = CsvFileLoader::with_glob("data/*.tsv")?
    ///     .delimiter(b'\t')
    ///     .rows()
    ///     .ignore_errors()
    ///     .into_iter()
    ///     .flatten()
    ///     .collect();
    /// # Ok(())
    /// # }
    /// ```
    pub fn rows(self) -> CsvFileLoader<'a, Result<Vec<String>, CsvLoaderError>> {
        let format = self.format;
        CsvFileLoader {
            iterator: Box::new(self.iterator.map(move |item| format.rows(&item.load()?))),
            format,
        }
    }

    /// Renders each file as one document per row, paired with its path.
    /// In-memory inputs use the path `<memory>`.
    pub fn rows_with_path(
        self,
    ) -> CsvFileLoader<'a, Result<(PathBuf, Vec<String>), CsvLoaderError>> {
        let format = self.format;
        CsvFileLoader {
            iterator: Box::new(self.iterator.map(move |item| {
                let (path, bytes) = item.load_with_path()?;
                Ok((path, format.rows(&bytes)?))
            })),
            format,
        }
    }
}

loader_scaffold!(CsvFileLoader, CsvLoaderError, dir: files_only, settings: format);
loader_from_bytes!(CsvFileLoader, settings: format);

#[cfg(test)]
mod tests;
