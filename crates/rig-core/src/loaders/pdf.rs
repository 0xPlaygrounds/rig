//! Lazy PDF loading and text extraction from paths or bytes.
//!
//! ```no_run
//! use rig_core::loaders::PdfFileLoader;
//!
//! let documents = PdfFileLoader::with_glob("documents/*.pdf")?
//!     .read().into_iter().collect::<Result<Vec<_>, _>>()?;
//! # let _ = documents;
//! # Ok::<(), rig_core::loaders::pdf::PdfLoaderError>(())
//! ```

use std::path::PathBuf;

use lopdf::{Document, Error as LopdfError};
use thiserror::Error;

use super::file::FileLoaderError;

#[derive(Error, Debug)]
pub enum PdfLoaderError {
    #[error("{0}")]
    FileLoaderError(#[from] FileLoaderError),

    #[error("UTF-8 conversion error: {0}")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),

    #[error("IO error: {0}")]
    PdfError(#[from] LopdfError),
}

loadable_trait!(Loadable, PdfLoaderError, Document, load, load_with_path);

impl Loadable for PathBuf {
    fn load(self) -> Result<Document, PdfLoaderError> {
        Document::load(self).map_err(PdfLoaderError::PdfError)
    }
    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError> {
        let contents = Document::load(&self);
        Ok((self, contents?))
    }
}

impl Loadable for Vec<u8> {
    fn load(self) -> Result<Document, PdfLoaderError> {
        Document::load_mem(&self).map_err(PdfLoaderError::PdfError)
    }

    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError> {
        let doc = Document::load_mem(&self).map_err(PdfLoaderError::PdfError)?;
        Ok((PathBuf::from("<memory>"), doc))
    }
}

/// Iterator pipeline for loading PDF documents and extracting text synchronously.
/// Loading and extraction errors are yielded per item unless filtered.
pub struct PdfFileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

#[allow(private_bounds)] // `Loadable` deliberately seals which states expose these methods
impl<'a, T: Loadable + 'a> PdfFileLoader<'a, T> {
    /// Parses each input as a PDF during iteration, yielding document-loading errors.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(doc) => println!("{doc:?}"),
    ///         Err(e) => eprintln!("Error reading pdf: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(self) -> PdfFileLoader<'a, Result<Document, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(Loadable::load)),
        }
    }

    /// Parses each PDF and pairs it with its path, yielding loading errors.
    /// In-memory inputs use the path `<memory>`.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{path:?} {doc:?}"),
    ///         Err(e) => eprintln!("Error reading pdf: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, Document), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(Loadable::load_with_path)),
        }
    }
}

/// Extract each page's text, paired with its zero-based page number.
fn page_texts(doc: &Document) -> Vec<(usize, Result<String, PdfLoaderError>)> {
    doc.page_iter()
        .enumerate()
        .map(|(page_no, _)| {
            (
                page_no,
                doc.extract_text(&[page_no as u32 + 1])
                    .map_err(PdfLoaderError::PdfError),
            )
        })
        .collect()
}

/// Concatenate the text of every page, failing on the first unreadable page.
fn all_text(doc: &Document) -> Result<String, PdfLoaderError> {
    page_texts(doc).into_iter().map(|(_, text)| text).collect()
}

#[allow(private_bounds)] // `Loadable` deliberately seals which states expose these methods
impl<'a, T: Loadable + 'a> PdfFileLoader<'a, T> {
    /// Loads each PDF and concatenates its page text without separators.
    /// Yields a loading error or the first page-extraction error for each document.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{content}"),
    ///         Err(e) => eprintln!("Error reading pdf: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(self) -> PdfFileLoader<'a, Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| all_text(&res.load()?))),
        }
    }

    /// Loads each PDF and pairs its path with concatenated page text.
    /// Yields loading or extraction errors; in-memory inputs use `<memory>`.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{path:?} {content}"),
    ///         Err(e) => eprintln!("Error reading pdf: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, String), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let (path, doc) = res.load_with_path()?;
                let content = all_text(&doc)?;
                Ok((path, content))
            })),
        }
    }
}

impl<'a> PdfFileLoader<'a, Document> {
    /// Yields page text in document order, flattening all documents into one
    /// sequence of per-page extraction results.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?
    ///     .load()
    ///     .ignore_errors()
    ///     .by_page()
    ///     .into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(page) => println!("{page}"),
    ///         Err(e) => eprintln!("Error reading pdf: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn by_page(self) -> PdfFileLoader<'a, Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(
                self.iterator
                    .flat_map(|doc| page_texts(&doc).into_iter().map(|(_, text)| text)),
            ),
        }
    }
}

type ByPage = (PathBuf, Vec<(usize, Result<String, PdfLoaderError>)>);
impl<'a> PdfFileLoader<'a, (PathBuf, Document)> {
    /// Pairs each source path with zero-based page numbers and extraction results.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?
    ///     .load_with_path()
    ///     .ignore_errors()
    ///     .by_page()
    ///     .into_iter();
    ///
    /// for (path, pages) in content {
    ///     println!("{}", path.display());
    ///     for (pageno, result) in pages {
    ///         match result {
    ///             Ok(content) => println!("Page {pageno}: {content}"),
    ///             Err(e) => eprintln!("Error reading page: {e}"),
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn by_page(self) -> PdfFileLoader<'a, ByPage> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|(path, doc)| (path, page_texts(&doc)))),
        }
    }
}

impl<'a> PdfFileLoader<'a, ByPage> {
    /// Drops failed pages while retaining each document's path and original
    /// zero-based page numbers. Documents with no successful pages remain.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?
    ///     .load_with_path()
    ///     .ignore_errors()
    ///     .by_page()
    ///     .ignore_errors();
    /// for (_path, pages) in content {
    ///     println!("{}", pages.len())
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn ignore_errors(self) -> PdfFileLoader<'a, (PathBuf, Vec<(usize, String)>)> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|(path, pages)| {
                let pages = pages
                    .into_iter()
                    .filter_map(|(page_no, res)| res.ok().map(|content| (page_no, content)))
                    .collect::<Vec<_>>();
                (path, pages)
            })),
        }
    }
}

loader_scaffold!(PdfFileLoader, PdfLoaderError, dir: all_entries);
loader_from_bytes!(PdfFileLoader);

#[cfg(test)]
mod tests;
