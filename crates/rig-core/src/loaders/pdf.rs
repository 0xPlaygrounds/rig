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

// ================================================================
// Implementing Loadable trait for loading pdfs
// ================================================================

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

// ================================================================
// PdfFileLoader definitions and implementations
// ================================================================

/// [PdfFileLoader] is a utility for loading pdf files from the filesystem using glob patterns or
///  directory paths. It provides methods to read file contents and handle errors gracefully.
///
/// # Errors
///
/// This module defines a custom error type [PdfLoaderError] which can represent various errors
///  that might occur during file loading operations, such as any [FileLoaderError] alongside
///  specific PDF-related errors.
///
/// # Example Usage
///
/// ```no_run
/// use rig_core::loaders::PdfFileLoader;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a FileLoader using a glob pattern
///     let loader = PdfFileLoader::with_glob("tests/data/*.pdf")?;
///
///     // Load pdf file contents by page, ignoring any errors
///     let contents: Vec<String> = loader
///         .load()
///         .ignore_errors()
///         .by_page()
///         .ignore_errors()
///         .into_iter()
///         .collect();
///
///     for content in contents {
///         println!("{}", content);
///     }
///
///     Ok(())
/// }
/// ```
///
/// [PdfFileLoader] uses strict typing between the iterator methods to ensure that transitions
///  between different implementations of the loaders and it's methods are handled properly by
///  the compiler.
pub struct PdfFileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

#[allow(private_bounds)] // `Loadable` deliberately seals which states expose these methods
impl<'a, T: Loadable + 'a> PdfFileLoader<'a, T> {
    /// Loads the contents of the pdfs within the iterator returned by [PdfFileLoader::with_glob]
    ///  or [PdfFileLoader::with_dir]. Loaded PDF documents are raw PDF instances that can be
    ///  further processed (by page, etc).
    ///
    /// # Example
    /// Load pdfs in directory "tests/data/*.pdf" and return the loaded documents
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(doc) => println!("{:?}", doc),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(self) -> PdfFileLoader<'a, Result<Document, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load())),
        }
    }

    /// Loads the contents of the pdfs within the iterator returned by [PdfFileLoader::with_glob]
    ///  or [PdfFileLoader::with_dir]. Loaded PDF documents are raw PDF instances with their path
    ///  that can be further processed.
    ///
    /// # Example
    /// Load pdfs in directory "tests/data/*.pdf" and return the loaded documents
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{:?} {:?}", path, doc),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, Document), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load_with_path())),
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
    /// Directly reads the contents of the pdfs within the iterator returned by
    ///  [PdfFileLoader::with_glob] or [PdfFileLoader::with_dir].
    ///
    /// # Example
    /// Read pdfs in directory "tests/data/*.pdf" and return the contents of the documents.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{}", content),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
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

    /// Directly reads the contents of the pdfs within the iterator returned by
    ///  [PdfFileLoader::with_glob] or [PdfFileLoader::with_dir] and returns the path along with
    ///  the content.
    ///
    /// # Example
    /// Read pdfs in directory "tests/data/*.pdf" and return the content and paths of the documents.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{:?} {}", path, content),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
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
    /// Chunks the pages of a loaded document by page, flattened as a single vector.
    ///
    /// # Example
    /// Load pdfs in directory "tests/data/*.pdf" and chunk all document into it's pages.
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
    ///         Ok(page) => println!("{}", page),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
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
    /// Chunks the pages of a loaded document by page, processed as a vector of documents by path
    ///  which each document container an inner vector of pages by page number.
    ///
    /// # Example
    /// Read pdfs in directory "tests/data/*.pdf" and chunk all documents by path by it's pages.
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
    ///             Ok(content) => println!("Page {}: {}", pageno, content),
    ///             Err(e) => eprintln!("Error reading page: {}", e),
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
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [PdfFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.pdf" and ignore errors from unreadable files.
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
mod tests {
    use crate::loaders::test_fixtures::{fixture_glob, fixture_path};

    use super::PdfFileLoader;

    #[test]
    fn test_pdf_loader() {
        let glob = fixture_glob("*.pdf");
        let loader = PdfFileLoader::with_glob(&glob).unwrap();
        let actual = loader
            .load_with_path()
            .ignore_errors()
            .by_page()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        let mut actual = actual
            .into_iter()
            .map(|result| {
                let (path, pages) = result;
                pages.iter().for_each(|(page_no, content)| {
                    println!("{path:?} Page {page_no}: {content:?}");
                });
                (path, pages)
            })
            .collect::<Vec<_>>();

        let mut expected = vec![
            (
                fixture_path("dummy.pdf"),
                vec![(0, "Test\nPDF\nDocument\n".to_string())],
            ),
            (
                fixture_path("file-id-verifiers.pdf"),
                vec![
                    (0, "rig-file-id-page-one-verifier-3a91\n".to_string()),
                    (1, "rig-file-id-page-two-verifier-8c27\n".to_string()),
                    (2, "rig-file-id-page-three-verifier-f54e\n".to_string()),
                ],
            ),
            (
                fixture_path("pages.pdf"),
                vec![
                    (0, "Page\n1\n".to_string()),
                    (1, "Page\n2\n".to_string()),
                    (2, "Page\n3\n".to_string()),
                ],
            ),
        ];

        actual.sort();
        expected.sort();

        assert!(!actual.is_empty());
        assert!(expected == actual)
    }

    #[test]
    fn test_pdf_loader_bytes() {
        // this should never fail!
        let bytes = std::fs::read(fixture_path("dummy.pdf")).unwrap();

        let loader = PdfFileLoader::from_bytes(bytes);

        let actual = loader
            .load()
            .ignore_errors()
            .by_page()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);
        assert_eq!(actual, vec!["Test\nPDF\nDocument\n".to_string()]);

        // this should never fail!
        let bytes = std::fs::read(fixture_path("pages.pdf")).unwrap();

        let loader = PdfFileLoader::from_bytes(bytes);

        let actual = loader
            .load()
            .ignore_errors()
            .by_page()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 3);
        assert_eq!(
            actual,
            vec![
                "Page\n1\n".to_string(),
                "Page\n2\n".to_string(),
                "Page\n3\n".to_string(),
            ]
        );
    }

    #[test]
    fn test_pdf_loader_bytes_multi() {
        let dummy = std::fs::read(fixture_path("dummy.pdf")).unwrap();
        let pages = std::fs::read(fixture_path("pages.pdf")).unwrap();

        let loader = PdfFileLoader::from_bytes_multi(vec![dummy, pages]);

        let actual = loader
            .load()
            .ignore_errors()
            .by_page()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                "Test\nPDF\nDocument\n".to_string(),
                "Page\n1\n".to_string(),
                "Page\n2\n".to_string(),
                "Page\n3\n".to_string(),
            ]
        );
    }
}
