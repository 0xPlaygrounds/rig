use std::{fs, path::PathBuf};

use glob::glob;
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

pub(crate) trait Loadable {
    fn load(self) -> Result<Document, PdfLoaderError>;
    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError>;
}

impl Loadable for PathBuf {
    fn load(self) -> Result<Document, PdfLoaderError> {
        Document::load(self).map_err(PdfLoaderError::PdfError)
    }
    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError> {
        let contents = Document::load(&self);
        Ok((self, contents?))
    }
}

impl<T> Loadable for Result<T, PdfLoaderError>
where
    T: Loadable,
{
    fn load(self) -> Result<Document, PdfLoaderError> {
        self.map(|t| t.load())?
    }
    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError> {
        self.map(|t| t.load_with_path())?
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
///
/// # Evaluation is eager
///
/// Each stage ([`load`](Self::load), [`by_page`](Self::by_page), …) runs to
/// completion and materialises a `Vec` before the next one starts. The loader
/// previously held a `Box<dyn Iterator>`, which let stages compose lazily;
/// de-erasing that field costs the laziness, because the closure types the
/// stages produce cannot be named. Loading a large corpus therefore holds every
/// parsed [`Document`] in memory at once — chunk the glob if that matters.
pub struct PdfFileLoader<T> {
    iterator: std::vec::IntoIter<T>,
}

/// Collects a pipeline stage into a nameable iterator.
///
/// The loader used to hold a `Box<dyn Iterator<Item = T> + 'a>` so each stage
/// could wrap the previous one lazily. Closures have no nameable type, so
/// de-erasing the field means each stage materialises into a `Vec`. See the
/// laziness note on [`PdfFileLoader`].
fn eager<T>(iterator: impl Iterator<Item = T>) -> std::vec::IntoIter<T> {
    iterator.collect::<Vec<_>>().into_iter()
}

impl PdfFileLoader<Result<PathBuf, PdfLoaderError>> {
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
    pub fn load(self) -> PdfFileLoader<Result<Document, PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|res| res.load())),
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
    pub fn load_with_path(self) -> PdfFileLoader<Result<(PathBuf, Document), PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|res| res.load_with_path())),
        }
    }
}

impl PdfFileLoader<Result<PathBuf, PdfLoaderError>> {
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
    pub fn read(self) -> PdfFileLoader<Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|res| {
                let doc = res.load()?;
                Ok(doc
                    .page_iter()
                    .enumerate()
                    .map(|(page_no, _)| {
                        doc.extract_text(&[page_no as u32 + 1])
                            .map_err(PdfLoaderError::PdfError)
                    })
                    .collect::<Result<Vec<String>, PdfLoaderError>>()?
                    .into_iter()
                    .collect::<String>())
            })),
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
    pub fn read_with_path(self) -> PdfFileLoader<Result<(PathBuf, String), PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|res| {
                let (path, doc) = res.load_with_path()?;
                println!(
                    "Loaded {:?} PDF: {:?}",
                    path,
                    doc.page_iter().collect::<Vec<_>>()
                );
                let content = doc
                    .page_iter()
                    .enumerate()
                    .map(|(page_no, _)| {
                        doc.extract_text(&[page_no as u32 + 1])
                            .map_err(PdfLoaderError::PdfError)
                    })
                    .collect::<Result<Vec<String>, PdfLoaderError>>()?
                    .into_iter()
                    .collect::<String>();

                Ok((path, content))
            })),
        }
    }
}

impl PdfFileLoader<Document> {
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
    pub fn by_page(self) -> PdfFileLoader<Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.flat_map(|doc| {
                doc.page_iter()
                    .enumerate()
                    .map(|(page_no, _)| {
                        doc.extract_text(&[page_no as u32 + 1])
                            .map_err(PdfLoaderError::PdfError)
                    })
                    .collect::<Vec<_>>()
            })),
        }
    }
}

type ByPage = (PathBuf, Vec<(usize, Result<String, PdfLoaderError>)>);
impl PdfFileLoader<(PathBuf, Document)> {
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
    pub fn by_page(self) -> PdfFileLoader<ByPage> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|(path, doc)| {
                (
                    path,
                    doc.page_iter()
                        .enumerate()
                        .map(|(page_no, _)| {
                            (
                                page_no,
                                doc.extract_text(&[page_no as u32 + 1])
                                    .map_err(PdfLoaderError::PdfError),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })),
        }
    }
}

impl PdfFileLoader<ByPage> {
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
    pub fn ignore_errors(self) -> PdfFileLoader<(PathBuf, Vec<(usize, String)>)> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|(path, pages)| {
                let pages = pages
                    .into_iter()
                    .filter_map(|(page_no, res)| res.ok().map(|content| (page_no, content)))
                    .collect::<Vec<_>>();
                (path, pages)
            })),
        }
    }
}

impl<T> PdfFileLoader<Result<T, PdfLoaderError>> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [PdfFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.pdf" and ignore errors from unreadable files.
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read().ignore_errors();
    /// for content in content {
    ///     println!("{}", content)
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn ignore_errors(self) -> PdfFileLoader<T> {
        PdfFileLoader {
            iterator: eager(self.iterator.filter_map(|res| res.ok())),
        }
    }
}

impl PdfFileLoader<Result<PathBuf, FileLoaderError>> {
    /// Creates a new [PdfFileLoader] using a glob pattern to match files.
    ///
    /// # Example
    /// Create a [PdfFileLoader] for all `.pdf` files that match the glob "tests/data/*.pdf".
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let loader = PdfFileLoader::with_glob("tests/data/*.pdf")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_glob(
        pattern: &str,
    ) -> Result<PdfFileLoader<Result<PathBuf, PdfLoaderError>>, PdfLoaderError> {
        let paths = glob(pattern).map_err(FileLoaderError::PatternError)?;
        Ok(PdfFileLoader {
            iterator: eager(paths.into_iter().map(|path| {
                path.map_err(FileLoaderError::GlobError)
                    .map_err(PdfLoaderError::FileLoaderError)
            })),
        })
    }

    /// Creates a new [PdfFileLoader] on all files within a directory.
    ///
    /// # Example
    /// Create a [PdfFileLoader] for all files that are in the directory "files".
    ///
    /// ```no_run
    /// # use rig_core::loaders::PdfFileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let loader = PdfFileLoader::with_dir("files")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_dir(
        directory: &str,
    ) -> Result<PdfFileLoader<Result<PathBuf, PdfLoaderError>>, PdfLoaderError> {
        Ok(PdfFileLoader {
            iterator: eager(
                fs::read_dir(directory)
                    .map_err(FileLoaderError::IoError)?
                    .map(|entry| Ok(entry.map_err(FileLoaderError::IoError)?.path())),
            ),
        })
    }
}

impl PdfFileLoader<Vec<u8>> {
    /// Ingest a PDF as a byte array.
    pub fn from_bytes(bytes: Vec<u8>) -> PdfFileLoader<Vec<u8>> {
        PdfFileLoader {
            iterator: eager(vec![bytes].into_iter()),
        }
    }

    /// Ingest multiple byte arrays.
    pub fn from_bytes_multi(bytes_vec: Vec<Vec<u8>>) -> PdfFileLoader<Vec<u8>> {
        PdfFileLoader {
            iterator: eager(bytes_vec.into_iter()),
        }
    }

    /// Use this once you've created the loader to load the document in.
    pub fn load(self) -> PdfFileLoader<Result<Document, PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|res| res.load())),
        }
    }

    /// Use this once you've created the loader to load the document in (and get the path).
    pub fn load_with_path(self) -> PdfFileLoader<Result<(PathBuf, Document), PdfLoaderError>> {
        PdfFileLoader {
            iterator: eager(self.iterator.map(|res| res.load_with_path())),
        }
    }
}

// ================================================================
// PDFFileLoader iterator implementations
// ================================================================

impl<T> IntoIterator for PdfFileLoader<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iterator
    }
}

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
}
