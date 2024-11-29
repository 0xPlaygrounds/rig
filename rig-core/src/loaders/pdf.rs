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
impl<T: Loadable> Loadable for Result<T, PdfLoaderError> {
    fn load(self) -> Result<Document, PdfLoaderError> {
        self.map(|t| t.load())?
    }
    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError> {
        self.map(|t| t.load_with_path())?
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
/// ```rust
/// use rig:loaders::PdfileLoader;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a FileLoader using a glob pattern
///     let loader = PdfFileLoader::with_glob("tests/data/*.pdf")?;
///
///     // Load pdf file contents by page, ignoring any errors
///     let contents: Vec<String> = loader
///         .load_with_path()
///         .ignore_errors()
///         .by_page()
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

impl<'a> PdfFileLoader<'a, Result<PathBuf, PdfLoaderError>> {
    /// Loads the contents of the pdfs within the iterator returned by [PdfFileLoader::with_glob]
    ///  or [PdfFileLoader::with_dir]. Loaded PDF documents are raw PDF instances that can be
    ///  further processed (by page, etc).
    ///
    /// # Example
    /// Load pdfs in directory "tests/data/*.pdf" and return the loaded documents
    ///
    /// ```rust
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{:?} {}", path, doc),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
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
    /// ```rust
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{:?} {}", path, doc),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// ```
    pub fn load_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, Document), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load_with_path())),
        }
    }
}

impl<'a> PdfFileLoader<'a, Result<PathBuf, PdfLoaderError>> {
    /// Directly reads the contents of the pdfs within the iterator returned by
    ///  [PdfFileLoader::with_glob] or [PdfFileLoader::with_dir].
    ///
    /// # Example
    /// Read pdfs in directory "tests/data/*.pdf" and return the contents of the documents.
    ///
    /// ```rust
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{}", content),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read(self) -> PdfFileLoader<'a, Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
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
    /// ```rust
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{:?} {}", path, content),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, String), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
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

impl<'a> PdfFileLoader<'a, Document> {
    /// Chunks the pages of a loaded document by page, flattened as a single vector.
    ///
    /// # Example
    /// Load pdfs in directory "tests/data/*.pdf" and chunk all document into it's pages.
    ///
    /// ```rust
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?.load().by_page().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(page) => println!("{}", page),
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// ```
    pub fn by_page(self) -> PdfFileLoader<'a, Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.flat_map(|doc| {
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
impl<'a> PdfFileLoader<'a, (PathBuf, Document)> {
    /// Chunks the pages of a loaded document by page, processed as a vector of documents by path
    ///  which each document container an inner vector of pages by page number.
    ///
    /// # Example
    /// Read pdfs in directory "tests/data/*.pdf" and chunk all documents by path by it's pages.
    ///
    /// ```rust
    /// let content = PdfFileLoader::with_glob("tests/data/*.pdf")?
    ///     .load_with_path()
    ///     .by_page()
    ///     .into_iter();
    ///
    /// for result in content {
    ///     match result {
    ///         Ok(documents) => {
    ///             for doc in documents {
    ///                 match doc {
    ///                     Ok((pageno, content)) => println!("Page {}: {}", pageno, content),
    ///                     Err(e) => eprintln!("Error reading page: {}", e),
    ///                }
    ///             }
    ///         },
    ///         Err(e) => eprintln!("Error reading pdf: {}", e),
    ///     }
    /// }
    /// ```
    pub fn by_page(self) -> PdfFileLoader<'a, ByPage> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|(path, doc)| {
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

impl<'a> PdfFileLoader<'a, ByPage> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [PdfFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.pdf" and ignore errors from unreadable files.
    ///
    /// ```rust
    /// let content = FileLoader::with_glob("tests/data/*.pdf")?.read().ignore_errors().into_iter();
    /// for result in content {
    ///     println!("{}", content)
    /// }
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

impl<'a, T: 'a> PdfFileLoader<'a, Result<T, PdfLoaderError>> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [PdfFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.pdf" and ignore errors from unreadable files.
    ///
    /// ```rust
    /// let content = FileLoader::with_glob("tests/data/*.pdf")?.read().ignore_errors().into_iter();
    /// for result in content {
    ///     println!("{}", content)
    /// }
    /// ```
    pub fn ignore_errors(self) -> PdfFileLoader<'a, T> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
        }
    }
}

impl PdfFileLoader<'_, Result<PathBuf, FileLoaderError>> {
    /// Creates a new [PdfFileLoader] using a glob pattern to match files.
    ///
    /// # Example
    /// Create a [PdfFileLoader] for all `.pdf` files that match the glob "tests/data/*.pdf".
    ///
    /// ```rust
    /// let loader = FileLoader::with_glob("tests/data/*.txt")?;
    /// ```
    pub fn with_glob(
        pattern: &str,
    ) -> Result<PdfFileLoader<Result<PathBuf, PdfLoaderError>>, PdfLoaderError> {
        let paths = glob(pattern).map_err(FileLoaderError::PatternError)?;
        Ok(PdfFileLoader {
            iterator: Box::new(paths.into_iter().map(|path| {
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
    /// ```rust
    /// let loader = PdfFileLoader::with_dir("files")?;
    /// ```
    pub fn with_dir(
        directory: &str,
    ) -> Result<PdfFileLoader<Result<PathBuf, PdfLoaderError>>, PdfLoaderError> {
        Ok(PdfFileLoader {
            iterator: Box::new(
                fs::read_dir(directory)
                    .map_err(FileLoaderError::IoError)?
                    .map(|entry| Ok(entry.map_err(FileLoaderError::IoError)?.path())),
            ),
        })
    }
}

// ================================================================
// PDFFileLoader iterator implementations
// ================================================================

pub struct IntoIter<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

impl<'a, T> IntoIterator for PdfFileLoader<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iterator: self.iterator,
        }
    }
}

impl<T> Iterator for IntoIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::PdfFileLoader;

    #[test]
    fn test_pdf_loader() {
        let loader = PdfFileLoader::with_glob("tests/data/*.pdf").unwrap();
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
                    println!("{:?} Page {}: {:?}", path, page_no, content);
                });
                (path, pages)
            })
            .collect::<Vec<_>>();

        let mut expected = vec![
            (
                PathBuf::from("tests/data/dummy.pdf"),
                vec![(0, "Test\nPDF\nDocument\n".to_string())],
            ),
            (
                PathBuf::from("tests/data/pages.pdf"),
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
}
