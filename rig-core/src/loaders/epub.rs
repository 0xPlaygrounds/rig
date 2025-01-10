use super::file::FileLoaderError;
use epub::doc::{DocError, EpubDoc};
use thiserror::Error;

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Error, Debug)]
pub enum EpubLoaderError {
    #[error("IO error: {0}")]
    EpubError(#[from] DocError),

    #[error("File loader error: {0}")]
    FileLoaderError(#[from] FileLoaderError),
}

// ================================================================
// Implementing Loadable trait for loading epubs
// ================================================================

pub(crate) trait Loadable {
    fn load(self) -> Result<EpubDoc<BufReader<File>>, EpubLoaderError>;
    fn load_with_path(self) -> Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError>;
}

impl Loadable for PathBuf {
    fn load(self) -> Result<EpubDoc<BufReader<File>>, EpubLoaderError> {
        EpubDoc::new(self).map_err(EpubLoaderError::EpubError)
    }

    fn load_with_path(self) -> Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError> {
        let contents = EpubDoc::new(&self).map_err(EpubLoaderError::EpubError);
        Ok((self, contents?))
    }
}

impl<T: Loadable> Loadable for Result<T, EpubLoaderError> {
    fn load(self) -> Result<EpubDoc<BufReader<File>>, EpubLoaderError> {
        self.map(|t| t.load())?
    }
    fn load_with_path(self) -> Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError> {
        self.map(|t| t.load_with_path())?
    }
}

// ================================================================
// EpubFileLoader definitions and implementations
// ================================================================

/// [EpubFileLoader] is a utility for loading epub files from the filesystem using glob patterns or
///  directory paths. It provides methods to read file contents and handle errors gracefully.
///
/// # Errors
///
/// This module defines a custom error type [EpubLoaderError] which can represent various errors
///  that might occur during file loading operations, such as any [FileLoaderError] alongside
///  specific EPUB-related errors.
///
/// # Example Usage
///
/// ```rust
/// use rig::loaders::EpubFileLoader;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a FileLoader using a glob pattern
///     let loader = EpubFileLoader::with_glob("tests/data/*.epub")?;
///
///     // Load epub file contents by chapter, ignoring any errors
///     let contents = loader
///         .load_with_path()
///         .ignore_errors()
///         .by_chapter();
///
///     for (path, chapters) in contents {
///         println!("{}", path.display());
///         for (idx, chapter) in chapters {
///             println!("Chapter {} begins", idx);
///             println!("{}", chapter);
///             println!("Chapter {} ends", idx);
///         }
///     }
///
///     Ok(())
/// }
/// ```
///
/// [EpubFileLoader] uses strict typing between the iterator methods to ensure that transitions
///  between different implementations of the loaders and it's methods are handled properly by
///  the compiler.
pub struct EpubFileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

type EpubLoaded = Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError>;

impl<'a> EpubFileLoader<'a, Result<PathBuf, EpubLoaderError>> {
    /// Loads the contents of the epub files within the iterator returned by [EpubFileLoader::with_glob]
    ///  or [EpubFileLoader::with_dir]. Loaded EPUB documents are raw EPUB instances that can be
    ///  further processed (by chapter, etc).
    ///
    /// # Example
    /// Load epub files in directory "tests/data/*.epub" and return the loaded documents
    ///
    /// ```rust
    /// use rig::loaders::EpubFileLoader;
    ///
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub")?.load().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(doc) => println!("{:?}", doc),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn load(self) -> EpubFileLoader<'a, Result<EpubDoc<BufReader<File>>, EpubLoaderError>> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load())),
        }
    }

    /// Loads the contents of the epub files within the iterator returned by [EpubFileLoader::with_glob]
    ///  or [EpubFileLoader::with_dir]. Loaded EPUB documents are raw EPUB instances with their path
    ///  that can be further processed.
    ///
    /// # Example
    /// Load epub files in directory "tests/data/*.epub" and return the loaded documents
    ///
    /// ```rust
    /// use rig::loaders::EpubFileLoader;
    ///
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub").unwrap().load_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{:?} {:?}", path, doc),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn load_with_path(self) -> EpubFileLoader<'a, EpubLoaded> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load_with_path())),
        }
    }
}

impl<'a> EpubFileLoader<'a, Result<PathBuf, EpubLoaderError>> {
    /// Directly reads the contents of the epub files within the iterator returned by
    ///  [EpubFileLoader::with_glob] or [EpubFileLoader::with_dir].
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and return the contents of the documents.
    ///
    /// ```rust
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub")?.read().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{}", content),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read(self) -> EpubFileLoader<'a, Result<String, EpubLoaderError>> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let doc = res.load().map(EpubChapterIterator::from)?;

                Ok(doc.into_iter().collect::<String>())
            })),
        }
    }

    /// Directly reads the contents of the epub files within the iterator returned by
    ///  [EpubFileLoader::with_glob] or [EpubFileLoader::with_dir] and returns the path along with
    ///  the content.
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and return the content and paths of the documents.
    ///
    /// ```rust
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{:?} {}", path, content),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read_with_path(self) -> EpubFileLoader<'a, Result<(PathBuf, String), EpubLoaderError>> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let (path, doc) = res.load_with_path()?;
                Ok((path, EpubChapterIterator::from(doc).collect::<String>()))
            })),
        }
    }
}

impl<'a> EpubFileLoader<'a, EpubDoc<BufReader<File>>> {
    /// Chunks the chapters of a loaded document by chapter, flattened as a single vector.
    ///
    /// # Example
    /// Load epub files in directory "tests/data/*.epub" and chunk all document into it's chapters.
    ///
    /// ```rust
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub")?.load().by_chapter().into_iter();
    /// for result in content {
    ///     println!("{}", result);
    /// }
    /// ```
    pub fn by_chapter(self) -> EpubFileLoader<'a, String> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.flat_map(|doc| EpubChapterIterator::from(doc))),
        }
    }
}

type ByChapter = (PathBuf, Vec<(usize, String)>);
impl<'a> EpubFileLoader<'a, (PathBuf, EpubDoc<BufReader<File>>)> {
    /// Chunks the chapters of a loaded document by chapter, processed as a vector of documents by path
    ///  which each document container an inner vector of chapters by chapter number.
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and chunk all documents by path by it's chapters.
    ///
    /// ```rust
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub")?
    ///     .load_with_path()
    ///     .ignore_errors()
    ///     .by_chapter()
    ///     .into_iter();
    ///
    /// for result in content {
    ///     println!("{:?}", result);
    /// }
    /// ```
    pub fn by_chapter(self) -> EpubFileLoader<'a, ByChapter> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|doc| {
                let (path, doc) = doc;

                (
                    path,
                    EpubChapterIterator::from(doc)
                        .enumerate()
                        .collect::<Vec<_>>(),
                )
            })),
        }
    }
}

impl<'a, T: 'a> EpubFileLoader<'a, Result<T, EpubLoaderError>> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [EpubFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.epub" and ignore errors from unreadable files.
    ///
    /// ```rust
    /// let content = EpubFileLoader::with_glob("tests/data/*.epub")?.read().ignore_errors().into_iter();
    /// for result in content {
    ///     println!("{}", content)
    /// }
    /// ```
    pub fn ignore_errors(self) -> EpubFileLoader<'a, T> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
        }
    }
}

impl EpubFileLoader<'_, Result<PathBuf, FileLoaderError>> {
    /// Creates a new [EpubFileLoader] using a glob pattern to match files.
    ///
    /// # Example
    /// Create a [EpubFileLoader] for all `.epub` files that match the glob "tests/data/*.epub".
    ///
    /// ```rust
    /// let loader = EpubFileLoader::with_glob("tests/data/*.epub")?;
    /// ```
    pub fn with_glob(
        pattern: &str,
    ) -> Result<EpubFileLoader<Result<PathBuf, EpubLoaderError>>, EpubLoaderError> {
        let paths = glob::glob(pattern).map_err(FileLoaderError::PatternError)?;

        Ok(EpubFileLoader {
            iterator: Box::new(paths.into_iter().map(|path| {
                path.map_err(FileLoaderError::GlobError)
                    .map_err(EpubLoaderError::FileLoaderError)
            })),
        })
    }

    /// Creates a new [EpubFileLoader] on all files within a directory.
    ///
    /// # Example
    /// Create a [EpubFileLoader] for all files that are in the directory "files".
    ///
    /// ```rust
    /// let loader = EpubFileLoader::with_dir("files")?;
    /// ```
    pub fn with_dir(
        directory: &str,
    ) -> Result<EpubFileLoader<Result<PathBuf, EpubLoaderError>>, EpubLoaderError> {
        let paths = std::fs::read_dir(directory).map_err(FileLoaderError::IoError)?;

        Ok(EpubFileLoader {
            iterator: Box::new(
                paths
                    .into_iter()
                    .map(|entry| Ok(entry.map_err(FileLoaderError::IoError)?.path())),
            ),
        })
    }
}

// ================================================================
// EpubFileLoader iterator implementations
// ================================================================
pub struct IntoIter<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

impl<'a, T> IntoIterator for EpubFileLoader<'a, T> {
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

// ================================================================
// EpubChapterIterator definitions and implementations
// ================================================================

struct EpubChapterIterator {
    epub: EpubDoc<BufReader<File>>,
    finished: bool,
}

impl From<EpubDoc<BufReader<File>>> for EpubChapterIterator {
    fn from(epub: EpubDoc<BufReader<File>>) -> Self {
        Self::new(epub)
    }
}

impl EpubChapterIterator {
    fn new(epub: EpubDoc<BufReader<File>>) -> Self {
        Self {
            epub,
            finished: false,
        }
    }
}

impl Iterator for EpubChapterIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // ignore empty chapters if they exist
        while !self.finished {
            let chapter = self.epub.get_current_str();

            if !self.epub.go_next() {
                self.finished = true;
            }

            if let Some((text, _)) = chapter {
                return Some(text);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::EpubFileLoader;

    #[test]
    fn test_epub_loader() {
        let loader = EpubFileLoader::with_glob("tests/data/*.epub").unwrap();
        let actual = loader
            .load_with_path()
            .ignore_errors()
            .by_chapter()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);

        let (_, chapters) = &actual[0];
        assert_eq!(chapters.len(), 3);
    }
}
