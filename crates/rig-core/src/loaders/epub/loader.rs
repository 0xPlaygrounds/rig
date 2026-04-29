use crate::loaders::file::FileLoaderError;
use epub::doc::EpubDoc;

use std::fs::File;
use std::io::BufReader;
use std::marker::PhantomData;
use std::path::PathBuf;

use super::RawTextProcessor;
use super::errors::EpubLoaderError;
use super::text_processors::TextProcessor;

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
/// use rig::loaders::{EpubFileLoader, RawTextProcessor, StripXmlProcessor};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a FileLoader using a glob pattern
///     let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?;
///
///     // Load epub file contents by chapter, ignoring any errors
///     let contents = loader
///         .load_with_path()
///         .ignore_errors()
///         .by_chapter()
///         .ignore_errors();
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
///     // Create a FileLoader using a glob pattern with stripping xml
///     let loader = EpubFileLoader::<_, StripXmlProcessor>::with_glob("tests/data/*.epub")?;
///
///     // Load epub file contents by chapter, ignoring any errors
///     let contents = loader
///         .load_with_path()
///         .ignore_errors()
///         .by_chapter()
///         .ignore_errors();
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
pub struct EpubFileLoader<'a, T, P = RawTextProcessor> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
    _processor: PhantomData<P>,
}

type EpubLoaded = Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError>;

impl<'a, P> EpubFileLoader<'a, Result<PathBuf, EpubLoaderError>, P> {
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
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.load().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(doc) => println!("{:?}", doc),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn load(self) -> EpubFileLoader<'a, Result<EpubDoc<BufReader<File>>, EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load())),
            _processor: PhantomData,
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
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub").unwrap().load_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{:?} {:?}", path, doc),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn load_with_path(self) -> EpubFileLoader<'a, EpubLoaded, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load_with_path())),
            _processor: PhantomData,
        }
    }
}

impl<'a, P> EpubFileLoader<'a, Result<PathBuf, EpubLoaderError>, P>
where
    P: TextProcessor,
{
    /// Directly reads the contents of the epub files within the iterator returned by
    ///  [EpubFileLoader::with_glob] or [EpubFileLoader::with_dir].
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and return the contents of the documents.
    ///
    /// ```rust
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{}", content),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read(self) -> EpubFileLoader<'a, Result<String, EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let doc = res.load().map(EpubChapterIterator::<P>::from)?;

                Ok(doc
                    .into_iter()
                    .collect::<Result<Vec<String>, EpubLoaderError>>()?
                    .into_iter()
                    .collect::<String>())
            })),
            _processor: PhantomData,
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
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{:?} {}", path, content),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read_with_path(
        self,
    ) -> EpubFileLoader<'a, Result<(PathBuf, String), EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let (path, doc) = res.load_with_path()?;

                let content = EpubChapterIterator::<P>::from(doc)
                    .collect::<Result<Vec<String>, EpubLoaderError>>()?
                    .into_iter()
                    .collect::<String>();
                Ok((path, content))
            })),
            _processor: PhantomData,
        }
    }
}

impl<'a, P> EpubFileLoader<'a, EpubDoc<BufReader<File>>, P>
where
    P: TextProcessor + 'a,
{
    /// Chunks the chapters of a loaded document by chapter, flattened as a single vector.
    ///
    /// # Example
    /// Load epub files in directory "tests/data/*.epub" and chunk all document into it's chapters.
    ///
    /// ```rust
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.load().by_chapter().into_iter();
    /// for result in content {
    ///     println!("{}", result);
    /// }
    /// ```
    pub fn by_chapter(self) -> EpubFileLoader<'a, Result<String, EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.flat_map(EpubChapterIterator::<P>::from)),
            _processor: PhantomData,
        }
    }
}

type ByChapter = (PathBuf, Vec<(usize, Result<String, EpubLoaderError>)>);
impl<'a, P: TextProcessor> EpubFileLoader<'a, (PathBuf, EpubDoc<BufReader<File>>), P> {
    /// Chunks the chapters of a loaded document by chapter, processed as a vector of documents by path
    ///  which each document container an inner vector of chapters by chapter number.
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and chunk all documents by path by it's chapters.
    ///
    /// ```rust
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?
    ///     .load_with_path()
    ///     .ignore_errors()
    ///     .by_chapter()
    ///     .ignore_errors()
    ///     .into_iter();
    ///
    /// for result in content {
    ///     println!("{:?}", result);
    /// }
    /// ```
    pub fn by_chapter(self) -> EpubFileLoader<'a, ByChapter, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|doc| {
                let (path, doc) = doc;

                (
                    path,
                    EpubChapterIterator::<P>::from(doc)
                        .enumerate()
                        .collect::<Vec<_>>(),
                )
            })),
            _processor: PhantomData,
        }
    }
}

impl<'a, P> EpubFileLoader<'a, ByChapter, P>
where
    P: TextProcessor,
{
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [EpubFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.epub" and ignore errors from unreadable files.
    ///
    /// ```rust
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read().ignore_errors().into_iter();
    /// for result in content {
    ///     println!("{}", content)
    /// }
    /// ```
    pub fn ignore_errors(self) -> EpubFileLoader<'a, (PathBuf, Vec<(usize, String)>), P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.map(|(path, chapters)| {
                let chapters = chapters
                    .into_iter()
                    .filter_map(|(idx, res)| res.ok().map(|content| (idx, content)))
                    .collect::<Vec<_>>();
                (path, chapters)
            })),
            _processor: PhantomData,
        }
    }
}

impl<'a, P, T: 'a> EpubFileLoader<'a, Result<T, EpubLoaderError>, P> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [EpubFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.epub" and ignore errors from unreadable files.
    ///
    /// ```rust
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read().ignore_errors().into_iter();
    /// for result in content {
    ///     println!("{}", content)
    /// }
    /// ```
    pub fn ignore_errors(self) -> EpubFileLoader<'a, T, P> {
        EpubFileLoader {
            iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
            _processor: PhantomData,
        }
    }
}

impl<P> EpubFileLoader<'_, Result<PathBuf, FileLoaderError>, P> {
    /// Creates a new [EpubFileLoader] using a glob pattern to match files.
    ///
    /// # Example
    /// Create a [EpubFileLoader] for all `.epub` files that match the glob "tests/data/*.epub".
    ///
    /// ```rust
    /// let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?;
    /// ```
    pub fn with_glob(
        pattern: &str,
    ) -> Result<EpubFileLoader<'_, Result<PathBuf, EpubLoaderError>, P>, EpubLoaderError> {
        let paths = glob::glob(pattern).map_err(FileLoaderError::PatternError)?;

        Ok(EpubFileLoader {
            iterator: Box::new(paths.into_iter().map(|path| {
                path.map_err(FileLoaderError::GlobError)
                    .map_err(EpubLoaderError::FileLoaderError)
            })),
            _processor: PhantomData,
        })
    }

    /// Creates a new [EpubFileLoader] on all files within a directory.
    ///
    /// # Example
    /// Create a [EpubFileLoader] for all files that are in the directory "files".
    ///
    /// ```rust
    /// let loader = EpubFileLoader::<_, RawTextProcessor>::with_dir("files")?;
    /// ```
    pub fn with_dir(
        directory: &str,
    ) -> Result<EpubFileLoader<'_, Result<PathBuf, EpubLoaderError>, P>, EpubLoaderError> {
        let paths = std::fs::read_dir(directory).map_err(FileLoaderError::IoError)?;

        Ok(EpubFileLoader {
            iterator: Box::new(
                paths
                    .into_iter()
                    .map(|entry| Ok(entry.map_err(FileLoaderError::IoError)?.path())),
            ),
            _processor: PhantomData,
        })
    }
}

// ================================================================
// EpubFileLoader iterator implementations
// ================================================================
pub struct IntoIter<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

impl<'a, T, P> IntoIterator for EpubFileLoader<'a, T, P> {
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

struct EpubChapterIterator<P> {
    epub: EpubDoc<BufReader<File>>,
    finished: bool,
    _processor: PhantomData<P>,
}

impl<P> From<EpubDoc<BufReader<File>>> for EpubChapterIterator<P> {
    fn from(epub: EpubDoc<BufReader<File>>) -> Self {
        Self::new(epub)
    }
}

impl<P> EpubChapterIterator<P> {
    fn new(epub: EpubDoc<BufReader<File>>) -> Self {
        Self {
            epub,
            finished: false,
            _processor: PhantomData,
        }
    }
}

impl<P> Iterator for EpubChapterIterator<P>
where
    P: TextProcessor,
{
    type Item = Result<String, EpubLoaderError>;

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
                return Some(
                    P::process(&text)
                        .map_err(|err| EpubLoaderError::TextProcessorError(Box::new(err))),
                );
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::loaders::epub::RawTextProcessor;

    use super::EpubFileLoader;

    #[test]
    fn test_epub_loader_with_errors() {
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub").unwrap();
        let actual = loader
            .load_with_path()
            .ignore_errors()
            .by_chapter()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);

        let (_, chapters) = &actual[0];
        assert_eq!(chapters.len(), 3);

        for chapter in chapters {
            assert!(chapter.1.is_ok());
        }
    }

    #[test]
    fn test_epub_loader_with_ignoring_errors() {
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub").unwrap();
        let actual = loader
            .load_with_path()
            .ignore_errors()
            .by_chapter()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);

        let (_, chapters) = &actual[0];
        assert_eq!(chapters.len(), 3);
    }

    #[test]
    fn test_single_file() {
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub").unwrap();

        let actual = loader
            .read()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);
    }

    #[test]
    fn test_single_file_with_path() {
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub").unwrap();

        let actual = loader
            .read_with_path()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);

        let (path, _) = &actual[0];
        assert_eq!(path, &PathBuf::from("tests/data/dummy.epub"));
    }
}
