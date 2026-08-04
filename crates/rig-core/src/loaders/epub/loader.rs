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
/// ```no_run
/// use rig_core::loaders::{EpubFileLoader, RawTextProcessor, StripXmlProcessor};
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
///
/// # Evaluation is eager
///
/// Each stage ([`load`](Self::load), [`by_chapter`](Self::by_chapter), …) runs
/// to completion and materialises a `Vec` before the next one starts. The
/// loader previously held a `Box<dyn Iterator>`, which let stages compose
/// lazily; de-erasing that field costs the laziness, because the closure types
/// the stages produce cannot be named. Loading a large corpus therefore holds
/// every parsed document in memory at once — chunk the glob if that matters.
pub struct EpubFileLoader<T, P = RawTextProcessor> {
    iterator: std::vec::IntoIter<T>,
    _processor: PhantomData<P>,
}

/// Collects a pipeline stage into a nameable iterator. See the laziness note
/// on [`EpubFileLoader`].
fn eager<T>(iterator: impl Iterator<Item = T>) -> std::vec::IntoIter<T> {
    iterator.collect::<Vec<_>>().into_iter()
}

type EpubLoaded = Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError>;

impl<P> EpubFileLoader<Result<PathBuf, EpubLoaderError>, P> {
    /// Loads the contents of the epub files within the iterator returned by [EpubFileLoader::with_glob]
    ///  or [EpubFileLoader::with_dir]. Loaded EPUB documents are raw EPUB instances that can be
    ///  further processed (by chapter, etc).
    ///
    /// # Example
    /// Load epub files in directory "tests/data/*.epub" and return the loaded documents
    ///
    /// ```no_run
    /// use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.load().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(doc) => println!("{:?}", doc),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(self) -> EpubFileLoader<Result<EpubDoc<BufReader<File>>, EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.map(|res| res.load())),
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
    /// ```no_run
    /// use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.load_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, doc)) => println!("{:?} {:?}", path, doc),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_with_path(self) -> EpubFileLoader<EpubLoaded, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.map(|res| res.load_with_path())),
            _processor: PhantomData,
        }
    }
}

impl<P> EpubFileLoader<Result<PathBuf, EpubLoaderError>, P>
where
    P: TextProcessor,
{
    /// Directly reads the contents of the epub files within the iterator returned by
    ///  [EpubFileLoader::with_glob] or [EpubFileLoader::with_dir].
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and return the contents of the documents.
    ///
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{}", content),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(self) -> EpubFileLoader<Result<String, EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.map(|res| {
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
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read_with_path().into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{:?} {}", path, content),
    ///         Err(e) => eprintln!("Error reading epub: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_with_path(self) -> EpubFileLoader<Result<(PathBuf, String), EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.map(|res| {
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

impl<P> EpubFileLoader<EpubDoc<BufReader<File>>, P>
where
    P: TextProcessor,
{
    /// Chunks the chapters of a loaded document by chapter, flattened as a single vector.
    ///
    /// # Example
    /// Load epub files in directory "tests/data/*.epub" and chunk all document into it's chapters.
    ///
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?
    ///     .load()
    ///     .ignore_errors()
    ///     .by_chapter()
    ///     .into_iter();
    /// for result in content {
    ///     match result {
    ///         Ok(chapter) => println!("{}", chapter),
    ///         Err(e) => eprintln!("Error reading chapter: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn by_chapter(self) -> EpubFileLoader<Result<String, EpubLoaderError>, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.flat_map(EpubChapterIterator::<P>::from)),
            _processor: PhantomData,
        }
    }
}

type ByChapter = (PathBuf, Vec<(usize, Result<String, EpubLoaderError>)>);
impl<P: TextProcessor> EpubFileLoader<(PathBuf, EpubDoc<BufReader<File>>), P> {
    /// Chunks the chapters of a loaded document by chapter, processed as a vector of documents by path
    ///  which each document container an inner vector of chapters by chapter number.
    ///
    /// # Example
    /// Read epub files in directory "tests/data/*.epub" and chunk all documents by path by it's chapters.
    ///
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
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
    /// # Ok(())
    /// # }
    /// ```
    pub fn by_chapter(self) -> EpubFileLoader<ByChapter, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.map(|doc| {
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

impl<P> EpubFileLoader<ByChapter, P>
where
    P: TextProcessor,
{
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [EpubFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.epub" and ignore errors from unreadable files.
    ///
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?
    ///     .load_with_path()
    ///     .ignore_errors()
    ///     .by_chapter()
    ///     .ignore_errors();
    /// for (_path, chapters) in content {
    ///     println!("{}", chapters.len())
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn ignore_errors(self) -> EpubFileLoader<(PathBuf, Vec<(usize, String)>), P> {
        EpubFileLoader {
            iterator: eager(self.iterator.map(|(path, chapters)| {
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

impl<P, T> EpubFileLoader<Result<T, EpubLoaderError>, P> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [EpubFileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "tests/data/*.epub" and ignore errors from unreadable files.
    ///
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?.read().ignore_errors();
    /// for content in content {
    ///     println!("{}", content)
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn ignore_errors(self) -> EpubFileLoader<T, P> {
        EpubFileLoader {
            iterator: eager(self.iterator.filter_map(|res| res.ok())),
            _processor: PhantomData,
        }
    }
}

impl<P> EpubFileLoader<Result<PathBuf, FileLoaderError>, P> {
    /// Creates a new [EpubFileLoader] using a glob pattern to match files.
    ///
    /// # Example
    /// Create a [EpubFileLoader] for all `.epub` files that match the glob "tests/data/*.epub".
    ///
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob("tests/data/*.epub")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_glob(
        pattern: &str,
    ) -> Result<EpubFileLoader<Result<PathBuf, EpubLoaderError>, P>, EpubLoaderError> {
        let paths = glob::glob(pattern).map_err(FileLoaderError::PatternError)?;

        Ok(EpubFileLoader {
            iterator: eager(paths.into_iter().map(|path| {
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
    /// ```no_run
    /// # use rig_core::loaders::{EpubFileLoader, RawTextProcessor};
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let loader = EpubFileLoader::<_, RawTextProcessor>::with_dir("files")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_dir(
        directory: &str,
    ) -> Result<EpubFileLoader<Result<PathBuf, EpubLoaderError>, P>, EpubLoaderError> {
        let paths = std::fs::read_dir(directory).map_err(FileLoaderError::IoError)?;

        Ok(EpubFileLoader {
            iterator: eager(
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

impl<T, P> IntoIterator for EpubFileLoader<T, P> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iterator
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
    use crate::loaders::epub::RawTextProcessor;
    use crate::loaders::test_fixtures::{fixture_glob, fixture_path};

    use super::EpubFileLoader;

    #[test]
    fn test_epub_loader_with_errors() {
        let glob = fixture_glob("*.epub");
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();
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
        let glob = fixture_glob("*.epub");
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();
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
        let glob = fixture_glob("*.epub");
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();

        let actual = loader
            .read()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);
    }

    #[test]
    fn test_single_file_with_path() {
        let glob = fixture_glob("*.epub");
        let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();

        let actual = loader
            .read_with_path()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), 1);

        let (path, _) = &actual[0];
        assert_eq!(path, &fixture_path("dummy.epub"));
    }
}
