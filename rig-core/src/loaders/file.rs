use std::{fs, path::PathBuf};

use glob::glob;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FileLoaderError {
    #[error("Invalid glob pattern: {0}")]
    InvalidGlobPattern(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Pattern error: {0}")]
    PatternError(#[from] glob::PatternError),

    #[error("Glob error: {0}")]
    GlobError(#[from] glob::GlobError),
}

// ================================================================
// Implementing Readable trait for reading file contents
// ================================================================
pub(crate) trait Readable {
    fn read(self) -> Result<String, FileLoaderError>;
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError>;
}

impl<'a> FileLoader<'a, PathBuf> {
    pub fn read(self) -> FileLoader<'a, Result<String, FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(|res| res.read())),
        }
    }
    pub fn read_with_path(self) -> FileLoader<'a, Result<(PathBuf, String), FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(|res| res.read_with_path())),
        }
    }
}

impl Readable for PathBuf {
    fn read(self) -> Result<String, FileLoaderError> {
        fs::read_to_string(self).map_err(FileLoaderError::IoError)
    }
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError> {
        let contents = fs::read_to_string(&self);
        Ok((self, contents?))
    }
}
impl<T: Readable> Readable for Result<T, FileLoaderError> {
    fn read(self) -> Result<String, FileLoaderError> {
        self.map(|t| t.read())?
    }
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError> {
        self.map(|t| t.read_with_path())?
    }
}

// ================================================================
// FileLoader definitions and implementations
// ================================================================

/// [FileLoader] is a utility for loading files from the filesystem using glob patterns or directory
///  paths. It provides methods to read file contents and handle errors gracefully.
///
/// # Errors
///
/// This module defines a custom error type [FileLoaderError] which can represent various errors
///  that might occur during file loading operations, such as invalid glob patterns, IO errors, and
///  glob errors.
///
/// # Example Usage
///
/// ```rust
/// use rig:loaders::FileLoader;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a FileLoader using a glob pattern
///     let loader = FileLoader::with_glob("path/to/files/*.txt")?;
///
///     // Read file contents, ignoring any errors
///     let contents: Vec<String> = loader
///         .read()
///         .ignore_errors()
///
///     for content in contents {
///         println!("{}", content);
///     }
///
///     Ok(())
/// }
/// ```
///
/// [FileLoader] uses strict typing between the iterator methods to ensure that transitions between
///   different implementations of the loaders and it's methods are handled properly by the compiler.
pub struct FileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

impl<'a> FileLoader<'a, Result<PathBuf, FileLoaderError>> {
    /// Reads the contents of the files within the iterator returned by [FileLoader::with_glob] or
    ///  [FileLoader::with_dir].
    ///
    /// # Example
    /// Read files in directory "files/*.txt" and print the content for each file
    ///
    /// ```rust
    /// let content = FileLoader::with_glob(...)?.read();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{}", content),
    ///         Err(e) => eprintln!("Error reading file: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read(self) -> FileLoader<'a, Result<String, FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(|res| res.read())),
        }
    }
    /// Reads the contents of the files within the iterator returned by [FileLoader::with_glob] or
    ///  [FileLoader::with_dir] and returns the path along with the content.
    ///
    /// # Example
    /// Read files in directory "files/*.txt" and print the content for corresponding path for each
    ///  file.
    ///
    /// ```rust
    /// let content = FileLoader::with_glob("files/*.txt")?.read();
    /// for (path, result) in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{:?} {}", path, content),
    ///         Err(e) => eprintln!("Error reading file: {}", e),
    ///     }
    /// }
    /// ```
    pub fn read_with_path(self) -> FileLoader<'a, Result<(PathBuf, String), FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(|res| res.read_with_path())),
        }
    }
}

impl<'a, T: 'a> FileLoader<'a, Result<T, FileLoaderError>> {
    /// Ignores errors in the iterator, returning only successful results. This can be used on any
    ///  [FileLoader] state of iterator whose items are results.
    ///
    /// # Example
    /// Read files in directory "files/*.txt" and ignore errors from unreadable files.
    ///
    /// ```rust
    /// let content = FileLoader::with_glob("files/*.txt")?.read().ignore_errors();
    /// for result in content {
    ///     println!("{}", content)
    /// }
    /// ```
    pub fn ignore_errors(self) -> FileLoader<'a, T> {
        FileLoader {
            iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
        }
    }
}

impl FileLoader<'_, Result<PathBuf, FileLoaderError>> {
    /// Creates a new [FileLoader] using a glob pattern to match files.
    ///
    /// # Example
    /// Create a [FileLoader] for all `.txt` files that match the glob "files/*.txt".
    ///
    /// ```rust
    /// let loader = FileLoader::with_glob("files/*.txt")?;
    /// ```
    pub fn with_glob(
        pattern: &str,
    ) -> Result<FileLoader<Result<PathBuf, FileLoaderError>>, FileLoaderError> {
        let paths = glob(pattern)?;
        Ok(FileLoader {
            iterator: Box::new(
                paths
                    .into_iter()
                    .map(|path| path.map_err(FileLoaderError::GlobError)),
            ),
        })
    }

    /// Creates a new [FileLoader] on all files within a directory.
    ///
    /// # Example
    /// Create a [FileLoader] for all files that are in the directory "files" (ignores subdirectories).
    ///
    /// ```rust
    /// let loader = FileLoader::with_dir("files")?;
    /// ```
    pub fn with_dir(
        directory: &str,
    ) -> Result<FileLoader<Result<PathBuf, FileLoaderError>>, FileLoaderError> {
        Ok(FileLoader {
            iterator: Box::new(fs::read_dir(directory)?.filter_map(|entry| {
                let path = entry.ok()?.path();
                if path.is_file() {
                    Some(Ok(path))
                } else {
                    None
                }
            })),
        })
    }
}

// ================================================================
// Iterators for FileLoader
// ================================================================

pub struct IntoIter<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

impl<'a, T> IntoIterator for FileLoader<'a, T> {
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
    use assert_fs::prelude::{FileTouch, FileWriteStr, PathChild};

    use super::FileLoader;

    #[test]
    fn test_file_loader() {
        let temp = assert_fs::TempDir::new().expect("Failed to create temp dir");
        let foo_file = temp.child("foo.txt");
        let bar_file = temp.child("bar.txt");

        foo_file.touch().expect("Failed to create foo.txt");
        bar_file.touch().expect("Failed to create bar.txt");

        foo_file.write_str("foo").expect("Failed to write to foo");
        bar_file.write_str("bar").expect("Failed to write to bar");

        let glob = temp.path().to_string_lossy().to_string() + "/*.txt";

        let loader = FileLoader::with_glob(&glob).unwrap();
        let mut actual = loader
            .ignore_errors()
            .read()
            .ignore_errors()
            .into_iter()
            .collect::<Vec<_>>();
        let mut expected = vec!["foo".to_string(), "bar".to_string()];

        actual.sort();
        expected.sort();

        assert!(!actual.is_empty());
        assert!(expected == actual)
    }
}
