use std::{fs, path::PathBuf, string::FromUtf8Error};

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

    #[error("String conversion error: {0}")]
    StringUtf8Error(#[from] FromUtf8Error),
}

// ================================================================
// Implementing Readable trait for reading file contents
// ================================================================
loadable_trait!(Readable, FileLoaderError, String, read, read_with_path);

impl Readable for PathBuf {
    fn read(self) -> Result<String, FileLoaderError> {
        fs::read_to_string(self).map_err(FileLoaderError::IoError)
    }
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError> {
        let contents = fs::read_to_string(&self);
        Ok((self, contents?))
    }
}

impl Readable for Vec<u8> {
    fn read(self) -> Result<String, FileLoaderError> {
        Ok(String::from_utf8(self)?)
    }

    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError> {
        let res = String::from_utf8(self)?;

        Ok((PathBuf::from("<memory>"), res))
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
/// ```no_run
/// use rig_core::loaders::FileLoader;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a FileLoader using a glob pattern
///     let loader = FileLoader::with_glob("path/to/files/*.txt")?;
///
///     // Read file contents, ignoring any errors
///     let contents: Vec<String> = loader
///         .read()
///         .ignore_errors()
///         .into_iter()
///         .collect();
///
///     for content in contents {
///         println!("{content}");
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

#[allow(private_bounds)] // `Readable` deliberately seals which states expose these methods
impl<'a, T: Readable + 'a> FileLoader<'a, T> {
    /// Reads the contents of the files within the iterator returned by [FileLoader::with_glob] or
    ///  [FileLoader::with_dir].
    ///
    /// # Example
    /// Read files in directory "files/*.txt" and print the content for each file
    ///
    /// ```no_run
    /// # use rig_core::loaders::FileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = FileLoader::with_glob("files/*.txt")?.read();
    /// for result in content {
    ///     match result {
    ///         Ok(content) => println!("{content}"),
    ///         Err(e) => eprintln!("Error reading file: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(self) -> FileLoader<'a, Result<String, FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(Readable::read)),
        }
    }
    /// Reads the contents of the files within the iterator returned by [FileLoader::with_glob] or
    ///  [FileLoader::with_dir] and returns the path along with the content.
    ///
    /// # Example
    /// Read files in directory "files/*.txt" and print the content for corresponding path for each
    ///  file.
    ///
    /// ```no_run
    /// # use rig_core::loaders::FileLoader;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let content = FileLoader::with_glob("files/*.txt")?.read_with_path();
    /// for result in content {
    ///     match result {
    ///         Ok((path, content)) => println!("{path:?} {content}"),
    ///         Err(e) => eprintln!("Error reading file: {e}"),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_with_path(self) -> FileLoader<'a, Result<(PathBuf, String), FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(Readable::read_with_path)),
        }
    }
}

loader_scaffold!(FileLoader, FileLoaderError, dir: files_only);
loader_from_bytes!(FileLoader);

#[cfg(test)]
mod tests;
