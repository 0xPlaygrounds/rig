//! Lazy UTF-8 loading from filesystem paths or in-memory bytes.
//!
//! ```
//! use rig_core::loaders::FileLoader;
//!
//! let documents = FileLoader::from_bytes(b"hello".to_vec())
//!     .read().into_iter().collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(documents, vec!["hello"]);
//! # Ok::<(), rig_core::loaders::file::FileLoaderError>(())
//! ```

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

/// Iterator pipeline for loading UTF-8 documents. Reads happen synchronously
/// during iteration; per-item I/O and decoding errors are yielded unless filtered.
pub struct FileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>,
}

#[allow(private_bounds)] // `Readable` deliberately seals which states expose these methods
impl<'a, T: Readable + 'a> FileLoader<'a, T> {
    /// Decodes each input as UTF-8 during iteration, yielding I/O or decoding errors.
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
    /// Decodes each input as UTF-8 and pairs it with its source path, yielding
    /// I/O or decoding errors. In-memory inputs use the path `<memory>`.
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
