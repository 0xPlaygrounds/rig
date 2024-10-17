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

pub struct FileLoader<'a, State> {
    iterator: Box<dyn Iterator<Item = State> + 'a>,
}

pub(crate) trait Readable {
    fn read(self) -> Result<String, FileLoaderError>;
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError>;
}

impl<'a> FileLoader<'a, Result<PathBuf, FileLoaderError>> {
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

impl<'a, T: 'a> FileLoader<'a, Result<T, FileLoaderError>> {
    pub fn ignore_errors(self) -> FileLoader<'a, T> {
        FileLoader {
            iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
        }
    }
}

impl<'a> FileLoader<'a, PathBuf> {
    pub fn new(
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
}

impl<'a, State> FileLoader<'a, State> {
    pub fn iter(self) -> Box<dyn Iterator<Item = State> + 'a> {
        self.iterator
    }
}

#[cfg(test)]
mod tests {
    use super::FileLoader;

    #[test]
    fn test_file_loader() {
        let loader = FileLoader::new("src/*.rs").unwrap();
        loader
            .ignore_errors()
            .read_with_path()
            .iter()
            .for_each(|file| println!("{:?}", file));
    }
}
