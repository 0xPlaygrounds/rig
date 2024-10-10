use std::fs;

use futures::Stream;
use glob::{glob, GlobError};
use lopdf::{Document, Error as LopdfError};
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

struct FileLoader<State> {
    iter_generator: Box<dyn Fn(String) -> Result<Box<dyn Iterator<Item = State>>, FileLoaderError>>,
}

type FileLoaderWithPath = FileLoader<(String, String)>;

// struct WithPath;
// struct IgnoreErrors;
// struct IgnoreErrorsWithPath;

impl FileLoader<String> {
    pub fn new() -> Self {}
}

impl FileLoader<String> {
    pub fn new() -> Self {
        Self {
            iter_generator: Box::new(
                |pattern: String| -> Result<Box<dyn Iterator<Item = String>>, FileLoaderError> {
                    let paths = glob(&pattern).map_err(|e| FileLoaderError::PatternError(e))?;
                    let iter = paths.map(|path| {
                        fs::read_to_string(path)
                            .map_err(|e| FileLoaderError::IoError(e))
                            .unwrap_or_default()
                    });
                    Ok(Box::new(iter))
                },
            ),
        }
    }

    pub fn with_path(self) -> FileLoaderWithPath {
        FileLoader::<(String, String)> {
            iter_generator:
                Box::new(
                    move |pattern: String| -> Result<
                        Box<dyn Iterator<Item = (String, String)>>,
                        FileLoaderError,
                    > {
                        let paths = glob(&pattern).map_err(|e| FileLoaderError::PatternError(e))?;
                        let iter = paths.filter_map(Result::ok).map(|path| {
                            let content = fs::read_to_string(&path)
                                .map_err(|e| FileLoaderError::IoError(e))
                                .unwrap_or_default();
                            (path.to_string_lossy().into_owned(), content)
                        });
                        Ok(Box::new(iter))
                    },
                ),
        }
    }
}

impl<State> FileLoader<State> {
    pub fn glob(self, pattern: &str) -> Result<Box<dyn Iterator<Item = State>>, FileLoaderError> {
        (self.iter_generator)(pattern.to_string())
    }
}
