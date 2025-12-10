use std::error::Error;

use epub::doc::DocError;

use crate::loaders::file::FileLoaderError;

#[derive(thiserror::Error, Debug)]
pub enum EpubLoaderError {
    #[error("IO error: {0}")]
    EpubError(#[from] DocError),

    #[error("File loader error: {0}")]
    FileLoaderError(#[from] FileLoaderError),

    #[error("Text processor error: {0}")]
    TextProcessorError(#[from] Box<dyn Error>),
}
