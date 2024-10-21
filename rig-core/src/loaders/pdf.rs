use std::path::PathBuf;

use glob::glob;
use lopdf::{Document, Error as LopdfError};
use thiserror::Error;

use super::file::FileLoaderError;

#[derive(Error, Debug)]
pub enum PdfLoaderError {
    #[error("{0}")]
    FileLoaderError(#[from] FileLoaderError),

    #[error("IO error: {0}")]
    PdfError(#[from] LopdfError),
}

pub struct PdfFileLoader<'a, State> {
    iterator: Box<dyn Iterator<Item = State> + 'a>,
}

trait Loadable {
    fn load(self) -> Result<Document, PdfLoaderError>;
    fn load_with_path(self) -> Result<(PathBuf, Document), PdfLoaderError>;
}

impl<'a> PdfFileLoader<'a, Result<PathBuf, PdfLoaderError>> {
    pub fn load(self) -> PdfFileLoader<'a, Result<Document, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load())),
        }
    }
    pub fn load_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, Document), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| res.load_with_path())),
        }
    }
}

impl<'a> PdfFileLoader<'a, Result<PathBuf, PdfLoaderError>> {
    pub fn read(self) -> PdfFileLoader<'a, Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let doc = res.load()?;
                doc.page_iter()
                    .map(|(i, _)| doc.extract_text(&[i]).map_err(PdfLoaderError::PdfError))
                    .collect::<Result<String, PdfLoaderError>>()
            })),
        }
    }
    pub fn read_with_path(self) -> PdfFileLoader<'a, Result<(PathBuf, String), PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|res| {
                let (path, doc) = res.load_with_path()?;
                let contents = doc
                    .page_iter()
                    .map(|(i, _)| doc.extract_text(&[i]).map_err(PdfLoaderError::PdfError))
                    .collect::<Result<String, PdfLoaderError>>()?;

                Ok((path, contents))
            })),
        }
    }
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

impl<'a> PdfFileLoader<'a, Document> {
    pub fn by_page(self) -> PdfFileLoader<'a, Result<String, PdfLoaderError>> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.flat_map(|doc| {
                doc.page_iter()
                    .map(|(i, _)| doc.extract_text(&[i]).map_err(PdfLoaderError::PdfError))
                    .collect::<Vec<_>>()
            })),
        }
    }
}

type ByPage = (PathBuf, Vec<Result<(usize, String), PdfLoaderError>>);
impl<'a> PdfFileLoader<'a, (PathBuf, Document)> {
    pub fn by_page(self) -> PdfFileLoader<'a, ByPage> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.map(|(path, doc)| {
                (
                    path,
                    doc.page_iter()
                        .map(|(i, _)| {
                            doc.extract_text(&[i])
                                .map(|text| (i as usize, text))
                                .map_err(PdfLoaderError::PdfError)
                        })
                        .collect::<Vec<_>>(),
                )
            })),
        }
    }
}

impl<'a, T: 'a> PdfFileLoader<'a, Result<T, PdfLoaderError>> {
    pub fn ignore_errors(self) -> PdfFileLoader<'a, T> {
        PdfFileLoader {
            iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
        }
    }
}

impl<'a> PdfFileLoader<'a, PathBuf> {
    pub fn new(
        pattern: &str,
    ) -> Result<PdfFileLoader<Result<PathBuf, PdfLoaderError>>, FileLoaderError> {
        let paths = glob(pattern)?;
        Ok(PdfFileLoader {
            iterator: Box::new(paths.into_iter().map(|path| {
                path.map_err(FileLoaderError::GlobError)
                    .map_err(PdfLoaderError::FileLoaderError)
            })),
        })
    }
}

impl<'a, State> PdfFileLoader<'a, State> {
    pub fn iter(self) -> Box<dyn Iterator<Item = State> + 'a> {
        self.iterator
    }
}

#[cfg(test)]
mod tests {
    use super::PdfFileLoader;

    #[test]
    fn test_pdf_loader() {
        let loader = PdfFileLoader::new("*.md").unwrap();
        let actual = loader
            .ignore_errors()
            .read_with_path()
            .ignore_errors()
            .iter()
            .map(|(_, content)| content.split("\n").next().unwrap().to_string())
            .collect::<Vec<_>>();

        let expected = vec!["# Changelog".to_string(), "# Rig".to_string()];

        assert!(!actual.is_empty());
        assert!(expected == actual)
    }
}
