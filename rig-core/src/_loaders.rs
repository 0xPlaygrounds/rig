

#[derive(Error, Debug)]
pub enum PdfLoaderError {
    #[error("Loader error: {0}")]
    LoaderError(#[from] FileLoaderError),

    #[error("PDF error: {0}")]
    PdfError(#[from] LopdfError),
}

struct PdfLoader<State> {
    iter_generator: Box<dyn Fn(String) -> Result<Box<dyn Iterator<Item = State>>, PdfLoaderError>>,
}

type StateWithPath = (String, String);
type StateByPage = (usize, String);
type StateWithPathByPage = (String, Vec<(usize, String)>);

impl PdfLoader<String> {
    pub fn new() -> Self {
        Self {
            iter_generator: Box::new(
                |pattern: String| -> Result<Box<dyn Iterator<Item = String>>, PdfLoaderError> {
                    let paths = glob(&pattern).map_err(FileLoaderError::PatternError)?;
                    let iter = paths.filter_map(Result::ok).map(|path| {
                        let doc = Document::load(&path).unwrap_or_default();
                        let content = doc
                            .page_iter()
                            .map(|(i, _)| doc.extract_text(&[i]).unwrap_or_default())
                            .collect::<String>();
                        content
                    });
                    Ok(Box::new(iter))
                },
            ),
        }
    }

    pub fn with_path(self) -> PdfLoader<StateWithPath> {
        PdfLoader::<StateWithPath> {
            iter_generator: Box::new(
                move |pattern: String| -> Result<Box<dyn Iterator<Item = StateWithPath>>, PdfLoaderError> {
                    let paths = glob(&pattern).map_err(FileLoaderError::PatternError)?;
                    let iter = paths.filter_map(Result::ok).map(|path| {
                        let doc = Document::load(&path).map_err(|e| PdfLoaderError::PdfError(e)).unwrap();
                        let content = doc
                            .page_iter()
                            .map(|(i, _)| doc.extract_text(&[i]).unwrap_or_default())
                            .collect::<String>();
                        (path.to_string_lossy().into_owned(), content)
                    });
                    Ok(Box::new(iter))
                },
            ),
        }
    }

    pub fn by_page(self) -> PdfLoader<StateByPage> {
        PdfLoader::<StateByPage> {
            iter_generator: Box::new(
                move |pattern: String| -> Result<
                    Box<dyn Iterator<Item = StateByPage>>,
                    PdfLoaderError,
                > {
                    let paths = glob(&pattern).map_err(FileLoaderError::PatternError)?;
                    let iter = paths.filter_map(Result::ok).flat_map(|path| {
                        let doc = Document::load(&path).map_err(|e| PdfLoaderError::PdfError(e)).unwrap();
                        doc
                            .page_iter()
                            .map(|(i, _)| (i as usize, doc.extract_text(&[i]).unwrap_or_default()))
                            .collect::<Vec<_>>()
                    });
                    Ok(Box::new(iter))
                },
            ),
        }
    }
}

impl PdfLoader<StateByPage> {
    pub fn with_path(self) -> PdfLoader<StateWithPathByPage> {
        PdfLoader::<StateWithPathByPage> {
            iter_generator:
                Box::new(
                    move |pattern: String| -> Result<
                        Box<dyn Iterator<Item = StateWithPathByPage>>,
                        PdfLoaderError,
                    > {
                        let paths = glob(&pattern).map_err(|e| {
                            PdfLoaderError::LoaderError(FileLoaderError::PatternError(e))
                        })?;
                        let iter = paths.filter_map(Result::ok).map(|path| {
                            let doc = Document::load(&path)
                                .map_err(|e| PdfLoaderError::PdfError(e))
                                .unwrap();
                            let page_iterator = doc
                                .page_iter()
                                .map(|(i, _)| {
                                    (i as usize, doc.extract_text(&[i]).unwrap_or_default())
                                })
                                .collect::<Vec<_>>();
                            (path.to_string_lossy().into_owned(), page_iterator)
                        });
                        Ok(Box::new(iter))
                    },
                ),
        }
    }
}

impl PdfLoader<StateWithPath> {
    pub fn by_page(self) -> PdfLoader<StateWithPathByPage> {
        PdfLoader::new().by_page().with_path()
    }
}

impl<State> PdfLoader<State> {
    pub fn glob(self, pattern: &str) -> Result<Box<(dyn Iterator<Item = State> + 'static)>, PdfLoaderError> {
        (self.iter_generator)(pattern.to_string())
    }
}

mod tests {
    use super::{FileLoader, PdfLoader};
    use glob::glob;
    use lopdf::{Document, Error as LopdfError};

    #[test]
    fn test_file_loader() {
        let loader = FileLoader::new();
        let files = loader.glob("src/*.rs").unwrap().for_each(|file| {
            if let Some(first_line) = file.lines().next() {
                println!("{}", first_line);
            }
        });
    }

    #[test]
    fn test_file_loader_with_path() {
        let loader = FileLoader::new().with_path();
        let files = loader.glob("src/*.rs").unwrap().collect::<Vec<_>>();
        assert_eq!(files.len(), 1);
    }
    #[test]
    fn test_pdf_loader() {
        let loader = PdfLoader::new();
        let pdfs = loader.glob("docs/*.pdf").unwrap().collect::<Vec<_>>();
        pdfs.iter().for_each(|content| {
            println!("{}", content);
        });
    }

    #[test]
    fn test_pdf_loader_with_path() {
        for file in glob("*.pdf").unwrap() {
            println!("{:?}", file);
        }
        let loader = PdfLoader::new().with_path();
        let pdfs = loader.glob("*.pdf").unwrap().collect::<Vec<_>>();
        pdfs.iter().for_each(|(path, content)| {
            println!("{}: {}", path, content);
        });
    }

    #[test]
    fn test_pdf_loader_by_page() {
        let loader = PdfLoader::new().with_path().by_page();
        loader
            .glob("*.pdf")
            .expect("no pdfs")
            .for_each(|(path, pages)| {
                println!("{}:", path);
                pages.iter().for_each(|page| {
                    println!("{}", page);
                });
            });
    }
}
