//! File loading utilities for preparing local documents as model or embedding input.
//!
//! [`FileLoader`] provides a common interface for reading files from disk, glob
//! matches, directories, or in-memory bytes. It can return content alone or pair
//! content with source paths, and it can optionally skip per-file errors.
//!
//! `PdfFileLoader` is available with the `pdf` feature. It loads PDFs and can
//! split extracted text by page while preserving page numbers.
//!
//! `EpubFileLoader` is available with the `epub` feature. It loads EPUB files
//! and can split extracted text by chapter while preserving chapter numbers.

// ================================================================
// Shared scaffolding for the typestate loaders (file, pdf, epub)
// ================================================================

/// Defines the `pub(crate)` source trait for a loader (e.g. `Readable` /
/// `Loadable`) together with its blanket impl for `Result`, which lets loader
/// states whose items are results be consumed transparently.
macro_rules! loadable_trait {
    ($Trait:ident, $Err:ty, $Out:ty, $get:ident, $get_with_path:ident) => {
        pub(crate) trait $Trait {
            fn $get(self) -> Result<$Out, $Err>;
            fn $get_with_path(self) -> Result<(std::path::PathBuf, $Out), $Err>;
        }

        impl<T: $Trait> $Trait for Result<T, $Err> {
            fn $get(self) -> Result<$Out, $Err> {
                self.map(|t| t.$get())?
            }
            fn $get_with_path(self) -> Result<(std::path::PathBuf, $Out), $Err> {
                self.map(|t| t.$get_with_path())?
            }
        }
    };
}

/// The `with_dir` doc line matching each `loader_dir_entries!` kind.
macro_rules! loader_dir_doc {
    (files_only) => {
        "Creates a new loader on all files within a directory (ignores subdirectories)."
    };
    (all_entries) => {
        "Creates a new loader on all entries within a directory. Entries are not \
         filtered: a non-file entry's path is yielded too and surfaces as an error \
         when loaded."
    };
}

/// Expands to the directory-entry iterator used by a loader's `with_dir`.
///
/// - `files_only`: skips unreadable entries and non-files (the [`file::FileLoader`]
///   behavior).
/// - `all_entries`: yields every entry's path, surfacing entry errors as items
///   (the pdf/epub behavior).
macro_rules! loader_dir_entries {
    (files_only, $entries:expr, $Err:ty) => {
        $entries.filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.is_file() { Some(Ok(path)) } else { None }
        })
    };
    (all_entries, $entries:expr, $Err:ty) => {
        $entries.map(|entry| {
            Ok(entry
                .map_err($crate::loaders::file::FileLoaderError::IoError)
                .map_err(<$Err>::from)?
                .path())
        })
    };
}

/// Generates the shared typestate plumbing for a loader struct with a boxed
/// `iterator` field: the `IntoIter` type with its `IntoIterator`/`Iterator`
/// impls, `ignore_errors` on `Result` states, and the `with_glob`/`with_dir`
/// constructors. Pass `extra: P` for loaders carrying an extra type parameter
/// held in a `_processor: PhantomData<P>` field.
macro_rules! loader_scaffold {
    ($Loader:ident, $Err:ty, dir: $dir_kind:ident $(, extra: $P:ident)?) => {
        pub struct IntoIter<'a, T> {
            iterator: Box<dyn Iterator<Item = T> + 'a>,
        }

        impl<'a, T $(, $P)?> IntoIterator for $Loader<'a, T $(, $P)?> {
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

        impl<'a, T: 'a $(, $P)?> $Loader<'a, Result<T, $Err> $(, $P)?> {
            /// Ignores errors in the iterator, returning only successful results. This
            ///  can be used on any loader state of iterator whose items are results.
            pub fn ignore_errors(self) -> $Loader<'a, T $(, $P)?> {
                $Loader {
                    iterator: Box::new(self.iterator.filter_map(|res| res.ok())),
                    $(_processor: std::marker::PhantomData::<$P>,)?
                }
            }
        }

        impl<'a $(, $P)?> $Loader<'a, Result<std::path::PathBuf, $Err> $(, $P)?> {
            /// Creates a new loader using a glob pattern to match files.
            pub fn with_glob(
                pattern: &str,
            ) -> Result<$Loader<'_, Result<std::path::PathBuf, $Err> $(, $P)?>, $Err> {
                let paths = ::glob::glob(pattern)
                    .map_err($crate::loaders::file::FileLoaderError::PatternError)
                    .map_err(<$Err>::from)?;
                Ok($Loader {
                    iterator: Box::new(paths.map(|path| {
                        path.map_err($crate::loaders::file::FileLoaderError::GlobError)
                            .map_err(<$Err>::from)
                    })),
                    $(_processor: std::marker::PhantomData::<$P>,)?
                })
            }

            #[doc = loader_dir_doc!($dir_kind)]
            pub fn with_dir(
                directory: &str,
            ) -> Result<$Loader<'_, Result<std::path::PathBuf, $Err> $(, $P)?>, $Err> {
                let entries = std::fs::read_dir(directory)
                    .map_err($crate::loaders::file::FileLoaderError::IoError)
                    .map_err(<$Err>::from)?;
                Ok($Loader {
                    iterator: Box::new(loader_dir_entries!($dir_kind, entries, $Err)),
                    $(_processor: std::marker::PhantomData::<$P>,)?
                })
            }
        }
    };
}

/// Generates the byte-ingestion constructors for a loader (file and pdf).
macro_rules! loader_from_bytes {
    ($Loader:ident) => {
        impl<'a> $Loader<'a, Vec<u8>> {
            /// Ingest a document as a byte array.
            pub fn from_bytes(bytes: Vec<u8>) -> $Loader<'a, Vec<u8>> {
                $Loader {
                    iterator: Box::new(vec![bytes].into_iter()),
                }
            }

            /// Ingest multiple byte arrays.
            pub fn from_bytes_multi(bytes_vec: Vec<Vec<u8>>) -> $Loader<'a, Vec<u8>> {
                $Loader {
                    iterator: Box::new(bytes_vec.into_iter()),
                }
            }
        }
    };
}

pub mod file;

pub use file::FileLoader;

// Test-only helpers for resolving on-disk fixtures in a CWD-independent way.
// Gated to the features whose tests use them so it never warns as dead code.
#[cfg(all(test, any(feature = "pdf", feature = "epub")))]
mod test_fixtures;

#[cfg(feature = "pdf")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdf")))]
pub mod pdf;

#[cfg(feature = "pdf")]
pub use pdf::PdfFileLoader;

#[cfg(feature = "epub")]
#[cfg_attr(docsrs, doc(cfg(feature = "epub")))]
pub mod epub;

#[cfg(feature = "epub")]
pub use epub::{EpubFileLoader, RawTextProcessor, StripXmlProcessor, TextProcessor};
