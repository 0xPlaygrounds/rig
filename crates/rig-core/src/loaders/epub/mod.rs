//! EPUB loading with raw or XML-stripped chapter text.
//!
//! ```no_run
//! use rig_core::loaders::{EpubFileLoader, StripXmlProcessor};
//!
//! let documents = EpubFileLoader::<_, StripXmlProcessor>::with_glob("books/*.epub")?
//!     .read().into_iter().collect::<Result<Vec<_>, _>>()?;
//! # let _ = documents;
//! # Ok::<(), rig_core::loaders::epub::EpubLoaderError>(())
//! ```

mod errors;
mod loader;
mod text_processors;

pub use errors::EpubLoaderError;
pub use loader::{EpubFileLoader, IntoIter};
pub use text_processors::{RawTextProcessor, StripXmlProcessor, TextProcessor};
