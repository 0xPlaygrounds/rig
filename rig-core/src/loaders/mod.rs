pub mod file;

pub use file::FileLoader;

#[cfg(feature = "pdf")]
pub mod pdf;

#[cfg(feature = "pdf")]
pub use pdf::PdfFileLoader;
