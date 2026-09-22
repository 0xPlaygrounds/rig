//! Conventional Azure OpenAI deployment names. Deployments are account-defined;
//! replace these constants with the names configured on your resource.
//!
//! [`crate::providers::openai::wire::AZURE`] reads `AZURE_ENDPOINT`,
//! `AZURE_API_VERSION`, and either `AZURE_API_KEY` or `AZURE_TOKEN`.
//!
//! ```no_run
//! use rig_core::providers::azure;
//! use rig_core::providers::openai::wire::{AZURE, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let gpt4o = OpenAI::from_env_with(&AZURE)?.chat(azure::GPT_4O);
//! # Ok(())
//! # }
//! ```

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

/// `o1` completion model
pub const O1: &str = "o1";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-mini` completion model
pub const O1_MINI: &str = "o1-mini";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-realtime-preview` completion model
pub const GPT_4O_REALTIME_PREVIEW: &str = "gpt-4o-realtime-preview";
/// Conventional `gpt-4` deployment name; this constant does not select a turbo version.
pub const GPT_4_TURBO: &str = "gpt-4";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";
/// `gpt-3.5-turbo-16k` completion model
pub const GPT_35_TURBO_16K: &str = "gpt-3.5-turbo-16k";
