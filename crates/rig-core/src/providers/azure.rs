//! Azure OpenAI's API versions and deployment-name constants.
//!
//! Azure OpenAI is an OpenAI chat-completions dialect, so it has no client
//! and no models of its own:
//! [`openai::wire::AZURE`](crate::providers::openai::wire::AZURE) carries
//! everything that makes it Azure. The deployment is in the URL rather than
//! the request body, the `api-version` rides as a query parameter, and the
//! credential is either an account key sent as `api-key` (`AZURE_API_KEY`)
//! or an Entra bearer token (`AZURE_TOKEN`) — two different credentials with
//! two different headers, which is why the dialect records which one it
//! holds instead of guessing from the value.
//!
//! There is no shared host: the base URL is the account's own resource
//! endpoint, read from `AZURE_ENDPOINT` or set with
//! [`OpenAI::with_base_url`](crate::providers::openai::wire::OpenAI::with_base_url).
//!
//! The constants below are Azure *deployment* names only by convention —
//! a deployment is named by whoever created it, so these are the names the
//! portal offers by default rather than identifiers Azure will recognize on
//! every account.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::azure;
//! use rig_core::providers::openai::wire::{AZURE, AZURE_DEFAULT_API_VERSION, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // From `AZURE_API_KEY`/`AZURE_TOKEN`, `AZURE_ENDPOINT` and `AZURE_API_VERSION`;
//! // the wire, which `.bind(transport)` joins to a socket.
//! let gpt4o = OpenAI::from_env_with(&AZURE)?.chat(azure::GPT_4O);
//!
//! // Or spelled out, with the endpoint and version supplied directly.
//! let explicit = OpenAI::with_key(&AZURE, "YOUR_API_KEY")
//!     .with_base_url("https://your-resource-name.openai.azure.com")
//!     .with_api_version(AZURE_DEFAULT_API_VERSION)
//!     .chat(azure::GPT_4O);
//! # Ok(())
//! # }
//! ```

// ================================================================
// Azure OpenAI Embedding API
// ================================================================

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

// ================================================================
// Azure OpenAI Completion API
// ================================================================

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
/// `gpt-4-turbo` completion model
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
