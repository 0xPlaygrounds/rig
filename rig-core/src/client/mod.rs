//! Provider client traits and utilities for LLM integration.
//!
//! This module defines the core abstractions for creating and managing provider clients
//! that interface with different LLM services (OpenAI, Anthropic, Cohere, etc.). It provides
//! both static and dynamic dispatch mechanisms for working with multiple providers.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │           ProviderClient (Base Trait)               │
//! │  Conversion: AsCompletion, AsEmbeddings, etc.       │
//! └────────────┬────────────────────────────────────────┘
//!              │
//!              ├─> CompletionClient     (text generation)
//!              ├─> EmbeddingsClient     (vector embeddings)
//!              ├─> TranscriptionClient  (audio to text)
//!              ├─> ImageGenerationClient (text to image)
//!              └─> AudioGenerationClient (text to speech)
//! ```
//!
//! # Quick Start
//!
//! ## Using a Static Client
//!
//! ```no_run
//! use rig::prelude::*;
//! use rig::providers::openai::{Client, self};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an OpenAI client
//! let client = Client::new("your-api-key");
//!
//! // Create a completion model
//! let model = client.completion_model(openai::GPT_4O);
//!
//! // Generate a completion
//! let response = model
//!     .completion_request("What is the capital of France?")
//!     .send()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Using a Dynamic Client Builder
//!
//! ```no_run
//! use rig::client::builder::DynClientBuilder;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a dynamic client builder
//! let builder = DynClientBuilder::new();
//!
//! // Build agents for different providers
//! let openai_agent = builder.agent("openai", "gpt-4o")?
//!     .preamble("You are a helpful assistant")
//!     .build();
//!
//! let anthropic_agent = builder.agent("anthropic", "claude-3-7-sonnet")?
//!     .preamble("You are a helpful assistant")
//!     .build();
//! # Ok(())
//! # }
//! ```
//!
//! # Main Components
//!
//! ## Core Traits
//!
//! - [`ProviderClient`] - Base trait for all provider clients
//!   - **Use when:** Implementing a new LLM provider integration
//!
//! ## Capability Traits
//!
//! - [`CompletionClient`] - Text generation capabilities
//!   - **Use when:** You need to generate text completions or build agents
//!
//! - [`EmbeddingsClient`] - Vector embedding capabilities
//!   - **Use when:** You need to convert text to vector embeddings for search or similarity
//!
//! - [`TranscriptionClient`] - Audio transcription capabilities
//!   - **Use when:** You need to convert audio to text
//!
//! - [`ImageGenerationClient`] - Image generation capabilities (feature: `image`)
//!   - **Use when:** You need to generate images from text prompts
//!
//! - [`AudioGenerationClient`] - Audio generation capabilities (feature: `audio`)
//!   - **Use when:** You need to generate audio from text (text-to-speech)
//!
//! ## Dynamic Client Builder
//!
//! - [`builder::DynClientBuilder`] - Dynamic client factory for multiple providers
//!   - **Use when:** You need to support multiple providers at runtime
//!
//! # Common Patterns
//!
//! ## Pattern 1: Single Provider with Static Dispatch
//!
//! ```no_run
//! use rig::prelude::*;
//! use rig::providers::openai::{Client, self};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Client::new("api-key");
//! let agent = client.agent(openai::GPT_4O)
//!     .preamble("You are a helpful assistant")
//!     .build();
//!
//! let response = agent.prompt("Hello!").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Pattern 2: Multiple Providers with Dynamic Dispatch
//!
//! ```no_run
//! use rig::client::builder::DynClientBuilder;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let builder = DynClientBuilder::new();
//!
//! // User selects provider at runtime
//! let provider = "openai";
//! let model = "gpt-4o";
//!
//! let agent = builder.agent(provider, model)?
//!     .preamble("You are a helpful assistant")
//!     .build();
//! # Ok(())
//! # }
//! ```
//!
//! ## Pattern 3: Provider Capability Detection
//!
//! ```no_run
//! use rig::client::{ProviderClient, AsCompletion, AsEmbeddings};
//! use rig::providers::openai::Client;
//!
//! # fn example() {
//! let client = Client::from_env();
//!
//! // Check if provider supports completions
//! if let Some(completion_client) = client.as_completion() {
//!     let model = completion_client.completion_model("gpt-4o");
//!     // Use model...
//! }
//!
//! // Check if provider supports embeddings
//! if let Some(embeddings_client) = client.as_embeddings() {
//!     let model = embeddings_client.embedding_model("text-embedding-3-large");
//!     // Use model...
//! }
//! # }
//! ```
//!
//! ## Pattern 4: Creating Provider from Environment
//!
//! ```no_run
//! use rig::client::ProviderClient;
//! use rig::providers::openai::Client;
//!
//! # fn example() {
//! // Reads API key from OPENAI_API_KEY environment variable
//! let client = Client::from_env();
//! # }
//! ```
//!
//! # Performance Characteristics
//!
//! - **Client creation**: Lightweight, typically O(1) memory allocation for configuration
//! - **Model cloning**: Inexpensive due to internal `Arc` usage (reference counting only)
//! - **Static vs Dynamic Dispatch**:
//!   - Static dispatch (using concrete types): Zero runtime overhead
//!   - Dynamic dispatch (trait objects): Small vtable lookup overhead
//!   - Use static dispatch when working with a single provider
//!   - Use dynamic dispatch when provider selection happens at runtime
//!
//! For most applications, the network latency of API calls (100-1000ms) far exceeds
//! any client-side overhead
//!
//! # Implementing a New Provider
//!
//! ```rust
//! use rig::client::{ProviderClient, ProviderValue, CompletionClient};
//! use rig::completion::CompletionModel;
//! # use std::fmt::Debug;
//!
//! // Step 1: Define your client struct
//! #[derive(Clone, Debug)]
//! struct MyProvider {
//!     api_key: String,
//!     base_url: String,
//! }
//!
//! // Step 2: Implement ProviderClient
//! impl ProviderClient for MyProvider {
//!     fn from_env() -> Self {
//!         Self {
//!             api_key: std::env::var("MY_API_KEY")
//!                 .expect("MY_API_KEY environment variable not set"),
//!             base_url: std::env::var("MY_BASE_URL")
//!                 .unwrap_or_else(|_| "https://api.myprovider.com".to_string()),
//!         }
//!     }
//!
//!     fn from_val(input: ProviderValue) -> Self {
//!         match input {
//!             ProviderValue::Simple(api_key) => Self {
//!                 api_key,
//!                 base_url: "https://api.myprovider.com".to_string(),
//!             },
//!             ProviderValue::ApiKeyWithOptionalKey(api_key, Some(base_url)) => Self {
//!                 api_key,
//!                 base_url,
//!             },
//!             _ => panic!("Invalid ProviderValue for MyProvider"),
//!         }
//!     }
//! }
//!
//! // Step 3: Implement capability traits (e.g., CompletionClient)
//! // impl CompletionClient for MyProvider { ... }
//!
//! // Step 4: Register the capabilities
//! // rig::impl_conversion_traits!(AsCompletion for MyProvider);
//! ```
//!
//! # Troubleshooting
//!
//! ## Provider Not Found
//!
//! ```compile_fail
//! // ❌ BAD: Provider not registered
//! let builder = DynClientBuilder::empty();
//! let agent = builder.agent("openai", "gpt-4o")?; // Error: UnknownProvider
//! ```
//!
//! ```no_run
//! // ✅ GOOD: Use default registry or register manually
//! use rig::client::builder::DynClientBuilder;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let builder = DynClientBuilder::new(); // Includes all providers
//! let agent = builder.agent("openai", "gpt-4o")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Unsupported Feature for Provider
//!
//! Not all providers support all features. Check the provider capability table below.
//!
//! ### Provider Capabilities
//!
//! | Provider   | Completions | Embeddings | Transcription | Image Gen | Audio Gen |
//! |------------|-------------|------------|---------------|-----------|-----------|
//! | OpenAI     | ✅          | ✅         | ✅            | ✅        | ✅        |
//! | Anthropic  | ✅          | ❌         | ❌            | ❌        | ❌        |
//! | Cohere     | ✅          | ✅         | ❌            | ❌        | ❌        |
//! | Gemini     | ✅          | ✅         | ✅            | ❌        | ❌        |
//! | Azure      | ✅          | ✅         | ✅            | ✅        | ✅        |
//! | Together   | ✅          | ✅         | ❌            | ❌        | ❌        |
//!
//! ### Handling Unsupported Features
//!
//! ```no_run
//! use rig::client::builder::{DynClientBuilder, ClientBuildError};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let builder = DynClientBuilder::new();
//!
//! // Attempt to use embeddings with a provider
//! match builder.embeddings("anthropic", "any-model") {
//!     Ok(model) => {
//!         // Use embeddings
//!     },
//!     Err(ClientBuildError::UnsupportedFeature(provider, feature)) => {
//!         eprintln!("{} doesn't support {}", provider, feature);
//!         // Fallback to a provider that supports embeddings
//!         let model = builder.embeddings("openai", "text-embedding-3-large")?;
//!     },
//!     Err(e) => return Err(e.into()),
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Flags
//!
//! The client module supports optional features for additional functionality:
//!
//! ## `image` - Image Generation Support
//!
//! Enables [`ImageGenerationClient`] and image generation capabilities:
//!
//! ```toml
//! [dependencies]
//! rig-core = { version = "0.x", features = ["image"] }
//! ```
//!
//! ## `audio` - Audio Generation Support
//!
//! Enables [`AudioGenerationClient`] for text-to-speech:
//!
//! ```toml
//! [dependencies]
//! rig-core = { version = "0.x", features = ["audio"] }
//! ```
//!
//! ## `derive` - Derive Macros
//!
//! Enables the `#[derive(ProviderClient)]` macro for automatic trait implementation:
//!
//! ```toml
//! [dependencies]
//! rig-core = { version = "0.x", features = ["derive"] }
//! ```
//!
//! Without these features, the corresponding traits and functionality are not available.
//!
//! # See Also
//!
//! - [`crate::completion`] - Text completion functionality
//! - [`crate::embeddings`] - Vector embedding functionality
//! - [`crate::agent`] - High-level agent abstraction
//! - [`crate::providers`] - Available provider implementations

pub mod audio_generation;
pub mod builder;
pub mod completion;
pub mod embeddings;
pub mod image_generation;
pub mod transcription;
pub mod verify;

#[cfg(feature = "derive")]
pub use rig_derive::ProviderClient;
use std::fmt::Debug;
use thiserror::Error;

/// Errors that can occur when building a client.
///
/// This enum represents errors that may arise during client construction,
/// such as HTTP configuration errors or invalid property values.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientBuilderError {
    /// An error occurred in the HTTP client (reqwest).
    ///
    /// This typically indicates issues with network configuration,
    /// TLS setup, or invalid URLs.
    #[error("reqwest error: {0}")]
    HttpError(
        #[from]
        #[source]
        reqwest::Error,
    ),

    /// An invalid property value was provided during client construction.
    ///
    /// # Examples
    ///
    /// This error may be returned when:
    /// - An API key format is invalid
    /// - A required configuration field is missing
    /// - A property value is outside acceptable bounds
    #[error("invalid property: {0}")]
    InvalidProperty(&'static str),
}

/// Base trait for all LLM provider clients.
///
/// This trait defines the common interface that all provider clients must implement.
/// It includes methods for creating clients from environment variables or explicit values,
/// and provides automatic conversion to capability-specific clients through the
/// `As*` conversion traits.
///
/// # Implementing ProviderClient
///
/// When implementing a new provider, you must:
/// 1. Implement [`ProviderClient`] with `from_env()` and `from_val()`
/// 2. Implement capability traits as needed ([`CompletionClient`], [`EmbeddingsClient`], etc.)
/// 3. Use the [`impl_conversion_traits!`] macro to register capabilities
///
/// # Examples
///
/// ```rust
/// use rig::client::{ProviderClient, ProviderValue};
/// # use std::fmt::Debug;
///
/// #[derive(Clone, Debug)]
/// struct MyProvider {
///     api_key: String,
/// }
///
/// impl ProviderClient for MyProvider {
///     fn from_env() -> Self {
///         Self {
///             api_key: std::env::var("MY_API_KEY")
///                 .expect("MY_API_KEY environment variable not set"),
///         }
///     }
///
///     fn from_val(input: ProviderValue) -> Self {
///         match input {
///             ProviderValue::Simple(api_key) => Self { api_key },
///             _ => panic!("Expected simple API key"),
///         }
///     }
/// }
/// ```
///
/// # Panics
///
/// The `from_env()` and `from_env_boxed()` methods panic if the environment
/// is improperly configured (e.g., required environment variables are missing).
pub trait ProviderClient:
    AsCompletion + AsTranscription + AsEmbeddings + AsImageGeneration + AsAudioGeneration + Debug
{
    /// Creates a client from the process environment variables.
    ///
    /// This method reads configuration (typically API keys) from environment variables.
    /// Each provider defines its own required environment variables.
    ///
    /// # Panics
    ///
    /// Panics if required environment variables are not set or have invalid values.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::client::ProviderClient;
    /// use rig::providers::openai::Client;
    ///
    /// // Reads from OPENAI_API_KEY environment variable
    /// let client = Client::from_env();
    /// ```
    fn from_env() -> Self
    where
        Self: Sized;

    /// Wraps this client in a `Box<dyn ProviderClient>`.
    ///
    /// This is a convenience method for converting a concrete client type
    /// into a trait object for dynamic dispatch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::client::ProviderClient;
    /// use rig::providers::openai::Client;
    ///
    /// let client = Client::from_env().boxed();
    /// // client is now Box<dyn ProviderClient>
    /// ```
    fn boxed(self) -> Box<dyn ProviderClient>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// Creates a boxed client from the process environment variables.
    ///
    /// This is equivalent to `Self::from_env().boxed()` but more convenient.
    ///
    /// # Panics
    ///
    /// Panics if required environment variables are not set or have invalid values.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::client::ProviderClient;
    /// use rig::providers::openai::Client;
    ///
    /// let client = Client::from_env_boxed();
    /// // client is Box<dyn ProviderClient>
    /// ```
    fn from_env_boxed<'a>() -> Box<dyn ProviderClient + 'a>
    where
        Self: Sized,
        Self: 'a,
    {
        Box::new(Self::from_env())
    }

    /// Creates a client from a provider-specific value.
    ///
    /// This method allows creating clients with explicit configuration values
    /// rather than reading from environment variables.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::client::{ProviderClient, ProviderValue};
    /// use rig::providers::openai::Client;
    ///
    /// let client = Client::from_val(ProviderValue::Simple("api-key".to_string()));
    /// ```
    fn from_val(input: ProviderValue) -> Self
    where
        Self: Sized;

    /// Creates a boxed client from a provider-specific value.
    ///
    /// This is equivalent to `Self::from_val(input).boxed()` but more convenient.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::client::{ProviderClient, ProviderValue};
    /// use rig::providers::openai::Client;
    ///
    /// let client = Client::from_val_boxed(ProviderValue::Simple("api-key".to_string()));
    /// // client is Box<dyn ProviderClient>
    /// ```
    fn from_val_boxed<'a>(input: ProviderValue) -> Box<dyn ProviderClient + 'a>
    where
        Self: Sized,
        Self: 'a,
    {
        Box::new(Self::from_val(input))
    }
}

/// Configuration values for creating provider clients.
///
/// This enum supports different provider authentication schemes,
/// allowing flexibility in how clients are configured.
///
/// # Examples
///
/// ```rust
/// use rig::client::ProviderValue;
///
/// // Simple API key
/// let simple = ProviderValue::Simple("sk-abc123".to_string());
///
/// // API key with optional organization ID (e.g., OpenAI)
/// let with_org = ProviderValue::ApiKeyWithOptionalKey(
///     "sk-abc123".to_string(),
///     Some("org-xyz".to_string())
/// );
///
/// // API key with version and header (e.g., Azure)
/// let azure = ProviderValue::ApiKeyWithVersionAndHeader(
///     "api-key".to_string(),
///     "2024-01-01".to_string(),
///     "api-key".to_string()
/// );
/// ```
#[derive(Clone)]
pub enum ProviderValue {
    /// Simple API key authentication.
    ///
    /// This is the most common authentication method, used by providers
    /// that only require a single API key.
    Simple(String),

    /// API key with an optional secondary key.
    ///
    /// Used by providers that support optional organization IDs or
    /// project keys (e.g., OpenAI with organization ID).
    ///
    /// # Format
    /// `(api_key, optional_key)`
    ApiKeyWithOptionalKey(String, Option<String>),

    /// API key with version and header name.
    ///
    /// Used by providers that require API versioning and custom
    /// header names (e.g., Azure OpenAI).
    ///
    /// # Format
    /// `(api_key, version, header_name)`
    ApiKeyWithVersionAndHeader(String, String, String),
}

impl From<&str> for ProviderValue {
    fn from(value: &str) -> Self {
        Self::Simple(value.to_string())
    }
}

impl From<String> for ProviderValue {
    fn from(value: String) -> Self {
        Self::Simple(value)
    }
}

impl<P> From<(P, Option<P>)> for ProviderValue
where
    P: AsRef<str>,
{
    fn from((api_key, optional_key): (P, Option<P>)) -> Self {
        Self::ApiKeyWithOptionalKey(
            api_key.as_ref().to_string(),
            optional_key.map(|x| x.as_ref().to_string()),
        )
    }
}

impl<P> From<(P, P, P)> for ProviderValue
where
    P: AsRef<str>,
{
    fn from((api_key, version, header): (P, P, P)) -> Self {
        Self::ApiKeyWithVersionAndHeader(
            api_key.as_ref().to_string(),
            version.as_ref().to_string(),
            header.as_ref().to_string(),
        )
    }
}

/// Trait for converting a [`ProviderClient`] to a [`CompletionClient`].
///
/// This trait enables capability detection and conversion for text completion features.
/// It is automatically implemented for types that implement [`CompletionClientDyn`].
///
/// # Examples
///
/// ```no_run
/// use rig::client::{ProviderClient, AsCompletion};
/// use rig::providers::openai::Client;
///
/// let client = Client::from_env();
///
/// if let Some(completion_client) = client.as_completion() {
///     let model = completion_client.completion_model("gpt-4o");
///     // Use the completion model...
/// }
/// ```
pub trait AsCompletion {
    /// Attempts to convert this client to a completion client.
    ///
    /// Returns `Some` if the provider supports completion features,
    /// `None` otherwise.
    fn as_completion(&self) -> Option<Box<dyn CompletionClientDyn>> {
        None
    }
}

/// Trait for converting a [`ProviderClient`] to a [`TranscriptionClient`].
///
/// This trait enables capability detection and conversion for audio transcription features.
/// It is automatically implemented for types that implement [`TranscriptionClientDyn`].
///
/// # Examples
///
/// ```no_run
/// use rig::client::{ProviderClient, AsTranscription};
/// use rig::providers::openai::Client;
///
/// let client = Client::from_env();
///
/// if let Some(transcription_client) = client.as_transcription() {
///     let model = transcription_client.transcription_model("whisper-1");
///     // Use the transcription model...
/// }
/// ```
pub trait AsTranscription {
    /// Attempts to convert this client to a transcription client.
    ///
    /// Returns `Some` if the provider supports transcription features,
    /// `None` otherwise.
    fn as_transcription(&self) -> Option<Box<dyn TranscriptionClientDyn>> {
        None
    }
}

/// Trait for converting a [`ProviderClient`] to an [`EmbeddingsClient`].
///
/// This trait enables capability detection and conversion for vector embedding features.
/// It is automatically implemented for types that implement [`EmbeddingsClientDyn`].
///
/// # Examples
///
/// ```no_run
/// use rig::client::{ProviderClient, AsEmbeddings};
/// use rig::providers::openai::Client;
///
/// let client = Client::from_env();
///
/// if let Some(embeddings_client) = client.as_embeddings() {
///     let model = embeddings_client.embedding_model("text-embedding-3-large");
///     // Use the embedding model...
/// }
/// ```
pub trait AsEmbeddings {
    /// Attempts to convert this client to an embeddings client.
    ///
    /// Returns `Some` if the provider supports embedding features,
    /// `None` otherwise.
    fn as_embeddings(&self) -> Option<Box<dyn EmbeddingsClientDyn>> {
        None
    }
}

/// Trait for converting a [`ProviderClient`] to an [`AudioGenerationClient`].
///
/// This trait enables capability detection and conversion for audio generation (TTS) features.
/// It is automatically implemented for types that implement [`AudioGenerationClientDyn`].
/// Only available with the `audio` feature enabled.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "audio")]
/// # {
/// use rig::client::{ProviderClient, AsAudioGeneration};
/// use rig::providers::openai::Client;
///
/// let client = Client::from_env();
///
/// if let Some(audio_client) = client.as_audio_generation() {
///     let model = audio_client.audio_generation_model("tts-1");
///     // Use the audio generation model...
/// }
/// # }
/// ```
pub trait AsAudioGeneration {
    /// Attempts to convert this client to an audio generation client.
    ///
    /// Returns `Some` if the provider supports audio generation features,
    /// `None` otherwise. Only available with the `audio` feature enabled.
    #[cfg(feature = "audio")]
    fn as_audio_generation(&self) -> Option<Box<dyn AudioGenerationClientDyn>> {
        None
    }
}

/// Trait for converting a [`ProviderClient`] to an [`ImageGenerationClient`].
///
/// This trait enables capability detection and conversion for image generation features.
/// It is automatically implemented for types that implement [`ImageGenerationClientDyn`].
/// Only available with the `image` feature enabled.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "image")]
/// # {
/// use rig::client::{ProviderClient, AsImageGeneration};
/// use rig::providers::openai::Client;
///
/// let client = Client::from_env();
///
/// if let Some(image_client) = client.as_image_generation() {
///     let model = image_client.image_generation_model("dall-e-3");
///     // Use the image generation model...
/// }
/// # }
/// ```
pub trait AsImageGeneration {
    /// Attempts to convert this client to an image generation client.
    ///
    /// Returns `Some` if the provider supports image generation features,
    /// `None` otherwise. Only available with the `image` feature enabled.
    #[cfg(feature = "image")]
    fn as_image_generation(&self) -> Option<Box<dyn ImageGenerationClientDyn>> {
        None
    }
}

/// Trait for converting a [`ProviderClient`] to a [`VerifyClient`].
///
/// This trait enables capability detection and conversion for client verification features.
/// It is automatically implemented for types that implement [`VerifyClientDyn`].
///
/// # Examples
///
/// ```no_run
/// use rig::client::{ProviderClient, AsVerify};
/// use rig::providers::openai::Client;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::from_env();
///
/// if let Some(verify_client) = client.as_verify() {
///     verify_client.verify().await?;
///     println!("Client configuration is valid");
/// }
/// # Ok(())
/// # }
/// ```
pub trait AsVerify {
    /// Attempts to convert this client to a verify client.
    ///
    /// Returns `Some` if the provider supports verification features,
    /// `None` otherwise.
    fn as_verify(&self) -> Option<Box<dyn VerifyClientDyn>> {
        None
    }
}

#[cfg(not(feature = "audio"))]
impl<T: ProviderClient> AsAudioGeneration for T {}

#[cfg(not(feature = "image"))]
impl<T: ProviderClient> AsImageGeneration for T {}

/// Registers capability traits for a provider client.
///
/// This macro automatically implements the conversion traits ([`AsCompletion`],
/// [`AsEmbeddings`], etc.) for your provider client type, enabling capability
/// detection and dynamic dispatch through the [`ProviderClient`] trait.
///
/// # When to Use
///
/// Use this macro after implementing [`ProviderClient`] and the specific
/// capability traits (like [`CompletionClient`], [`EmbeddingsClient`]) for
/// your provider. This macro wires up the automatic conversions that allow
/// your client to work with the dynamic client builder and capability detection.
///
/// # What This Does
///
/// For each trait listed, this macro implements the empty conversion trait,
/// which signals to the Rig system that your client supports that capability.
/// The actual conversion logic is provided by blanket implementations.
///
/// # Syntax
///
/// ```text
/// impl_conversion_traits!(Trait1, Trait2, ... for YourType);
/// ```
///
/// # Examples
///
/// ## Complete Provider Implementation
///
/// ```rust
/// use rig::client::{ProviderClient, ProviderValue, CompletionClient};
/// use rig::completion::CompletionModel;
/// # use std::fmt::Debug;
///
/// // 1. Define your client
/// #[derive(Clone, Debug)]
/// pub struct MyClient {
///     api_key: String,
/// }
///
/// // 2. Implement ProviderClient
/// impl ProviderClient for MyClient {
///     fn from_env() -> Self {
///         Self {
///             api_key: std::env::var("MY_API_KEY")
///                 .expect("MY_API_KEY not set"),
///         }
///     }
///
///     fn from_val(input: ProviderValue) -> Self {
///         match input {
///             ProviderValue::Simple(key) => Self { api_key: key },
///             _ => panic!("Invalid value"),
///         }
///     }
/// }
///
/// // 3. Implement capability traits
/// // impl CompletionClient for MyClient {
/// //     type CompletionModel = MyCompletionModel;
/// //     fn completion_model(&self, model: &str) -> Self::CompletionModel {
/// //         MyCompletionModel { /* ... */ }
/// //     }
/// // }
///
/// // 4. Register the capabilities with this macro
/// rig::impl_conversion_traits!(AsCompletion for MyClient);
/// ```
///
/// ## Multiple Capabilities
///
/// ```rust
/// # use rig::client::{ProviderClient, ProviderValue};
/// # use std::fmt::Debug;
/// # #[derive(Clone, Debug)]
/// # pub struct MultiClient;
/// # impl ProviderClient for MultiClient {
/// #     fn from_env() -> Self { MultiClient }
/// #     fn from_val(_: ProviderValue) -> Self { MultiClient }
/// # }
/// // Register multiple capabilities at once
/// rig::impl_conversion_traits!(AsCompletion, AsEmbeddings, AsTranscription for MultiClient);
/// ```
///
/// ## With Feature Gates
///
/// ```rust
/// # use rig::client::{ProviderClient, ProviderValue};
/// # use std::fmt::Debug;
/// # #[derive(Clone, Debug)]
/// # pub struct MyClient;
/// # impl ProviderClient for MyClient {
/// #     fn from_env() -> Self { MyClient }
/// #     fn from_val(_: ProviderValue) -> Self { MyClient }
/// # }
/// // The macro automatically handles feature gates for image/audio
/// rig::impl_conversion_traits!(
///     AsCompletion,
///     AsEmbeddings,
///     AsImageGeneration,  // Only available with "image" feature
///     AsAudioGeneration   // Only available with "audio" feature
///     for MyClient
/// );
/// ```
///
/// # See Also
///
/// - [`ProviderClient`] - The base trait to implement first
/// - [`CompletionClient`], [`EmbeddingsClient`] - Capability traits to implement
/// - [`AsCompletion`], [`AsEmbeddings`] - Conversion traits this macro implements
#[macro_export]
macro_rules! impl_conversion_traits {
    ($( $trait_:ident ),* for $struct_:ident ) => {
        $(
            impl_conversion_traits!(@impl $trait_ for $struct_);
        )*
    };

    (@impl AsAudioGeneration for $struct_:ident ) => {
        rig::client::impl_audio_generation!($struct_);
    };

    (@impl AsImageGeneration for $struct_:ident ) => {
        rig::client::impl_image_generation!($struct_);
    };

    (@impl $trait_:ident for $struct_:ident) => {
        impl rig::client::$trait_ for $struct_ {}
    };
}

#[cfg(feature = "audio")]
#[macro_export]
macro_rules! impl_audio_generation {
    ($struct_:ident) => {
        impl rig::client::AsAudioGeneration for $struct_ {}
    };
}

#[cfg(not(feature = "audio"))]
#[macro_export]
macro_rules! impl_audio_generation {
    ($struct_:ident) => {};
}

#[cfg(feature = "image")]
#[macro_export]
macro_rules! impl_image_generation {
    ($struct_:ident) => {
        impl rig::client::AsImageGeneration for $struct_ {}
    };
}

#[cfg(not(feature = "image"))]
#[macro_export]
macro_rules! impl_image_generation {
    ($struct_:ident) => {};
}

pub use impl_audio_generation;
pub use impl_conversion_traits;
pub use impl_image_generation;

#[cfg(feature = "audio")]
use crate::client::audio_generation::AudioGenerationClientDyn;
use crate::client::completion::CompletionClientDyn;
use crate::client::embeddings::EmbeddingsClientDyn;
#[cfg(feature = "image")]
use crate::client::image_generation::ImageGenerationClientDyn;
use crate::client::transcription::TranscriptionClientDyn;
use crate::client::verify::VerifyClientDyn;

#[cfg(feature = "audio")]
pub use crate::client::audio_generation::AudioGenerationClient;
pub use crate::client::completion::CompletionClient;
pub use crate::client::embeddings::EmbeddingsClient;
#[cfg(feature = "image")]
pub use crate::client::image_generation::ImageGenerationClient;
pub use crate::client::transcription::TranscriptionClient;
pub use crate::client::verify::{VerifyClient, VerifyError};

#[cfg(test)]
mod tests {
    use crate::OneOrMany;
    use crate::client::ProviderClient;
    use crate::completion::{Completion, CompletionRequest, ToolDefinition};
    use crate::image_generation::ImageGenerationRequest;
    use crate::message::AssistantContent;
    use crate::providers::{
        anthropic, azure, cohere, deepseek, galadriel, gemini, huggingface, hyperbolic, mira,
        moonshot, openai, openrouter, together, xai,
    };
    use crate::streaming::StreamingCompletion;
    use crate::tool::Tool;
    use crate::transcription::TranscriptionRequest;
    use futures::StreamExt;
    use rig::message::Message;
    use rig::providers::{groq, ollama, perplexity};
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::fs::File;
    use std::io::Read;

    use super::ProviderValue;

    struct ClientConfig {
        name: &'static str,
        factory_env: Box<dyn Fn() -> Box<dyn ProviderClient>>,
        // Not sure where we're going to be using this but I've added it for completeness
        #[allow(dead_code)]
        factory_val: Box<dyn Fn(ProviderValue) -> Box<dyn ProviderClient>>,
        env_variable: &'static str,
        completion_model: Option<&'static str>,
        embeddings_model: Option<&'static str>,
        transcription_model: Option<&'static str>,
        image_generation_model: Option<&'static str>,
        audio_generation_model: Option<(&'static str, &'static str)>,
    }

    impl Default for ClientConfig {
        fn default() -> Self {
            Self {
                name: "",
                factory_env: Box::new(|| panic!("Not implemented")),
                factory_val: Box::new(|_| panic!("Not implemented")),
                env_variable: "",
                completion_model: None,
                embeddings_model: None,
                transcription_model: None,
                image_generation_model: None,
                audio_generation_model: None,
            }
        }
    }

    impl ClientConfig {
        fn is_env_var_set(&self) -> bool {
            self.env_variable.is_empty() || std::env::var(self.env_variable).is_ok()
        }

        fn factory_env(&self) -> Box<dyn ProviderClient + '_> {
            self.factory_env.as_ref()()
        }
    }

    fn providers() -> Vec<ClientConfig> {
        vec![
            ClientConfig {
                name: "Anthropic",
                factory_env: Box::new(anthropic::Client::from_env_boxed),
                factory_val: Box::new(anthropic::Client::from_val_boxed),
                env_variable: "ANTHROPIC_API_KEY",
                completion_model: Some(anthropic::CLAUDE_3_5_SONNET),
                ..Default::default()
            },
            ClientConfig {
                name: "Cohere",
                factory_env: Box::new(cohere::Client::from_env_boxed),
                factory_val: Box::new(cohere::Client::from_val_boxed),
                env_variable: "COHERE_API_KEY",
                completion_model: Some(cohere::COMMAND_R),
                embeddings_model: Some(cohere::EMBED_ENGLISH_LIGHT_V2),
                ..Default::default()
            },
            ClientConfig {
                name: "Gemini",
                factory_env: Box::new(gemini::Client::from_env_boxed),
                factory_val: Box::new(gemini::Client::from_val_boxed),
                env_variable: "GEMINI_API_KEY",
                completion_model: Some(gemini::completion::GEMINI_2_0_FLASH),
                embeddings_model: Some(gemini::embedding::EMBEDDING_001),
                transcription_model: Some(gemini::transcription::GEMINI_2_0_FLASH),
                ..Default::default()
            },
            ClientConfig {
                name: "Huggingface",
                factory_env: Box::new(huggingface::Client::from_env_boxed),
                factory_val: Box::new(huggingface::Client::from_val_boxed),
                env_variable: "HUGGINGFACE_API_KEY",
                completion_model: Some(huggingface::PHI_4),
                transcription_model: Some(huggingface::WHISPER_SMALL),
                image_generation_model: Some(huggingface::STABLE_DIFFUSION_3),
                ..Default::default()
            },
            ClientConfig {
                name: "OpenAI",
                factory_env: Box::new(openai::Client::from_env_boxed),
                factory_val: Box::new(openai::Client::from_val_boxed),
                env_variable: "OPENAI_API_KEY",
                completion_model: Some(openai::GPT_4O),
                embeddings_model: Some(openai::TEXT_EMBEDDING_ADA_002),
                transcription_model: Some(openai::WHISPER_1),
                image_generation_model: Some(openai::DALL_E_2),
                audio_generation_model: Some((openai::TTS_1, "onyx")),
            },
            ClientConfig {
                name: "OpenRouter",
                factory_env: Box::new(openrouter::Client::from_env_boxed),
                factory_val: Box::new(openrouter::Client::from_val_boxed),
                env_variable: "OPENROUTER_API_KEY",
                completion_model: Some(openrouter::CLAUDE_3_7_SONNET),
                ..Default::default()
            },
            ClientConfig {
                name: "Together",
                factory_env: Box::new(together::Client::from_env_boxed),
                factory_val: Box::new(together::Client::from_val_boxed),
                env_variable: "TOGETHER_API_KEY",
                completion_model: Some(together::ALPACA_7B),
                embeddings_model: Some(together::BERT_BASE_UNCASED),
                ..Default::default()
            },
            ClientConfig {
                name: "XAI",
                factory_env: Box::new(xai::Client::from_env_boxed),
                factory_val: Box::new(xai::Client::from_val_boxed),
                env_variable: "XAI_API_KEY",
                completion_model: Some(xai::GROK_3_MINI),
                embeddings_model: None,
                ..Default::default()
            },
            ClientConfig {
                name: "Azure",
                factory_env: Box::new(azure::Client::from_env_boxed),
                factory_val: Box::new(azure::Client::from_val_boxed),
                env_variable: "AZURE_API_KEY",
                completion_model: Some(azure::GPT_4O),
                embeddings_model: Some(azure::TEXT_EMBEDDING_ADA_002),
                transcription_model: Some("whisper-1"),
                image_generation_model: Some("dalle-2"),
                audio_generation_model: Some(("tts-1", "onyx")),
            },
            ClientConfig {
                name: "Deepseek",
                factory_env: Box::new(deepseek::Client::from_env_boxed),
                factory_val: Box::new(deepseek::Client::from_val_boxed),
                env_variable: "DEEPSEEK_API_KEY",
                completion_model: Some(deepseek::DEEPSEEK_CHAT),
                ..Default::default()
            },
            ClientConfig {
                name: "Galadriel",
                factory_env: Box::new(galadriel::Client::from_env_boxed),
                factory_val: Box::new(galadriel::Client::from_val_boxed),
                env_variable: "GALADRIEL_API_KEY",
                completion_model: Some(galadriel::GPT_4O),
                ..Default::default()
            },
            ClientConfig {
                name: "Groq",
                factory_env: Box::new(groq::Client::from_env_boxed),
                factory_val: Box::new(groq::Client::from_val_boxed),
                env_variable: "GROQ_API_KEY",
                completion_model: Some(groq::MIXTRAL_8X7B_32768),
                transcription_model: Some(groq::DISTIL_WHISPER_LARGE_V3),
                ..Default::default()
            },
            ClientConfig {
                name: "Hyperbolic",
                factory_env: Box::new(hyperbolic::Client::from_env_boxed),
                factory_val: Box::new(hyperbolic::Client::from_val_boxed),
                env_variable: "HYPERBOLIC_API_KEY",
                completion_model: Some(hyperbolic::LLAMA_3_1_8B),
                image_generation_model: Some(hyperbolic::SD1_5),
                audio_generation_model: Some(("EN", "EN-US")),
                ..Default::default()
            },
            ClientConfig {
                name: "Mira",
                factory_env: Box::new(mira::Client::from_env_boxed),
                factory_val: Box::new(mira::Client::from_val_boxed),
                env_variable: "MIRA_API_KEY",
                completion_model: Some("gpt-4o"),
                ..Default::default()
            },
            ClientConfig {
                name: "Moonshot",
                factory_env: Box::new(moonshot::Client::from_env_boxed),
                factory_val: Box::new(moonshot::Client::from_val_boxed),
                env_variable: "MOONSHOT_API_KEY",
                completion_model: Some(moonshot::MOONSHOT_CHAT),
                ..Default::default()
            },
            ClientConfig {
                name: "Ollama",
                factory_env: Box::new(ollama::Client::from_env_boxed),
                factory_val: Box::new(ollama::Client::from_val_boxed),
                env_variable: "OLLAMA_ENABLED",
                completion_model: Some("llama3.1:8b"),
                embeddings_model: Some(ollama::NOMIC_EMBED_TEXT),
                ..Default::default()
            },
            ClientConfig {
                name: "Perplexity",
                factory_env: Box::new(perplexity::Client::from_env_boxed),
                factory_val: Box::new(perplexity::Client::from_val_boxed),
                env_variable: "PERPLEXITY_API_KEY",
                completion_model: Some(perplexity::SONAR),
                ..Default::default()
            },
        ]
    }

    async fn test_completions_client(config: &ClientConfig) {
        let client = config.factory_env();

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have completion_model set", config.name));

        let model = client.completion_model(model);

        let resp = model
            .completion_request(Message::user("Whats the capital of France?"))
            .send()
            .await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when prompting, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        match resp.choice.first() {
            AssistantContent::Text(text) => {
                assert!(text.text.to_lowercase().contains("paris"));
            }
            _ => {
                unreachable!(
                    "[{}]: First choice wasn't a Text message, {:?}",
                    config.name,
                    resp.choice.first()
                );
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_completions() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_completions_client(&p).await;
        }
    }

    async fn test_tools_client(config: &ClientConfig) {
        let client = config.factory_env();
        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have the model set.", config.name));

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = client.agent(model)
            .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let request = model.completion("Calculate 2 - 5", vec![]).await;

        assert!(
            request.is_ok(),
            "[{}]: Error occurred when building prompt, {}",
            config.name,
            request.err().unwrap()
        );

        let resp = request.unwrap().send().await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when prompting, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            resp.choice.iter().any(|content| match content {
                AssistantContent::ToolCall(tc) => {
                    if tc.function.name != Subtract::NAME {
                        return false;
                    }

                    let arguments =
                        serde_json::from_value::<OperationArgs>((tc.function.arguments).clone())
                            .expect("Error parsing arguments");

                    arguments.x == 2.0 && arguments.y == 5.0
                }
                _ => false,
            }),
            "[{}]: Model did not use the Subtract tool.",
            config.name
        )
    }

    #[tokio::test]
    #[ignore]
    async fn test_tools() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_tools_client(&p).await;
        }
    }

    async fn test_streaming_client(config: &ClientConfig) {
        let client = config.factory_env();

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have the model set.", config.name));

        let model = client.completion_model(model);

        let resp = model.stream(CompletionRequest {
            preamble: None,
            tools: vec![],
            documents: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            tool_choice: None,
            chat_history: OneOrMany::one(Message::user("What is the capital of France?")),
        });

        let mut resp = resp.await.unwrap();

        let mut received_chunk = false;

        while let Some(chunk) = resp.next().await {
            received_chunk = true;
            assert!(chunk.is_ok());
        }

        assert!(
            received_chunk,
            "[{}]: Failed to receive a chunk from stream",
            config.name
        );

        for choice in resp.choice {
            match choice {
                AssistantContent::Text(text) => {
                    assert!(
                        text.text.to_lowercase().contains("paris"),
                        "[{}]: Did not answer with Paris",
                        config.name
                    );
                }
                AssistantContent::ToolCall(_) => {}
                AssistantContent::Reasoning(_) => {}
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_streaming() {
        for provider in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_streaming_client(&provider).await;
        }
    }

    async fn test_streaming_tools_client(config: &ClientConfig) {
        let client = config.factory_env();
        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have the model set.", config.name));

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = client.agent(model)
            .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let request = model.stream_completion("Calculate 2 - 5", vec![]).await;

        assert!(
            request.is_ok(),
            "[{}]: Error occurred when building prompt, {}",
            config.name,
            request.err().unwrap()
        );

        let resp = request.unwrap().stream().await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when prompting, {}",
            config.name,
            resp.err().unwrap()
        );

        let mut resp = resp.unwrap();

        let mut received_chunk = false;

        while let Some(chunk) = resp.next().await {
            received_chunk = true;
            assert!(chunk.is_ok());
        }

        assert!(
            received_chunk,
            "[{}]: Failed to receive a chunk from stream",
            config.name
        );

        assert!(
            resp.choice.iter().any(|content| match content {
                AssistantContent::ToolCall(tc) => {
                    if tc.function.name != Subtract::NAME {
                        return false;
                    }

                    let arguments =
                        serde_json::from_value::<OperationArgs>((tc.function.arguments).clone())
                            .expect("Error parsing arguments");

                    arguments.x == 2.0 && arguments.y == 5.0
                }
                _ => false,
            }),
            "[{}]: Model did not use the Subtract tool.",
            config.name
        )
    }

    #[tokio::test]
    #[ignore]
    async fn test_streaming_tools() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_streaming_tools_client(&p).await;
        }
    }

    async fn test_audio_generation_client(config: &ClientConfig) {
        let client = config.factory_env();

        let Some(client) = client.as_audio_generation() else {
            return;
        };

        let (model, voice) = config
            .audio_generation_model
            .unwrap_or_else(|| panic!("{} doesn't have the model set", config.name));

        let model = client.audio_generation_model(model);

        let request = model
            .audio_generation_request()
            .text("Hello world!")
            .voice(voice);

        let resp = request.send().await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            !resp.audio.is_empty(),
            "[{}]: Returned audio was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_audio_generation() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_audio_generation_client(&p).await;
        }
    }

    fn assert_feature<F, M>(
        name: &str,
        feature_name: &str,
        model_name: &str,
        feature: Option<F>,
        model: Option<M>,
    ) {
        assert_eq!(
            feature.is_some(),
            model.is_some(),
            "{} has{} implemented {} but config.{} is {}.",
            name,
            if feature.is_some() { "" } else { "n't" },
            feature_name,
            model_name,
            if model.is_some() { "some" } else { "none" }
        );
    }

    #[test]
    #[ignore]
    pub fn test_polymorphism() {
        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            let client = config.factory_env();
            assert_feature(
                config.name,
                "AsCompletion",
                "completion_model",
                client.as_completion(),
                config.completion_model,
            );

            assert_feature(
                config.name,
                "AsEmbeddings",
                "embeddings_model",
                client.as_embeddings(),
                config.embeddings_model,
            );

            assert_feature(
                config.name,
                "AsTranscription",
                "transcription_model",
                client.as_transcription(),
                config.transcription_model,
            );

            assert_feature(
                config.name,
                "AsImageGeneration",
                "image_generation_model",
                client.as_image_generation(),
                config.image_generation_model,
            );

            assert_feature(
                config.name,
                "AsAudioGeneration",
                "audio_generation_model",
                client.as_audio_generation(),
                config.audio_generation_model,
            )
        }
    }

    async fn test_embed_client(config: &ClientConfig) {
        const TEST: &str = "Hello world.";

        let client = config.factory_env();

        let Some(client) = client.as_embeddings() else {
            return;
        };

        let model = config.embeddings_model.unwrap();

        let model = client.embedding_model(model);

        let resp = model.embed_text(TEST).await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert_eq!(resp.document, TEST);

        assert!(
            !resp.vec.is_empty(),
            "[{}]: Returned embed was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_embed() {
        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_embed_client(&config).await;
        }
    }

    async fn test_image_generation_client(config: &ClientConfig) {
        let client = config.factory_env();
        let Some(client) = client.as_image_generation() else {
            return;
        };

        let model = config.image_generation_model.unwrap();

        let model = client.image_generation_model(model);

        let resp = model
            .image_generation(ImageGenerationRequest {
                prompt: "A castle sitting on a large hill.".to_string(),
                width: 256,
                height: 256,
                additional_params: None,
            })
            .await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            !resp.image.is_empty(),
            "[{}]: Generated image was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_image_generation() {
        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_image_generation_client(&config).await;
        }
    }

    async fn test_transcription_client(config: &ClientConfig, data: Vec<u8>) {
        let client = config.factory_env();
        let Some(client) = client.as_transcription() else {
            return;
        };

        let model = config.image_generation_model.unwrap();

        let model = client.transcription_model(model);

        let resp = model
            .transcription(TranscriptionRequest {
                data,
                filename: "audio.mp3".to_string(),
                language: "en".to_string(),
                prompt: None,
                temperature: None,
                additional_params: None,
            })
            .await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            !resp.text.is_empty(),
            "[{}]: Returned transcription was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_transcription() {
        let mut file = File::open("examples/audio/en-us-natural-speech.mp3").unwrap();

        let mut data = Vec::new();
        let _ = file.read(&mut data);

        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_transcription_client(&config, data.clone()).await;
        }
    }

    #[derive(Deserialize)]
    struct OperationArgs {
        x: f32,
        y: f32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct Adder;
    impl Tool for Adder {
        const NAME: &'static str = "add";

        type Error = MathError;
        type Args = OperationArgs;
        type Output = f32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y together".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The first number to add"
                        },
                        "y": {
                            "type": "number",
                            "description": "The second number to add"
                        }
                    }
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> anyhow::Result<Self::Output, Self::Error> {
            println!("[tool-call] Adding {} and {}", args.x, args.y);
            let result = args.x + args.y;
            Ok(result)
        }
    }

    #[derive(Deserialize, Serialize)]
    struct Subtract;
    impl Tool for Subtract {
        const NAME: &'static str = "subtract";

        type Error = MathError;
        type Args = OperationArgs;
        type Output = f32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            serde_json::from_value(json!({
                "name": "subtract",
                "description": "Subtract y from x (i.e.: x - y)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to subtract from"
                        },
                        "y": {
                            "type": "number",
                            "description": "The number to subtract"
                        }
                    }
                }
            }))
            .expect("Tool Definition")
        }

        async fn call(&self, args: Self::Args) -> anyhow::Result<Self::Output, Self::Error> {
            println!("[tool-call] Subtracting {} from {}", args.y, args.x);
            let result = args.x - args.y;
            Ok(result)
        }
    }
}
