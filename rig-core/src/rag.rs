//! This module contains the implementation of the [RagAgent] struct and its builder.
//!
//! The [RagAgent] struct defines a fully featured RAG system, combining and agent with
//! dynamic context documents and tools which are used to enhance the completion model at
//! prompt-time.
//!
//! The [RagAgentBuilder] implements the builder pattern for creating instances of [RagAgent].
//! It allows configuring the model, preamble, dynamic context documents and tools, as well
//! as all the other parameters that are available in the [crate::agent::AgentBuilder].
//!
//! # Example
//! ```rust
//! use rig::{
//!     completion::Prompt,
//!     embeddings::EmbeddingsBuilder,
//!     providers::openai,
//!     vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
//! };
//!
//! // Initialize OpenAI client
//! let openai = openai::Client::from_env();
//!
//! // Initialize OpenAI embedding model
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
//!
//! // Create vector store, compute embeddings and load them in the store
//! let mut vector_store = InMemoryVectorStore::default();
//!
//! let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
//!     .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
//!     .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
//!     .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
//!     .build()
//!     .await
//!     .expect("Failed to build embeddings");
//!
//! vector_store.add_documents(embeddings)
//!     .await
//!     .expect("Failed to add documents");
//!
//! // Create vector store index
//! let index = vector_store.index(embedding_model);
//!
//! let rag_agent = openai.context_rag_agent(openai::GPT_4O)
//!     .preamble("
//!         You are a dictionary assistant here to assist the user in understanding the meaning of words.
//!         You will find additional non-standard word definitions that could be useful below.
//!     ")
//!     .dynamic_context(1, index)
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await
//!     .expect("Failed to prompt the agent");
//! ```

use crate::agent::{Agent, AgentBuilder};

#[deprecated(
    since = "0.2.0",
    note = "Please use the `Agent` type directly instead of the `RagAgent` type."
)]
pub type RagAgent<M> = Agent<M>;

#[deprecated(
    since = "0.2.0",
    note = "Please use the `AgentBuilder` type directly instead of the `RagAgentBuilder` type."
)]
pub type RagAgentBuilder<M> = AgentBuilder<M>;
