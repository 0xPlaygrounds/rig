//! `gem`, a "research and illustrate" assistant on Google Gemini over gRPC,
//! built on rig's library crates only: [`rig_gemini_grpc`] for the
//! transport, [`rig_agent`] for the streamed tool-calling chat,
//! [`rig_memory`] for the conversation kept on disk, and rig-core's
//! embeddings and in-memory vector store for the notes index.
//!
//! Model names are plain strings everywhere: nothing here lists models.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use base64::Engine as _;
use futures::StreamExt;
use rig_agent::prelude::*;
use rig_core::completion::{
    AssistantContent, CompletionRequest, Effort, FinishReason, GenerationOptions, Reasoning, Usage,
};
use rig_core::embeddings::{EmbedError, TextEmbedder};
use rig_core::error::ErrorReport;
use rig_core::message::{DocumentSourceKind, MimeType as _};
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::gemini::extension::{GeminiOptions, ResponseModality};
use rig_core::streaming::{Item, StreamEvent};
use rig_core::tool::{ToolContext, ToolExecutionError};
use rig_core::vector_store::in_memory_store::InMemoryVectorIndex;
use rig_gemini_grpc::GeminiGrpc;
use rig_gemini_grpc::extension::{GeminiGrpcExt, GeminiGrpcOptions};
use serde::{Deserialize, Serialize};

/// How much a request thinks: off, a token budget, or a level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thinking {
    /// No thinking, where the model can turn it off.
    Off,
    /// At most this many thinking tokens (Gemini 2.5).
    Budget(u32),
    /// A thinking level such as low or high (Gemini 3).
    Level(Effort),
}

impl Thinking {
    /// Read `off`, a number (a budget) or a level word (`minimal`, `low`,
    /// `medium`, `high`).
    pub fn parse(text: &str) -> Option<Self> {
        if let Ok(tokens) = text.parse() {
            return Some(Self::Budget(tokens));
        }
        Some(match text {
            "off" => Self::Off,
            "minimal" => Self::Level(Effort::Minimal),
            "low" => Self::Level(Effort::Low),
            "medium" => Self::Level(Effort::Medium),
            "high" => Self::Level(Effort::High),
            _ => return None,
        })
    }
}

impl From<Thinking> for Reasoning {
    fn from(thinking: Thinking) -> Self {
        match thinking {
            Thinking::Off => Self::Off,
            Thinking::Budget(tokens) => Self::Budget { tokens },
            Thinking::Level(effort) => Self::Effort(effort),
        }
    }
}

/// One call's token counts. A count the provider did not give is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallUsage {
    /// Every input token.
    pub input: Option<u64>,
    /// Every output token, thinking included.
    pub output: Option<u64>,
    /// The part of `output` spent thinking.
    pub thinking: Option<u64>,
    /// The part of `input` read from the cache.
    pub cached: Option<u64>,
}

impl From<&Usage> for CallUsage {
    fn from(usage: &Usage) -> Self {
        Self {
            input: usage.input_tokens,
            output: usage.output_tokens,
            thinking: usage.reasoning_tokens,
            cached: usage.cached_input_tokens,
        }
    }
}

impl std::fmt::Display for CallUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = |count: Option<u64>| count.map_or_else(|| "?".to_owned(), |n| n.to_string());
        write!(
            f,
            "input {} (cached {}), output {} (thinking {})",
            count(self.input),
            count(self.cached),
            count(self.output),
            count(self.thinking)
        )
    }
}

/// Why a call failed, sorted by what the app does about it.
#[derive(Debug, thiserror::Error)]
pub enum GemError {
    /// Out of quota or rate limited: retry, after `retry_after` when the
    /// provider gave a delay.
    #[error("rate limited ({code}): {message}")]
    RateLimited {
        /// The provider's status code name.
        code: String,
        /// The provider's message.
        message: String,
        /// The wait the provider asked for, when it gave one.
        retry_after: Option<Duration>,
    },
    /// The prompt or the reply was blocked by a safety filter.
    #[error("blocked ({reason}){}", categories_suffix(.categories))]
    Blocked {
        /// The block reason the provider gave (`SAFETY`, `IMAGE_SAFETY`).
        reason: String,
        /// The harm categories that blocked it, when the provider named them.
        categories: Vec<String>,
    },
    /// The reply stopped part way: `partial` is the answer text that arrived.
    #[error("the reply was cut off after {} characters: {message}", .partial.len())]
    Cut {
        /// The answer text that arrived before the cut.
        partial: String,
        /// What ended it.
        message: String,
        /// Whether sending again may succeed.
        retryable: bool,
    },
    /// Any other provider failure, such as an unknown model or an option
    /// the model does not take, with the provider's own message.
    #[error("{message}")]
    Provider {
        /// The provider's status code name, when it gave one.
        code: Option<String>,
        /// The provider's message.
        message: String,
        /// Whether sending again may succeed.
        retryable: bool,
    },
    /// The transport could not be set up.
    #[error(transparent)]
    Connect(#[from] rig_gemini_grpc::GeminiGrpcError),
    /// The conversation store failed.
    #[error(transparent)]
    Memory(#[from] rig_memory::SessionError),
    /// A note produced nothing to embed.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// A file could not be read or written.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// An image in the reply was not valid base64.
    #[error("an image in the reply does not decode: {0}")]
    Image(#[from] base64::DecodeError),
}

fn categories_suffix(categories: &[String]) -> String {
    match categories.is_empty() {
        true => String::new(),
        false => format!(": {}", categories.join(", ")),
    }
}

impl GemError {
    /// Whether sending the same request again may succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimited { .. } => true,
            Self::Cut { retryable, .. } | Self::Provider { retryable, .. } => *retryable,
            _ => false,
        }
    }

    /// The wait the provider asked for before a retry.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Sort a failure `report`, after `partial` answer text arrived.
    fn from_report(report: &ErrorReport, partial: String) -> Self {
        let code = report.code.clone();
        let message = report
            .provider_response
            .as_ref()
            .map_or_else(|| report.message.clone(), |reply| reply.body.clone());
        if report.refusal {
            return Self::Blocked {
                reason: code.unwrap_or_else(|| "SAFETY".to_owned()),
                // The block's harm categories reach rig only inside the
                // message text; see the report.
                categories: Vec::new(),
            };
        }
        if code.as_deref() == Some("RESOURCE_EXHAUSTED") || report.http_status == Some(429) {
            return Self::RateLimited {
                code: code.unwrap_or_else(|| "RESOURCE_EXHAUSTED".to_owned()),
                message,
                retry_after: report.retry_after(SystemTime::now()),
            };
        }
        if !partial.is_empty() {
            return Self::Cut {
                partial,
                message,
                retryable: report.retryable,
            };
        }
        Self::Provider {
            code,
            message,
            retryable: report.retryable,
        }
    }

    fn from_prompt(error: &PromptError, partial: String) -> Self {
        let report = match error {
            PromptError::Provider(error) => error.report(),
            PromptError::Report(report) => report.clone(),
            PromptError::Memory { error, .. } => {
                return Self::Provider {
                    code: None,
                    message: error.to_string(),
                    retryable: false,
                };
            }
            other => {
                return Self::Provider {
                    code: None,
                    message: other.to_string(),
                    retryable: false,
                };
            }
        };
        Self::from_report(&report, partial)
    }
}

/// The app's Gemini connection.
#[derive(Clone, Debug)]
pub struct Gem {
    transport: GeminiGrpc,
}

impl Gem {
    /// Connect to `config`'s endpoint.
    pub async fn connect(config: &GeminiConfig) -> Result<Self, GemError> {
        Ok(Self {
            transport: GeminiGrpc::connect(config).await?,
        })
    }

    /// Send through `channel`, a channel the host built (a test's fake).
    pub fn with_channel(
        config: &GeminiConfig,
        channel: tonic::transport::Channel,
    ) -> Result<Self, GemError> {
        Ok(Self {
            transport: GeminiGrpc::with_channel(config, channel)?,
        })
    }

    /// A chat over `store`'s conversation `settings.conversation`, with a
    /// `search_notes` tool over `notes`.
    pub fn chat(
        &self,
        settings: ChatSettings,
        store: impl rig_memory::AsyncStore + Send + Sync + 'static,
        notes: Option<Notes>,
    ) -> Chat {
        let mut builder = AgentBuilder::new(self.transport.completion(settings.model))
            .preamble(
                "You are a research assistant. Use search_notes to look things up in the \
                 user's notes before answering questions about them.",
            )
            .options(GenerationOptions::new().reasoning(settings.thinking))
            .default_max_turns(8)
            .memory(store)
            .conversation(settings.conversation)
            .tool(SearchNotes { notes });
        if settings.thinking != Thinking::Off {
            builder = builder.provider_option(GeminiGrpcOptions::from(
                GeminiOptions::new().include_thoughts(true),
            ));
        }
        Chat {
            agent: builder.build(),
        }
    }

    /// Generate images for `prompt` with `model`, thinking as `thinking`
    /// when set.
    pub async fn draw(
        &self,
        model: &str,
        prompt: &str,
        thinking: Option<Thinking>,
    ) -> Result<Drawing, GemError> {
        let mut options = GeminiOptions::new()
            .response_modalities([ResponseModality::Text, ResponseModality::Image]);
        let mut request = CompletionRequest::new(prompt);
        if let Some(thinking) = thinking {
            request = request.reasoning(thinking);
            options = options.include_thoughts(thinking != Thinking::Off);
        }
        let request = request.provider_option(GeminiGrpcOptions::from(options));
        let response = self
            .transport
            .completion(model)
            .call(request)
            .await
            .map_err(|error| GemError::from_report(&error.report(), String::new()))?;
        if response.finish_reason() == Some(FinishReason::ContentFilter) {
            let extras = response.extras_lossy::<GeminiGrpcExt>();
            return Err(GemError::Blocked {
                reason: extras
                    .finish_message
                    .unwrap_or_else(|| "content filter".to_owned()),
                categories: extras
                    .safety_ratings
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|rating| rating.blocked == Some(true))
                    .filter_map(|rating| rating.category)
                    .collect(),
            });
        }
        let mut parts = Vec::new();
        let mut thoughts = String::new();
        for content in &response.choice {
            match content {
                AssistantContent::Text(text) => parts.push(DrawnPart::Text(text.text.clone())),
                AssistantContent::Reasoning(reasoning) => thoughts.push_str(&reasoning.text),
                AssistantContent::Image(image) => {
                    let bytes = match &image.data {
                        DocumentSourceKind::Base64(data) => {
                            base64::engine::general_purpose::STANDARD.decode(data)?
                        }
                        DocumentSourceKind::Raw(bytes) => bytes.clone(),
                        _ => continue,
                    };
                    let mime = image
                        .media_type
                        .as_ref()
                        .map_or("application/octet-stream", |media| media.to_mime_type());
                    parts.push(DrawnPart::Image {
                        mime: mime.to_owned(),
                        bytes,
                    });
                }
                _ => {}
            }
        }
        Ok(Drawing {
            parts,
            thoughts,
            usage: CallUsage::from(&response.usage),
        })
    }

    /// Embed `texts` with `model`, `dims` wide, in order.
    pub async fn embed(
        &self,
        model: &str,
        dims: usize,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f64>>, GemError> {
        let embedded = EmbeddingsBuilder::new(self.transport.embedding(model, Some(dims)))
            .documents(texts)?
            .build()
            .await
            .map_err(|error| GemError::from_report(&error.report(), String::new()))?;
        Ok(embedded
            .into_iter()
            .map(|(_, embeddings)| {
                embeddings
                    .into_iter()
                    .flat_map(|embedding| embedding.vec)
                    .collect()
            })
            .collect())
    }

    /// Embed every text file in `dir` with `model`, `dims` wide, into an
    /// in-memory index.
    pub async fn index(&self, model: &str, dims: usize, dir: &Path) -> Result<Notes, GemError> {
        let mut notes = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_file()
                && let Ok(text) = std::fs::read_to_string(&path)
                && !text.trim().is_empty()
            {
                notes.push(Note {
                    path: path.display().to_string(),
                    text,
                });
            }
        }
        let embedding = self.transport.embedding(model, Some(dims));
        let embedded = EmbeddingsBuilder::new(embedding.clone())
            .documents(notes)?
            .build()
            .await
            .map_err(|error| GemError::from_report(&error.report(), String::new()))?;
        let count = embedded.len();
        let store = InMemoryVectorStore::from_documents(embedded);
        Ok(Notes {
            index: Arc::new(store.index(embedding)),
            count,
        })
    }
}

/// What a [`Chat`] talks to and how.
#[derive(Clone, Debug)]
pub struct ChatSettings {
    /// The model name, such as `gemini-2.5-pro`.
    pub model: String,
    /// The thinking every turn asks for.
    pub thinking: Thinking,
    /// The conversation's name in the store.
    pub conversation: String,
}

/// What a chat turn streams to the UI as it arrives.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatEvent {
    /// A fragment of a thought summary, shown dimmed.
    Thought(String),
    /// A fragment of the answer.
    Answer(String),
    /// The model called a tool.
    ToolCall(String),
    /// One model call finished with this usage.
    Usage(CallUsage),
}

/// A chat turn's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatReply {
    /// The final answer, without any thought summary.
    pub answer: String,
    /// The thought summaries of the turn, joined.
    pub thoughts: String,
    /// The usage of each model call of the turn.
    pub usage: Vec<CallUsage>,
}

/// A streamed chat with thinking and a `search_notes` tool, kept in a store.
pub struct Chat {
    agent: Agent,
}

impl Chat {
    /// Send `prompt`, streaming thoughts and answer text to `on_event`.
    pub async fn send(
        &self,
        prompt: &str,
        mut on_event: impl FnMut(ChatEvent),
    ) -> Result<ChatReply, GemError> {
        let mut stream = self.agent.prompt(prompt).stream();
        let mut partial = String::new();
        let mut thoughts = String::new();
        let mut usage = Vec::new();
        while let Some(item) = stream.next().await {
            let item = item.map_err(|error| GemError::from_prompt(&error, partial.clone()))?;
            match item {
                MultiTurnStreamItem::StreamAssistantItem(Item::Event(event)) => match event {
                    StreamEvent::Reasoning { text, .. } => {
                        thoughts.push_str(&text);
                        on_event(ChatEvent::Thought(text));
                    }
                    StreamEvent::Text { text, .. } => {
                        partial.push_str(&text);
                        on_event(ChatEvent::Answer(text));
                    }
                    _ => {}
                },
                MultiTurnStreamItem::ToolCall { tool_call } => {
                    on_event(ChatEvent::ToolCall(tool_call.function.name.to_string()));
                }
                MultiTurnStreamItem::CompletionCall(call) => {
                    let call = CallUsage::from(&call.usage);
                    usage.push(call);
                    on_event(ChatEvent::Usage(call));
                }
                MultiTurnStreamItem::ModelTurnRetried { .. } => partial.clear(),
                MultiTurnStreamItem::FinalResponse(response) => {
                    return Ok(ChatReply {
                        answer: response.output(),
                        thoughts,
                        usage,
                    });
                }
                _ => {}
            }
        }
        Err(GemError::Cut {
            partial,
            message: "the stream ended without a final response".to_owned(),
            retryable: true,
        })
    }
}

/// One part of a drawing reply, in reply order.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawnPart {
    /// Text, such as a caption.
    Text(String),
    /// An image's bytes and MIME type.
    Image {
        /// The MIME type, such as `image/png`.
        mime: String,
        /// The image file's bytes.
        bytes: Vec<u8>,
    },
}

/// A drawing reply.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawing {
    /// The text and images, in reply order.
    pub parts: Vec<DrawnPart>,
    /// The thought summary, when the model shared one.
    pub thoughts: String,
    /// The call's usage.
    pub usage: CallUsage,
}

impl Drawing {
    /// Write each image to `dir` as `<stem>-<n>.<ext>`, numbered from 1.
    pub fn save(&self, dir: &Path, stem: &str) -> std::io::Result<Vec<PathBuf>> {
        let mut saved = Vec::new();
        for (mime, bytes) in self.parts.iter().filter_map(|part| match part {
            DrawnPart::Image { mime, bytes } => Some((mime, bytes)),
            DrawnPart::Text(_) => None,
        }) {
            let ext = mime.rsplit('/').next().unwrap_or("bin");
            let ext = if ext == "jpeg" { "jpg" } else { ext };
            let path = dir.join(format!("{stem}-{}.{ext}", saved.len() + 1));
            std::fs::write(&path, bytes)?;
            saved.push(path);
        }
        Ok(saved)
    }
}

/// A file name stem for `prompt`: its first word that is not an article.
pub fn stem_for(prompt: &str) -> String {
    prompt
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .find(|word| !word.is_empty() && !["a", "an", "the"].contains(&word.as_str()))
        .unwrap_or_else(|| "image".to_owned())
}

/// One embedded text file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Note {
    /// Where it was read from.
    pub path: String,
    /// Its text.
    pub text: String,
}

impl rig_core::Embed for Note {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

/// An in-memory index of embedded notes.
#[derive(Clone)]
pub struct Notes {
    index: Arc<InMemoryVectorIndex<Note>>,
    count: usize,
}

impl Notes {
    /// How many notes it holds.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// The `search_notes` tool.
#[derive(Clone)]
struct SearchNotes {
    notes: Option<Notes>,
}

/// The arguments of `search_notes`.
#[derive(Deserialize, schemars::JsonSchema)]
struct SearchArgs {
    /// What to look for.
    query: String,
}

impl Tool for SearchNotes {
    const NAME: &'static str = "search_notes";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Search the user's notes; returns the most relevant passages with their file paths."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(SearchArgs)).unwrap_or_default()
    }

    async fn call(&self, _: &mut ToolContext, args: SearchArgs) -> Result<String, Self::Error> {
        let Some(notes) = &self.notes else {
            return Ok("No notes are indexed; run `gem index <dir>`.".to_owned());
        };
        let request = VectorSearchRequest::builder()
            .query(args.query)
            .samples(3)
            .build();
        let hits = notes
            .index
            .top_n::<Note>(request)
            .await
            .map_err(ToolExecutionError::from_error)?;
        if hits.is_empty() {
            return Ok("No notes match.".to_owned());
        }
        Ok(hits
            .into_iter()
            .map(|hit| format!("{}:\n{}", hit.document.path, hit.document.text))
            .collect::<Vec<_>>()
            .join("\n\n"))
    }
}

#[cfg(test)]
mod tests;
