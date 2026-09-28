//! Explicit caching of the prefix every request repeats: the system
//! instruction, tools and tool config. A cache is rendered by the same code
//! that renders a `generateContent` body, so a request reading it can prove it
//! matches and leave that prefix out.
//!
//! ```no_run
//! use std::time::Duration;
//! use rig_core::providers::gemini::{self, CacheExpiry, Gemini, NewCachedContent};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::from_env()?;
//! let prefix = gemini
//!     .cached_contents()
//!     .create(
//!         NewCachedContent::new(gemini::GEMINI_3_8_FLASH)
//!             .system_instruction("A long, stable preamble.")
//!             .expiry(CacheExpiry::ttl(Duration::from_secs(3600))),
//!     )
//!     .await?;
//! let model = gemini.completion(gemini::GEMINI_3_8_FLASH).cached_content(prefix);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use super::api;
use super::cached_content::CacheExpiry;
use super::generate_content::{convert, mixes_tools, tool_list};
use crate::completion::ToolDefinition;
use crate::error::EncodeError;

/// A cache to create: the prefix an agent sends every turn, and how long it
/// lives. Gemini refuses caches under its minimum size (1,024 tokens on 3.x
/// models).
#[derive(Clone, Debug, Default)]
pub struct NewCachedContent {
    request: api::CachedContent,
    tools: Vec<ToolDefinition>,
    hosted: Vec<api::HostedTool>,
    tool_config: api::ToolConfigSettings,
}

impl NewCachedContent {
    /// A cache for `model`, the id or `models/<id>`.
    pub fn new(model: impl AsRef<str>) -> Self {
        let model = model.as_ref();
        let model = if model.starts_with("models/") {
            model.to_owned()
        } else {
            format!("models/{model}")
        };
        Self {
            request: api::CachedContent {
                model: Some(model),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// The preamble requests would send as their system instruction.
    pub fn system_instruction(mut self, text: impl Into<String>) -> Self {
        self.request.system_instruction = Some(api::Content {
            parts: vec![api::Part {
                text: Some(text.into()),
                ..Default::default()
            }],
            ..Default::default()
        });
        self
    }

    /// The function tools requests would declare, in the order they declare
    /// them.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// The hosted tools the model's settings list, in their order.
    pub fn hosted_tools(mut self, tools: impl IntoIterator<Item = api::HostedTool>) -> Self {
        self.hosted.extend(tools);
        self
    }

    /// The tool config the model's settings carry. A request reading a cache
    /// cannot send one of its own.
    pub fn tool_config(mut self, tool_config: api::ToolConfigSettings) -> Self {
        self.tool_config = tool_config;
        self
    }

    /// A user-role text block every request reads before its own contents.
    pub fn content(mut self, text: impl Into<String>) -> Self {
        self.request.contents.push(api::Content {
            role: Some("user".to_owned()),
            parts: vec![api::Part {
                text: Some(text.into()),
                ..Default::default()
            }],
            ..Default::default()
        });
        self
    }

    /// A name for listings.
    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.request.display_name = Some(name.into());
        self
    }

    /// When the cache expires. Without one Gemini keeps it an hour.
    pub fn expiry(mut self, expiry: CacheExpiry) -> Self {
        match expiry {
            CacheExpiry::Ttl(ttl) => {
                self.request.ttl = Some(CacheExpiry::ttl_string(ttl));
                self.request.expire_time = None;
            }
            CacheExpiry::ExpireTime(at) => {
                self.request.expire_time = Some(at);
                self.request.ttl = None;
            }
        }
        self
    }

    /// The `cachedContents` body. Tools and tool config render exactly as a
    /// `generateContent` request renders them.
    pub(crate) fn render(self) -> Result<api::CachedContent, EncodeError> {
        let mut request = self.request;
        if request.contents.is_empty() && request.system_instruction.is_none() {
            return Err(EncodeError::request(
                "a cached content needs a system instruction or contents",
            ));
        }
        request.tools = tool_list(&self.tools, &self.hosted)?;
        let mut tool_config: api::ToolConfig = convert(&self.tool_config)?;
        tool_config.include_server_side_tool_invocations =
            mixes_tools(&request.tools).then_some(true);
        request.tool_config = (tool_config != api::ToolConfig::default()).then_some(tool_config);
        Ok(request)
    }
}

/// A created cache and the body that created it. Serializable, so a
/// checkpoint can carry it and [`ensure`](crate::driver::Model::ensure)
/// can recreate it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedPrefix {
    /// The resource Gemini returned: name, model, expiry and token count.
    pub resource: api::CachedContent,
    /// The body that created it, hosted tools and tool config included.
    pub request: api::CachedContent,
}

impl CachedPrefix {
    /// The cache's handle, `cachedContents/<id>`.
    pub fn name(&self) -> &str {
        self.resource.name.as_deref().unwrap_or_default()
    }

    /// Replace `request`'s system instruction, tools and tool config with
    /// this cache, which must hold exactly those.
    pub(crate) fn strip(
        &self,
        request: &mut api::GenerateContentRequest,
    ) -> Result<(), EncodeError> {
        let conflict = |what: &str| {
            EncodeError::request(format!(
                "cached content `{}` conflicts with the request's {what}",
                self.name()
            ))
        };
        if request.system_instruction != self.request.system_instruction {
            return Err(conflict("system instruction"));
        }
        if request.tools != self.request.tools {
            return Err(conflict("tools"));
        }
        if request.tool_config != self.request.tool_config {
            return Err(conflict("tool config"));
        }
        request.system_instruction = None;
        request.tools = Vec::new();
        request.tool_config = None;
        request.cached_content = Some(self.name().to_owned());
        Ok(())
    }
}

#[cfg(test)]
mod tests;
