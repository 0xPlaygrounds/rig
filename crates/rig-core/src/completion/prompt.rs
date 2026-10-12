//! A system prompt as data: a role and named sections, rendered into one
//! system message only when the request is sent, so a layer in between can
//! read, change or move a section by its tag.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, Section, SystemPrompt};
//!
//! let mut prompt = SystemPrompt {
//!     role: "You are a coding agent.".into(),
//!     sections: vec![Section::new("environment", "Today is Monday.")],
//! };
//! assert_eq!(
//!     prompt.text(),
//!     "You are a coding agent.\n\n<environment>\nToday is Monday.\n</environment>"
//! );
//! let request = CompletionRequest::new("Hello").with_system(&prompt);
//! assert_eq!(request.system_instructions(), Some(prompt.text().as_str()));
//!
//! let moved = prompt.take("environment");
//! assert_eq!(moved.map(|section| section.text), Some("Today is Monday.".into()));
//! assert_eq!(prompt.text(), "You are a coding agent.");
//! ```

use serde::{Deserialize, Serialize};

use super::request::CompletionRequest;

/// A system prompt in parts: the role, then each non-empty section between
/// `<tag>` and `</tag>`, sent as one system message.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemPrompt {
    /// The text the prompt opens with: who the model is and how it works.
    pub role: String,
    /// The sections after the role, in the order they are sent.
    pub sections: Vec<Section>,
}

/// A named part of a [`SystemPrompt`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    /// The tag around the text, such as `project_instructions`.
    pub tag: String,
    /// The text. Blank text leaves the section out of the rendering.
    pub text: String,
}

impl Section {
    /// A section of `text` between `<tag>` and `</tag>`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

impl SystemPrompt {
    /// The prompt as one text: the role, then each non-empty section tagged,
    /// a blank line apart, each trimmed. The one rendering; a user who wants
    /// another renders `role` and `sections` itself.
    pub fn text(&self) -> String {
        let mut prompt = self.role.trim().to_owned();
        for section in &self.sections {
            let text = section.text.trim();
            if text.is_empty() {
                continue;
            }
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            let tag = &section.tag;
            prompt.push_str(&format!("<{tag}>\n{text}\n</{tag}>"));
        }
        prompt
    }

    /// The first section tagged `tag`, if there is one.
    pub fn section(&self, tag: &str) -> Option<&Section> {
        self.sections.iter().find(|section| section.tag == tag)
    }

    /// Takes the first section tagged `tag` out, if there is one.
    pub fn take(&mut self, tag: &str) -> Option<Section> {
        let at = self
            .sections
            .iter()
            .position(|section| section.tag == tag)?;
        Some(self.sections.remove(at))
    }
}

impl CompletionRequest {
    /// This request with `prompt`, rendered by [`SystemPrompt::text`], as its
    /// leading system message, ahead of any system message already there.
    #[must_use]
    pub fn with_system(self, prompt: &SystemPrompt) -> Self {
        self.preamble(prompt.text())
    }
}

#[cfg(test)]
mod tests;
