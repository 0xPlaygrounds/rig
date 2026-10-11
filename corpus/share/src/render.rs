//! The session as a minimal HTML page, with every secret found in it
//! redacted.

use std::fmt::Write as _;

use rig_harness::rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};

/// What a secret's prefix is followed by, at least, to count as one.
const MIN_SECRET_TAIL: usize = 16;

/// Prefixes of well-known API keys and tokens.
const SECRET_PREFIXES: &[&str] = &[
    "sk-",
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "ghp_",
    "gho_",
    "ghs_",
    "ghu_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "AKIA",
    "AIza",
    "hf_",
];

/// What replaces a secret in the page.
const REDACTED: &str = "[REDACTED]";

/// A rendered session: the page and the secrets it no longer holds, each
/// shown redacted.
pub struct Rendered {
    /// The HTML page.
    pub html: String,
    /// The messages in it.
    pub messages: usize,
    /// Each secret found, redacted for display (its start and length).
    pub secrets: Vec<String>,
}

/// Renders `messages` as an HTML page titled `title`, secrets redacted.
pub fn render(title: &str, messages: &[Message]) -> Rendered {
    let mut page = Page::default();
    for message in messages {
        match message {
            Message::System { content } => page.section("system", "System", content),
            Message::User { content } => {
                for item in content {
                    match item {
                        UserContent::Text(text) => page.section("user", "User", &text.text),
                        UserContent::ToolResult(result) => {
                            let text: Vec<String> =
                                result.content.iter().map(result_text).collect();
                            let heading = format!("Result of {}", result.name.as_str());
                            page.section("result", &heading, &text.join("\n"));
                        }
                        _ => page.section("user", "User", "(attachment)"),
                    }
                }
            }
            Message::Assistant(assistant) => {
                for item in &assistant.content {
                    match item {
                        AssistantContent::Text(text) => {
                            page.section("assistant", "Assistant", &text.text);
                        }
                        AssistantContent::Reasoning(reasoning) if !reasoning.redacted => {
                            page.section("reasoning", "Reasoning", &reasoning.text);
                        }
                        AssistantContent::ToolCall(call) => {
                            let heading = format!("Call of {}", call.function.name.as_str());
                            let args = serde_json::to_string_pretty(&call.function.arguments)
                                .unwrap_or_default();
                            page.section("call", &heading, &args);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    Rendered {
        html: page.finish(title),
        messages: messages.len(),
        secrets: page.secrets,
    }
}

/// A tool result item as text.
fn result_text(item: &ToolResultContent) -> String {
    match item {
        ToolResultContent::Text(text) => text.text.clone(),
        ToolResultContent::Json { value } => value.to_string(),
        ToolResultContent::Image(_) => "(image)".to_owned(),
    }
}

/// The page being written.
#[derive(Default)]
struct Page {
    body: String,
    secrets: Vec<String>,
}

impl Page {
    /// A section of class `class` under `heading`, its text redacted.
    fn section(&mut self, class: &str, heading: &str, text: &str) {
        let text = self.redact(text);
        let heading = self.redact(heading);
        let _ = writeln!(
            self.body,
            "<section class=\"{class}\"><h2>{}</h2><pre>{}</pre></section>",
            escape(&heading),
            escape(&text),
        );
    }

    /// `text` with each secret replaced, remembered for the confirmation.
    fn redact(&mut self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find(is_token_char) {
            let (before, from) = rest.split_at(start);
            out.push_str(before);
            let end = from.find(|c| !is_token_char(c)).unwrap_or(from.len());
            let (token, after) = from.split_at(end);
            if is_secret(token) {
                let shown: String = token.chars().take(6).collect();
                self.secrets
                    .push(format!("{shown}… ({} chars)", token.chars().count()));
                out.push_str(REDACTED);
            } else {
                out.push_str(token);
            }
            rest = after;
        }
        out.push_str(rest);
        out
    }

    /// The whole page.
    fn finish(&self, title: &str) -> String {
        let title = escape(title);
        format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>{title}</title>\
             <style>body{{font-family:sans-serif;max-width:60em;margin:auto;padding:1em}}\
             pre{{white-space:pre-wrap}}section{{border-left:3px solid #888;padding-left:1em}}\
             .user{{border-color:#36c}}.assistant{{border-color:#3a3}}</style></head>\n\
             <body><h1>{title}</h1>\n{}</body></html>\n",
            self.body
        )
    }
}

/// Whether `c` may be part of a key.
fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_'
}

/// Whether `token` looks like a key: a known prefix and a long tail.
fn is_secret(token: &str) -> bool {
    SECRET_PREFIXES.iter().any(|prefix| {
        token
            .strip_prefix(prefix)
            .is_some_and(|tail| tail.len() >= MIN_SECRET_TAIL)
    })
}

/// `text` safe inside HTML.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}
