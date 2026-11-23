//! What the agent's effect records become on Linear: agent activities, cut
//! to size and redacted, and the pull request link a `gh pr create`
//! command printed.

use rig_core::completion::CompletionResponse;
use rig_core::effect::Answered;
use rig_core::error::ErrorReport;
use rig_core::message::AssistantContent;
use rig_core::tool::{ExitStatus, ToolResultContext};
use rig_harness::links::SessionLink;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// The most characters of a note.
const NOTE_MOST: usize = 2000;

/// The most characters of an action's parameter, and of the output a
/// failed action keeps.
const ACTION_MOST: usize = 300;

/// What stands for a secret in what is sent.
const REDACTED: &str = "[REDACTED]";

/// What an activity says: the `content` of `agentActivityCreate`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Content {
    /// A thought: a note, or reasoning.
    Thought(String),
    /// A question to the people watching.
    Elicitation(String),
    /// The run's result.
    Response(String),
    /// Why the run failed.
    Error(String),
    /// A tool call.
    Action {
        /// A short verb, such as `Ran`.
        action: String,
        /// What it acted on, such as the command.
        parameter: String,
        /// How it failed, when it did.
        result: Option<String>,
    },
}

impl Content {
    /// The JSON Linear takes.
    pub(crate) fn json(&self) -> Value {
        match self {
            Self::Thought(body) => json!({ "type": "thought", "body": body }),
            Self::Elicitation(body) => json!({ "type": "elicitation", "body": body }),
            Self::Response(body) => json!({ "type": "response", "body": body }),
            Self::Error(body) => json!({ "type": "error", "body": body }),
            Self::Action {
                action,
                parameter,
                result: None,
            } => json!({ "type": "action", "action": action, "parameter": parameter }),
            Self::Action {
                action,
                parameter,
                result: Some(result),
            } => json!({
                "type": "action",
                "action": action,
                "parameter": parameter,
                "result": result,
            }),
        }
    }

    /// Whether this ends the run on Linear's side: a response, an error or
    /// a question.
    pub(crate) fn ends(&self) -> bool {
        matches!(
            self,
            Self::Response(_) | Self::Error(_) | Self::Elicitation(_)
        )
    }
}

/// One agent activity. Only thoughts and actions are ever ephemeral: the
/// constructors keep it so.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Activity {
    /// What it says.
    pub(crate) content: Content,
    /// Whether the next activity replaces it.
    pub(crate) ephemeral: bool,
}

impl Activity {
    /// A note that stays.
    pub(crate) fn note(body: String) -> Self {
        Self {
            content: Content::Thought(body),
            ephemeral: false,
        }
    }

    /// A thought the next activity replaces.
    pub(crate) fn passing(body: String) -> Self {
        Self {
            content: Content::Thought(body),
            ephemeral: true,
        }
    }

    /// A tool call, which the next activity replaces.
    pub(crate) fn action(action: String, parameter: String, result: Option<String>) -> Self {
        Self {
            content: Content::Action {
                action,
                parameter,
                result,
            },
            ephemeral: true,
        }
    }

    /// The run's end: a response, an error or a question, which stays.
    pub(crate) fn end(content: Content) -> Self {
        Self {
            content,
            ephemeral: false,
        }
    }
}

/// Redacts what is sent: every secret rig-core's redaction finds, and each
/// of `secrets`, such as the Linear token itself.
#[derive(Clone, Debug, Default)]
pub(crate) struct Redactor {
    secrets: Vec<String>,
}

impl Redactor {
    /// Redacts these values besides what rig-core finds; empty ones are
    /// left out.
    pub(crate) fn new(secrets: impl IntoIterator<Item = String>) -> Self {
        Self {
            secrets: secrets
                .into_iter()
                .filter(|secret| !secret.is_empty())
                .collect(),
        }
    }

    /// Redacts `secret` too.
    pub(crate) fn add(&mut self, secret: String) {
        if !secret.is_empty() {
            self.secrets.push(secret);
        }
    }

    /// `text` redacted.
    pub(crate) fn text(&self, text: &str) -> String {
        let mut text = rig_core::redact::text(text).into_owned();
        for secret in &self.secrets {
            if text.contains(secret.as_str()) {
                text = text.replace(secret.as_str(), REDACTED);
            }
        }
        text
    }

    /// Every string in `value` redacted.
    pub(crate) fn json(&self, value: &mut Value) {
        match value {
            Value::String(text) => *text = self.text(text),
            Value::Array(items) => items.iter_mut().for_each(|item| self.json(item)),
            Value::Object(fields) => fields.values_mut().for_each(|field| self.json(field)),
            _ => {}
        }
    }

    /// Every string of `activity` redacted.
    pub(crate) fn activity(&self, activity: Activity) -> Activity {
        let content = match activity.content {
            Content::Thought(body) => Content::Thought(self.text(&body)),
            Content::Elicitation(body) => Content::Elicitation(self.text(&body)),
            Content::Response(body) => Content::Response(self.text(&body)),
            Content::Error(body) => Content::Error(self.text(&body)),
            Content::Action {
                action,
                parameter,
                result,
            } => Content::Action {
                action: self.text(&action),
                parameter: self.text(&parameter),
                result: result.map(|result| self.text(&result)),
            },
        };
        Activity {
            content,
            ..activity
        }
    }
}

/// `text` cut to `most` characters, ending with `…` when cut.
pub(crate) fn head(text: &str, most: usize) -> String {
    let text = text.trim();
    match text.char_indices().nth(most) {
        Some((at, _)) => format!("{}…", text.get(..at).unwrap_or_default().trim_end()),
        None => text.to_owned(),
    }
}

/// The last `most` characters of `text`, starting with `…` when cut.
pub(crate) fn tail(text: &str, most: usize) -> String {
    let text = text.trim();
    let count = text.chars().count();
    if count <= most {
        return text.to_owned();
    }
    let skip = count - most;
    match text.char_indices().nth(skip) {
        Some((at, _)) => format!("…{}", text.get(at..).unwrap_or_default().trim_start()),
        None => String::new(),
    }
}

/// What a turn's model reply says: its text, and its reasoning when the
/// provider returned any.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reply {
    /// The reply's text, trimmed; empty when it has none.
    pub(crate) text: String,
    /// The reasoning shown, trimmed; empty when none.
    pub(crate) reasoning: String,
    /// Whether it asked for tool calls, so the turn goes on.
    pub(crate) calls_tools: bool,
}

impl Reply {
    /// What `response` says.
    pub(crate) fn of(response: &CompletionResponse) -> Self {
        let mut reply = Self::default();
        let mut reasoning = Vec::new();
        let mut text = Vec::new();
        for content in &response.choice {
            match content {
                AssistantContent::Text(part) => text.push(part.text.as_str()),
                AssistantContent::Reasoning(part) if !part.redacted => {
                    reasoning.push(part.text.as_str());
                }
                AssistantContent::ToolCall(_) => reply.calls_tools = true,
                _ => {}
            }
        }
        reply.text = text.concat().trim().to_owned();
        reply.reasoning = reasoning.join("\n\n").trim().to_owned();
        reply
    }

    /// The activities it becomes: the reasoning as a passing thought, then
    /// the text as a note.
    pub(crate) fn activities(&self) -> Vec<Activity> {
        let mut activities = Vec::new();
        if !self.reasoning.is_empty() {
            activities.push(Activity::passing(head(&self.reasoning, NOTE_MOST)));
        }
        if !self.text.is_empty() {
            activities.push(Activity::note(head(&self.text, NOTE_MOST)));
        }
        activities
    }
}

/// A settled tool call as its record has it.
pub(crate) struct Call<'a> {
    /// The tool's name.
    pub(crate) name: &'a str,
    /// The raw JSON arguments.
    pub(crate) args: &'a str,
    /// Its answer.
    pub(crate) outcome: &'a Result<Answered, ErrorReport>,
    /// What it published, such as its exit status.
    pub(crate) published: Option<&'a ToolResultContext>,
}

impl Call<'_> {
    /// The arguments as JSON, or `null` when they do not read.
    fn args(&self) -> Value {
        serde_json::from_str(self.args).unwrap_or(Value::Null)
    }

    /// The string argument `field`, if the call has it.
    pub(crate) fn arg(&self, field: &str) -> Option<String> {
        self.args()
            .get(field)
            .and_then(Value::as_str)
            .map(str::to_owned)
    }

    /// The output it answered with, as text: its error's or refusal's
    /// message when it has no output.
    pub(crate) fn output(&self) -> String {
        match self.outcome {
            Ok(Answered::ToolResult { result }) => match result.output().as_text() {
                Some(text) if !text.is_empty() => text.to_owned(),
                _ => (result.error().or(result.refusal()))
                    .map(|error| error.message().to_owned())
                    .unwrap_or_default(),
            },
            Ok(_) => String::new(),
            Err(report) => report.message.clone(),
        }
    }

    /// How it failed, if it did: its exit status or error kind, then the
    /// end of its output.
    fn failure(&self) -> Option<String> {
        let status = self
            .published
            .and_then(|published| published.get::<ExitStatus>().ok().flatten());
        let why = match (self.outcome, status) {
            (Err(_), _) => "failed".to_owned(),
            (Ok(Answered::ToolResult { result }), _) if result.is_refused() => "refused".to_owned(),
            (Ok(Answered::ToolResult { result }), _) if result.is_error() => result
                .error()
                .map_or("failed", |error| error.kind().as_str())
                .to_owned(),
            (_, Some(ExitStatus::Exited(0)) | None) => return None,
            (_, Some(ExitStatus::Exited(code))) => format!("exit code {code}"),
            (_, Some(ExitStatus::Signaled)) => "killed by a signal".to_owned(),
            (_, Some(ExitStatus::TimedOut)) => "timed out".to_owned(),
            (_, Some(ExitStatus::Killed)) => "stopped".to_owned(),
        };
        let output = tail(&self.output(), ACTION_MOST);
        Some(match output.is_empty() {
            true => why,
            false => format!("{why}\n{output}"),
        })
    }

    /// The action it becomes: a verb by tool, what it acted on, and how it
    /// failed, if it did.
    pub(crate) fn activity(&self) -> Activity {
        let path = || self.arg("path").unwrap_or_default();
        let (action, parameter) = match self.name {
            "shell" => ("Ran".to_owned(), self.arg("command").unwrap_or_default()),
            "read" => ("Read".to_owned(), path()),
            "write" | "edit" => ("Edited".to_owned(), path()),
            "search" => {
                let pattern = self.arg("pattern").unwrap_or_default();
                let parameter = match self.arg("path") {
                    Some(path) => format!("{pattern} in {path}"),
                    None => pattern,
                };
                ("Searched".to_owned(), parameter)
            }
            "task" => ("Delegated".to_owned(), self.compact()),
            name => match name.split_once("__") {
                Some((_, tool)) => (format!("Called {tool}"), self.compact()),
                None => (name.to_owned(), self.compact()),
            },
        };
        Activity::action(action, head(&parameter, ACTION_MOST), self.failure())
    }

    /// The arguments on one line.
    fn compact(&self) -> String {
        match self.args() {
            Value::Null => self.args.trim().to_owned(),
            Value::Object(fields) if fields.is_empty() => String::new(),
            args => args.to_string(),
        }
    }
}

/// The pull request `output` names, when `command` created one with `gh pr
/// create`: the last `https://github.com/<owner>/<repo>/pull/<n>` in it.
pub(crate) fn pull_request(command: &str, output: &str) -> Option<SessionLink> {
    if !command.contains("gh pr create") {
        return None;
    }
    const PREFIX: &str = "https://github.com/";
    output
        .match_indices(PREFIX)
        .filter_map(|(at, _)| {
            let rest = output.get(at + PREFIX.len()..)?;
            let mut parts = rest.splitn(4, '/');
            let owner = parts.next().filter(|part| is_name(part))?;
            let repo = parts.next().filter(|part| is_name(part))?;
            (parts.next()? == "pull").then_some(())?;
            let number: String = parts
                .next()?
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            (!number.is_empty()).then_some(())?;
            let url = format!("{PREFIX}{owner}/{repo}/pull/{number}");
            Some(SessionLink::new(format!("Pull request #{number}"), url))
        })
        .last()
}

/// Whether `part` reads as a GitHub owner or repository name.
fn is_name(part: &str) -> bool {
    !part.is_empty()
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}
