//! Least privilege: what a session may do, and who answers its asks.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{RpcError, codes};

/// What a session may use. A child's grant is its parent's narrowed by
/// [`Grant::narrow`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Grant {
    /// The tools it may call; `None` is every tool the parent has.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// The directories it may touch; `None` is unrestricted (owner sessions only).
    #[serde(default)]
    pub dirs: Option<Vec<DirGrant>>,
    /// Shell network access.
    #[serde(default)]
    pub net: bool,
    /// The most children it may run at once; `None` inherits.
    #[serde(default)]
    pub children: Option<u32>,
    /// Up-front answers to its asks.
    #[serde(default)]
    pub approve: Vec<Rule>,
    /// The calls this session must ask about before they run, unless an
    /// up-front rule (`approve`) answers them: "shell", "edits" (edit,
    /// write, delete, move) or a tool's name. Empty: nothing asks (the
    /// default).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ask: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirGrant {
    pub path: PathBuf,
    pub write: bool,
}

/// An up-front answer: allow a tool, a shell command prefix or edits under a
/// path, or deny a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Rule {
    Tool { name: String },
    ShellPrefix { prefix: String },
    EditsUnder { path: PathBuf },
    Deny { name: String },
}

/// What a requested grant asked for beyond its parent's.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Wider {
    #[error("tools the parent lacks: {}", .0.join(", "))]
    Tools(Vec<String>),
    #[error("{} is outside the parent's directories", .0.display())]
    Dir(PathBuf),
    #[error("{} is read-only for the parent", .0.display())]
    Write(PathBuf),
    #[error("network access the parent lacks")]
    Net,
    #[error("{requested} children where the parent may run {limit}")]
    Children { requested: u32, limit: u32 },
    #[error("an up-front rule the parent may not answer itself: {0:?}")]
    Rule(Rule),
}

impl From<Wider> for RpcError {
    fn from(wider: Wider) -> RpcError {
        RpcError::new(codes::GRANT_WIDER, wider.to_string())
    }
}

impl Grant {
    /// The child's effective grant: the intersection of this (the parent's)
    /// grant and `requested`, where a `None` in `requested` inherits the
    /// parent's. Err names the first thing `requested` wants beyond it.
    /// The child keeps every `Deny` rule of the parent.
    pub fn narrow(&self, requested: &Grant) -> Result<Grant, Wider> {
        let tools = match (&self.tools, &requested.tools) {
            (_, None) => self.tools.clone(),
            (None, Some(asked)) => Some(asked.clone()),
            (Some(has), Some(asked)) => {
                let extra: Vec<String> =
                    asked.iter().filter(|t| !has.contains(t)).cloned().collect();
                if !extra.is_empty() {
                    return Err(Wider::Tools(extra));
                }
                Some(asked.clone())
            }
        };
        let dirs = match (&self.dirs, &requested.dirs) {
            (_, None) => self.dirs.clone(),
            (None, Some(asked)) => Some(asked.clone()),
            (Some(has), Some(asked)) => {
                for dir in asked {
                    let Some(within) = has.iter().find(|h| dir.path.starts_with(&h.path)) else {
                        return Err(Wider::Dir(dir.path.clone()));
                    };
                    if dir.write && !has.iter().any(|h| h.write && dir.path.starts_with(&h.path)) {
                        return Err(Wider::Write(within.path.clone()));
                    }
                }
                Some(asked.clone())
            }
        };
        if requested.net && !self.net {
            return Err(Wider::Net);
        }
        let children = match (self.children, requested.children) {
            (limit, None) => limit,
            (None, Some(asked)) => Some(asked),
            (Some(limit), Some(asked)) if asked > limit => {
                return Err(Wider::Children {
                    requested: asked,
                    limit,
                });
            }
            (Some(_), Some(asked)) => Some(asked),
        };
        let mut approve: Vec<Rule> = self
            .approve
            .iter()
            .filter(|rule| matches!(rule, Rule::Deny { .. }))
            .cloned()
            .collect();
        for rule in &requested.approve {
            if !self.answers(rule) {
                return Err(Wider::Rule(rule.clone()));
            }
            if !approve.contains(rule) {
                approve.push(rule.clone());
            }
        }
        // A child asks at least what its parent asks: asking more is never wider.
        let mut ask = self.ask.clone();
        for call in &requested.ask {
            if !ask.contains(call) {
                ask.push(call.clone());
            }
        }
        Ok(Grant {
            tools,
            dirs,
            net: requested.net,
            children,
            approve,
            ask,
        })
    }

    /// Whether this grant's own rules already answer what `rule` answers.
    fn answers(&self, rule: &Rule) -> bool {
        match rule {
            Rule::Deny { .. } => true,
            Rule::Tool { name } => self
                .approve
                .iter()
                .any(|r| matches!(r, Rule::Tool { name: n } if n == name)),
            Rule::ShellPrefix { prefix } => self.approve.iter().any(
                |r| matches!(r, Rule::ShellPrefix { prefix: p } if prefix.starts_with(p.as_str())),
            ),
            Rule::EditsUnder { path } => self.approve.iter().any(
                |r| matches!(r, Rule::EditsUnder { path: p } if Path::new(path).starts_with(p)),
            ),
        }
    }
}

/// Who answers a session's asks, and what happens when nobody does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub answerers: Answerers,
    pub timeout_s: u64,
    pub on_timeout: TimeoutOutcome,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy {
            answerers: Answerers::AnyOwner,
            timeout_s: 600,
            on_timeout: TimeoutOutcome::Deny,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Answerers {
    AnyOwner,
    Driver,
    Parent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutOutcome {
    Deny,
}
