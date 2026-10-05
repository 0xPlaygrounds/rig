//! A live view of one completion's stream. Each tool call is addressed by
//! its part: the start names the tool, every argument fragment prints with
//! the best-effort object the text so far states, and the end brings the
//! call's id and final arguments. The summary checks that each call's
//! joined fragments are the arguments its end states.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rig::message::AssistantContent;
use rig::streaming::{PartKind, StreamEvent, parse_partial_arguments};
use serde_json::Value;

/// One tool call as it streamed.
struct CallView {
    tool: String,
    fragments: usize,
    arguments: String,
    /// When each fragment arrived.
    arrivals: Vec<Duration>,
    ended: Option<(String, Value)>,
}

impl CallView {
    /// How evenly the fragments arrived: the 50 ms windows they landed in,
    /// and the longest wait between two of them. A provider that holds an
    /// argument back sends it in a few windows after a long wait.
    fn pacing(&self) -> (usize, Duration) {
        let windows: std::collections::BTreeSet<u128> =
            self.arrivals.iter().map(|at| at.as_millis() / 50).collect();
        let longest = self
            .arrivals
            .windows(2)
            .filter_map(|pair| match pair {
                [earlier, later] => Some(later.saturating_sub(*earlier)),
                _ => None,
            })
            .max()
            .unwrap_or_default();
        (windows.len(), longest)
    }
}

/// The tool calls of one completion, by part.
pub struct StreamView {
    started: Instant,
    calls: BTreeMap<usize, CallView>,
}

impl StreamView {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            calls: BTreeMap::new(),
        }
    }

    fn stamp(&self) -> String {
        format!("[{:>6}ms]", self.started.elapsed().as_millis())
    }

    /// Print `event` and fold it into the view.
    pub fn event(&mut self, event: &StreamEvent) {
        let stamp = self.stamp();
        let at = self.started.elapsed();
        match event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name,
            } => {
                let tool = name.as_ref().map_or("?", |name| name.as_str()).to_owned();
                println!("{stamp} part {} start  tool call `{tool}`", part.index());
                self.calls.insert(
                    part.index(),
                    CallView {
                        tool,
                        fragments: 0,
                        arguments: String::new(),
                        arrivals: Vec::new(),
                        ended: None,
                    },
                );
            }
            StreamEvent::Start { part, kind, .. } => {
                println!("{stamp} part {} start  {kind:?}", part.index());
            }
            StreamEvent::Arguments { part, json } => {
                let Some(call) = self.calls.get_mut(&part.index()) else {
                    return;
                };
                call.fragments += 1;
                call.arguments.push_str(json);
                call.arrivals.push(at);
                let partial = Value::Object(parse_partial_arguments(&call.arguments));
                println!(
                    "{stamp} part {} args   +{:>4}B {:?}  now {}",
                    part.index(),
                    json.len(),
                    clip(json, 24),
                    preview(&partial),
                );
            }
            StreamEvent::Text { part, text } => {
                println!("{stamp} part {} text   {:?}", part.index(), clip(text, 60));
            }
            StreamEvent::Reasoning { part, text } => {
                println!("{stamp} part {} think  {:?}", part.index(), clip(text, 60));
            }
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(tool_call),
            } => {
                let id = tool_call.id.to_string();
                println!(
                    "{stamp} part {} end    `{}` id {id}",
                    part.index(),
                    tool_call.function.name
                );
                if let Some(call) = self.calls.get_mut(&part.index()) {
                    call.ended = Some((id, tool_call.function.arguments_value()));
                }
            }
            StreamEvent::End { part, .. } => {
                println!("{stamp} part {} end", part.index());
            }
        }
    }

    /// Print one row per tool call and whether its joined fragments are
    /// the arguments its end states. Returns whether every ended call's
    /// fragments matched.
    pub fn summary(&self) -> bool {
        if self.calls.is_empty() {
            return true;
        }
        println!(
            "\n{:<5} {:<12} {:>9} {:>14} {:>8} {:>9} {:<32} matches end",
            "part", "tool", "fragments", "first..last", "windows", "max gap", "call id"
        );
        let mut all_match = true;
        for (part, call) in &self.calls {
            let window = match (call.arrivals.first(), call.arrivals.last()) {
                (Some(first), Some(last)) => {
                    format!("{}..{}ms", first.as_millis(), last.as_millis())
                }
                _ => "-".to_owned(),
            };
            let (windows, longest) = call.pacing();
            let (id, matches) = match &call.ended {
                Some((id, arguments)) => {
                    let joined = parse_partial_arguments(&call.arguments);
                    // The end's arguments are the joined text, read whole.
                    let matches = serde_json::from_str::<Value>(&call.arguments).ok().as_ref()
                        == Some(arguments)
                        && Value::Object(joined) == *arguments;
                    all_match &= matches;
                    (id.as_str(), if matches { "yes" } else { "NO" })
                }
                None => ("(never ended)", "-"),
            };
            println!(
                "{part:<5} {:<12} {:>9} {window:>14} {windows:>8} {:>7}ms {:<32} {matches}",
                call.tool,
                call.fragments,
                longest.as_millis(),
                clip(id, 32)
            );
        }
        all_match
    }
}

/// `text` cut to `max` characters.
fn clip(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text.to_owned(),
    }
}

/// `value` with long strings shortened to their head and length.
fn preview(value: &Value) -> String {
    fn shorten(value: &Value) -> Value {
        match value {
            Value::String(text) if text.chars().count() > 24 => Value::String(format!(
                "{} ({} chars)",
                clip(text, 16),
                text.chars().count()
            )),
            Value::Array(items) => Value::Array(items.iter().map(shorten).collect()),
            Value::Object(object) => Value::Object(
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), shorten(value)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    shorten(value).to_string()
}
