//! Plugin corpus (held-out): H30: Cut looping and role-play replies
//! mid-stream (spec `H30-degenerate-stream-cutter.md`).
//!
//! Small local models sometimes repeat a line forever or start writing the
//! user's next message. This plugin puts an [`Adapter`] on every model
//! call; for a watched model it rewrites the reply stream ([`map_stream`])
//! and watches its text. When a line (or a block of up to 400 characters)
//! comes a fourth time in a row, or a role marker such as `User:` starts a
//! line, the rewrite ends the reply there: it ends the text part with the
//! text cut back and passes on a response made of the parts that ended
//! before the cut. The kernel takes that as the reply and drops the
//! provider's stream, so nothing more is read. A tool call whose arguments
//! were still streaming is left out; finished ones run as usual. The view
//! draws the reply from the conversation once it is in, so what streamed
//! past the cut point is taken back. The cut is published on the call, and
//! an observer counts it per model (`/cuts`) and tells the user.
//!
//! Settings, `[stream_cutter]` in `RIG_HOME/settings.toml`:
//!
//! ```toml
//! [stream_cutter]
//! models = ["ollama/*"]   # the models watched; default all
//! markers = ["User:", "Human:", "<|im_start|>user", "### Instruction"]
//! repeats = 4             # a line or block this many times in a row
//! min_chars = 20          # a repeated line or block at least this long, trimmed
//! max_block = 400         # the longest block looked for
//! ```
//!
//! [`map_stream`]: rig_harness::rig_core::serve::ErasedHandler::map_stream

use std::collections::HashMap;
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, Usage};
use rig_harness::rig_core::message::{Origin, Text};
use rig_harness::rig_core::streaming::{Item, Part, Relayed, StreamEvent, delivered};
use rig_harness::rig_core::tool::{ContextValue, PublishedContext};
use rig_harness::rig_ecs::effects::Handler;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

const NAME: &str = "stream-cutter";

/// Which models are watched and what cuts a reply: the `[stream_cutter]`
/// settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct StreamCutter {
    /// Patterns of the models watched (`vendor/model`, `*` for any run of
    /// characters).
    pub models: Vec<String>,
    /// What, at the start of a line, means the model writes the next
    /// message itself.
    pub markers: Vec<String>,
    /// How many times in a row a line or block comes before it is cut.
    pub repeats: usize,
    /// The shortest repeated line or block, in characters, trimmed.
    pub min_chars: usize,
    /// The longest repeated block looked for, in characters.
    pub max_block: usize,
}

impl Default for StreamCutter {
    fn default() -> Self {
        Self {
            models: vec!["*".to_owned()],
            markers: ["User:", "Human:", "<|im_start|>user", "### Instruction"]
                .map(str::to_owned)
                .to_vec(),
            repeats: 4,
            min_chars: 20,
            max_block: 400,
        }
    }
}

impl StreamCutter {
    fn watches(&self, model: &str) -> bool {
        self.models.iter().any(|pattern| glob(pattern, model))
    }
}

/// How many replies were cut, per model, kept with the session.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
pub struct Cuts(pub HashMap<String, u32>);

/// The cutter's stage system, which adapts each model call, in
/// [`PrepareSystems::Adapt`], for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamCutterSystems;

#[derive(Default)]
pub struct StreamCutterPlugin;

impl Plugin for StreamCutterPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<StreamCutter>("stream_cutter")
            .init_resource::<Cuts>()
            .configure_sets(AgentPass, StreamCutterSystems.in_set(PrepareSystems::Adapt))
            .add_systems(AgentPass, adapt.in_set(StreamCutterSystems))
            .add_observer(announce)
            .add_command("cuts", "How many replies were cut, per model", cuts);
    }
}

/// What the cut tells a model call: why its reply was cut.
#[derive(Serialize, Deserialize)]
struct Cut(String);

impl ContextValue for Cut {
    const KEY: &'static str = NAME;
}

/// Puts the cutter on each new model call, for whichever model it goes to.
fn adapt(
    mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>,
    settings: Res<StreamCutter>,
) {
    if settings.models.is_empty() {
        return;
    }
    let rules = Arc::new(Rules::from(&*settings));
    for mut adapted in &mut calls {
        let Adapted(connection) = &mut *adapted;
        if !rules.settings.watches(&connection.spec.reference()) {
            continue;
        }
        let rules = rules.clone();
        let cut = move |published| {
            let mut cutter = Cutter::new(rules.clone(), published);
            move |relayed| cutter.relay(relayed)
        };
        connection.handler = Handler(connection.handler.0.clone().map_stream(NAME, cut));
    }
}

/// Counts each reply that was cut and tells the user, once its call
/// settled.
fn announce(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &SentTo, Option<&ForAgent>)>,
    mut cuts: ResMut<Cuts>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((published, sent, agent)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(Cut(why))) = published.0.get::<Cut>() else {
        return;
    };
    *cuts.0.entry(sent.0.spec.reference()).or_default() += 1;
    notices.write(Notice::info(agent.map(|agent| agent.0), why));
}

/// `/cuts`: the replies cut, per model.
fn cuts(In(args): In<CommandArgs>, cuts: Res<Cuts>, mut notices: MessageWriter<Notice>) {
    let mut counts: Vec<_> = cuts.0.iter().collect();
    counts.sort();
    let text = match counts.is_empty() {
        true => "No reply was cut.".to_owned(),
        false => counts
            .iter()
            .map(|(model, count)| format!("{model}: {count} cut"))
            .collect::<Vec<_>>()
            .join("\n"),
    };
    notices.write(Notice::info(args.agent, text));
}

/// The settings as the watch reads them.
struct Rules {
    settings: StreamCutter,
    markers: Vec<Vec<char>>,
    repeats: usize,
}

impl From<&StreamCutter> for Rules {
    fn from(settings: &StreamCutter) -> Self {
        Self {
            markers: settings
                .markers
                .iter()
                .filter(|marker| !marker.is_empty())
                .map(|marker| marker.chars().collect())
                .collect(),
            repeats: settings.repeats.max(2),
            settings: settings.clone(),
        }
    }
}

/// One reply being watched.
struct Cutter {
    rules: Arc<Rules>,
    /// Where the call is told its reply was cut.
    published: Arc<PublishedContext>,
    /// Who the reply is from, which the cut response names.
    origin: Option<Origin>,
    /// The items passed on, which the cut response is folded from.
    items: Vec<Item<StreamEvent>>,
    /// The text part streaming now.
    text: Option<(Part, Watch)>,
    /// The reply was cut: nothing more goes on.
    cut: bool,
}

impl Cutter {
    fn new(rules: Arc<Rules>, published: Arc<PublishedContext>) -> Self {
        Self {
            rules,
            published,
            origin: None,
            items: Vec::new(),
            text: None,
            cut: false,
        }
    }

    /// What `relayed` becomes: itself, or the end of the reply where it
    /// is cut.
    fn relay(&mut self, relayed: Relayed) -> Vec<Relayed> {
        if self.cut {
            return Vec::new();
        }
        match &relayed {
            Relayed::Origin(origin) => self.origin = Some(origin.clone()),
            Relayed::Item(item) => {
                if let Item::Event(StreamEvent::Text { part, text }) = item {
                    if self.text.as_ref().is_none_or(|(open, _)| open != part) {
                        self.text = Some((*part, Watch::new(&self.rules)));
                    }
                    if let Some((_, watch)) = &mut self.text
                        && let Some(found) = watch.push(text, &self.rules)
                        && let Some(origin) = self.origin.clone()
                    {
                        return self.cut_at(*part, found, origin);
                    }
                }
                self.items.push(item.clone());
            }
            Relayed::Done(_) => {}
        }
        vec![relayed]
    }

    /// Ends the reply in the text `part` at `found`: what of the fragment
    /// comes before the cut, the part's end with the text kept, and the
    /// response of what was delivered, which leaves out a tool call still
    /// streaming.
    fn cut_at(&mut self, part: Part, found: Found, origin: Origin) -> Vec<Relayed> {
        self.cut = true;
        let mut items = Vec::new();
        if !found.rest.is_empty() {
            let text = found.rest;
            items.push(Item::Event(StreamEvent::Text { part, text }));
        }
        let content = AssistantContent::Text(Text::new(found.kept));
        items.push(Item::Event(StreamEvent::End { part, content }));
        self.items.extend(items.iter().cloned());
        let choice = delivered(&self.items)
            .into_iter()
            .filter(|block| !matches!(block, AssistantContent::Text(text) if text.text.is_empty()))
            .collect();
        let response =
            CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
        // Encoding a string cannot fail.
        self.published.insert(Cut(found.why)).ok();
        let mut relayed: Vec<Relayed> = items.into_iter().map(Relayed::Item).collect();
        relayed.push(Relayed::Done(Arc::new(response)));
        relayed
    }
}

/// Where a text part is cut.
struct Found {
    /// The part's text kept.
    kept: String,
    /// What of the last fragment comes before the cut and was not passed on.
    rest: String,
    /// The notice.
    why: String,
}

/// The text of one part so far, watched for a repeat or a marker.
struct Watch {
    chars: Vec<char>,
    /// For each block length from `min_chars` up, how many characters in a
    /// row equal the one that length earlier.
    runs: Vec<usize>,
    /// Where the line being written starts.
    line_start: usize,
    /// The last line long enough to count, trimmed; how many times in a
    /// row it came; and where its first time ends.
    line: (String, usize, usize),
}

impl Watch {
    fn new(rules: &Rules) -> Self {
        let settings = &rules.settings;
        let lengths = settings
            .max_block
            .saturating_add(1)
            .saturating_sub(settings.min_chars.max(1));
        Self {
            chars: Vec::new(),
            runs: vec![0; lengths],
            line_start: 0,
            line: (String::new(), 0, 0),
        }
    }

    /// Adds `fragment`; where to cut, if it completes a repeat or a marker.
    fn push(&mut self, fragment: &str, rules: &Rules) -> Option<Found> {
        let before = self.chars.len();
        for char in fragment.chars() {
            self.chars.push(char);
            let found = self.block(rules).or_else(|| self.line(char, rules));
            let found = found.or_else(|| self.marker(rules));
            if let Some((cut, why)) = found {
                let kept = self.text(..cut);
                let kept = match why {
                    Why::Marker(_) => kept.trim_end().to_owned(),
                    _ => kept,
                };
                let rest: String = kept.chars().skip(before).collect();
                let why = why.describe(rules.repeats);
                return Some(Found { kept, rest, why });
            }
        }
        None
    }

    fn text(&self, range: impl std::slice::SliceIndex<[char], Output = [char]>) -> String {
        self.chars.get(range).unwrap_or_default().iter().collect()
    }

    /// A block of `min_chars..=max_block` characters that just came the
    /// `repeats`th time in a row: where its first time ends.
    fn block(&mut self, rules: &Rules) -> Option<(usize, Why)> {
        let min = rules.settings.min_chars.max(1);
        let end = self.chars.len();
        let last = end.checked_sub(1)?;
        let now = self.chars.get(last).copied();
        let mut found = None;
        for (offset, run) in self.runs.iter_mut().enumerate() {
            let length = min + offset;
            let earlier = last
                .checked_sub(length)
                .and_then(|at| self.chars.get(at).copied());
            *run = if earlier.is_some() && earlier == now {
                *run + 1
            } else {
                0
            };
            if found.is_none() && *run >= (rules.repeats - 1) * length {
                found = Some((length, end - *run));
            }
        }
        let (length, start) = found?;
        // The text from `start - length` on repeats with that period, so any
        // `length` characters of it are one time of the block: the first
        // that ends a line, else the first.
        let cut = (start..start + length)
            .find(|&at| at > 0 && self.chars.get(at - 1) == Some(&'\n'))
            .unwrap_or(start);
        let block = self.text(cut - length..cut);
        if block.trim().chars().count() < min {
            return None;
        }
        let one_line = block.ends_with('\n') && block.matches('\n').count() == 1;
        Some((cut, if one_line { Why::Line } else { Why::Block }))
    }

    /// A line of at least `min_chars` that just came the `repeats`th time
    /// in a row, blank lines between left aside: where its first time ends.
    fn line(&mut self, now: char, rules: &Rules) -> Option<(usize, Why)> {
        if now != '\n' {
            return None;
        }
        let end = self.chars.len();
        let text = self.text(self.line_start..end);
        self.line_start = end;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        if trimmed.chars().count() < rules.settings.min_chars {
            self.line = (String::new(), 0, 0);
            return None;
        }
        if self.line.0 == trimmed {
            self.line.1 += 1;
        } else {
            self.line = (trimmed.to_owned(), 1, end);
        }
        (self.line.1 >= rules.repeats).then_some((self.line.2, Why::Line))
    }

    /// A marker the line being written just became: where the line starts.
    fn marker(&self, rules: &Rules) -> Option<(usize, Why)> {
        let line = self.chars.get(self.line_start..)?;
        let marker = rules
            .markers
            .iter()
            .find(|marker| marker.as_slice() == line)?;
        Some((self.line_start, Why::Marker(marker.iter().collect())))
    }
}

/// Why a reply was cut.
enum Why {
    Line,
    Block,
    Marker(String),
}

impl Why {
    fn describe(&self, repeats: usize) -> String {
        match self {
            Self::Line => format!("cut a looping reply: line repeated {repeats}×"),
            Self::Block => format!("cut a looping reply: block repeated {repeats}×"),
            Self::Marker(marker) => format!("cut a reply that wrote the next message: `{marker}`"),
        }
    }
}

/// Whether `text` matches `pattern`, where `*` stands for any run of
/// characters.
fn glob(pattern: &str, text: &str) -> bool {
    let mut pieces = pattern.split('*');
    let first = pieces.next().unwrap_or_default();
    let Some(mut rest) = text.strip_prefix(first) else {
        return false;
    };
    let pieces: Vec<&str> = pieces.collect();
    let Some((last, middle)) = pieces.split_last() else {
        return rest.is_empty();
    };
    for piece in middle {
        let Some((_, after)) = rest.split_once(piece) else {
            return false;
        };
        rest = after;
    }
    rest.ends_with(last)
}
