//! Each helper this session started, as an entity following that child's
//! journal. What a helper is doing is folded from its events alone, so a
//! worker that restarts (an idle unload, a crash) rebuilds it by reading the
//! journal again.
//!
//! "A message is a message and the final answer is the report" holds by
//! construction here: a wait settles only on a `turn.ended` whose `through`
//! covers the last message the child was sent, or on an open ask, or on the
//! child's end. The child's progress `message`s are kept as its last words
//! and never settle a wait.

use std::collections::BTreeMap;

use rig_harness::prelude::*;
use rig_protocol::event::{AskOpened, Event, TurnEnded, TurnStatus};
use rig_protocol::methods::{Send, SessionInfo, Start};
use rig_protocol::{CommandSeq, EventRecord, Method, Seq, SessionId};

use crate::link::Feed;

/// A helper (child session) this session started, following its journal.
#[derive(Component)]
pub struct Helper {
    pub id: SessionId,
    pub name: String,
    pub journal: HelperJournal,
}

/// The events of a helper's journal not read yet.
#[derive(Component)]
pub struct HelperFeed(pub Box<dyn Feed>);

/// What a helper's journal says, folded event by event.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HelperJournal {
    pub last_seq: Seq,
    /// The last message given to it (its task or a later `send`).
    pub asked: CommandSeq,
    /// Its last turn's end.
    pub ended: Option<TurnEnded>,
    /// Its asks not yet resolved, oldest first.
    pub asks: Vec<AskOpened>,
    /// The last text it said, for progress.
    pub said: Option<String>,
    pub stopped: bool,
    pub faulted: Option<String>,
    /// Whether its turn is running (a turn started after the last end).
    pub working: bool,
}

impl HelperJournal {
    /// Folds one event in.
    pub fn apply(&mut self, record: &EventRecord) {
        if record.seq <= self.last_seq {
            return;
        }
        self.last_seq = record.seq;
        let Some(event) = Event::decode(record) else {
            return;
        };
        match event {
            Event::CommandAccepted(accepted)
                if accepted.method == Send::NAME || accepted.method == Start::NAME =>
            {
                self.asked = self.asked.max(accepted.command_seq);
                self.working = true;
            }
            Event::TurnStarted(_) => self.working = true,
            Event::Message(message) if message.role == "assistant" => {
                self.said = Some(message.text);
            }
            Event::AskOpened(ask) => self.asks.push(ask),
            Event::AskResolved(resolved) => self.asks.retain(|a| a.ask_id != resolved.ask_id),
            Event::TurnEnded(ended) => {
                if ended.through >= self.asked {
                    self.working = false;
                }
                self.ended = Some(ended);
            }
            Event::SessionStopped(_) => {
                self.stopped = true;
                self.working = false;
                self.asks.clear();
            }
            Event::SessionFaulted(faulted) => {
                self.faulted = Some(faulted.reason);
                self.working = false;
            }
            _ => {}
        }
    }

    /// Where the helper is, for a wait that has already been given its
    /// reports through `delivered`.
    pub fn status(&self, delivered: CommandSeq) -> Status {
        if let Some(ask) = self.asks.first() {
            return Status::Asks(ask.clone());
        }
        if let Some(reason) = &self.faulted {
            return Status::Faulted(reason.clone());
        }
        match &self.ended {
            Some(ended) if ended.through >= self.asked && self.asked > delivered => {
                Status::Report(ended.clone())
            }
            _ if self.stopped => Status::Stopped,
            Some(ended) if ended.through >= self.asked => Status::Reported,
            None if self.asked == CommandSeq(0) && !self.working => Status::Reported,
            _ => Status::Working,
        }
    }
}

/// Where a helper is.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    /// Its report on the last message it was given, not yet delivered.
    Report(TurnEnded),
    /// It waits for an approval or an answer.
    Asks(AskOpened),
    /// It stopped.
    Stopped,
    /// Its worker failed for good.
    Faulted(String),
    /// Its last report was delivered; it is idle until messaged.
    Reported,
    /// It is working on the last message it was given.
    Working,
}

impl Status {
    /// Whether a wait ends on it.
    pub fn settles(&self) -> bool {
        !matches!(self, Status::Working | Status::Reported)
    }
}

/// On the main agent: the last command of each helper whose report the
/// model has been given, so a restart never gives it twice.
#[derive(Component, Reflect, Default, Debug, Clone)]
#[reflect(Component, Saved)]
pub struct DeliveredReports(pub BTreeMap<String, u64>);

impl DeliveredReports {
    pub fn of(&self, id: &SessionId) -> CommandSeq {
        CommandSeq(self.0.get(&id.0).copied().unwrap_or(0))
    }
}

/// Reads each helper's new events into its journal.
pub fn read_feeds(mut helpers: Query<(&mut Helper, &mut HelperFeed)>) {
    for (mut helper, mut feed) in &mut helpers {
        let mut seen = 0;
        while seen < 1024 {
            match feed.0.try_next() {
                Some(Ok(record)) => {
                    helper.journal.apply(&record);
                    seen += 1;
                }
                Some(Err(_)) | None => break,
            }
        }
    }
}

/// How one helper reads in a list or a wait's answer.
pub fn line(helper: &Helper, delivered: CommandSeq) -> String {
    let what = match helper.journal.status(delivered) {
        Status::Report(ended) => match ended.status {
            TurnStatus::Answered => "has answered (its report is waiting)".to_owned(),
            TurnStatus::Failed => "failed (its report is waiting)".to_owned(),
            TurnStatus::Stopped => "was stopped mid-turn".to_owned(),
        },
        Status::Asks(ask) => format!("waits on its ask `{}`: {}", ask.ask_id.0, ask.title),
        Status::Stopped => "stopped".to_owned(),
        Status::Faulted(reason) => format!("failed: {reason}"),
        Status::Reported => "idle, report delivered".to_owned(),
        Status::Working => match &helper.journal.said {
            Some(said) => format!("working; last said: {}", clip(said, 160)),
            None => "working".to_owned(),
        },
    };
    format!("`{}` ({}) {what}", helper.name, helper.id.0)
}

/// `text` cut to `max` characters, on one line.
pub fn clip(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", flat.get(..at).unwrap_or_default()),
        None => flat,
    }
}

/// A tree of sessions from `rig/session/list {tree}` as indented lines,
/// starting at `root`'s children.
pub fn tree_lines(sessions: &[SessionInfo], root: &SessionId) -> Vec<String> {
    fn walk(sessions: &[SessionInfo], at: &SessionId, depth: usize, out: &mut Vec<String>) {
        if depth > 8 {
            return;
        }
        for child in sessions.iter().filter(|s| s.parent.as_ref() == Some(at)) {
            let name = child
                .name
                .clone()
                .unwrap_or_else(|| child.session_id.0.clone());
            out.push(format!(
                "{}`{name}` ({}) {:?}, {} tokens",
                "  ".repeat(depth),
                child.session_id.0,
                child.state,
                child.usage.tokens
            ));
            walk(sessions, &child.session_id, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    walk(sessions, root, 0, &mut out);
    out
}

/// The status line's `helpers` item, on each agent: how many helpers work,
/// wait on an answer, and are done. Shown only while there are helpers,
/// and changed only when its text does.
pub fn show_status(
    helpers: Query<&Helper>,
    mut agents: Query<(&mut rig_harness::status::StatusItems, &DeliveredReports), With<Agent>>,
) {
    use rig_harness::status::{Side, StatusItem, Tone};
    const ITEM: StatusItem = StatusItem::new("helpers", Side::Left, 70, 20);
    for (mut items, delivered) in &mut agents {
        let (mut working, mut asking, mut done) = (0, 0, 0);
        for helper in &helpers {
            match helper.journal.status(delivered.of(&helper.id)) {
                Status::Working => working += 1,
                Status::Asks(_) => asking += 1,
                _ => done += 1,
            }
        }
        let text = match (working, asking, done) {
            (0, 0, 0) => None,
            _ => {
                let mut parts = Vec::new();
                if working > 0 {
                    parts.push(format!("{working} working"));
                }
                if asking > 0 {
                    parts.push(format!("{asking} asking"));
                }
                if done > 0 {
                    parts.push(format!("{done} done"));
                }
                Some(format!("helpers: {}", parts.join(", ")))
            }
        };
        let shown = items
            .iter()
            .find(|i| i.name == ITEM.name)
            .map(|i| i.text.clone());
        match text {
            Some(text) if shown.as_deref() != Some(text.as_str()) => {
                let tone = if asking > 0 { Tone::Yellow } else { Tone::Dim };
                items.show(ITEM.says(text, tone));
            }
            None if shown.is_some() => items.hide("helpers"),
            _ => {}
        }
    }
}
