//! CT-07: conversation chapters.
//!
//! A session is divided into named chapters, each a run of turns. The
//! first turn starts one; after each turn a cheap model call of the
//! plugin's own (no turn's, so it never delays the next) says whether that
//! turn started a new topic and titles it; `/chapter <title>` starts one at
//! the current turn (or the next, when idle) and always wins.
//!
//! Chapters live in the agent's conversation as asides of kind [`KIND`],
//! each following the first message of its chapter, beside a transcript
//! [`Note`] that fronts draw. Asides never reach a model, and are saved and
//! restored with the conversation.
//!
//! `/fold <n>` has a model (`[chapters] summary_model`, else the agent's)
//! summarize chapter `n`; from then on a Prepare system replaces that
//! chapter's messages in every request's `history` (a view, never saved)
//! with the summary, which is the only place a chapter title reaches a
//! model. `/unfold <n>` brings the originals back. The stored conversation
//! never changes.

use std::ops::Range;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::conversation::{Aside, AsideId, Entry};
use rig_harness::rig_core::message::Message;
use rig_harness::rig_memory::{
    CompactReason, HeuristicTokenCounter, Summarizer, SummaryLimits, SummaryPrompts, SummaryState,
    TokenCounter, TrackArgument,
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The plugin's name: its author, its settings table and its calls'
/// purpose.
const NAME: &str = "chapters";

/// The kind of the asides holding a [`Chapter`].
pub const KIND: &str = "chapters.chapter";

/// How much of a request or an answer the boundary call reads.
const SNIPPET_CHARS: usize = 400;

/// How many earlier requests of the current chapter the boundary call reads.
const EARLIER: usize = 3;

/// The longest title kept.
const TITLE_CHARS: usize = 60;

/// What the boundary model is told.
const BOUNDARY_PREAMBLE: &str = "You split a long coding session into chapters by topic, \
    such as a bug fix, a refactor or a release. Given the latest exchange, and the requests \
    that came before it in the current chapter if any, say whether the latest exchange starts \
    a new topic, and title its topic in 2 to 5 words. Answer with JSON only: \
    {\"new_chapter\": true or false, \"title\": \"...\"}";

/// The tool arguments that name a file a chapter changed.
const CHANGED: &[TrackArgument] = &[
    TrackArgument::new("edit", "path", "files-changed"),
    TrackArgument::new("write", "path", "files-changed"),
];

/// What the summary model is asked when a chapter is folded.
const FOLD_SUMMARIZER: Summarizer = Summarizer {
    prompts: SummaryPrompts {
        system: std::borrow::Cow::Borrowed(
            "You summarize one finished chapter of a session between a user and a coding \
             agent, so the agent can go on without reading it. Output only the summary.",
        ),
        initial: std::borrow::Cow::Borrowed(
            "The conversation above is one finished chapter of the session. Summarize it.",
        ),
        update: std::borrow::Cow::Borrowed(
            "The conversation above is one finished chapter of the session. Summarize it.",
        ),
        format: std::borrow::Cow::Borrowed(
            "\n\nUse exactly this format, short, keeping exact paths and names:\n\n\
             ## What was done\n- ...\n\n## Files changed\n- [path, or \"(none)\"]\n\n\
             ## Decisions\n- **[Decision]**: [Why]",
        ),
        ..SummaryPrompts::DEFAULT
    },
    limits: SummaryLimits::DEFAULT,
};

/// Divides each top-level agent's conversation into chapters, with
/// `/chapters`, `/chapter`, `/fold` and `/unfold`.
#[derive(Default)]
pub struct ConversationChaptersPlugin;

impl Plugin for ConversationChaptersPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ChaptersConfig>(NAME)
            .configure_sets(AgentPass, ChapterSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, fold_in_requests.in_set(ChapterSystems))
            .add_command(
                "chapters",
                "The session's chapters; choosing one shows it",
                chapters_command,
            )
            .add_command(
                "chapter",
                "/chapter <title> starts a chapter at this turn; /chapter rename <n> <title>",
                chapter_command,
            )
            .add_command(
                "fold",
                "/fold <n>: the model reads a summary of chapter n in its place",
                fold_command,
            )
            .add_command(
                "unfold",
                "/unfold <n>: the model reads chapter n again",
                unfold_command,
            )
            .add_observer(on_turn_start)
            .add_observer(ask_for_boundary)
            .add_observer(take_boundary)
            .add_observer(take_fold)
            .add_observer(follow_successor::<Boundary>)
            .add_observer(follow_successor::<Folding>);
    }
}

/// The plugin's Prepare system, which replaces folded chapters in each
/// request's messages, in [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChapterSystems;

/// The `[chapters]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChaptersConfig {
    /// The catalog model (`vendor/model`) that suggests boundaries and
    /// titles; none uses the agent's own.
    pub boundary_model: Option<String>,
    /// The catalog model that writes fold summaries; none uses the
    /// agent's own.
    pub summary_model: Option<String>,
}

/// A chapter, as its aside holds it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chapter {
    /// Its title; empty until the boundary model titled it.
    pub title: String,
    /// Whether the user started it with `/chapter`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub manual: bool,
    /// Whether the user named it, so no model renames it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub renamed: bool,
    /// The summary the model reads in its place while it is folded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl Chapter {
    /// The chapter `aside` holds, when it is one of this plugin's.
    pub fn of(aside: &Aside) -> Option<Self> {
        (aside.kind == KIND && aside.author == author())
            .then(|| serde_json::from_value(aside.value.clone()).ok())
            .flatten()
    }

    /// Its title, or a stand-in while it has none.
    pub fn shown_title(&self) -> &str {
        if self.title.is_empty() {
            "(untitled)"
        } else {
            &self.title
        }
    }

    /// Whether the model reads its summary in its place.
    pub fn folded(&self) -> bool {
        self.summary.is_some()
    }
}

/// One chapter of an agent's conversation, as the outline shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChapterLine {
    /// Its number in the outline, from 1.
    pub number: usize,
    /// The first message of the chapter.
    pub start: MessageId,
    /// The aside holding it.
    pub aside: AsideId,
    /// The chapter.
    pub chapter: Chapter,
    /// Its messages in the conversation now.
    pub messages: Range<usize>,
    /// Its first and last turn, counted from 1; `None` before its first
    /// turn is recorded.
    pub turns: Option<(u64, u64)>,
    /// The estimated tokens it puts in the model's context: its summary's
    /// while folded, else its messages'.
    pub tokens: u64,
}

impl ChapterLine {
    /// Its turns, as `4–6`.
    pub fn turn_range(&self) -> String {
        match self.turns {
            Some((first, last)) if first == last => format!("turn {first}"),
            Some((first, last)) => format!("turns {first}–{last}"),
            None => "no turns yet".to_owned(),
        }
    }

    /// The message the model reads in place of the chapter while folded.
    fn fold_message(&self) -> Option<Message> {
        let summary = self.chapter.summary.as_ref()?;
        Some(Message::user(format!(
            "[Chapter {} \"{}\" ({}) is folded: this summary stands for its messages.]\n\n{}",
            self.number,
            self.chapter.shown_title(),
            self.turn_range(),
            summary.trim()
        )))
    }
}

/// The plugin's author, on its asides and its edits of requests.
pub fn author() -> Author {
    Author::plugin(NAME)
}

/// The chapters of `conversation`, in order: what `/chapters` shows.
pub fn outline(conversation: &Conversation) -> Vec<ChapterLine> {
    let mut found: Vec<(usize, AsideId, MessageId, Chapter)> = Vec::new();
    for (id, aside) in conversation.asides() {
        let (Some(start), Some(chapter)) = (aside.after, Chapter::of(aside)) else {
            continue;
        };
        if found.iter().all(|(_, _, known, _)| *known != start) {
            found.push((conversation.start_of(start), id, start, chapter));
        }
    }
    found.sort_by_key(|(at, id, ..)| (*at, *id));
    let len = conversation.messages().len();
    let ends: Vec<usize> = found
        .iter()
        .skip(1)
        .map(|(at, ..)| *at)
        .chain([len])
        .collect();
    let mut turns: Vec<Option<(u64, u64)>> = vec![None; found.len()];
    for (_, record) in conversation.turns() {
        let at = conversation.start_of(record.start);
        let Some(of) = found.iter().rposition(|(start, ..)| *start <= at) else {
            continue;
        };
        if let Some(range) = turns.get_mut(of) {
            let number = record.index.saturating_add(1);
            *range = Some(range.map_or((number, number), |(first, last)| {
                (first.min(number), last.max(number))
            }));
        }
    }
    let counter = HeuristicTokenCounter::default();
    let count = |messages: &[std::sync::Arc<Message>]| -> u64 {
        messages
            .iter()
            .map(|message| counter.count(message))
            .sum::<usize>() as u64
    };
    let mut lines = Vec::with_capacity(found.len());
    for (index, ((at, aside, start, chapter), end)) in found.into_iter().zip(ends).enumerate() {
        let messages = at..end.max(at);
        let mut line = ChapterLine {
            number: index.saturating_add(1),
            start,
            aside,
            chapter,
            turns: turns.get(index).copied().flatten(),
            tokens: 0,
            messages: messages.clone(),
        };
        line.tokens = match line.fold_message() {
            Some(message) => count(&[message.into()]),
            None => count(conversation.messages().get(messages).unwrap_or_default()),
        };
        lines.push(line);
    }
    lines
}

/// Puts `chapter` at `start` in `conversation`, in place of the one there,
/// with its transcript note.
fn put(conversation: &mut Conversation, start: MessageId, chapter: &Chapter) {
    let value = serde_json::to_value(chapter).unwrap_or_default();
    let aside = Aside::new(KIND, author(), value).after(Some(start));
    let mine = |aside: &Aside, chapter: bool| {
        aside.after == Some(start)
            && aside.author == author()
            && if chapter {
                aside.kind == KIND
            } else {
                Note::of(aside).is_some()
            }
    };
    let find = |conversation: &Conversation, chapter: bool| {
        let mut asides = conversation.asides();
        asides
            .find(|(_, aside)| mine(aside, chapter))
            .map(|(id, _)| id)
    };
    match find(conversation, true) {
        Some(id) => conversation.replace_aside(id, aside),
        None => {
            conversation.add_aside(aside);
            true
        }
    };
    let note = match &chapter.summary {
        Some(summary) => Note::new(
            format!("Chapter: {} (folded)", chapter.shown_title()),
            format!("The model reads this summary in place of the chapter:\n\n{summary}"),
        ),
        None => Note::new(
            format!("Chapter: {}", chapter.shown_title()),
            "A new chapter starts here.",
        ),
    }
    .aside(author(), Some(start));
    match find(conversation, false) {
        Some(id) => conversation.replace_aside(id, note),
        None => {
            conversation.add_aside(note);
            true
        }
    };
}

/// A title the model or the user gave, on one line and not too long.
fn clean_title(title: &str) -> String {
    let line = title.lines().map(str::trim).find(|line| !line.is_empty());
    let line = line
        .unwrap_or_default()
        .trim_matches(['"', '\'', '*', '#', '.', ' ']);
    line.chars().take(TITLE_CHARS).collect()
}

/// `text`, cut to [`SNIPPET_CHARS`].
fn snippet(text: &str) -> String {
    let mut cut: String = text.chars().take(SNIPPET_CHARS).collect();
    if cut.len() < text.len() {
        cut.push('…');
    }
    cut
}

/// The user's chapter for the agent's next turn, given by `/chapter` while
/// it was idle.
#[derive(Component, Clone, Debug)]
struct PendingChapter(String);

/// On a boundary call, and whichever call takes it over: the first message
/// of the turn it judges.
#[derive(Component, Clone, Debug)]
pub struct Boundary {
    /// The turn's first message.
    pub start: MessageId,
}

/// On a fold's summary call, and whichever call takes it over: the first
/// message of the chapter it summarizes, and the files its tool calls
/// changed.
#[derive(Component, Clone, Debug)]
pub struct Folding {
    /// The chapter's first message.
    pub start: MessageId,
    /// The files the chapter's tool calls changed.
    pub changed: Vec<String>,
}

/// On the outline picker `/chapters` opened.
#[derive(Component)]
struct OutlinePicker;

/// Starts the first chapter with the first turn, and the user's chapter
/// with the turn after `/chapter` while idle. Only turns that answer a
/// request count (not a compaction's).
fn on_turn_start(
    added: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, &Turn)>,
    mut agents: Query<(&mut Conversation, Option<&PendingChapter>), Without<SpawnedBy>>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), turn)) = turns.get(added.entity) else {
        return;
    };
    let Ok((mut conversation, pending)) = agents.get_mut(agent) else {
        return;
    };
    let start = turn.start;
    if conversation.request(start).is_none() {
        return;
    }
    if let Some(PendingChapter(title)) = pending {
        let chapter = Chapter {
            title: title.clone(),
            manual: true,
            renamed: true,
            summary: None,
        };
        put(&mut conversation, start, &chapter);
        commands.entity(agent).remove::<PendingChapter>();
    } else if outline(&conversation).is_empty() {
        put(&mut conversation, start, &Chapter::default());
    }
}

/// After each turn that answered a request, asks the boundary model
/// whether it began a new chapter, in a call of the plugin's own that no
/// turn waits for. What it says about a turn that begins a chapter the
/// user started is ignored ([`take_boundary`]).
fn ask_for_boundary(
    settled: On<Add<Settled>>,
    turns: Query<(&Turn, &ForAgent), With<TurnOutcome>>,
    agents: Query<&Conversation, (With<Agent>, Without<SpawnedBy>)>,
    (config, models): (Res<ChaptersConfig>, Res<Models>),
    mut commands: Commands,
) {
    let Ok((turn, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(conversation) = agents.get(agent) else {
        return;
    };
    let start = turn.start;
    let Some(request) = conversation.request(start) else {
        return;
    };
    let lines = outline(conversation);
    let at = conversation.start_of(start);
    let Some(current) = lines.iter().rev().find(|line| line.messages.start <= at) else {
        return;
    };
    let starts_here = current.start == start;
    let asked = request.message.rag_text().unwrap_or_default();
    let range = at..current.messages.end;
    let answer = conversation.messages().get(range).unwrap_or_default();
    let answer = answer
        .iter()
        .rev()
        .find_map(|message| Message::final_answer(message));
    let mut prompt = String::new();
    if !starts_here {
        let earlier: Vec<String> = conversation
            .turns()
            .map(|(_, record)| record.start)
            .filter(|id| {
                let begins = conversation.start_of(*id);
                begins >= current.messages.start && begins < at
            })
            .filter_map(|id| conversation.request(id)?.message.rag_text())
            .collect();
        let earlier = earlier.iter().rev().take(EARLIER).rev();
        prompt.push_str("Earlier requests in the current chapter:\n");
        for text in earlier {
            prompt.push_str(&format!("- {}\n", snippet(text)));
        }
        prompt.push('\n');
    } else {
        prompt.push_str("This is the first exchange of a chapter: set new_chapter to true.\n\n");
    }
    prompt.push_str(&format!(
        "The latest exchange:\nUser: {}\nAssistant: {}",
        snippet(&asked),
        snippet(&answer.unwrap_or_default())
    ));
    let mut call = ModelRequest::new(
        CompletionRequest::new(Message::user(prompt)).preamble(BOUNDARY_PREAMBLE),
    );
    if let Some(connection) = config
        .boundary_model
        .as_deref()
        .and_then(|reference| models.connect(reference).ok())
    {
        call = call.on(connection);
    }
    commands.spawn((
        Name::new("chapter boundary"),
        Boundary { start },
        call,
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
    ));
}

/// What the boundary model answers.
#[derive(Deserialize, Default)]
#[serde(default)]
struct Verdict {
    new_chapter: bool,
    title: String,
}

/// Reads a boundary call's verdict: titles the chapter its turn begins if
/// none titled it, or starts one there. A chapter the user started or
/// named always stays as it is.
fn take_boundary(
    settled: On<Add<Settled>>,
    calls: Query<(&Boundary, &ForAgent, &Outcome)>,
    mut agents: Query<&mut Conversation>,
) {
    let Ok((&Boundary { start }, &ForAgent(agent), outcome)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let text = outcome
        .completion()
        .map(|reply| reply.text())
        .unwrap_or_default();
    let json = match (text.find('{'), text.rfind('}')) {
        (Some(from), Some(to)) if from < to => text.get(from..=to).unwrap_or_default(),
        _ => "",
    };
    let verdict: Verdict = serde_json::from_str(json).unwrap_or_default();
    let title = clean_title(&verdict.title);
    let lines = outline(&conversation);
    let at = conversation.start_of(start);
    match lines.iter().rev().find(|line| line.messages.start <= at) {
        Some(line) if line.start == start => {
            if line.chapter.title.is_empty() && !line.chapter.renamed && !title.is_empty() {
                let chapter = Chapter {
                    title,
                    ..line.chapter.clone()
                };
                put(&mut conversation, start, &chapter);
            }
        }
        // A folded chapter's summary covers its messages: it is not split.
        Some(line) if line.chapter.folded() => {}
        _ if verdict.new_chapter => {
            let chapter = Chapter {
                title,
                ..Chapter::default()
            };
            put(&mut conversation, start, &chapter);
        }
        _ => {}
    }
}

/// `/chapters`: the outline, as a picker whose choice shows the chapter.
fn chapters_command(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    open: Query<Entity, With<OutlinePicker>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(conversation) = agents.get(args.agent) else {
        return;
    };
    let lines = outline(conversation);
    if lines.is_empty() {
        notices.write(Notice::info(args.agent, "No chapters yet."));
        return;
    }
    for picker in &open {
        commands.entity(picker).despawn();
    }
    let last = lines.len();
    let mut picker = Picker::new("Chapters")
        .for_agent(args.agent)
        .body("/fold <n> gives the model a summary in place of chapter n; /unfold <n> undoes it.");
    for line in &lines {
        let state = match (line.chapter.folded(), line.number == last) {
            (true, _) => "folded",
            (false, true) => "current",
            (false, false) => "open",
        };
        picker.items.push(PickItem {
            label: format!("{}. {}", line.number, line.chapter.shown_title()),
            command: format!("chapter goto {}", line.number),
            detail: Some(format!(
                "{} · {state} · ~{} tokens in context",
                line.turn_range(),
                line.tokens
            )),
            more: None,
            checked: false,
        });
    }
    picker.selected = last.saturating_sub(1);
    commands.spawn((picker, OutlinePicker));
}

/// The chapter `/fold n`, `/unfold n` or `/chapter rename n` names, or the
/// error notice why there is none.
fn numbered(agent: Entity, lines: &[ChapterLine], word: &str) -> Result<ChapterLine, Notice> {
    let number: Option<usize> = word.parse().ok();
    number
        .and_then(|number| lines.iter().find(|line| line.number == number))
        .cloned()
        .ok_or_else(|| {
            let why = format!("No chapter {word}: /chapters lists {}.", lines.len());
            Notice::error(agent, why)
        })
}

/// `/chapter <title>` starts a chapter at the running turn, else the next;
/// `/chapter rename <n> <title>` renames chapter `n`; `/chapter goto <n>`
/// shows it in the transcript.
fn chapter_command(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, Option<&ActiveTurn>)>,
    turns: Query<&Turn>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((mut conversation, active)) = agents.get_mut(agent) else {
        return;
    };
    let mut words = args.args.splitn(3, char::is_whitespace);
    let first = words.next().unwrap_or_default();
    let lines = outline(&conversation);
    match first {
        "" => {
            notices.write(Notice::error(agent, "/chapter needs a title."));
        }
        "rename" => {
            let line = numbered(agent, &lines, words.next().unwrap_or_default());
            let title = clean_title(words.next().unwrap_or_default());
            match line {
                Err(notice) => {
                    notices.write(notice);
                }
                Ok(_) if title.is_empty() => {
                    notices.write(Notice::error(agent, "/chapter rename <n> <title>"));
                }
                Ok(line) => {
                    let chapter = Chapter {
                        title,
                        renamed: true,
                        ..line.chapter
                    };
                    put(&mut conversation, line.start, &chapter);
                }
            }
        }
        "goto" => match numbered(agent, &lines, words.next().unwrap_or_default()) {
            Err(notice) => {
                notices.write(notice);
            }
            Ok(line) => show(agent, &conversation, &line, &mut commands),
        },
        _ => {
            let title = clean_title(&args.args);
            let running = active.and_then(|active| turns.get(active.turn()).ok());
            match running {
                Some(turn) if conversation.request(turn.start).is_some() => {
                    let start = turn.start;
                    let chapter = Chapter {
                        title,
                        manual: true,
                        renamed: true,
                        summary: None,
                    };
                    put(&mut conversation, start, &chapter);
                }
                _ => {
                    let text = format!("Chapter \"{title}\" starts with the next turn.");
                    notices.write(Notice::info(agent, text));
                    commands.entity(agent).insert(PendingChapter(title));
                }
            }
        }
    }
}

/// Shows `agent` in the terminal view's main pane, scrolled to `line`'s
/// note.
#[cfg(feature = "tui")]
fn show(agent: Entity, conversation: &Conversation, line: &ChapterLine, commands: &mut Commands) {
    use rig_tui::{ItemRow, MainPane, ScrollTo, ShowAgent, TranscriptItem};
    let note = conversation.asides().find(|(_, aside)| {
        aside.after == Some(line.start) && aside.author == author() && Note::of(aside).is_some()
    });
    let item = match note {
        Some((id, _)) => TranscriptItem::Aside(id),
        None => TranscriptItem::Message(line.start),
    };
    commands.trigger(ShowAgent { entity: agent });
    commands.queue(move |world: &mut World| {
        let mut main = world.query_filtered::<Entity, With<MainPane>>();
        if let Ok(pane) = main.single(world) {
            world.trigger(ScrollTo {
                pane,
                item: Some(item),
                row: ItemRow::Top(0),
            });
        }
    });
}

/// Without the terminal view there is nothing to scroll.
#[cfg(not(feature = "tui"))]
fn show(_: Entity, _: &Conversation, _: &ChapterLine, _: &mut Commands) {}

/// `/fold <n>`: asks the summary model for a summary of chapter `n`, which
/// [`take_fold`] puts in its place. The current chapter is refused.
fn fold_command(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, Option<&Connection>)>,
    folding: Query<(&Folding, &ForAgent), Without<Settled>>,
    (config, models): (Res<ChaptersConfig>, Res<Models>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((conversation, own)) = agents.get(agent) else {
        return;
    };
    let lines = outline(conversation);
    let line = match numbered(agent, &lines, args.args.trim()) {
        Ok(line) => line,
        Err(notice) => {
            notices.write(notice);
            return;
        }
    };
    let n = line.number;
    let busy = folding
        .iter()
        .any(|(folding, &ForAgent(of))| of == agent && folding.start == line.start);
    let refusal = if n == lines.len() {
        Some(format!(
            "Chapter {n} is the current one; it cannot be folded."
        ))
    } else if line.chapter.folded() {
        Some(format!("Chapter {n} is folded already."))
    } else if busy {
        Some(format!("Chapter {n} is being folded."))
    } else {
        None
    };
    if let Some(why) = refusal {
        notices.write(Notice::error(agent, why));
        return;
    }
    let configured = config
        .summary_model
        .as_deref()
        .map(|reference| models.connect(reference));
    let connection = match configured {
        Some(Ok(connection)) => Some(connection),
        Some(Err(why)) => {
            notices.write(Notice::error(agent, format!("Cannot fold: {why}.")));
            return;
        }
        None => None,
    };
    let Some(spec) = connection.as_ref().or(own).map(|it| it.spec.clone()) else {
        notices.write(Notice::no_model(agent));
        return;
    };
    let messages = conversation
        .messages()
        .get(line.messages.clone())
        .unwrap_or_default();
    let request = match FOLD_SUMMARIZER.request(messages, "", &CompactReason::Threshold, &spec) {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot fold: {why}.")));
            return;
        }
    };
    let mut state = SummaryState::default();
    state.track(messages, CHANGED);
    let changed = state
        .tracked
        .into_iter()
        .flat_map(|set| set.values)
        .collect();
    let mut call = ModelRequest::new(request);
    if let Some(connection) = connection {
        call = call.on(connection);
    }
    commands.spawn((
        Name::new("chapter summary"),
        Folding {
            start: line.start,
            changed,
        },
        call,
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
    ));
    let text = format!("Summarizing chapter {n} to fold it.");
    notices.write(Notice::info(agent, text));
}

/// Folds the chapter a settled summary call summarized.
fn take_fold(
    settled: On<Add<Settled>>,
    calls: Query<(&Folding, &ForAgent, &Outcome)>,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((folding, &ForAgent(agent), outcome)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let summary = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(reply)) => Summarizer::summary_text(reply).map_err(|why| why.to_string()),
    };
    let lines = outline(&conversation);
    let line = lines.iter().find(|line| line.start == folding.start);
    match (summary, line) {
        (Ok(mut summary), Some(line)) => {
            if !folding.changed.is_empty() {
                summary.push_str("\n\nFiles changed by its tool calls: ");
                summary.push_str(&folding.changed.join(", "));
            }
            let before = line.tokens;
            let chapter = Chapter {
                summary: Some(summary),
                ..line.chapter.clone()
            };
            put(&mut conversation, line.start, &chapter);
            let after = outline(&conversation)
                .iter()
                .find(|line| line.start == folding.start)
                .map_or(0, |line| line.tokens);
            let text = format!(
                "Folded chapter {}: ~{before} tokens became a ~{after}-token summary.",
                line.number
            );
            notices.write(Notice::info(agent, text));
        }
        (Ok(_), None) => {
            notices.write(Notice::error(agent, "Cannot fold: the chapter is gone."));
        }
        (Err(why), _) => {
            notices.write(Notice::error(agent, format!("Cannot fold: {why}.")));
        }
    }
}

/// `/unfold <n>`: the model reads chapter `n`'s messages again.
fn unfold_command(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let lines = outline(&conversation);
    match numbered(agent, &lines, args.args.trim()) {
        Err(notice) => {
            notices.write(notice);
        }
        Ok(line) if !line.chapter.folded() => {
            let why = format!("Chapter {} is not folded.", line.number);
            notices.write(Notice::error(agent, why));
        }
        Ok(line) => {
            let chapter = Chapter {
                summary: None,
                ..line.chapter
            };
            put(&mut conversation, line.start, &chapter);
            let text = format!("Unfolded chapter {}.", line.number);
            notices.write(Notice::info(agent, text));
        }
    }
}

/// Replaces each folded chapter's messages in every request built from
/// the agent's conversation with its summary, keeping tool calls and their
/// results together: a request whose messages would not pair them is left
/// as it is.
fn fold_in_requests(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls)>,
    agents: Query<&Conversation>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Ok(conversation) = agents.get(agent) else {
            continue;
        };
        let lines = outline(conversation);
        if !lines.iter().any(|line| line.chapter.folded()) {
            continue;
        }
        let Some(mut history) = call.history.clone() else {
            continue;
        };
        for (line, next) in lines.iter().zip(lines.iter().skip(1)).rev() {
            let Some(message) = line.fold_message() else {
                continue;
            };
            let (Some(from), Some(to)) =
                (history.index_of(line.start), history.index_of(next.start))
            else {
                continue;
            };
            if from < to {
                let summary = Entry::new(message).by(author());
                history.splice(from..to, [summary], author());
            }
        }
        if history.validate().is_ok() {
            call.history = Some(history);
        }
    }
}
