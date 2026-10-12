//! FH-14: side questions while the agent works.
//!
//! `/btw <question>` starts a side agent at once, even while the agent
//! shown is mid-turn: a helper subagent ([`SpawnSubagent::helper`]) seeded
//! with a copy of the agent's conversation as it is at that moment (its
//! tool calls still running answered as such), offered only the tools
//! whose footprint only reads, on the agent's model unless the `[btw]`
//! table of the settings names another. The `/btw` line is a command, so
//! it never reaches the agent, and the side agent's conversation is its
//! own: the agent's requests never carry the side question.
//!
//! The answer streams into a box under the terminal view's transcript
//! ([`SideChat`], drawn in its panel), which takes the keys: what is typed there and sent
//! with Enter is a follow-up to the same side agent, as is `/btw` while it
//! is open (`/btw new <question>` replaces it), Ctrl+Y copies the last
//! answer into the main input without sending it and closes the box, and
//! Esc closes it. Closing retires the side agent. Each side conversation is
//! kept with the session on the agent it was asked beside
//! ([`SideConversations`], saved), apart from its conversation, and the
//! transcript gets one line about it when it closes. Its model calls are
//! the side agent's, so its cost is that agent's spending, with the
//! purpose [`PURPOSE`]. Interrupting either agent stops only that one.
//!
//! ```toml
//! [btw]
//! model = "deepseek/deepseek-flash"
//! ```

use rig_harness::prelude::*;
use rig_harness::rig_core::transcript::{close_pending_with, pending_calls};
use rig_subagents::{Helper, SpawnSubagent, SpawnedAgent};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::{Block, Paragraph, Wrap};
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, PanelCanvas,
    Placement, RequestRedraw, TerminalEvent, TuiInput, TuiPanel, TuiPlugin, TuiSystems,
};
use serde::Deserialize;

/// The side agent's helper title, which names it.
pub const TITLE: &str = "btw";

/// The purpose of the side agent's model calls in the spending.
pub const PURPOSE: &str = "btw";

/// What a still-running tool call of the agent reads as in the side
/// agent's copy of the conversation.
const STILL_RUNNING: &str = "Still running in the main agent; no result yet.";

/// What the side agent is told about its job.
const SIDE_PROMPT: &str = "You answer the user's side question about the conversation above, \
    which another agent is still working on. Answer briefly. Do not carry on that agent's task: \
    you may only read.";

/// The `[btw]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BtwSettings {
    /// The side agent's model, as `vendor/model`; the agent's own when
    /// not set.
    pub model: Option<String>,
}

/// A side question and its answer.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub struct SideExchange {
    /// What the user asked.
    pub question: String,
    /// The side agent's answer, as streamed so far.
    pub answer: String,
}

/// On an agent: the side conversations asked beside it, oldest first,
/// each one entry of exchanges. Saved with the session, never part of its
/// conversation.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct SideConversations(pub Vec<Vec<SideExchange>>);

/// On a side agent this run started.
#[derive(Component, Clone, Copy, Debug)]
pub struct SideAgent;

/// On a side agent's spawn request: the conversation it starts from.
#[derive(Component, Clone, Debug)]
struct Snapshot(Vec<Entry>);

/// Ask the side agent of `agent` a question: a follow-up when one is open
/// beside it, else a new side agent; `fresh` replaces an open one.
#[derive(Event, Clone, Debug)]
pub struct AskSide {
    /// The agent the question is asked beside.
    pub agent: Entity,
    /// The question.
    pub question: String,
    /// Whether it starts a new side agent even when one is open.
    pub fresh: bool,
}

/// The side conversation open in the panel, if any.
#[derive(Resource, Debug)]
pub struct SideChat {
    /// The panel's entity, a [`TuiPanel`] while the side conversation is
    /// open.
    pub panel: Entity,
    /// The agent it is beside.
    pub main: Option<Entity>,
    /// The side agent, once spawned.
    pub agent: Option<Entity>,
    /// Its spawn request, until it is spawned.
    request: Option<Entity>,
    /// The questions and answers so far.
    pub exchanges: Vec<SideExchange>,
    /// Questions asked before the side agent was spawned.
    waiting: Vec<String>,
    /// What is typed in the panel.
    pub draft: String,
    /// Its entry in the agent's [`SideConversations`], once saved.
    saved_at: Option<usize>,
}

impl SideChat {
    /// A closed side conversation drawn in `panel`.
    fn closed(panel: Entity) -> Self {
        Self {
            panel,
            main: None,
            agent: None,
            request: None,
            exchanges: Vec::new(),
            waiting: Vec::new(),
            draft: String::new(),
            saved_at: None,
        }
    }

    /// The last answer.
    pub fn last_answer(&self) -> Option<&str> {
        self.exchanges
            .last()
            .map(|exchange| exchange.answer.as_str())
    }
}

/// `/btw`, the side agent, its panel and keys.
#[derive(Default)]
pub struct SideQuestionPlugin;

impl Plugin for SideQuestionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        let panel = app.world_mut().spawn(Name::new("panel:btw")).id();
        let scope = KeyScope::Panel(panel);
        let close = KeyBinding::new(KeyPress::key(KeyCode::Esc), "Close the side question");
        let send = KeyBinding::new(KeyPress::ctrl('y'), "Send the side answer to the agent");
        app.add_key_binding(close.in_scope(scope), close_on_key);
        app.add_key_binding(send.in_scope(scope), send_to_agent);
        app.add_setting::<BtwSettings>("btw")
            .register_type::<SideExchange>()
            .register_type::<Vec<SideExchange>>()
            .register_type::<Vec<Vec<SideExchange>>>()
            .register_type::<SideConversations>()
            .insert_resource(SideChat::closed(panel))
            .add_command(
                "btw",
                "Ask a side question while the agent works: /btw <question>, /btw new <question>",
                btw,
            )
            .add_observer(ask_side)
            .add_observer(seed_side_agent)
            .add_observer(side_purpose)
            .add_observer(answered)
            .add_systems(
                Update,
                (
                    type_follow_up.before(AgentLoop),
                    stream_answer.after(AgentLoop),
                    discard_restored.run_if(resource_added::<SessionRestored>),
                ),
            )
            .add_systems(
                PostUpdate,
                (
                    redraw.run_if(resource_changed::<SideChat>),
                    draw.in_set(TuiSystems::Draw),
                ),
            );
    }
}

/// `/btw <question>`, `/btw new <question>`, or `/btw` to give the open
/// side conversation the keys again.
fn btw(
    In(args): In<CommandArgs>,
    chat: Res<SideChat>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let text = args.args.trim();
    let (fresh, question) = match text.strip_prefix("new") {
        Some(rest) if rest.is_empty() || rest.starts_with(char::is_whitespace) => {
            (true, rest.trim())
        }
        _ => (false, text),
    };
    if question.is_empty() {
        if !fresh && chat.main == Some(args.agent) {
            focus.0 = Some(chat.panel);
        } else {
            let usage = "/btw needs a question: /btw <question>, or /btw new <question>";
            notices.write(Notice::error(args.agent, usage));
        }
        return;
    }
    commands.trigger(AskSide {
        agent: args.agent,
        question: question.to_owned(),
        fresh,
    });
}

/// Asks the open side agent a follow-up, or starts one from a copy of the
/// agent's conversation as it is now, offered the tools it may use that
/// only read, and opens the panel with the keys.
fn ask_side(
    asked: On<AskSide>,
    mut chat: ResMut<SideChat>,
    agents: Query<(&Conversation, &ToolAccess)>,
    tools: Query<(&ToolDef, &Footprint)>,
    settings: Res<BtwSettings>,
    mut focus: ResMut<KeyFocus>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let (main, question) = (asked.agent, asked.question.trim().to_owned());
    let exchange = SideExchange {
        question: question.clone(),
        answer: String::new(),
    };
    let open = chat.main == Some(main) && (chat.agent.is_some() || chat.request.is_some());
    if open && !asked.fresh {
        chat.exchanges.push(exchange);
        match chat.agent {
            Some(side) => {
                let follow_up = Deliver::new(side, question, DeliveryMode::Queue);
                commands.trigger(follow_up.with_origin(Origin::user().titled(TITLE)));
            }
            None => chat.waiting.push(question),
        }
        focus.0 = Some(chat.panel);
        return;
    }
    close(&mut chat, &mut commit, &mut commands, &mut notices);
    let Ok((conversation, access)) = agents.get(main) else {
        notices.write(Notice::error(main, "/btw needs an agent to ask beside."));
        return;
    };
    let mut snapshot: Vec<Entry> = (0..conversation.messages().len())
        .filter_map(|at| conversation.entry(at))
        .map(|entry| Entry { id: None, ..entry })
        .collect();
    let running = pending_calls(conversation.messages());
    if !running.is_empty() {
        let results = close_pending_with(&running, STILL_RUNNING);
        snapshot.push(<Entry as From<_>>::from(results));
    }
    let reading = tools
        .iter()
        .filter(|(def, footprint)| footprint.0.reads_only() && access.allows(def.0.name.as_str()))
        .map(|(def, _)| def.0.name.to_string())
        .collect();
    let mut request = SpawnSubagent::helper(TITLE);
    request.tools = Some(reading);
    request.model.clone_from(&settings.model);
    let request = commands
        .spawn((request, ForAgent(main), Snapshot(snapshot)))
        .id();
    let panel = chat.panel;
    *chat = SideChat {
        main: Some(main),
        request: Some(request),
        exchanges: vec![exchange],
        waiting: vec![question],
        ..SideChat::closed(panel)
    };
    // Under the transcript, which keeps streaming above it.
    let below = Placement::Bottom(Constraint::Percentage(40));
    commands.entity(panel).insert(TuiPanel::new(below));
    focus.0 = Some(panel);
}

/// Gives the side agent just spawned its copy of the conversation and the
/// questions asked meanwhile; retires one whose side conversation was
/// closed first.
fn seed_side_agent(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &Snapshot)>,
    mut agents: Query<&mut Conversation>,
    mut chat: ResMut<SideChat>,
    mut commit: Commit,
    mut commands: Commands,
) {
    let Ok((&SpawnedAgent(side), Snapshot(snapshot))) = requests.get(spawned.entity) else {
        return;
    };
    if chat.request != Some(spawned.entity) {
        commit.retire(side);
        return;
    }
    if let Ok(mut conversation) = agents.get_mut(side) {
        commit.splice(side, &mut conversation, 0..0, snapshot.iter().cloned());
    }
    commands.entity(side).insert(SideAgent);
    let section = PromptSection::new(PromptSection::ORDER_ROLE + 1, TITLE, SIDE_PROMPT);
    commands.spawn((section, SectionOf(side)));
    for question in chat.waiting.drain(..) {
        let question = Deliver::new(side, question, DeliveryMode::Queue);
        commands.trigger(question.with_origin(Origin::user().titled(TITLE)));
    }
    chat.agent = Some(side);
    chat.request = None;
}

/// Files a side agent's model calls under [`PURPOSE`] in the spending,
/// before the kernel builds them as a turn's.
fn side_purpose(
    admitted: On<Add<Admitted>>,
    mut calls: Query<(&mut Effect, &CallOf), With<TurnCall>>,
    turns: Query<&TurnOf>,
    sides: Query<(), With<SideAgent>>,
) {
    let Ok((mut effect, &CallOf(turn))) = calls.get_mut(admitted.entity) else {
        return;
    };
    if turns
        .get(turn)
        .is_ok_and(|&TurnOf(agent)| sides.contains(agent))
    {
        effect.purpose = Some(PURPOSE.into());
    }
}

/// Adds the side agent's answer text to the last exchange as it streams,
/// from its model calls still answering: the final answer of one that
/// settled is the turn's to put there.
fn stream_answer(
    mut deltas: MessageReader<StreamDelta>,
    answering: Query<(), (With<ModelRequest>, Without<Outcome>)>,
    mut chat: ResMut<SideChat>,
) {
    for delta in deltas.read() {
        let side = delta.agent.is_some() && delta.agent == chat.agent;
        if delta.kind != DeltaKind::Text || !side || !answering.contains(delta.effect) {
            continue;
        }
        if let Some(last) = chat.exchanges.last_mut() {
            last.answer.push_str(&delta.text);
        }
    }
}

/// Puts the side agent's final answer in the last exchange as its turn
/// ends, and keeps the side conversation with the agent it is beside.
fn answered(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut chat: ResMut<SideChat>,
    mut kept: Query<&mut SideConversations>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Some(main) = chat.main.filter(|_| chat.agent == Some(agent)) else {
        return;
    };
    let answer = match outcome {
        TurnOutcome::Answered(message) => final_answer(message).unwrap_or_default(),
        TurnOutcome::Failed(why) => format!("(failed: {why})"),
        TurnOutcome::Stopped(_) => "(stopped)".to_owned(),
    };
    if let Some(last) = chat.exchanges.last_mut() {
        last.answer = answer;
    }
    let exchanges = chat.exchanges.clone();
    match kept.get_mut(main) {
        Ok(mut kept) => match chat.saved_at.and_then(|at| kept.0.get_mut(at)) {
            Some(entry) => *entry = exchanges,
            None => {
                chat.saved_at = Some(kept.0.len());
                kept.0.push(exchanges);
            }
        },
        Err(_) => {
            chat.saved_at = Some(0);
            commands
                .entity(main)
                .insert(SideConversations(vec![exchanges]));
        }
    }
}

/// Closes the side conversation: the side agent is retired, the panel
/// goes, and the transcript says what was asked.
fn close(
    chat: &mut SideChat,
    commit: &mut Commit,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    if let Some(side) = chat.agent.take() {
        commit.retire(side);
    }
    if let Some(request) = chat.request.take() {
        commands.entity(request).try_despawn();
    }
    if let (Some(main), Some(first)) = (chat.main, chat.exchanges.first()) {
        let more = chat.exchanges.len().saturating_sub(1);
        let more = match more {
            0 => String::new(),
            more => format!(" (+{more} follow-ups)"),
        };
        let text = format!("btw: {}{more}; kept with the session.", first.question);
        notices.write(Notice::info(main, text));
    }
    commands.entity(chat.panel).remove::<TuiPanel>();
    *chat = SideChat::closed(chat.panel);
}

/// Esc in the panel: closes the side conversation.
fn close_on_key(
    pressed: On<KeyPressed>,
    mut chat: ResMut<SideChat>,
    mut focus: ResMut<KeyFocus>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if pressed.released {
        return;
    }
    close(&mut chat, &mut commit, &mut commands, &mut notices);
    focus.0 = None;
}

/// Ctrl+Y in the panel: the last answer goes in the main input, not sent,
/// and the side conversation closes.
fn send_to_agent(
    pressed: On<KeyPressed>,
    mut chat: ResMut<SideChat>,
    mut input: ResMut<TuiInput>,
    mut focus: ResMut<KeyFocus>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if pressed.released {
        return;
    }
    if let Some(answer) = chat.last_answer().filter(|answer| !answer.is_empty()) {
        input.set(answer.to_owned());
    }
    close(&mut chat, &mut commit, &mut commands, &mut notices);
    focus.0 = None;
}

/// Types into the panel while it has the keys: characters, pastes and
/// Backspace edit the draft, and Enter asks it as a follow-up.
fn type_follow_up(
    mut events: MessageReader<TerminalEvent>,
    focus: Res<KeyFocus>,
    mut chat: ResMut<SideChat>,
    mut commands: Commands,
) {
    let Some(main) = chat.main.filter(|_| focus.0 == Some(chat.panel)) else {
        events.clear();
        return;
    };
    for TerminalEvent(event) in events.read() {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                KeyCode::Enter => {
                    let question = std::mem::take(&mut chat.draft);
                    if !question.trim().is_empty() {
                        commands.trigger(AskSide {
                            agent: main,
                            question,
                            fresh: false,
                        });
                    }
                }
                KeyCode::Backspace => {
                    chat.draft.pop();
                }
                KeyCode::Char(typed)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    chat.draft.push(typed);
                }
                _ => {}
            },
            Event::Paste(text) => chat.draft.push_str(text),
            _ => {}
        }
    }
}

/// Retires the side agents a restored session brings back: a side agent
/// lives only while its panel is open, and its exchanges are kept on the
/// agent it was beside.
fn discard_restored(helpers: Query<(Entity, &Helper)>, mut commit: Commit) {
    for (agent, helper) in &helpers {
        if helper.0 == TITLE {
            commit.retire(agent);
        }
    }
}

/// Draws the panel again when the side conversation changed.
fn redraw(mut redraw: MessageWriter<RequestRedraw>) {
    redraw.write(RequestRedraw);
}

/// Draws the side conversation, its latest lines, and the draft.
fn draw(chat: Res<SideChat>, mut canvases: Query<&mut PanelCanvas, With<TuiPanel>>) {
    let Ok(mut canvas) = canvases.get_mut(chat.panel) else {
        return;
    };
    let mut lines: Vec<Line> = Vec::new();
    for exchange in &chat.exchanges {
        lines.push(Line::from(format!("› {}", exchange.question)).bold());
        if exchange.answer.is_empty() {
            lines.push(Line::from("…").dim());
        }
        lines.extend(
            exchange
                .answer
                .lines()
                .map(|line| Line::from(line.to_owned())),
        );
        lines.push(Line::default());
    }
    lines.push(Line::from(format!("> {}", chat.draft)));
    let rows = usize::from(canvas.area().height.saturating_sub(2));
    let skip = lines.len().saturating_sub(rows);
    let block = Block::bordered()
        .title(" btw ")
        .title_bottom(" Enter ask · Ctrl+Y to the agent's input · Esc close ")
        .border_style(Style::new().dim());
    let text: Vec<Line> = lines.into_iter().skip(skip).collect();
    canvas.render(Paragraph::new(text).wrap(Wrap { trim: false }).block(block));
}

#[cfg(test)]
mod tests;
