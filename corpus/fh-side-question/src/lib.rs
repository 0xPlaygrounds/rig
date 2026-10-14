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
//! ([`SideChat`], drawn in its panel), which takes the keys: what is typed
//! there and sent with Enter is a follow-up to the same side agent, as is
//! `/btw` while it is open (`/btw new <question>` replaces it), Ctrl+Y
//! copies the last answer into the main input without sending it and
//! closes the box, and Esc closes it. Closing retires the side agent. Each
//! side conversation is a [`Note`] aside of the agent it was asked beside,
//! after the message it was asked at: titled by its first question
//! and drawn cut until expanded, kept with the session, never part of the
//! conversation. Its model calls are the side agent's, so its cost is that
//! agent's spending, with the purpose [`PURPOSE`]. Interrupting either
//! agent stops only that one.
//!
//! ```toml
//! [btw]
//! model = "deepseek/deepseek-flash"
//! ```

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::pairing::{close_pending_with, pending_calls};
use rig_subagents::{SpawnSubagent, SpawnedAgent};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::{Block, Paragraph, Wrap};
use rig_tui::{
    AppKeyBindingsExt, InputSystems, KeyBinding, KeyFocus, KeyInput, KeyPress, KeyPressed,
    KeyScope, MainInput, PanelCanvas, Placement, RequestRedraw, TerminalEvents, TuiPanel,
    TuiPlugin, TuiSystems,
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SideExchange {
    /// What the user asked.
    pub question: String,
    /// The side agent's answer, once its turn ended.
    pub answer: String,
}

/// On a side agent, kept with the session: one a restore brings back is
/// retired, its panel being gone.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default)]
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
    /// The message of the agent it was asked after.
    after: Option<MessageId>,
    /// Its [`Note`] aside in the agent's conversation, once answered.
    entry: Option<AsideId>,
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
            after: None,
            entry: None,
        }
    }

    /// The last answer.
    pub fn last_answer(&self) -> Option<&str> {
        self.exchanges
            .last()
            .map(|exchange| exchange.answer.as_str())
    }

    /// Its note aside: titled by the first question, holding every
    /// exchange.
    fn entry(&self) -> Aside {
        let first = self.exchanges.first();
        let title = first
            .map(|first| first.question.as_str())
            .unwrap_or_default();
        let title = match self.exchanges.len().saturating_sub(1) {
            0 => format!("btw: {title}"),
            more => format!("btw: {title} (+{more} follow-ups)"),
        };
        let text: Vec<String> = self
            .exchanges
            .iter()
            .map(|exchange| format!("› {}\n{}", exchange.question, exchange.answer))
            .collect();
        Note::new(title, text.join("\n\n")).aside(Author::plugin(TITLE), self.after)
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
            .register_type::<SideAgent>()
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
                rig_tui::TuiKeys,
                type_follow_up.in_set(InputSystems::Filter),
            )
            .add_systems(
                Update,
                discard_restored.run_if(resource_added::<SessionRestored>),
            )
            .add_systems(PostUpdate, (redraw, draw.in_set(TuiSystems::Draw)));
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
                commands.trigger(follow_up.with_author(Author::user().titled(TITLE)));
            }
            None => chat.waiting.push(question),
        }
        focus.0 = Some(chat.panel);
        return;
    }
    close(&mut chat, &mut commands);
    let Ok((conversation, access)) = agents.get(main) else {
        notices.write(Notice::error(main, "/btw needs an agent to ask beside."));
        return;
    };
    let messages = conversation.messages().len();
    let mut snapshot: Vec<Entry> = (0..messages)
        .filter_map(|at| conversation.entry(at))
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
        after: messages
            .checked_sub(1)
            .and_then(|last| conversation.id(last)),
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
    mut commands: Commands,
) {
    let Ok((&SpawnedAgent(side), Snapshot(snapshot))) = requests.get(spawned.entity) else {
        return;
    };
    if chat.request != Some(spawned.entity) {
        commands.trigger(Retire { entity: side });
        return;
    }
    if let Ok(mut conversation) = agents.get_mut(side) {
        conversation.splice(0..0, snapshot.iter().cloned(), Author::plugin(TITLE));
    }
    commands.entity(side).insert(SideAgent);
    let section = PromptSection::new(PromptSection::ORDER_ROLE + 1, TITLE, SIDE_PROMPT);
    commands.spawn((section, SectionOf(side)));
    for question in chat.waiting.drain(..) {
        let question = Deliver::new(side, question, DeliveryMode::Queue);
        commands.trigger(question.with_author(Author::user().titled(TITLE)));
    }
    chat.agent = Some(side);
    chat.request = None;
}

/// Files a side agent's model calls under [`PURPOSE`] in the spending,
/// before the kernel builds them as a turn's.
fn side_purpose(
    admitted: On<Add<Admitted>>,
    mut calls: Query<(&mut Effect, &ForAgent), With<TurnCall>>,
    sides: Query<(), With<SideAgent>>,
) {
    if let Ok((mut effect, &ForAgent(agent))) = calls.get_mut(admitted.entity)
        && sides.contains(agent)
    {
        effect.purpose = Some(PURPOSE.into());
    }
}

/// Puts the side agent's final answer in the last exchange as its turn
/// ends, and the side conversation in the transcript of the agent it is
/// beside.
fn answered(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut chat: ResMut<SideChat>,
    mut agents: Query<&mut Conversation>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Some(main) = chat.main.filter(|_| chat.agent == Some(agent)) else {
        return;
    };
    let answer = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().unwrap_or_default(),
        TurnOutcome::Failed(why) => format!("(failed: {why})"),
        TurnOutcome::Stopped(_) => "(stopped)".to_owned(),
    };
    if let Some(last) = chat.exchanges.last_mut() {
        last.answer = answer;
    }
    let entry = chat.entry();
    let Ok(mut conversation) = agents.get_mut(main) else {
        return;
    };
    match chat.entry {
        Some(shown) if conversation.replace_aside(shown, entry.clone()) => {}
        _ => chat.entry = Some(conversation.add_aside(entry)),
    }
}

/// Closes the side conversation: the side agent is retired and the panel
/// goes; its note in the transcript stays.
fn close(chat: &mut SideChat, commands: &mut Commands) {
    if let Some(side) = chat.agent.take() {
        commands.trigger(Retire { entity: side });
    }
    if let Some(request) = chat.request.take() {
        commands.entity(request).try_despawn();
    }
    commands.entity(chat.panel).remove::<TuiPanel>();
    *chat = SideChat::closed(chat.panel);
}

/// Esc in the panel: closes the side conversation.
fn close_on_key(
    pressed: On<KeyPressed>,
    mut chat: ResMut<SideChat>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released {
        return;
    }
    close(&mut chat, &mut commands);
    focus.0 = None;
}

/// Ctrl+Y in the panel: the last answer goes in the main input, not sent,
/// and the side conversation closes.
fn send_to_agent(
    pressed: On<KeyPressed>,
    mut chat: ResMut<SideChat>,
    input: Option<MainInput>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released {
        return;
    }
    if let (Some(answer), Some(mut input)) = (
        chat.last_answer().filter(|answer| !answer.is_empty()),
        input,
    ) {
        input.set(answer.to_owned());
    }
    close(&mut chat, &mut commands);
    focus.0 = None;
}

/// Types into the panel while it has the keys: characters, pastes and
/// Backspace edit the draft, and Enter asks it as a follow-up.
fn type_follow_up(
    events: Res<TerminalEvents>,
    focus: Res<KeyFocus>,
    mut chat: ResMut<SideChat>,
    mut commands: Commands,
) {
    let Some(main) = chat.main.filter(|_| focus.0 == Some(chat.panel)) else {
        return;
    };
    for item in &events.0 {
        let KeyInput::Event(event) = item else {
            continue;
        };
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
/// lives only while its panel is open, and its exchanges are a note aside
/// of the agent it was beside.
fn discard_restored(
    sides: Query<Entity, With<SideAgent>>,
    chat: Res<SideChat>,
    mut commands: Commands,
) {
    for side in sides.iter().filter(|&side| chat.agent != Some(side)) {
        commands.trigger(Retire { entity: side });
    }
}

/// The side agent's answer streaming so far, from its model call in flight.
type Streaming<'w, 's> =
    Query<'w, 's, (Ref<'static, Partial>, &'static ForAgent), (With<TurnCall>, Without<Settled>)>;

/// The text the side agent of `chat` has streamed so far.
fn streaming<'a>(chat: &SideChat, partials: &'a Streaming) -> Option<Ref<'a, Partial>> {
    let side = chat.agent?;
    partials
        .iter()
        .find_map(|(partial, &ForAgent(agent))| (agent == side).then_some(partial))
}

/// Draws the panel again when the side conversation or its streaming
/// answer changed.
fn redraw(chat: Res<SideChat>, partials: Streaming, mut redraw: MessageWriter<RequestRedraw>) {
    let streamed = streaming(&chat, &partials).is_some_and(|partial| partial.is_changed());
    if chat.is_changed() || streamed {
        redraw.write(RequestRedraw);
    }
}

/// Draws the side conversation, its latest lines, and the draft.
fn draw(
    chat: Res<SideChat>,
    partials: Streaming,
    mut canvases: Query<&mut PanelCanvas, With<TuiPanel>>,
) {
    let Ok(mut canvas) = canvases.get_mut(chat.panel) else {
        return;
    };
    let streamed = streaming(&chat, &partials);
    let mut lines: Vec<Line> = Vec::new();
    let last = chat.exchanges.len().saturating_sub(1);
    for (at, exchange) in chat.exchanges.iter().enumerate() {
        lines.push(Line::from(format!("› {}", exchange.question)).bold());
        // The turn's answer goes to the last question; it streams there.
        let answer = match &streamed {
            Some(partial) if at == last && exchange.answer.is_empty() => partial.text.as_str(),
            _ => exchange.answer.as_str(),
        };
        if answer.is_empty() {
            lines.push(Line::from("…").dim());
        }
        lines.extend(answer.lines().map(|line| Line::from(line.to_owned())));
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
