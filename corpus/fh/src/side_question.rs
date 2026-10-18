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
//! The agent half keeps the side conversation ([`SideChat`]) and
//! publishes it as an `x/btw/state` event ([`BtwState`]) at each change.
//! The view half ([`SideQuestionView`]) draws the side agent live in a
//! [`TranscriptPane`] under the terminal view's transcript, its answer
//! streaming in, over an [`Editor`] panel that takes the keys
//! ([`SideView`]): what is typed there and sent with Enter is a follow-up
//! to the same side agent (`/btw <question>`), as is `/btw` while it is
//! open (`/btw new <question>` replaces it), Ctrl+Y copies the last answer
//! into the main input without sending it and closes both (`/btw close`),
//! and Esc closes them. Closing retires the side agent. Each
//! side conversation is a [`Note`] aside of the agent it was asked beside,
//! after the message it was asked at: titled by its first question
//! and drawn whole, kept with the session, never part of the
//! conversation. Its model calls are the side agent's, so its cost is that
//! agent's spending, with the purpose [`PURPOSE`]. Interrupting either
//! agent stops only that one.
//!
//! ```toml
//! [btw]
//! model = "deepseek/deepseek-flash"
//! ```

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::channel::{ChannelAppExt, ChannelEvent};
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::pairing::{close_pending_with, pending_calls};
use rig_harness::view::{Applied, Mirror};
use rig_harness::worker::{Publish, WorkerOutbox};
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::{
    AppKeyBindingsExt, Editor, InputAction, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope,
    MainInput, Placement, TranscriptPane, TranscriptSource, TuiPanel,
};
use serde::{Deserialize, Serialize};

/// The side agent's helper title, which names it.
pub const TITLE: &str = "btw";

/// The purpose of the side agent's model calls in the spending.
pub const PURPOSE: &str = "btw";

/// What a still-running tool call of the agent reads as in the side
/// agent's copy of the conversation.
const STILL_RUNNING: &str = "Still running in the main agent; no result yet.";

/// The side agent's pane title: what its keys do.
const HELP: &str = "btw · Enter asks · Ctrl+Y to the agent's input · Esc closes";

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

/// The side conversation open, if any, in the agent half.
#[derive(Resource, Debug, Default)]
pub struct SideChat {
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
    /// The message of the agent it was asked after.
    after: Option<MessageId>,
    /// Its [`Note`] aside in the agent's conversation, once answered.
    entry: Option<AsideId>,
}

impl SideChat {
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

/// The side conversation as the agent half publishes it, for the views.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BtwState {
    /// The agent it is beside, while it is open.
    pub main: Option<String>,
    /// The side agent, once spawned.
    pub side: Option<String>,
    /// The last answer, once one came.
    pub last_answer: Option<String>,
}

impl ChannelEvent for BtwState {
    const KIND: &'static str = "x/btw/state";
}

/// `/btw` and the side agent.
#[derive(Default)]
pub struct SideQuestionPlugin;

impl Plugin for SideQuestionPlugin {
    fn build(&self, app: &mut App) {
        // The side agent is a helper subagent: without the plugin nothing
        // starts it and `/btw` would wait forever.
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.add_setting::<BtwSettings>("btw")
            .register_type::<SideAgent>()
            .init_resource::<SideChat>()
            .add_channel_event::<BtwState>()
            .add_command(
                "btw",
                "Ask a side question while the agent works: /btw <question>, /btw new \
                 <question>, /btw close",
                btw,
            )
            .add_observer(ask_side)
            .add_observer(seed_side_agent)
            .add_observer(side_purpose)
            .add_observer(answered)
            .add_systems(
                Update,
                discard_restored.run_if(resource_added::<SessionRestored>),
            )
            .add_systems(
                Last,
                publish
                    .run_if(resource_exists::<WorkerOutbox>)
                    .run_if(resource_changed::<SideChat>),
            );
    }
}

/// The side conversation in the view: its follow-up editor, a [`TuiPanel`]
/// with the keys while it is open, and the side agent's transcript pane,
/// a [`TranscriptPane`] once it is spawned; and the mirrored agents and
/// last answer the agent half's [`BtwState`] names.
#[derive(Resource, Debug)]
pub struct SideView {
    /// The follow-up editor.
    pub editor: Entity,
    /// The side agent's pane.
    pub pane: Entity,
    /// The agent it is beside, while open.
    pub main: Option<Entity>,
    /// The side agent, once spawned.
    pub side: Option<Entity>,
    /// The last answer.
    pub last_answer: Option<String>,
}

/// The view half: the side agent's pane, the follow-up editor and their
/// keys, from the agent's `x/btw/state` events; named in `plugins.toml` as
/// the entry's `view`.
#[derive(Default)]
pub struct SideQuestionView;

impl Plugin for SideQuestionView {
    fn build(&self, app: &mut App) {
        // Spawned in this order, the editor is laid out under the pane.
        let world = app.world_mut();
        let editor = world
            .spawn((Name::new("btw:editor"), Editor::default()))
            .id();
        let pane = world.spawn(Name::new("btw:pane")).id();
        let scope = KeyScope::Panel(editor);
        let backspace = KeyBinding::new(KeyPress::key(KeyCode::Backspace), "Delete back");
        world.spawn((
            backspace.in_scope(scope),
            InputAction::Backspace,
            ChildOf(editor),
        ));
        let ask = KeyBinding::new(KeyPress::key(KeyCode::Enter), "Ask the side agent");
        let close = KeyBinding::new(KeyPress::key(KeyCode::Esc), "Close the side question");
        let send = KeyBinding::new(KeyPress::ctrl('y'), "Send the side answer to the agent");
        app.add_key_binding(ask.in_scope(scope), ask_on_key);
        app.add_key_binding(close.in_scope(scope), close_on_key);
        app.add_key_binding(send.in_scope(scope), send_to_agent);
        app.insert_resource(SideView {
            editor,
            pane,
            main: None,
            side: None,
            last_answer: None,
        })
        .add_message::<Applied>()
        .add_systems(Update, show);
    }
}

/// Publishes the side conversation after a change.
fn publish(chat: Res<SideChat>, ids: Query<&AgentId>, mut out: Publish) {
    let id = |agent: Option<Entity>| {
        agent
            .and_then(|agent| ids.get(agent).ok())
            .map(|id| id.0.clone())
    };
    out.event(&BtwState {
        main: id(chat.main),
        side: id(chat.agent),
        last_answer: chat.last_answer().map(str::to_owned),
    });
}

/// Opens, fills and closes the panels as the agent half's [`BtwState`]
/// says, in the view: an open side conversation gets the keys.
fn show(
    mut applied: MessageReader<Applied>,
    mirror: Option<Res<Mirror>>,
    mut view: ResMut<SideView>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    let Some(mirror) = mirror else {
        return;
    };
    for Applied(record) in applied.read() {
        if record.kind != BtwState::KIND {
            continue;
        }
        let Ok(state) = serde_json::from_value::<BtwState>(record.data.clone()) else {
            continue;
        };
        let main = state.main.as_deref().and_then(|id| mirror.agent(id));
        let side = state.side.as_deref().and_then(|id| mirror.agent(id));
        let (editor, pane) = (view.editor, view.pane);
        if main.is_none() {
            commands
                .entity(editor)
                .remove::<TuiPanel>()
                .insert(Editor::default());
            commands.entity(pane).remove::<(TranscriptPane, TuiPanel)>();
            if focus.0 == Some(editor) {
                focus.0 = None;
            }
        } else {
            if view.main.is_none() {
                let below = Placement::Bottom(Constraint::Length(3));
                commands.entity(editor).insert(TuiPanel::new(below));
            }
            focus.0 = Some(editor);
            if let Some(side) = side.filter(|&side| view.side != Some(side)) {
                // Under the transcript, which keeps streaming above it.
                let shown = TranscriptPane::new(TranscriptSource::Agent(side)).titled(HELP);
                let below = Placement::Bottom(Constraint::Percentage(40));
                commands.entity(pane).insert((shown, TuiPanel::new(below)));
            }
        }
        view.main = main;
        view.side = side;
        view.last_answer = state.last_answer;
    }
}

/// `/btw <question>`, `/btw new <question>`, `/btw close`, or `/btw` to
/// give the open side conversation the keys again (its state, published
/// again, opens the views' panel).
fn btw(
    In(args): In<CommandArgs>,
    mut chat: ResMut<SideChat>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let text = args.args.trim();
    if text == "close" {
        close(&mut chat, &mut commands);
        return;
    }
    let (fresh, question) = match text.strip_prefix("new") {
        Some(rest) if rest.is_empty() || rest.starts_with(char::is_whitespace) => {
            (true, rest.trim())
        }
        _ => (false, text),
    };
    if question.is_empty() {
        if !fresh && chat.main == Some(args.agent) {
            chat.set_changed();
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
/// only read.
fn ask_side(
    asked: On<AskSide>,
    mut chat: ResMut<SideChat>,
    agents: Query<(&Conversation, &ToolAccess)>,
    tools: Query<(&ToolDef, &Footprint)>,
    settings: Res<BtwSettings>,
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
    *chat = SideChat {
        main: Some(main),
        request: Some(request),
        exchanges: vec![exchange],
        waiting: vec![question],
        after: messages
            .checked_sub(1)
            .and_then(|last| conversation.id(last)),
        ..SideChat::default()
    };
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

/// Closes the side conversation: the side agent is retired, and the
/// views' pane and editor go; its note in the transcript stays.
fn close(chat: &mut SideChat, commands: &mut Commands) {
    if let Some(side) = chat.agent.take() {
        commands.trigger(Retire { entity: side });
    }
    if let Some(request) = chat.request.take() {
        commands.entity(request).try_despawn();
    }
    *chat = SideChat::default();
}

/// Enter in the editor: what it holds is asked as a follow-up.
fn ask_on_key(
    pressed: On<KeyPressed>,
    mut editors: Query<&mut Editor>,
    view: Res<SideView>,
    mut commands: Commands,
) {
    let (Some(main), Ok(mut editor)) = (view.main, editors.get_mut(view.editor)) else {
        return;
    };
    if pressed.released || editor.text().trim().is_empty() {
        return;
    }
    let question = editor.take();
    commands.trigger(RunCommand::new(main, format!("btw {question}")));
}

/// Esc in the editor: closes the side conversation.
fn close_on_key(pressed: On<KeyPressed>, view: Res<SideView>, mut commands: Commands) {
    if let (false, Some(main)) = (pressed.released, view.main) {
        commands.trigger(RunCommand::new(main, "btw close"));
    }
}

/// Ctrl+Y in the editor: the last answer goes in the main input, not
/// sent, and the side conversation closes.
fn send_to_agent(
    pressed: On<KeyPressed>,
    view: Res<SideView>,
    input: Option<MainInput>,
    mut commands: Commands,
) {
    let Some(main) = view.main.filter(|_| !pressed.released) else {
        return;
    };
    if let (Some(answer), Some(mut input)) = (
        view.last_answer
            .as_deref()
            .filter(|answer| !answer.is_empty()),
        input,
    ) {
        input.set(answer.to_owned());
    }
    commands.trigger(RunCommand::new(main, "btw close"));
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

#[cfg(test)]
mod tests;
