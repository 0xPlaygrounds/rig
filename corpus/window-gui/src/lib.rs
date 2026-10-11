//! A window front: the agents on the left with their status and subagents
//! indented, the shown agent's transcript in the middle (markdown answers,
//! collapsible reasoning and tool calls, the reply streaming in), an input
//! box under it, and pickers from any plugin as dialogs with buttons.
//!
//! What the window shows is its model, four resources kept by systems that
//! need no window: [`GuiAgents`], [`GuiTranscript`], [`GuiInput`] and
//! [`GuiDialogs`]. The window, or a test, changes it only through the
//! requests [`Show`], [`Submit`], [`Stop`], [`Toggle`], [`Choose`] and
//! [`Dismiss`], which become the same kernel requests the terminal view
//! sends. With the default `window` feature, a Bevy UI window draws the
//! model, redrawing a part only when its resource changed.
//!
//! The window opens beside the terminal view in an interactive run, and as
//! the front when [`GUI_ENV`] is set: the launcher refuses an unknown
//! `--gui` flag.

#[cfg(feature = "window")]
mod window;

use std::collections::{HashSet, VecDeque};

use rig_activity::{Activity, ActivityPlugin, ActivitySystems, Status};
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{Message, ToolResult as CallResult};
use rig_harness::rig_ecs::agent::{STOPPED, primary_order};
use rig_tui::{PickItem, PickRequest};

/// Set to a non-empty value, it opens the window as the front, in place of
/// a `--gui` flag.
pub const GUI_ENV: &str = "RIG_GUI";

/// Lines of a tool call's arguments and result shown before it collapses.
pub const LONG_LINES: usize = 8;

/// Notices kept.
const KEPT_NOTICES: usize = 50;

/// The window front and its model.
#[derive(Default)]
pub struct WindowGuiPlugin;

impl Plugin for WindowGuiPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ActivityPlugin>() {
            app.add_plugins(ActivityPlugin);
        }
        app.init_resource::<GuiAgents>()
            .init_resource::<GuiTranscript>()
            .init_resource::<GuiInput>()
            .init_resource::<GuiDialogs>()
            .add_message::<PickRequest>()
            .configure_sets(PostUpdate, GuiSystems.after(ActivitySystems))
            .add_systems(Update, (open_dialogs, collect_notices, recall))
            .add_systems(
                PostUpdate,
                (list_agents, show_transcript, stream_reply)
                    .chain()
                    .in_set(GuiSystems),
            )
            .add_observer(show)
            .add_observer(submit)
            .add_observer(stop)
            .add_observer(toggle)
            .add_observer(choose)
            .add_observer(dismiss);
        #[cfg(feature = "window")]
        window::build(app);
    }

    #[cfg(feature = "window")]
    fn finish(&self, app: &mut App) {
        window::finish(app);
    }
}

/// The systems in `PostUpdate` that bring the model up to date, after the
/// agents' activity. A view draws after them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuiSystems;

/// One agent in the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentRow {
    /// The agent.
    pub agent: Entity,
    /// Its name.
    pub name: String,
    /// What it does: `idle`, `thinking`, ``running `shell` ``.
    pub status: String,
    /// Whether a turn runs.
    pub busy: bool,
    /// How deep it is among subagents: 0 for one the user started.
    pub depth: usize,
}

/// The agent list, subagents under their parent, and the agent shown.
#[derive(Resource, Default, Debug)]
pub struct GuiAgents {
    /// The agent shown and typed to.
    pub shown: Option<Entity>,
    /// Every agent, in tree order.
    pub rows: Vec<AgentRow>,
}

impl GuiAgents {
    /// The row of `agent`.
    pub fn row(&self, agent: Entity) -> Option<&AgentRow> {
        self.rows.iter().find(|row| row.agent == agent)
    }
}

/// Where a collapsible block is: its message and content item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockKey {
    /// The message's index in the conversation.
    pub message: usize,
    /// The item's index in the message.
    pub item: usize,
}

/// A tool call and its result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolBlock {
    /// Where it is.
    pub key: BlockKey,
    /// The tool.
    pub name: String,
    /// The arguments, as pretty JSON.
    pub arguments: String,
    /// The result's text, once the tool answered.
    pub result: Option<String>,
    /// Whether the tool answered with an error.
    pub failed: bool,
    /// Whether arguments and result together pass [`LONG_LINES`].
    pub long: bool,
    /// Whether a long call is shown whole.
    pub open: bool,
}

/// One part of a transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    /// What the user typed.
    User(String),
    /// What an agent or a plugin sent, without the header the model reads.
    Delivered {
        /// Who sent it.
        from: String,
        /// The text.
        text: String,
    },
    /// The model's answer, markdown.
    Answer(String),
    /// The model's reasoning, collapsed until opened.
    Reasoning {
        /// Where it is.
        key: BlockKey,
        /// The text.
        text: String,
        /// Whether it is shown.
        open: bool,
    },
    /// A tool call.
    Tool(ToolBlock),
    /// A request the user stopped before it was answered.
    Stopped,
}

/// A line for the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuiNotice {
    /// The agent it is about, or `None` for the whole app.
    pub agent: Option<Entity>,
    /// The text.
    pub text: String,
    /// Whether it reports a failure.
    pub error: bool,
}

/// The shown agent's transcript, the reply streaming in, and notices.
#[derive(Resource, Default, Debug)]
pub struct GuiTranscript {
    /// The agent it is of.
    pub agent: Option<Entity>,
    /// Its conversation, as blocks.
    pub blocks: Vec<Block>,
    /// The reply streaming in: its answer text so far.
    pub streaming: String,
    /// The reply streaming in: its reasoning so far.
    pub reasoning: String,
    /// Recent notices, oldest first.
    pub notices: Vec<GuiNotice>,
    /// The blocks opened by the user.
    opened: HashSet<BlockKey>,
}

impl GuiTranscript {
    /// The notices about the shown agent or the whole app.
    pub fn shown_notices(&self) -> impl Iterator<Item = &GuiNotice> {
        self.notices
            .iter()
            .filter(|notice| notice.agent.is_none() || notice.agent == self.agent)
    }

    /// Whether the transcript, the streamed reply included, holds `text`.
    pub fn contains(&self, text: &str) -> bool {
        self.streaming.contains(text)
            || self.blocks.iter().any(|block| match block {
                Block::User(said) | Block::Answer(said) => said.contains(text),
                Block::Delivered { text: said, .. } | Block::Reasoning { text: said, .. } => {
                    said.contains(text)
                }
                Block::Tool(tool) => {
                    tool.arguments.contains(text)
                        || tool.result.as_ref().is_some_and(|said| said.contains(text))
                }
                Block::Stopped => false,
            })
    }
}

/// The input box.
#[derive(Resource, Default, Debug)]
pub struct GuiInput {
    /// What is typed.
    pub text: String,
    /// A refused command put back in the input: sent again unchanged, it
    /// goes to the model as it is.
    pub refused: Option<String>,
}

/// A choice for the user, from a plugin's [`PickRequest`].
#[derive(Clone, Debug)]
pub struct Dialog {
    /// The agent the choice is for.
    pub agent: Entity,
    /// What is chosen.
    pub title: String,
    /// The choices; each runs its command for the agent.
    pub items: Vec<PickItem>,
}

/// The open dialogs; the first is shown.
#[derive(Resource, Default, Debug)]
pub struct GuiDialogs {
    /// The dialogs, oldest first.
    pub open: VecDeque<Dialog>,
}

/// Show `agent` and send what is typed to it.
#[derive(Event, Clone, Copy, Debug)]
pub struct Show {
    /// The agent.
    pub agent: Entity,
}

/// Send the input to the shown agent as the user's message, as the
/// terminal view's Enter does: a slash command, a new turn, or steering
/// for the running one.
#[derive(Event, Clone, Copy, Debug)]
pub struct Submit;

/// Stop the shown agent's turn, or what a plugin stops on it.
#[derive(Event, Clone, Copy, Debug)]
pub struct Stop;

/// Open or close a block of the transcript.
#[derive(Event, Clone, Copy, Debug)]
pub struct Toggle {
    /// The block.
    pub key: BlockKey,
}

/// Choose an item of the shown dialog, which runs its command.
#[derive(Event, Clone, Copy, Debug)]
pub struct Choose {
    /// The item's position.
    pub item: usize,
}

/// Close the shown dialog without choosing.
#[derive(Event, Clone, Copy, Debug)]
pub struct Dismiss;

/// What the agent does, for its row.
fn status_text(activity: &Activity) -> String {
    match &activity.status {
        Status::RunningTools if !activity.tools.is_empty() => {
            let names: Vec<String> = activity
                .tools
                .iter()
                .map(|tool| format!("`{}`", tool.name))
                .collect();
            format!("running {}", names.join(", "))
        }
        status => status.to_string(),
    }
}

type AgentItem<'a> = (
    Entity,
    &'a Name,
    &'a AgentId,
    &'a Activity,
    Has<SpawnedBy>,
    Option<&'a Spawned>,
);

/// Keeps the agent list and the agent shown: the first agent the user
/// started when none is shown or the shown one is gone.
fn list_agents(agents: Query<AgentItem, With<Agent>>, mut list: ResMut<GuiAgents>) {
    let mut roots: Vec<_> = agents.iter().filter(|agent| !agent.4).collect();
    roots.sort_by(|a, b| primary_order(a.4, a.2).cmp(&primary_order(b.4, b.2)));
    let mut rows = Vec::new();
    let mut stack: Vec<(Entity, usize)> = roots.iter().rev().map(|root| (root.0, 0)).collect();
    while let Some((agent, depth)) = stack.pop() {
        let Ok((agent, name, _, activity, _, spawned)) = agents.get(agent) else {
            continue;
        };
        rows.push(AgentRow {
            agent,
            name: name.as_str().to_owned(),
            status: status_text(activity),
            busy: activity.is_busy(),
            depth,
        });
        if let Some(spawned) = spawned {
            let children: Vec<Entity> = spawned.iter().collect();
            stack.extend(children.into_iter().rev().map(|child| (child, depth + 1)));
        }
    }
    let shown = list
        .shown
        .filter(|shown| agents.contains(*shown))
        .or_else(|| roots.first().map(|root| root.0));
    if list.shown != shown {
        list.shown = shown;
    }
    if list.rows != rows {
        list.rows = rows;
    }
}

/// Keeps the shown agent's transcript as blocks, rebuilt when its
/// conversation changed.
fn show_transcript(
    list: Res<GuiAgents>,
    conversations: Query<Ref<Conversation>>,
    mut transcript: ResMut<GuiTranscript>,
) {
    let shown = list
        .shown
        .and_then(|agent| Some((agent, conversations.get(agent).ok()?)));
    let Some((agent, conversation)) = shown else {
        if transcript.agent.is_some() {
            transcript.agent = None;
            transcript.blocks.clear();
        }
        return;
    };
    let switched = transcript.agent != Some(agent);
    if switched {
        transcript.agent = Some(agent);
        transcript.opened.clear();
    } else if !conversation.is_changed() {
        return;
    }
    let blocks = blocks_of(&conversation, &transcript.opened);
    if switched || transcript.blocks != blocks {
        transcript.blocks = blocks;
    }
}

/// The reply streaming in to the shown agent, from its model call.
fn stream_reply(
    list: Res<GuiAgents>,
    active: Query<&ActiveTurn>,
    turns: Query<&Calls>,
    partials: Query<&Partial>,
    mut transcript: ResMut<GuiTranscript>,
) {
    let partial = list
        .shown
        .and_then(|agent| active.get(agent).ok())
        .and_then(|turn| turns.get(turn.turn()).ok())
        .and_then(|calls| calls.iter().find_map(|call| partials.get(call).ok()));
    let (text, reasoning) = partial.map_or(("", ""), |partial| {
        (partial.text.as_str(), partial.reasoning.as_str())
    });
    if transcript.streaming != text {
        transcript.streaming = text.to_owned();
    }
    if transcript.reasoning != reasoning {
        transcript.reasoning = reasoning.to_owned();
    }
}

/// The conversation as blocks; a tool call carries the result its next
/// message holds.
fn blocks_of(conversation: &Conversation, opened: &HashSet<BlockKey>) -> Vec<Block> {
    let messages = conversation.messages();
    let mut blocks = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::System { .. } => {}
            Message::User { content } => {
                for (item, part) in content.iter().enumerate() {
                    match part {
                        UserContent::Text(text) if text.text == STOPPED => {
                            blocks.push(Block::Stopped);
                        }
                        UserContent::Text(text) => match conversation.origin(index, item) {
                            Some(origin) => blocks.push(Block::Delivered {
                                from: origin.label(),
                                text: origin
                                    .header()
                                    .and_then(|header| text.text.strip_prefix(&header))
                                    .map_or(text.text.as_str(), |body| {
                                        body.trim_start_matches('\n')
                                    })
                                    .to_owned(),
                            }),
                            None => blocks.push(Block::User(text.text.clone())),
                        },
                        UserContent::Image(_) => blocks.push(Block::User("[image]".to_owned())),
                        // Results go with their calls.
                        _ => {}
                    }
                }
            }
            Message::Assistant(reply) => {
                for (item, part) in reply.content.iter().enumerate() {
                    let key = BlockKey {
                        message: index,
                        item,
                    };
                    match part {
                        AssistantContent::Text(text) => {
                            blocks.push(Block::Answer(text.text.clone()))
                        }
                        AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                            blocks.push(Block::Reasoning {
                                key,
                                text: reasoning.text.clone(),
                                open: opened.contains(&key),
                            });
                        }
                        AssistantContent::ToolCall(call) => {
                            let result = messages
                                .get(index + 1)
                                .and_then(|next| result_of(next, call));
                            blocks.push(Block::Tool(tool_block(key, call, result, opened)));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    blocks
}

/// The result of `call` in `message`, if it carries one.
fn result_of<'a>(message: &'a Message, call: &ToolCall) -> Option<&'a CallResult> {
    let Message::User { content } = message else {
        return None;
    };
    content.iter().find_map(|part| match part {
        UserContent::ToolResult(result) if result.call == call.id => Some(result),
        _ => None,
    })
}

fn tool_block(
    key: BlockKey,
    call: &ToolCall,
    result: Option<&CallResult>,
    opened: &HashSet<BlockKey>,
) -> ToolBlock {
    let arguments = serde_json::to_string_pretty(&call.function.arguments).unwrap_or_default();
    let text = result.map(|result| {
        result
            .content
            .iter()
            .filter_map(|content| content.as_text())
            .collect::<Vec<_>>()
            .join("\n")
    });
    let lines = arguments.lines().count() + text.as_deref().map_or(0, |text| text.lines().count());
    ToolBlock {
        key,
        name: call.function.name.as_str().to_owned(),
        arguments,
        result: text,
        failed: result.is_some_and(|result| result.is_error),
        long: lines > LONG_LINES,
        open: opened.contains(&key),
    }
}

/// Opens the pickers plugins ask for as dialogs.
fn open_dialogs(mut requests: MessageReader<PickRequest>, mut dialogs: ResMut<GuiDialogs>) {
    for request in requests.read() {
        dialogs.open.push_back(Dialog {
            agent: request.agent,
            title: request.title.clone(),
            items: request.items.clone(),
        });
    }
}

/// Keeps the latest notices.
fn collect_notices(mut notices: MessageReader<Notice>, mut transcript: ResMut<GuiTranscript>) {
    if notices.is_empty() {
        return;
    }
    for notice in notices.read() {
        transcript.notices.push(GuiNotice {
            agent: notice.agent,
            text: notice.text.clone(),
            error: notice.level == NoticeLevel::Error,
        });
    }
    let excess = transcript.notices.len().saturating_sub(KEPT_NOTICES);
    transcript.notices.drain(..excess);
}

/// Puts what was typed to the shown agent and not sent back in the input,
/// and says why a refused command was.
fn recall(
    mut recalled: MessageReader<Recalled>,
    list: Res<GuiAgents>,
    mut input: ResMut<GuiInput>,
    mut transcript: ResMut<GuiTranscript>,
) {
    for recalled in recalled.read() {
        if list.shown != Some(recalled.agent) {
            continue;
        }
        if let Some(why) = &recalled.why {
            transcript.notices.push(GuiNotice {
                agent: Some(recalled.agent),
                text: format!("{why} Enter sends it to the model as it is."),
                error: true,
            });
            input.refused = Some(recalled.text.clone());
        }
        input.text = if input.text.trim().is_empty() {
            recalled.text.clone()
        } else {
            format!("{}\n\n{}", recalled.text, input.text)
        };
    }
}

fn show(show: On<Show>, agents: Query<(), With<Agent>>, mut list: ResMut<GuiAgents>) {
    if agents.contains(show.agent) && list.shown != Some(show.agent) {
        list.shown = Some(show.agent);
    }
}

fn submit(
    _submit: On<Submit>,
    list: Res<GuiAgents>,
    mut input: ResMut<GuiInput>,
    mut commands: Commands,
) {
    let Some(agent) = list.shown else {
        return;
    };
    if input.text.trim().is_empty() {
        return;
    }
    let text = std::mem::take(&mut input.text);
    let refused = input.refused.take().as_ref() == Some(&text);
    let typed = Deliver::new(agent, text, DeliveryMode::Steer);
    commands.trigger(if refused { typed.literal() } else { typed });
}

fn stop(_stop: On<Stop>, list: Res<GuiAgents>, mut commands: Commands) {
    if let Some(entity) = list.shown {
        commands.trigger(Interrupt { entity });
    }
}

fn toggle(toggle: On<Toggle>, mut transcript: ResMut<GuiTranscript>) {
    let key = toggle.key;
    let open = !transcript.opened.remove(&key);
    if open {
        transcript.opened.insert(key);
    }
    for block in &mut transcript.blocks {
        match block {
            Block::Reasoning {
                key: at, open: was, ..
            } if *at == key => *was = open,
            Block::Tool(tool) if tool.key == key => tool.open = open,
            _ => {}
        }
    }
}

fn choose(choose: On<Choose>, mut dialogs: ResMut<GuiDialogs>, mut commands: Commands) {
    let Some(dialog) = dialogs.open.pop_front() else {
        return;
    };
    if let Some(item) = dialog.items.get(choose.item) {
        commands.trigger(RunCommand {
            entity: dialog.agent,
            line: item.command.clone(),
        });
    }
}

fn dismiss(_dismiss: On<Dismiss>, mut dialogs: ResMut<GuiDialogs>) {
    if !dialogs.open.is_empty() {
        dialogs.open.pop_front();
    }
}

#[cfg(test)]
mod tests;
