//! A window front: the agents on the left with their status and subagents
//! indented, the shown agent's transcript in the middle (markdown answers,
//! collapsible reasoning and tool calls, the reply streaming in), an input
//! box under it, and every open question ([`Ask`]) and picker
//! ([`Picker`]) as a dialog with buttons, answered with [`Pick`].
//!
//! What the window shows is its model, four resources kept by systems that
//! need no window: [`GuiAgents`], [`GuiTranscript`], [`GuiInput`] and
//! [`GuiDialogs`]. The window, or a test, changes it only through the
//! requests [`Show`], [`Submit`], [`Stop`], [`Toggle`], [`Choose`] and
//! [`Dismiss`], which become the same requests the terminal view sends.
//! The agent shown is the harness's [`ShownAgent`], so the window and the
//! terminal view beside it show the same agent.
//! With the default `window` feature, a Bevy UI window draws the model,
//! redrawing a part only when its resource changed.
//!
//! `rig --gui` makes the window the run's front ([`FRONT`]); in an
//! interactive run without it, the window opens beside the terminal view,
//! and closing it leaves the agent running.

#[cfg(feature = "window")]
mod window;

use std::collections::HashSet;

use rig_basics::activity::{Activity, ActivityPlugin, ActivityStatus, ActivitySystems};
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{Message, ToolResult as CallResult};
use rig_harness::rig_memory::main_order;

/// The window's name as a front, and its option: `rig --gui`.
pub const FRONT: &str = "gui";

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
            .init_resource::<ShownAgent>()
            .add_front(Front::new(FRONT).on_flag(FRONT, "Run the agent in a window"))
            .configure_sets(PostUpdate, GuiSystems.after(ActivitySystems))
            .add_systems(Update, (collect_notices, recall))
            .add_systems(
                PostUpdate,
                (list_agents, show_transcript, stream_reply, list_dialogs)
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

/// The agent list, subagents under their parent.
#[derive(Resource, Default, Debug)]
pub struct GuiAgents {
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
    /// Where a stop cut a turn off, under its last message.
    Stopped {
        /// Whether that message is a reply, which the stop cut short;
        /// otherwise nothing answered the request.
        replied: bool,
    },
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
                Block::Stopped { .. } => false,
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

/// A question or a picker for the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dialog {
    /// The [`Ask`] or [`Picker`] entity.
    pub entity: Entity,
    /// What is asked.
    pub title: String,
    /// More about it; may be empty.
    pub body: String,
    /// A button per choice.
    pub choices: Vec<String>,
    /// Whether what is typed in the input answers it too.
    pub free_text: bool,
}

/// The open dialogs: pickers, then questions in the order they were asked.
/// The first is shown.
#[derive(Resource, Default, Debug)]
pub struct GuiDialogs {
    /// The dialogs.
    pub open: Vec<Dialog>,
}

/// Show `agent` and send what is typed to it: it becomes [`ShownAgent`].
#[derive(Event, Clone, Copy, Debug)]
pub struct Show {
    /// The agent.
    pub agent: Entity,
}

/// Send the input to the shown agent as the user's message, as the
/// terminal view's Enter does: a slash command, a new turn, steering for
/// the running one, or the answer to its open question that takes text.
/// With a picker that takes text shown, the input answers it instead.
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

/// Choose a button of the shown dialog: answer the question, or run the
/// picker item's command.
#[derive(Event, Clone, Copy, Debug)]
pub struct Choose {
    /// The item's position.
    pub item: usize,
}

/// Close the shown dialog without choosing: a question is refused.
#[derive(Event, Clone, Copy, Debug)]
pub struct Dismiss;

/// What the agent does, for its row.
fn status_text(activity: &Activity) -> String {
    match &activity.status {
        ActivityStatus::RunningTools if !activity.tools.is_empty() => {
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

/// Keeps the agent list.
fn list_agents(agents: Query<AgentItem, With<Agent>>, mut list: ResMut<GuiAgents>) {
    let mut roots: Vec<_> = agents.iter().filter(|agent| !agent.4).collect();
    roots.sort_by(|a, b| main_order(a.4, &a.2.0).cmp(&main_order(b.4, &b.2.0)));
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
    if list.rows != rows {
        list.rows = rows;
    }
}

/// Keeps the shown agent's transcript as blocks, rebuilt when its
/// conversation changed.
fn show_transcript(
    shown: Res<ShownAgent>,
    conversations: Query<Ref<Conversation>>,
    mut transcript: ResMut<GuiTranscript>,
) {
    let shown = shown
        .0
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

/// The reply streaming in to the shown agent: what its turn's model call
/// added up so far.
fn stream_reply(
    shown: Res<ShownAgent>,
    partials: Query<(&Partial, &ForAgent), (With<TurnCall>, Without<Settled>)>,
    mut transcript: ResMut<GuiTranscript>,
) {
    let partial = partials
        .iter()
        .find_map(|(partial, &ForAgent(agent))| (shown.0 == Some(agent)).then_some(partial));
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
/// message holds, and a stop follows the last message of the turn it cut
/// off, as its turn record says.
fn blocks_of(conversation: &Conversation, opened: &HashSet<BlockKey>) -> Vec<Block> {
    let messages = conversation.messages();
    let stops: HashSet<MessageId> = conversation
        .turns()
        .filter_map(|(_, record)| match record.end {
            Some(TurnEnd::Stopped { at, .. }) => at,
            _ => None,
        })
        .collect();
    let mut blocks = Vec::new();
    for entry in conversation.entries() {
        let index = entry.index;
        match entry.message {
            Message::System { .. } => {}
            // The note a stop leaves for the model.
            Message::User { .. } if entry.is_stop_marker() => {}
            Message::User { content } => {
                for part in content {
                    match part {
                        UserContent::Text(text) => match conversation.author(index) {
                            Some(author) => blocks.push(Block::Delivered {
                                from: author.label(),
                                text: author
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
        if stops.contains(&entry.id) {
            let replied = matches!(entry.message, Message::Assistant(_));
            blocks.push(Block::Stopped { replied });
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

/// Keeps the open pickers and questions as dialogs.
fn list_dialogs(
    pickers: Query<(Entity, &Picker)>,
    asks: Query<(Entity, &Ask, &Effect), (Without<Outcome>, Without<Settled>)>,
    mut dialogs: ResMut<GuiDialogs>,
) {
    let mut asked: Vec<_> = asks.iter().collect();
    asked.sort_by_key(|(.., effect)| effect.id());
    let open: Vec<Dialog> = pickers
        .iter()
        .map(|(entity, picker)| Dialog {
            entity,
            title: picker.title.clone(),
            body: picker.body.clone(),
            choices: picker.items.iter().map(|item| item.label.clone()).collect(),
            free_text: picker.free_text,
        })
        .chain(asked.into_iter().map(|(entity, ask, _)| {
            Dialog {
                entity,
                title: ask.title.clone(),
                body: ask.body.clone(),
                choices: ask
                    .choices
                    .iter()
                    .map(|choice| choice.label.clone())
                    .collect(),
                free_text: ask.free_text,
            }
        }))
        .collect();
    if dialogs.open != open {
        dialogs.open = open;
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
    shown: Res<ShownAgent>,
    mut input: ResMut<GuiInput>,
    mut transcript: ResMut<GuiTranscript>,
) {
    for recalled in recalled.read() {
        if shown.0 != Some(recalled.agent) {
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

fn show(show: On<Show>, agents: Query<(), With<Agent>>, mut shown: ResMut<ShownAgent>) {
    if agents.contains(show.agent) {
        shown.set_if_neq(ShownAgent(Some(show.agent)));
    }
}

fn submit(
    _submit: On<Submit>,
    shown: Res<ShownAgent>,
    dialogs: Res<GuiDialogs>,
    pickers: Query<&Picker>,
    mut input: ResMut<GuiInput>,
    mut commands: Commands,
) {
    let Some(agent) = shown.0 else {
        return;
    };
    if input.text.trim().is_empty() {
        return;
    }
    let text = std::mem::take(&mut input.text);
    if let Some(entity) = dialogs.open.first().map(|dialog| dialog.entity)
        && pickers.get(entity).is_ok_and(|picker| picker.free_text)
    {
        let chosen = Some(Chosen::text(text.trim().to_owned()));
        commands.trigger(Pick::person(entity, chosen));
        return;
    }
    let refused = input.refused.take().as_ref() == Some(&text);
    let typed = Deliver::new(agent, text, DeliveryMode::Steer);
    commands.trigger(if refused { typed.literal() } else { typed });
}

fn stop(_stop: On<Stop>, shown: Res<ShownAgent>, mut commands: Commands) {
    if let Some(entity) = shown.0 {
        commands.trigger(StopTurn::new(entity));
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

fn choose(choose: On<Choose>, dialogs: Res<GuiDialogs>, mut commands: Commands) {
    if let Some(dialog) = dialogs.open.first() {
        let chosen = Some(Chosen::item(choose.item));
        commands.trigger(Pick::person(dialog.entity, chosen));
    }
}

fn dismiss(_dismiss: On<Dismiss>, dialogs: Res<GuiDialogs>, mut commands: Commands) {
    if let Some(dialog) = dialogs.open.first() {
        commands.trigger(Pick::person(dialog.entity, None));
    }
}

#[cfg(test)]
mod tests;
