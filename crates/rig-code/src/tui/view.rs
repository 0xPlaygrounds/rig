//! View state, kept apart from the agent core: the focused agent, the input
//! editor and its completion, scrolling, the open overlay and recent
//! notices.

use bevy_ecs::prelude::*;
use rig_core::completion::Reasoning;

use super::complete::Completion;
use super::editor::Editor;
use crate::core::agent::{
    Agent, AgentId, CallOf, Connection, Conversation, Focus, Notice, NoticeLevel, PickKind,
    PickRequest, TurnOf,
};
use crate::core::approval::AwaitingApproval;
use crate::core::inbox::Recalled;
use crate::core::login::LoginProvider;
use crate::core::models;
use crate::core::rewind::{self, History};
use crate::core::save::SessionPaths;
use crate::core::subagents::{self, RosterQuery, SubagentOf};
use crate::host::reload::ReloadFailed;
use crate::host::sessions;

/// Notices kept for display.
const KEPT_NOTICES: usize = 50;

/// The terminal view's state. Never saved and never read by the core.
#[derive(Resource, Default)]
pub(crate) struct TuiView {
    /// The agent shown and typed to.
    pub(crate) agent: Option<Entity>,
    /// The input being typed, and the prompt history.
    pub(crate) editor: Editor,
    /// The open `/` or `@` completion list.
    pub(crate) completion: Option<Completion>,
    /// Where the token starts whose completion was closed with Esc.
    pub(crate) dismissed: Option<usize>,
    /// Lines scrolled up from the bottom of the transcript.
    pub(crate) scroll: usize,
    /// What is shown over the transcript and takes the keys.
    pub(crate) overlay: Option<Overlay>,
    /// Recent notices, oldest first.
    pub(crate) notices: Vec<ShownNotice>,
}

/// What is shown over the transcript, one at a time.
pub(crate) enum Overlay {
    /// A list to pick from.
    Picker(Picker),
    /// A failed rebuild's output, until Esc or Enter.
    ReloadFailure(String),
    /// A tool call of the shown agent waiting for the user's answer.
    Approval(ApprovalPrompt),
}

/// The choices of an [`ApprovalPrompt`], in order.
pub(crate) const APPROVAL_CHOICES: usize = 3;
/// The choice that refuses the call.
pub(crate) const DENY_CHOICE: usize = 2;

/// A tool call waiting for an answer: Yes, Yes and stop asking, or No
/// with what to tell the model.
pub(crate) struct ApprovalPrompt {
    /// The call entity.
    pub(crate) call: Entity,
    /// Its agent.
    pub(crate) agent: Entity,
    /// The tool.
    pub(crate) tool: String,
    /// What the call is about.
    pub(crate) subject: String,
    /// The highlighted choice.
    pub(crate) selected: usize,
    /// What to tell the model when refusing.
    pub(crate) reason: String,
}

/// A notice placed in a transcript.
pub(crate) struct ShownNotice {
    /// The agent it is about, or `None` for every agent.
    agent: Option<Entity>,
    /// The length of that agent's conversation (the focused one's, for an
    /// app notice) when it arrived; it is drawn after that many messages.
    pub(crate) after: usize,
    /// Whether it reports a failure.
    pub(crate) level: NoticeLevel,
    /// The text.
    pub(crate) text: String,
}

impl ShownNotice {
    /// Whether it belongs in `agent`'s transcript.
    pub(crate) fn is_for(&self, agent: Option<Entity>) -> bool {
        self.agent.is_none() || self.agent == agent
    }
}

/// What choosing a picker item sets.
#[derive(Clone, Debug)]
pub(crate) enum PickValue {
    /// A model reference.
    Model(String),
    /// A reasoning setting.
    Effort(Option<Reasoning>),
    /// A session id.
    Session(String),
    /// An agent to show.
    Agent(Entity),
    /// A checkpoint to go back to, and whether the files go back too.
    Rewind {
        /// The checkpoint's effect id.
        to: u64,
        /// Whether the files go back too.
        files: bool,
    },
    /// A checkpoint to fork at, or `None` for now.
    Fork(Option<u64>),
}

/// A filterable list to choose one item from.
pub(crate) struct Picker {
    /// The agent the choice is for.
    pub(crate) agent: Entity,
    /// The title.
    pub(crate) title: String,
    /// Every item: its label and value.
    items: Vec<(String, PickValue)>,
    /// The filter typed so far.
    pub(crate) filter: String,
    /// The selected position among the visible items.
    pub(crate) selected: usize,
}

impl Picker {
    /// The items whose label holds every word of the filter, ignoring case.
    pub(crate) fn visible(&self) -> Vec<&(String, PickValue)> {
        let filter = self.filter.to_lowercase();
        self.items
            .iter()
            .filter(|(label, _)| {
                let label = label.to_lowercase();
                filter.split_whitespace().all(|word| label.contains(word))
            })
            .collect()
    }

    /// The selected item's value.
    pub(crate) fn chosen(&self) -> Option<PickValue> {
        self.visible()
            .get(self.selected)
            .map(|(_, value)| value.clone())
    }
}

/// Focuses the first agent by id when the focused one is gone, one the
/// user started before any subagent.
pub(crate) fn focus_agent(
    mut view: ResMut<TuiView>,
    agents: Query<(Entity, &AgentId, Has<SubagentOf>), With<Agent>>,
) {
    if view.agent.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    view.agent = agents
        .iter()
        .min_by(|a, b| (a.2, &a.1.0).cmp(&(b.2, &b.1.0)))
        .map(|(entity, ..)| entity);
}

/// Shows the agent a [`Focus`] names.
pub(crate) fn on_focus(
    focus: On<Focus>,
    agents: Query<(), With<Agent>>,
    mut view: ResMut<TuiView>,
) {
    if !agents.contains(focus.entity) || view.agent == Some(focus.entity) {
        return;
    }
    view.agent = Some(focus.entity);
    view.scroll = 0;
    view.completion = None;
}

/// Opens the picker a command asked for.
pub(crate) fn open_pickers(
    mut requests: MessageReader<PickRequest>,
    agents: Query<&Connection>,
    histories: Query<(&Conversation, &History)>,
    roster: RosterQuery,
    paths: Option<Res<SessionPaths>>,
    mut view: ResMut<TuiView>,
    mut notices: MessageWriter<Notice>,
) {
    for request in requests.read() {
        let mut selected = 0;
        let current = agents
            .get(request.agent)
            .ok()
            .map(|connection| connection.spec);
        let (title, items) = match request.kind {
            PickKind::Model => {
                let items: Vec<(String, PickValue)> = models::available_models()
                    .into_iter()
                    .map(|spec| {
                        let reference = models::reference(spec);
                        let note = match LoginProvider::of(spec) {
                            Some(plan) => format!("  ({} plan)", plan.title()),
                            None if spec.provider.requires_credential() => String::new(),
                            None => "  (no key needed)".to_owned(),
                        };
                        (
                            format!("{reference}  {}{note}", spec.display_name),
                            PickValue::Model(reference),
                        )
                    })
                    .collect();
                if items.is_empty() {
                    notices.write(Notice::error(
                        request.agent,
                        "No provider with tool-calling models can be reached: set a key such \
                         as OPENAI_API_KEY, or sign in with /login chatgpt.",
                    ));
                    continue;
                }
                ("Model".to_owned(), items)
            }
            PickKind::Effort => {
                let Some(spec) = current else {
                    continue;
                };
                let items = models::effort_options(spec)
                    .into_iter()
                    .map(|option| (option.label(), PickValue::Effort(option.1)))
                    .collect();
                (format!("Reasoning for {}", spec.display_name), items)
            }
            PickKind::Session => {
                let Some(paths) = &paths else {
                    continue;
                };
                let items: Vec<(String, PickValue)> =
                    sessions::list(&rig::code_protocol::Home::from_env(), paths.path())
                        .into_iter()
                        .map(|session| {
                            (session.label(), PickValue::Session(session.id.to_string()))
                        })
                        .collect();
                if items.is_empty() {
                    notices.write(Notice::info(request.agent, "No earlier session to resume."));
                    continue;
                }
                ("Resume a session".to_owned(), items)
            }
            PickKind::Agent => {
                let entries = subagents::roster(&roster);
                selected = entries
                    .iter()
                    .position(|entry| Some(entry.agent) == view.agent)
                    .unwrap_or(0);
                let items = entries
                    .into_iter()
                    .enumerate()
                    .map(|(index, entry)| {
                        (
                            format!("{}. {}", index + 1, entry.label),
                            PickValue::Agent(entry.agent),
                        )
                    })
                    .collect();
                ("Show an agent".to_owned(), items)
            }
            PickKind::Rewind { files } => {
                let points = histories
                    .get(request.agent)
                    .map(|(conversation, history)| rewind::points(conversation, history))
                    .unwrap_or_default();
                let items = points
                    .into_iter()
                    .enumerate()
                    .map(|(index, point)| {
                        (
                            format!("{}. {}", index + 1, point.label),
                            PickValue::Rewind {
                                to: point.effect,
                                files,
                            },
                        )
                    })
                    .collect();
                let title = if files {
                    "Rewind to (conversation and files)"
                } else {
                    "Rewind the conversation to (files stay)"
                };
                (title.to_owned(), items)
            }
            PickKind::Fork => {
                let points = histories
                    .get(request.agent)
                    .map(|(conversation, history)| rewind::points(conversation, history))
                    .unwrap_or_default();
                let items = std::iter::once((
                    "now: the whole conversation".to_owned(),
                    PickValue::Fork(None),
                ))
                .chain(points.into_iter().enumerate().map(|(index, point)| {
                    (
                        format!("{}. {}", index + 1, point.label),
                        PickValue::Fork(Some(point.effect)),
                    )
                }))
                .collect();
                ("Fork a new agent at".to_owned(), items)
            }
        };
        view.overlay = Some(Overlay::Picker(Picker {
            agent: request.agent,
            title,
            items,
            filter: String::new(),
            selected,
        }));
    }
}

/// Asks about the shown agent's tool calls that wait for an answer, one at
/// a time once nothing else is open, and closes the question when its
/// call no longer waits: answered elsewhere, or stopped.
pub(crate) fn open_approvals(
    mut view: ResMut<TuiView>,
    waiting: Query<(Entity, &AwaitingApproval, &CallOf)>,
    turns: Query<&TurnOf>,
) {
    if let Some(Overlay::Approval(prompt)) = &view.overlay
        && !waiting.contains(prompt.call)
    {
        view.overlay = None;
    }
    let Some(agent) = view.agent else {
        return;
    };
    if view.overlay.is_some() {
        return;
    }
    let first = waiting
        .iter()
        .filter(|&(_, _, &CallOf(turn))| turns.get(turn).is_ok_and(|&TurnOf(of)| of == agent))
        .min_by_key(|(call, ..)| *call);
    if let Some((call, waiting, _)) = first {
        view.overlay = Some(Overlay::Approval(ApprovalPrompt {
            call,
            agent,
            tool: waiting.tool.clone(),
            subject: waiting.subject.clone(),
            selected: 0,
            reason: String::new(),
        }));
    }
}

/// Shows the output of a failed `/reload` until it is dismissed.
pub(crate) fn show_reload_failures(
    mut failures: MessageReader<ReloadFailed>,
    mut view: ResMut<TuiView>,
) {
    if let Some(failure) = failures.read().last() {
        view.overlay = Some(Overlay::ReloadFailure(failure.output.clone()));
    }
}

/// Keeps the latest notices for display, each placed after the messages
/// its agent had when it arrived.
pub(crate) fn collect_notices(
    mut notices: MessageReader<Notice>,
    conversations: Query<&Conversation>,
    mut view: ResMut<TuiView>,
) {
    if notices.is_empty() {
        return;
    }
    for notice in notices.read() {
        let after = notice
            .agent
            .or(view.agent)
            .and_then(|agent| conversations.get(agent).ok())
            .map_or(0, |conversation| conversation.0.len());
        view.notices.push(ShownNotice {
            agent: notice.agent,
            after,
            level: notice.level,
            text: notice.text.clone(),
        });
    }
    let excess = view.notices.len().saturating_sub(KEPT_NOTICES);
    view.notices.drain(..excess);
}

/// Puts the messages a turn did not send back in the input, before what
/// is typed there now.
pub(crate) fn recall_messages(mut recalled: MessageReader<Recalled>, mut view: ResMut<TuiView>) {
    for recalled in recalled.read() {
        if view.agent != Some(recalled.agent) {
            continue;
        }
        let typed = view.editor.take();
        let text = if typed.trim().is_empty() {
            recalled.text.clone()
        } else {
            format!("{}\n\n{typed}", recalled.text)
        };
        view.editor.set(text);
    }
}
