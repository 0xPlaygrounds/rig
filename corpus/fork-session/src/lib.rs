//! Plugin corpus: 43. Fork and branch sessions (spec `43-fork-session.md`).
//!
//! `/fork [name]` copies the running session into a new one with the
//! harness's [`ForkSession`] and switches to it. `/branch` picks one of the
//! shown agent's own messages and forks the session as of just before it
//! (the kernel cuts every log there, so saved plugin state comes back as it
//! was then); the new session starts with that message in the input line.
//! The harness records each fork's parent, fork point and the text of the
//! message it was branched before ([`ForkedFrom`]); `/resume` shows it,
//! and `/tree` lists the family in the harness's [`SessionPicker`]. Files
//! on disk are never branched, except that `/branch --files` (with the
//! `checkpoints` feature and spec 31's plugin) puts them back as that
//! plugin's checkpoint before the message kept them, through its `Rewind`
//! request, before switching.

#[cfg(feature = "checkpoints")]
use corpus_checkpoint_rewind::{Rewind, RewindScope, Rewound};
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_sessions::{NameSession, SessionsPlugin};
use rig_tools::{ago, shorten};
use rig_tui::{MainInput, ShowAgent};

/// Most characters of a message in the `/branch` picker.
const LABEL_CHARS: usize = 70;

/// What the user is told after a fork.
const FILES: &str = "Files on disk are not branched: the working tree stays as it is.";

/// `/fork`, `/branch` and `/tree`.
#[derive(Default)]
pub struct ForkSessionPlugin;

impl Plugin for ForkSessionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionsPlugin>() {
            app.add_plugins(SessionsPlugin);
        }
        app.add_message::<SessionForked>()
            .add_command(
                "fork",
                "Copy this session into a new one and switch to it: /fork [name]",
                fork,
            )
            .add_command(
                "branch",
                "Pick a message of yours to start a new session before it [--files]",
                branch,
            )
            .add_command("tree", "Show this session's forks and branches", tree)
            .add_systems(Startup, arrive)
            .add_systems(Update, forked);
        #[cfg(feature = "checkpoints")]
        app.add_observer(rewound);
    }
}

/// A fork this plugin asked for, until the harness wrote it: the
/// [`ForkSession::asker`] that [`SessionForked`] names.
#[derive(Component)]
struct Asked {
    agent: Entity,
    /// The name `/fork` gave it.
    name: Option<String>,
    /// The message to put the files back to before, for `/branch --files`.
    files: Option<MessageId>,
}

/// The messages the user typed in `conversation`, as [`ForkedFrom::text`]
/// keeps them.
fn typed_messages(conversation: &Conversation) -> Vec<(MessageId, String)> {
    (0..conversation.messages().len())
        .filter_map(|at| Some((conversation.id(at)?, typed(conversation, at)?)))
        .collect()
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Why nothing can be forked now, if so.
fn forkable(
    agent: Entity,
    turns: &Query<(), With<TurnOf>>,
    paths: Option<&SessionPaths>,
) -> Result<(), Notice> {
    if !turns.is_empty() {
        return Err(Notice::turn_running(agent));
    }
    match paths {
        Some(paths) if paths.is_saved() => Ok(()),
        _ => Err(Notice::error(
            agent,
            "Nothing of this session is saved yet.",
        )),
    }
}

/// `/fork [name]`.
fn fork(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if let Err(refused) = forkable(args.agent, &turns, paths.as_deref()) {
        notices.write(refused);
        return;
    }
    let name = args.args.trim();
    let asker = commands
        .spawn(Asked {
            agent: args.agent,
            name: (!name.is_empty()).then(|| name.to_owned()),
            files: None,
        })
        .id();
    commands.write_message(ForkSession {
        asker: Some(asker),
        ..ForkSession::default()
    });
}

/// `/branch [--files]`: a picker of the agent's typed messages.
fn branch(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    conversations: Query<&Conversation>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if let Err(refused) = forkable(agent, &turns, paths.as_deref()) {
        notices.write(refused);
        return;
    }
    let files = match args.args.trim() {
        "" => false,
        "--files" if cfg!(feature = "checkpoints") => true,
        "--files" => {
            let why = "/branch --files needs this plugin built with its `checkpoints` feature.";
            notices.write(Notice::error(agent, why));
            return;
        }
        other => {
            notices.write(Notice::error(agent, format!("Unknown option `{other}`.")));
            return;
        }
    };
    let typed = conversations
        .get(agent)
        .map(typed_messages)
        .unwrap_or_default();
    if typed.is_empty() {
        notices.write(Notice::info(agent, "No message of yours to branch from."));
        return;
    }
    let picker = typed.iter().fold(
        Picker::new("Branch a new session before this message").for_agent(agent),
        |picker, (_, text)| picker.item(shorten(&one_line(text), LABEL_CHARS), ""),
    );
    let selected = picker.items.len().saturating_sub(1);
    commands.spawn(Picker { selected, ..picker }).observe(
        move |picked: On<Picked>,
              ids: Query<&AgentId>,
              snapshots: Snapshots<'_, '_>,
              mut commands: Commands,
              mut notices: MessageWriter<Notice>| {
            let Some(&index) = picked
                .chosen
                .as_ref()
                .and_then(|chosen| chosen.items.first())
            else {
                return;
            };
            let (Some(&(message, _)), Ok(id)) = (typed.get(index), ids.get(agent)) else {
                return;
            };
            let files = match files.then(|| restore_files(&snapshots, agent, message)) {
                Some(Err(why)) => {
                    notices.write(Notice::error(agent, why));
                    return;
                }
                Some(Ok(message)) => Some(message),
                None => None,
            };
            let asker = commands
                .spawn(Asked {
                    agent,
                    name: None,
                    files,
                })
                .id();
            commands.write_message(ForkSession {
                from: None,
                cut: ForkCut::Before(id.clone(), message),
                asker: Some(asker),
            });
        },
    );
}

/// The checkpoints spec 31's plugin keeps with each agent.
#[cfg(feature = "checkpoints")]
type Snapshots<'w, 's> = Query<'w, 's, &'static corpus_checkpoint_rewind::Checkpoints>;
#[cfg(not(feature = "checkpoints"))]
type Snapshots<'w, 's> = ();

/// `message`, when spec 31's plugin kept the files before it.
#[cfg(feature = "checkpoints")]
fn restore_files(
    snapshots: &Snapshots<'_, '_>,
    agent: Entity,
    message: MessageId,
) -> Result<MessageId, String> {
    snapshots
        .get(agent)
        .ok()
        .and_then(|kept| kept.list.iter().find(|kept| kept.message == message))
        .map(|checkpoint| checkpoint.message)
        .ok_or_else(|| "No checkpoint of the files before that message.".to_owned())
}

#[cfg(not(feature = "checkpoints"))]
fn restore_files(_: &Snapshots<'_, '_>, _: Entity, _: MessageId) -> Result<MessageId, String> {
    Err("/branch --files needs spec 31's checkpoints.".to_owned())
}

/// On a [`Rewind`] of the files: the session to switch to once it ended.
#[cfg(feature = "checkpoints")]
#[derive(Component)]
struct Switching(SessionId);

/// Puts the files back as they were before `message`, then switches to
/// `to`.
#[cfg(feature = "checkpoints")]
fn switch_after_files(commands: &mut Commands, agent: Entity, message: MessageId, to: SessionId) {
    let scope = RewindScope::FilesOnly;
    commands.spawn((
        Rewind {
            agent,
            message,
            scope,
        },
        Switching(to),
    ));
}

#[cfg(not(feature = "checkpoints"))]
fn switch_after_files(commands: &mut Commands, _: Entity, _: MessageId, to: SessionId) {
    commands.trigger(Exit(ExitReason::Switch(Some(to))));
}

/// Switches once the files are back, or once the user kept theirs or
/// the rewind was refused (spec 31's plugin told them which).
#[cfg(feature = "checkpoints")]
fn rewound(
    ended: On<Add<Rewound>>,
    rewinds: Query<(&Rewind, &Rewound, &Switching)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((rewind, outcome, Switching(to))) = rewinds.get(ended.entity) else {
        return;
    };
    if *outcome != Rewound::Done {
        notices.write(Notice::info(rewind.agent, FILES));
    }
    commands.trigger(Exit(ExitReason::Switch(Some(to.clone()))));
    commands.entity(ended.entity).despawn();
}

/// Names each fork this plugin asked for, then switches to it; forgets
/// one that failed (the harness told the user why).
fn forked(
    (mut forks, mut failed): (
        MessageReader<SessionForked>,
        MessageReader<SessionForkFailed>,
    ),
    mut asked: Query<&mut Asked>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for asker in failed.read().filter_map(|failed| failed.asker) {
        if asked.contains(asker) {
            commands.entity(asker).despawn();
        }
    }
    for fork in forks.read() {
        let Some(asker) = fork.asker else {
            continue;
        };
        let Ok(mut asked) = asked.get_mut(asker) else {
            continue;
        };
        commands.entity(asker).despawn();
        if let Some(name) = asked.name.take() {
            commands.trigger(NameSession {
                session: fork.to.clone(),
                name,
            });
        }
        let to = fork.to.clone();
        if let Some(message) = asked.files.take() {
            let made = format!("Made session {to}; putting the files back first.");
            notices.write(Notice::info(asked.agent, made));
            switch_after_files(&mut commands, asked.agent, message, to);
        } else {
            notices.write(Notice::info(
                asked.agent,
                format!("Made session {to}. {FILES}"),
            ));
            commands.trigger(Exit(ExitReason::Switch(Some(to))));
        }
    }
}

/// How a session came from its parent: `fork`, `split`, `branch`, or
/// `branch before “text”`.
fn how(parent: &ForkedFrom) -> String {
    match (&parent.before, &parent.text) {
        (None, _) if parent.through.is_some() => "split".to_owned(),
        (None, _) => "fork".to_owned(),
        (Some(_), None) => "branch".to_owned(),
        (Some(_), Some(text)) => format!("branch before “{}”", shorten(&one_line(text), 40)),
    }
}

/// A branch's first start puts the message it was branched before in the
/// input line, showing its agent.
fn arrive(
    store: Option<Res<SessionStore>>,
    started: Option<Res<AppStart>>,
    agents: MainAgentQuery,
    input: Option<MainInput>,
    mut commands: Commands,
) {
    if started.is_none_or(|started| started.0 != StartReason::New) {
        return;
    }
    let parent = store.and_then(|store| ForkedFrom::read(&*store.0));
    let Some(ForkedFrom {
        before: Some((agent, _)),
        text: Some(text),
        ..
    }) = parent
    else {
        return;
    };
    let agent = agents.iter().find(|(_, id, _)| id.0 == agent);
    if let Some(entity) = agent
        .map(|(agent, ..)| agent)
        .or_else(|| main_agent(&agents))
    {
        commands.trigger(ShowAgent { entity });
    }
    if let Some(mut input) = input {
        input.set(text);
    }
}

/// `/tree`: this session's family (those it was forked or branched from,
/// and those forked or branched from them), each with how it came from
/// its parent, as the harness's session picker; choosing one resumes it.
fn tree(In(args): In<CommandArgs>, paths: Option<Res<SessionPaths>>, mut commands: Commands) {
    let Some(current) = paths.and_then(|paths| paths.id()) else {
        return;
    };
    commands.spawn((
        Picker::new("This session's family").for_agent(args.agent),
        SessionPicker {
            filter: SessionFilter {
                running: true,
                family: Some(current),
                ..SessionFilter::default()
            },
            command: "resume {session}".to_owned(),
            show: Some(family_member),
            ..SessionPicker::default()
        },
    ));
}

/// A session of the family: its label, and when it changed and how it
/// came from which session.
fn family_member(item: &SessionItem<'_>) -> (String, Option<String>) {
    let SessionItem::Session(session) = item else {
        return (String::new(), None);
    };
    let ago = ago(session.updated.elapsed().unwrap_or_default());
    let detail = match &session.listing.parent {
        Some(parent) => format!("{ago} · {} of {}", how(parent), parent.session),
        None => ago,
    };
    (session.label(), Some(detail))
}

#[cfg(test)]
mod tests;
