//! Splits of a conversation, built only on the harness's, rig-core's,
//! rig-memory's and rig-tui's public API: `/split` at any entry into a
//! branch of the session or into a new session, `/split move` taking a
//! range out and leaving a pointer behind, `/splits` listing the lines of
//! any saved session in the native session picker, and `/split switch`,
//! `/split compare` and `/split merge` for any line, a branch kept in the
//! session or a split kept in another.
//! It adds no event of its own: other plugins split through the same
//! primitives ([`ForkSession`] with [`ForkCut::Line`], rig-core's
//! `branch_off`, `move_out`, `merge_branch` and `merge_line`), or run its
//! commands with [`RunCommand`].
//!
//! One grammar names where: `--from <session>` (this one by default),
//! `--agent <agent>` (the one the command is for), `--line <line>` (the
//! active one) and `--at <message>` (the line's newest entry), a message
//! named by its plain id (`m4`). A line is named by its aside (`a3`), the
//! session a split of it is kept in, or its title.
//!
//! ```text
//! /split [new] [<where>] [title]
//! /split move <first> <last> [new] [<where>] [title]
//! /split switch [<line>] [<where>]
//! /split compare <line> [<where>]
//! /split merge <line> [<where>]
//! /splits [session]
//! ```

use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message as ChatMessage;
use rig_harness::rig_core::conversation::{self, BranchError, MergeAt, Spliced, shared_prefix};
use rig_harness::rig_core::message::UserContent;
use rig_harness::rig_memory::{self, LineId, SplitPointer, lines};

#[cfg(test)]
mod tests;

/// The `/split` and `/splits` commands.
#[derive(Default)]
pub struct SplitsPlugin;

impl Plugin for SplitsPlugin {
    fn build(&self, app: &mut App) {
        let asker = app.world_mut().spawn(Name::new("rig-splits")).id();
        app.insert_resource(Asker(asker))
            .add_command(
                "split",
                "Split the conversation at an entry into a branch, or with `new` a new session; \
                 /split move <first> <last>, /split switch|compare|merge <line>",
                split,
            )
            .add_command(
                "splits",
                "List the lines of this or a named saved session; picking one switches to it",
                splits,
            )
            .add_systems(Last, go_to_split.after(ForkSessions));
    }
}

/// The entity this plugin asks for forks as, so it knows its own.
#[derive(Resource)]
struct Asker(Entity);

/// Who makes the plugin's changes.
fn by() -> Author {
    Author::plugin("splits")
}

/// What `/split` was asked to do.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    /// Split at the point, into a branch or a new session.
    Split { new: bool },
    /// Move `first..=last` out, into a branch or a new session.
    Move {
        first: MessageId,
        last: MessageId,
        new: bool,
    },
    /// Make a line the active one; `None` names the active line, which
    /// in another session runs it.
    Switch(Option<String>),
    /// Show a line beside the active one.
    Compare(String),
    /// Bring a line back into the active one.
    Merge(String),
}

/// Where: a session, an agent, a line and an entry, each optional.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Where {
    from: Option<String>,
    agent: Option<String>,
    line: Option<String>,
    at: Option<MessageId>,
}

/// A `/split` line read.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Asked {
    action: Action,
    place: Where,
    title: String,
}

/// Reads the arguments of `/split`.
fn parse(args: &str) -> Result<Asked, String> {
    let mut words = args.split_whitespace().peekable();
    let message = |word: Option<&str>| -> Result<MessageId, String> {
        let word = word.ok_or("name the messages, such as `m4 m9`")?;
        word.parse()
            .map_err(|_| format!("`{word}` is not a message id, such as `m4`"))
    };
    let named = |word: Option<&str>| -> Result<String, String> {
        word.map(str::to_owned)
            .ok_or_else(|| "name the line, such as `a3`, a session or a title".to_owned())
    };
    let mut action = match words.peek().copied() {
        Some("move") => {
            words.next();
            let first = message(words.next())?;
            let last = message(words.next())?;
            Action::Move {
                first,
                last,
                new: false,
            }
        }
        Some("switch") => {
            words.next();
            let line = words.next_if(|word| !word.starts_with("--"));
            Action::Switch(line.map(str::to_owned))
        }
        Some("compare") => {
            words.next();
            Action::Compare(named(words.next())?)
        }
        Some("merge") => {
            words.next();
            Action::Merge(named(words.next())?)
        }
        _ => Action::Split { new: false },
    };
    let mut place = Where::default();
    let mut title = Vec::new();
    while let Some(word) = words.next() {
        let mut value = || {
            words
                .next()
                .map(str::to_owned)
                .ok_or_else(|| format!("`{word}` needs a value"))
        };
        match word {
            "--from" => place.from = Some(value()?),
            "--agent" => place.agent = Some(value()?),
            "--line" => place.line = Some(value()?),
            "--at" => place.at = Some(message(Some(&value()?))?),
            "new" if title.is_empty() => match &mut action {
                Action::Split { new } | Action::Move { new, .. } => *new = true,
                Action::Switch(_) | Action::Compare(_) | Action::Merge(_) => title.push(word),
            },
            _ => title.push(word),
        }
    }
    Ok(Asked {
        action,
        place,
        title: title.join(" "),
    })
}

/// `/split`: runs what it was asked and says what it did, or why not.
fn split(In(args): In<CommandArgs>, world: &mut World) {
    let done = parse(&args.args).and_then(|asked| run(world, args.agent, asked));
    let notice = match done {
        Ok(text) => Notice::info(args.agent, text),
        Err(why) => Notice::error(args.agent, why),
    };
    world.write_message(notice);
}

/// The session a command works on, and its agent.
struct Target {
    /// Another saved session; `None` for this one.
    session: Option<SessionId>,
    /// The agent, by id.
    agent: AgentId,
    /// The agent's entity, in this session.
    entity: Option<Entity>,
}

/// Finds the session and agent `place` names, `agent` by default.
fn target(world: &mut World, agent: Entity, place: &Where) -> Result<Target, String> {
    let here = world
        .get_resource::<SessionPaths>()
        .and_then(|paths| paths.id());
    let session = match &place.from {
        None => None,
        Some(name) => {
            let home = home(world);
            let id = home.find(name).map_err(|why| why.to_string())?;
            (Some(&id) != here.as_ref()).then_some(id)
        }
    };
    if let Some(session) = session {
        let read = read_session(world, &session)?;
        let agent = match &place.agent {
            Some(agent) => read.agent(agent).map(|agent| agent.id.clone()),
            None => read.main_agent().map(|agent| agent.id.clone()),
        };
        let agent = agent.ok_or_else(|| format!("session {session} has no such agent"))?;
        return Ok(Target {
            session: Some(session),
            agent: AgentId(agent),
            entity: None,
        });
    }
    let entity = match &place.agent {
        None => Some(agent),
        Some(name) => {
            let mut agents = world.query::<(Entity, &AgentId)>();
            let found = agents.iter(world).find(|(_, id)| id.0 == *name);
            found.map(|(entity, _)| entity)
        }
    };
    let entity = entity.ok_or("there is no such agent")?;
    let agent = world
        .get::<AgentId>(entity)
        .cloned()
        .ok_or("it is not an agent")?;
    Ok(Target {
        session: None,
        agent,
        entity: Some(entity),
    })
}

/// The app's home.
fn home(world: &World) -> Home {
    world
        .get_resource::<RigHome>()
        .map_or_else(Home::from_env, |home| home.0.clone())
}

/// Reads the saved session `id`.
fn read_session(world: &World, id: &SessionId) -> Result<Session, String> {
    let dir = home(world).session(id);
    let stores = world
        .get_resource::<StoreOpener>()
        .cloned()
        .unwrap_or_default();
    let store = stores.open(&dir).map_err(|failure| failure.to_string())?;
    Session::read(&*store).map_err(|failure| failure.to_string())
}

/// The line of `conversation` that `name` names.
fn find_line(
    conversation: &conversation::Conversation,
    name: &str,
) -> Result<rig_memory::Line, String> {
    let listed = lines(conversation);
    let found = rig_memory::Line::find(&listed, name).cloned();
    found.ok_or_else(|| format!("there is no line `{name}`"))
}

/// The kept branch `name` names in `conversation`; `None` for the active
/// line, or no name.
fn branch(
    conversation: &conversation::Conversation,
    name: Option<&str>,
) -> Result<Option<AsideId>, String> {
    let line = name.map(|name| find_line(conversation, name)).transpose()?;
    match line.map(|line| line.id) {
        None | Some(LineId::Active) => Ok(None),
        Some(LineId::Branch(id)) => Ok(Some(id)),
        Some(LineId::Split { session, .. }) => Err(format!(
            "that line is kept in session {session}: /split switch to it first"
        )),
    }
}

/// The newest entry of line `line` of `conversation`.
fn newest(
    conversation: &conversation::Conversation,
    line: Option<AsideId>,
) -> Result<MessageId, String> {
    let entries = conversation.line(line).unwrap_or_default();
    let last = entries.last().and_then(|entry| entry.id);
    last.ok_or_else(|| "the line has no entry to split at".to_owned())
}

/// Runs `asked` for `agent`; what it did.
fn run(world: &mut World, agent: Entity, asked: Asked) -> Result<String, String> {
    let target = target(world, agent, &asked.place)?;
    match asked.action {
        Action::Split { new } | Action::Move { new, .. } if new => fork(world, &target, &asked),
        Action::Split { .. } | Action::Move { .. } => {
            let go = matches!(asked.action, Action::Split { .. });
            edit(world, (&target, go), |conversation| {
                in_session(conversation, &asked)
            })
        }
        Action::Switch(name) => switch(world, &target, name.as_deref()),
        Action::Compare(name) => compare(world, &target, &name),
        Action::Merge(name) => merge(world, &target, &name),
    }
}

/// Splits or moves into a branch of `conversation`, as `asked` says.
fn in_session(
    conversation: &mut conversation::Conversation,
    asked: &Asked,
) -> Result<String, String> {
    let failed = |failure: BranchError| failure.to_string();
    if let Some(line) = branch(conversation, asked.place.line.as_deref())? {
        conversation.switch_branch(line, by()).map_err(failed)?;
    }
    let index = |id: MessageId| {
        conversation
            .index_of(id)
            .ok_or_else(|| format!("message {id} is not on the line"))
    };
    match asked.action {
        Action::Move { first, last, .. } => {
            let range = index(first)?..index(last)?.saturating_add(1);
            let title = match asked.title.as_str() {
                "" => format!("{first}–{last}"),
                title => title.to_owned(),
            };
            conversation
                .move_out(range, title.clone(), by())
                .map_err(failed)?;
            Ok(format!("Moved {first}–{last} out as “{title}”."))
        }
        _ => {
            let at = match asked.place.at {
                Some(at) => at,
                None => newest(conversation, None)?,
            };
            let point = index(at)?.saturating_add(1);
            let title = match asked.title.as_str() {
                "" => format!("split {}", lines(conversation).len()),
                title => title.to_owned(),
            };
            let left = conversation.line_title().to_owned();
            conversation
                .branch_off(point, title.clone(), by())
                .map_err(failed)?;
            Ok(format!(
                "Split after {at}: going on in “{title}”; “{left}” is kept."
            ))
        }
    }
}

/// Edits the conversation of `target`: the live one in this session, a
/// stopped session's through [`edit_stopped`], then, with `go`, runs that
/// session.
fn edit(
    world: &mut World,
    (target, go): (&Target, bool),
    change: impl FnOnce(&mut conversation::Conversation) -> Result<String, String>,
) -> Result<String, String> {
    match (&target.session, target.entity) {
        (None, Some(entity)) => {
            if world.get::<ActiveTurn>(entity).is_some() {
                return Err("A turn is running. Press Esc to stop it first.".to_owned());
            }
            let mut conversation = world
                .get_mut::<Conversation>(entity)
                .ok_or("the agent has no conversation")?;
            change(&mut conversation)
        }
        (Some(session), _) => {
            let paths = world
                .get_resource::<SessionPaths>()
                .cloned()
                .ok_or("this app keeps no session")?;
            let stores = world
                .get_resource::<StoreOpener>()
                .cloned()
                .unwrap_or_default();
            let dir = home(world).session(session);
            let mut said = String::new();
            edit_stopped(&stores, &paths, &dir, |read| {
                let agent = read
                    .agent_mut(&target.agent.0)
                    .ok_or("the session has no such agent")?;
                said = change(&mut agent.conversation)?;
                Ok(())
            })?;
            if !go {
                return Ok(format!("{said} (In session {session}.)"));
            }
            world.trigger(Exit(ExitReason::Switch(Some(session.clone()))));
            Ok(format!("{said} Running session {session}."))
        }
        (None, None) => Err("there is no such agent".to_owned()),
    }
}

/// Splits or moves into a new session: a [`ForkSession`] of the line,
/// which [`go_to_split`] follows.
fn fork(world: &mut World, target: &Target, asked: &Asked) -> Result<String, String> {
    let conversation = match (&target.session, target.entity) {
        (Some(session), _) => read_session(world, session)?
            .agent(&target.agent.0)
            .map(|agent| agent.conversation.clone())
            .ok_or("the session has no such agent")?,
        (None, Some(entity)) => world
            .get::<Conversation>(entity)
            .map(|conversation| conversation.view())
            .ok_or("the agent has no conversation")?,
        (None, None) => return Err("there is no such agent".to_owned()),
    };
    let line = branch(&conversation, asked.place.line.as_deref())?;
    let (through, take, title) = match asked.action {
        Action::Move { first, last, .. } => (last, Some(first), format!("{first}–{last}")),
        _ => {
            let at = match asked.place.at {
                Some(at) => at,
                None => newest(&conversation, line)?,
            };
            (at, None, format!("split {}", lines(&conversation).len()))
        }
    };
    let title = match asked.title.as_str() {
        "" => title,
        given => given.to_owned(),
    };
    let asker = world.get_resource::<Asker>().map(|asker| asker.0);
    world.write_message(ForkSession {
        from: target.session.clone(),
        cut: ForkCut::Line {
            agent: target.agent.clone(),
            line,
            through,
            move_from: take,
            title: title.clone(),
        },
        asker,
    });
    Ok(match take {
        Some(first) => format!("Moving {first}–{through} into a new session “{title}”…"),
        None => format!("Splitting after {through} into a new session “{title}”…"),
    })
}

/// Runs the new session a split made, and says where a move went.
fn go_to_split(
    mut forked: MessageReader<SessionForked>,
    asker: Res<Asker>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for forked in forked.read().filter(|forked| forked.asker == Some(asker.0)) {
        let ForkCut::Line {
            move_from, title, ..
        } = &forked.cut
        else {
            continue;
        };
        if let Some(note) = &forked.unfinished {
            notices.write(Notice::warn(None, format!("Split “{title}”: {note}.")));
        }
        if move_from.is_some() {
            notices.write(Notice::info(
                None,
                format!("Moved into session {} as “{title}”.", forked.to),
            ));
        } else {
            commands.trigger(Exit(ExitReason::Switch(Some(forked.to.clone()))));
        }
    }
}

/// `/split switch`: makes line `name` the active one, or runs the session
/// it is kept in; the active line (no name) of another session runs that
/// session.
fn switch(world: &mut World, target: &Target, name: Option<&str>) -> Result<String, String> {
    let conversation = conversation_of(world, target)?;
    let line = name
        .map(|name| find_line(&conversation, name))
        .transpose()?;
    match line.map_or(LineId::Active, |line| line.id) {
        LineId::Active => match &target.session {
            None => Ok("That is the active line.".to_owned()),
            Some(session) => {
                world.trigger(Exit(ExitReason::Switch(Some(session.clone()))));
                Ok(format!("Running session {session}."))
            }
        },
        LineId::Split { session, .. } => {
            let id: SessionId = session
                .parse()
                .map_err(|_| format!("`{session}` is no session id"))?;
            world.trigger(Exit(ExitReason::Switch(Some(id))));
            Ok(format!("Running session {session}."))
        }
        LineId::Branch(id) => edit(world, (target, true), |conversation| {
            conversation
                .switch_branch(id, by())
                .map_err(|failure| failure.to_string())?;
            Ok(format!("Now on “{}”.", conversation.line_title()))
        }),
    }
}

/// The conversation of `target`, as it is now.
fn conversation_of(
    world: &mut World,
    target: &Target,
) -> Result<conversation::Conversation, String> {
    match (&target.session, target.entity) {
        (Some(session), _) => read_session(world, session)?
            .agent(&target.agent.0)
            .map(|agent| agent.conversation.clone())
            .ok_or_else(|| "the session has no such agent".to_owned()),
        (None, Some(entity)) => world
            .get::<Conversation>(entity)
            .map(|conversation| conversation.view())
            .ok_or_else(|| "the agent has no conversation".to_owned()),
        (None, None) => Err("there is no such agent".to_owned()),
    }
}

/// `/split merge`: brings line `name` back into the active line: a kept
/// branch with `merge_branch`, a split kept in another session with
/// `merge_line` (back where a moved range was, else at the end), its
/// pointer then marked merged.
fn merge(world: &mut World, target: &Target, name: &str) -> Result<String, String> {
    let conversation = conversation_of(world, target)?;
    let failed = |failure: BranchError| failure.to_string();
    match find_line(&conversation, name)?.id {
        LineId::Active => Ok("That is the active line.".to_owned()),
        LineId::Branch(id) => edit(world, (target, false), |conversation| {
            let spliced = conversation.merge_branch(id, by()).map_err(failed)?;
            Ok(merged(&spliced))
        }),
        LineId::Split { session, agent } => {
            let pointer = SplitPointer::find(&conversation, &session, &agent);
            let pointer = pointer
                .and_then(|(id, pointer)| Some((id, pointer, conversation.aside(id)?.clone())));
            let (id, mut pointer, aside) = pointer.ok_or_else(|| {
                format!(
                    "the split in session {session} is merged already, or not in the active line"
                )
            })?;
            let line = split_line(world, &session, &agent)?;
            let at = match pointer.moved {
                0 => MergeAt::End,
                moved => MergeAt::Back { pointer: id, moved },
            };
            pointer.merged = true;
            let value = serde_json::to_value(&pointer).map_err(|failure| failure.to_string())?;
            edit(world, (target, false), |conversation| {
                let spliced = conversation.merge_line(&line, at, by()).map_err(failed)?;
                conversation.replace_aside(id, Aside { value, ..aside });
                Ok(merged(&spliced))
            })
        }
    }
}

/// What a merge says it did.
fn merged(spliced: &Spliced) -> String {
    format!("Merged {} entries back.", spliced.changed.len())
}

/// The active line of `agent` in session `session`, where a split of a
/// line is kept.
fn split_line(
    world: &World,
    session: &str,
    agent: &str,
) -> Result<conversation::Conversation, String> {
    let split: SessionId = session
        .parse()
        .map_err(|_| format!("`{session}` is no session id"))?;
    read_session(world, &split)?
        .agent(agent)
        .map(|agent| agent.conversation.clone())
        .ok_or_else(|| format!("session {session} has no agent {agent}"))
}

/// `/splits [session]`: the lines of the agent's conversation, or of a
/// named saved session's main agent, in the native session picker with a
/// preview; picking one runs `/split switch` for it.
fn splits(In(args): In<CommandArgs>, world: &mut World) {
    let name = args.args.trim();
    let place = Where {
        from: (!name.is_empty()).then(|| name.to_owned()),
        ..Where::default()
    };
    let target = match target(world, args.agent, &place) {
        Ok(target) => target,
        Err(why) => {
            world.write_message(Notice::error(args.agent, why));
            return;
        }
    };
    let title = match &target.session {
        None => "Lines of this conversation".to_owned(),
        Some(session) => format!("Lines of session {session}"),
    };
    world.spawn((
        Picker::new(title).for_agent(args.agent),
        SessionPicker {
            source: SessionSource::Lines {
                session: target.session,
                agent: Some(target.agent.0),
            },
            command: "split switch {line} --from {session} --agent {agent}".to_owned(),
            ..SessionPicker::default()
        },
    ));
}

/// `/split compare`: shows line `name` beside the active one, each from
/// the last entry they share.
fn compare(world: &mut World, target: &Target, name: &str) -> Result<String, String> {
    let conversation = conversation_of(world, target)?;
    let found = find_line(&conversation, name)?;
    let other = match &found.id {
        LineId::Active => return Err("That is the active line: name another one.".to_owned()),
        LineId::Branch(id) => {
            let length = conversation.line(Some(*id)).map_or(0, |line| line.len());
            conversation
                .line_conversation(Some(*id), length)
                .ok_or_else(|| format!("there is no line `{name}`"))?
        }
        LineId::Split { session, agent } => split_line(world, session, agent)?,
    };
    let ours = conversation.line(None).unwrap_or_default();
    let shared = shared_prefix(&ours, &other.line(None).unwrap_or_default());
    let sides = [
        (
            conversation.line_title().to_owned(),
            own(&conversation, shared),
        ),
        (found.title, own(&other, shared)),
    ];
    show(world, sides)?;
    Ok(format!("Comparing from entry {shared}."))
}

/// `line` from the last of its `shared` first entries on, moved back past
/// tool results so none is shown without its call.
fn own(line: &conversation::Conversation, shared: usize) -> conversation::Conversation {
    let results = |at: usize| match line.messages().get(at) {
        Some(ChatMessage::User { content }) => content
            .iter()
            .any(|part| matches!(part, UserContent::ToolResult(_))),
        _ => false,
    };
    let mut head = shared.saturating_sub(1);
    while head > 0 && results(head) {
        head -= 1;
    }
    let mut own = line.view();
    own.splice(..head, [], by());
    own
}

/// Shows two lines side by side, each in a transcript pane of one box
/// over the screen: Tab goes from one to the other, Esc closes them.
#[cfg(feature = "tui")]
fn show(world: &mut World, sides: [(String, conversation::Conversation); 2]) -> Result<(), String> {
    use rig_tui::crossterm::event::KeyCode;
    use rig_tui::ratatui::layout::Constraint;
    use rig_tui::{
        KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, PaneKeys, Placement, TranscriptPane,
        TranscriptSource, TuiPanel,
    };

    let over = Placement::Over {
        width: Constraint::Percentage(90),
        height: Constraint::Percentage(90),
    };
    let compared = world
        .spawn((Name::new("split compare"), TuiPanel::new(over)))
        .id();
    let panes: Vec<Entity> = (0..)
        .zip(sides)
        .map(|(part, (title, line))| {
            let within = Placement::Within {
                panel: compared,
                part,
                of: 2,
            };
            let pane = TranscriptPane::new(TranscriptSource::Value(line)).titled(title);
            let spawned = (pane, TuiPanel::new(within), PaneKeys, ChildOf(compared));
            world.spawn(spawned).id()
        })
        .collect();
    let (Some(&left), Some(&right)) = (panes.first(), panes.last()) else {
        return Err("no room to compare".to_owned());
    };
    for (pane, other) in [(left, right), (right, left)] {
        let key = |code, label| {
            let binding = KeyBinding::new(KeyPress::key(code), label);
            (binding.in_scope(KeyScope::Panel(pane)), ChildOf(pane))
        };
        world
            .spawn(key(KeyCode::Esc, "Close the comparison"))
            .observe(move |_: On<KeyPressed>, mut commands: Commands| {
                commands.entity(compared).try_despawn();
            });
        world.spawn(key(KeyCode::Tab, "The other line")).observe(
            move |_: On<KeyPressed>, mut focus: ResMut<KeyFocus>| {
                focus.0 = Some(other);
            },
        );
    }
    world.insert_resource(KeyFocus(Some(left)));
    Ok(())
}

/// Without the terminal view there is nowhere to show two lines.
#[cfg(not(feature = "tui"))]
fn show(_: &mut World, _: [(String, conversation::Conversation); 2]) -> Result<(), String> {
    Err("Comparing lines needs the terminal view.".to_owned())
}
