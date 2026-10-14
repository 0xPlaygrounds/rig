//! Splits of a conversation, built only on the harness's, rig-core's and
//! rig-memory's public API: `/split` at any entry into a branch of the
//! session or into a new session, `/split move` taking a range out and
//! leaving a pointer behind, and `/split switch` and `/split merge` for
//! any line, a branch kept in the session or a split kept in another.
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
//! /split switch <line> [<where>]
//! /split merge <line> [<where>]
//! ```

use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{BranchError, Conversation as Line, MergeAt, Spliced};
use rig_harness::rig_memory::{LineId, SPLIT, SplitPointer, lines};

#[cfg(test)]
mod tests;

/// The `/split` command.
#[derive(Default)]
pub struct SplitsPlugin;

impl Plugin for SplitsPlugin {
    fn build(&self, app: &mut App) {
        let asker = app.world_mut().spawn(Name::new("rig-splits")).id();
        app.insert_resource(Asker(asker))
            .init_resource::<RigHome>()
            .init_resource::<SessionStores>()
            .add_command(
                "split",
                "Split the conversation at an entry into a branch, or with `new` a new session; \
                 /split move <first> <last>, /split switch <line>, /split merge <line>",
                split,
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
    /// Make a line the active one.
    Switch(String),
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
            Action::Switch(named(words.next())?)
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
                Action::Switch(_) | Action::Merge(_) => title.push(word),
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
        .get_resource::<SessionStores>()
        .cloned()
        .unwrap_or_default();
    let store = stores.open(&dir).map_err(|failure| failure.to_string())?;
    Session::read(&*store).map_err(|failure| failure.to_string())
}

/// The line of `conversation` that `name` names.
fn find_line(conversation: &Line, name: &str) -> Result<LineId, String> {
    let listed = lines(conversation);
    let by_aside = name.parse::<AsideId>().ok();
    let found = listed.iter().find(|line| match &line.id {
        LineId::Branch(id) => Some(*id) == by_aside,
        LineId::Session { session, .. } => session == name,
        LineId::Active => false,
    });
    let found = found.or_else(|| listed.iter().find(|line| line.title == name));
    found
        .map(|line| line.id.clone())
        .ok_or_else(|| format!("there is no line `{name}`"))
}

/// The kept branch `name` names in `conversation`; `None` for the active
/// line, or no name.
fn branch(conversation: &Line, name: Option<&str>) -> Result<Option<AsideId>, String> {
    match name.map(|name| find_line(conversation, name)).transpose()? {
        None | Some(LineId::Active) => Ok(None),
        Some(LineId::Branch(id)) => Ok(Some(id)),
        Some(LineId::Session { session, .. }) => Err(format!(
            "that line is kept in session {session}: /split switch to it first"
        )),
    }
}

/// The newest entry of line `line` of `conversation`.
fn newest(conversation: &Line, line: Option<AsideId>) -> Result<MessageId, String> {
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
        Action::Switch(name) => switch(world, &target, &name),
        Action::Merge(name) => merge(world, &target, &name),
    }
}

/// Splits or moves into a branch of `conversation`, as `asked` says.
fn in_session(conversation: &mut Line, asked: &Asked) -> Result<String, String> {
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
            let left = conversation.branch_title().to_owned();
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
    change: impl FnOnce(&mut Line) -> Result<String, String>,
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
                .get_resource::<SessionStores>()
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
    world.trigger(ForkSession {
        from: target.session.clone(),
        cut: ForkCut::Line {
            agent: target.agent.clone(),
            line,
            through,
            take,
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
        let ForkCut::Line { take, title, .. } = &forked.cut else {
            continue;
        };
        if let Some(note) = &forked.note {
            notices.write(Notice::warn(None, format!("Split “{title}”: {note}.")));
        }
        if take.is_some() {
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
/// it is kept in.
fn switch(world: &mut World, target: &Target, name: &str) -> Result<String, String> {
    let conversation = conversation_of(world, target)?;
    match find_line(&conversation, name)? {
        LineId::Active => Ok("That is the active line.".to_owned()),
        LineId::Session { session, .. } => {
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
            Ok(format!("Now on “{}”.", conversation.branch_title()))
        }),
    }
}

/// The conversation of `target`, as it is now.
fn conversation_of(world: &mut World, target: &Target) -> Result<Line, String> {
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
    match find_line(&conversation, name)? {
        LineId::Active => Ok("That is the active line.".to_owned()),
        LineId::Branch(id) => edit(world, (target, false), |conversation| {
            let spliced = conversation.merge_branch(id, by()).map_err(failed)?;
            Ok(merged(&spliced))
        }),
        LineId::Session { session, agent } => {
            let pointer = conversation.asides().find_map(|(id, aside)| {
                let pointer: SplitPointer = (aside.kind == SPLIT)
                    .then(|| serde_json::from_value(aside.value.clone()).ok())
                    .flatten()?;
                (pointer.session == session && pointer.agent == agent && !pointer.merged)
                    .then_some((id, pointer, aside.clone()))
            });
            let (id, mut pointer, aside) = pointer.ok_or_else(|| {
                format!(
                    "the split in session {session} is merged already, or not in the active line"
                )
            })?;
            let split: SessionId = session
                .parse()
                .map_err(|_| format!("`{session}` is no session id"))?;
            let line = read_session(world, &split)?
                .agent(&agent)
                .map(|agent| agent.conversation.clone())
                .ok_or_else(|| format!("session {session} has no agent {agent}"))?;
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
