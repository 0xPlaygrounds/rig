//! Plugin corpus (held-out): H18: Shared blackboard for agents (spec `H18-blackboard.md`).
//!
//! A board every agent shares ([`Board`], saved with the session): agents
//! post findings to topics with `board_post` and read them with
//! `board_read`, and may claim a key (a path, a task) so the others leave
//! it alone. An agent is subscribed ([`Subscriptions`]) to the topics it
//! posted to or read; each new post goes to every other subscribed agent as
//! a note with its next request, which wakes no idle agent. A claim goes
//! when its holder releases it or is despawned. `/board` toggles the
//! board's panel in the terminal view, and `/board post <topic> <text>`
//! posts as the user, noted to every agent.

use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::{Paragraph, Wrap};
use rig_tui::{PanelCanvas, Placement, RequestRedraw, Theme, TuiPanel, TuiSystems};
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool that posts.
pub const BOARD_POST: &str = "board_post";
/// The tool that reads.
pub const BOARD_READ: &str = "board_read";

/// The posts a topic keeps; the oldest go first.
pub const MAX_POSTS: usize = 200;
/// The longest text a post may have, in characters.
pub const MAX_TEXT: usize = 2_000;

/// The panel's width.
const PANEL_WIDTH: u16 = 48;

/// Adds the `board_post` and `board_read` tools, the `/board` command and
/// the board's panel.
#[derive(Default)]
pub struct BlackboardPlugin;

impl Plugin for BlackboardPlugin {
    fn build(&self, app: &mut App) {
        let post = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        let read = ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.init_resource::<Board>()
            .register_required_components::<Agent, Subscriptions>()
            .add_message::<RequestRedraw>()
            .add_open_tool(
                BOARD_POST,
                "Post `text` to the shared board's `topic`, which the other agents see. \
                 `claim` marks a key (a path or a task) as yours; `release` gives one back. \
                 Other agents may work beside you: post what you find that others could use, \
                 `claim` a path or task you take on (`release` it when done), and leave alone \
                 what another agent claimed. New posts to topics you posted to or read reach \
                 you as notes.",
                post,
                on_post,
            )
            .add_open_tool(
                BOARD_READ,
                "Read the shared board: one `topic` or all, only the posts after the id \
                 `since`; and the claims held. Before starting on a file or task, read the \
                 topic it belongs to.",
                read,
                on_read,
            )
            .add_command(
                "board",
                "Show or hide the shared board; `/board post <topic> <text>` posts as you",
                board_command,
            )
            .add_observer(release_on_despawn)
            .add_systems(Startup, spawn_panel)
            .add_systems(Update, redraw.run_if(resource_changed::<Board>))
            .init_resource::<Theme>()
            .add_systems(PostUpdate, draw_panel.in_set(TuiSystems::Draw));
    }
}

/// The shared board: every topic's posts and the claims held. Saved with
/// the session.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default, Clone, Debug)]
pub struct Board {
    /// The topics, in the order they were first posted to.
    pub topics: Vec<Topic>,
    /// The keys taken, in the order they were claimed.
    pub claims: Vec<Claim>,
    /// The id of the last post.
    pub last_id: u64,
}

/// A topic of the [`Board`].
#[derive(Reflect, Clone, Debug, Default)]
#[reflect(Default, Clone, Debug)]
pub struct Topic {
    /// Its name.
    pub name: String,
    /// Its latest posts, oldest first, at most [`MAX_POSTS`].
    pub posts: Vec<Post>,
}

/// A post on the [`Board`].
#[derive(Reflect, Clone, Debug, Default)]
#[reflect(Default, Clone, Debug)]
pub struct Post {
    /// Its id, unique on the board, in the order posts came.
    pub id: u64,
    /// Who posted it.
    pub author: Author,
    /// When, in seconds since the Unix epoch.
    pub at: u64,
    /// What it says.
    pub text: String,
    /// The key it claimed, if the claim was granted.
    pub claimed: Option<String>,
    /// The key it released, if it held it.
    pub released: Option<String>,
}

/// Who posted or claims: an agent, by its stable id, or the user.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Default, Clone, Debug)]
pub struct Author {
    /// The agent's [`AgentId`]; `None` for the user.
    pub agent: Option<String>,
    /// Its name as shown, such as `A (agent 1a2b3c4d)` or `user`.
    pub label: String,
}

/// A key taken by an agent.
#[derive(Reflect, Clone, Debug, Default)]
#[reflect(Default, Clone, Debug)]
pub struct Claim {
    /// The key, such as a path.
    pub key: String,
    /// The holder.
    pub holder: Author,
}

/// The topics an agent posted to or read, whose new posts reach it as
/// notes. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Subscriptions(pub Vec<String>);

impl Subscriptions {
    fn add(&mut self, topic: &str) {
        if !self.0.iter().any(|known| known == topic) {
            self.0.push(topic.to_owned());
        }
    }
}

impl Board {
    /// Posts `text` to `topic` as `author`, releasing then claiming what
    /// it asks, and says what came of it.
    fn post(&mut self, author: &Author, args: &PostArgs) -> Result<(Post, String), String> {
        let topic = args.topic.trim();
        let text = args.text.trim();
        if topic.is_empty() || text.is_empty() {
            return Err("a post needs a `topic` and a `text`".to_owned());
        }
        let length = text.chars().count();
        if length > MAX_TEXT {
            return Err(format!(
                "the text has {length} characters; a post takes at most {MAX_TEXT}"
            ));
        }
        let mut said = Vec::new();
        let released = args.release.as_deref().map(str::trim).and_then(|key| {
            let held = self.claims.iter().position(|claim| claim.key == key);
            match held.and_then(|at| self.claims.get(at).map(|claim| (at, claim))) {
                Some((at, claim)) if claim.holder == *author => {
                    self.claims.remove(at);
                    said.push(format!("Released `{key}`."));
                    Some(key.to_owned())
                }
                Some((_, claim)) => {
                    let holder = &claim.holder.label;
                    said.push(format!("Release of `{key}` refused: {holder} holds it."));
                    None
                }
                None => {
                    said.push(format!("`{key}` was not claimed."));
                    None
                }
            }
        });
        let claimed = args.claim.as_deref().map(str::trim).and_then(|key| {
            match self.claims.iter().find(|claim| claim.key == key) {
                Some(claim) if claim.holder == *author => {
                    said.push(format!("You already hold `{key}`."));
                    None
                }
                Some(claim) => {
                    let holder = &claim.holder.label;
                    said.push(format!(
                        "Claim of `{key}` refused: {holder} holds it. Leave it to them."
                    ));
                    None
                }
                None => {
                    self.claims.push(Claim {
                        key: key.to_owned(),
                        holder: author.clone(),
                    });
                    said.push(format!("Claimed `{key}`."));
                    Some(key.to_owned())
                }
            }
        });
        self.last_id += 1;
        let post = Post {
            id: self.last_id,
            author: author.clone(),
            at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_secs()),
            text: text.to_owned(),
            claimed,
            released,
        };
        let at = match self.topics.iter().position(|known| known.name == topic) {
            Some(at) => at,
            None => {
                self.topics.push(Topic {
                    name: topic.to_owned(),
                    posts: Vec::new(),
                });
                self.topics.len() - 1
            }
        };
        if let Some(topic) = self.topics.get_mut(at) {
            topic.posts.push(post.clone());
            let over = topic.posts.len().saturating_sub(MAX_POSTS);
            topic.posts.drain(..over);
        }
        said.insert(0, format!("Posted #{} to `{topic}`.", post.id));
        Ok((post, said.join(" ")))
    }

    /// The posts of `topic`, or of every topic, after the id `since`, in
    /// the order they came, each with its topic.
    fn read(&self, topic: Option<&str>, since: u64) -> Vec<(&str, &Post)> {
        let mut posts: Vec<(&str, &Post)> = self
            .topics
            .iter()
            .filter(|known| topic.is_none_or(|topic| known.name == topic))
            .flat_map(|known| known.posts.iter().map(|post| (known.name.as_str(), post)))
            .filter(|(_, post)| post.id > since)
            .collect();
        posts.sort_unstable_by_key(|(_, post)| post.id);
        posts
    }

    /// The claims held, one a line.
    fn claims_text(&self) -> String {
        if self.claims.is_empty() {
            return "No claims are held.".to_owned();
        }
        let lines: Vec<String> = self
            .claims
            .iter()
            .map(|claim| format!("- `{}`: {}", claim.key, claim.holder.label))
            .collect();
        format!("Claims held:\n{}", lines.join("\n"))
    }
}

impl Post {
    /// Such as `#3 [bugs] A (agent 1a2b3c4d), 2026-10-11 12:00 UTC: text
    /// (claims x.rs)`.
    fn line(&self, topic: &str) -> String {
        let at = chrono::DateTime::from_timestamp(i64::try_from(self.at).unwrap_or(0), 0)
            .map(|at| at.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_default();
        let mut line = format!(
            "#{} [{topic}] {}, {at}: {}",
            self.id, self.author.label, self.text
        );
        if let Some(key) = &self.claimed {
            line.push_str(&format!(" (claims `{key}`)"));
        }
        if let Some(key) = &self.released {
            line.push_str(&format!(" (releases `{key}`)"));
        }
        line
    }
}

/// The author an agent posts as.
fn author_of(id: &AgentId, name: Option<&Name>) -> Author {
    let label = match name {
        Some(name) => format!("{name} (agent {})", id.short()),
        None => format!("agent {}", id.short()),
    };
    Author {
        agent: Some(id.0.clone()),
        label,
    }
}

/// The note a new post makes for the agents it reaches.
fn note(topic: &str, post: &Post) -> String {
    format!("New board post: {}", post.line(topic))
}

/// The arguments of a `board_post` call.
#[derive(Deserialize, JsonSchema, Debug)]
#[serde(deny_unknown_fields)]
struct PostArgs {
    /// The topic, such as `bugs`.
    topic: String,
    /// What you found, at most 2 000 characters.
    text: String,
    /// A key (a path or a task) to mark as yours.
    claim: Option<String>,
    /// A key you hold, to give back.
    release: Option<String>,
}

/// The arguments of a `board_read` call.
#[derive(Deserialize, JsonSchema, Debug)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    /// The topic to read; every topic when left out.
    topic: Option<String>,
    /// Only the posts after this post id.
    since: Option<u64>,
}

/// `board_post`: posts as the calling agent, subscribes it to the topic
/// and notes the post to every other agent subscribed to it.
fn on_post(
    called: On<ToolCalled<PostArgs>>,
    mut board: ResMut<Board>,
    mut agents: Query<(Entity, Option<&Name>, &mut Subscriptions)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let name = agents.get(called.agent).ok().and_then(|(_, name, _)| name);
    let author = author_of(&called.agent_id, name);
    let output = match board.post(&author, &called.args) {
        Ok((post, said)) => {
            let topic = called.args.topic.trim();
            if let Ok((_, _, mut subscribed)) = agents.get_mut(called.agent) {
                subscribed.add(topic);
            }
            let others = agents.iter().filter(|(agent, _, subscribed)| {
                *agent != called.agent && subscribed.0.iter().any(|known| known == topic)
            });
            let text = note(topic, &post);
            for (agent, ..) in others {
                send_note(&mut commands, agent, &text);
            }
            ToolResult::success(said.into())
        }
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("ho-blackboard"), output);
    }
}

/// `board_read`: the posts asked for and the claims held; subscribes the
/// calling agent to the topic read, or to every topic.
fn on_read(
    called: On<ToolCalled<ReadArgs>>,
    board: Res<Board>,
    mut agents: Query<&mut Subscriptions>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let topic = called.args.topic.as_deref().map(str::trim);
    if let Ok(mut subscribed) = agents.get_mut(called.agent) {
        match topic {
            Some(topic) => subscribed.add(topic),
            None => board
                .topics
                .iter()
                .for_each(|known| subscribed.add(&known.name)),
        }
    }
    let posts = board.read(topic, called.args.since.unwrap_or(0));
    let mut text = match (posts.is_empty(), topic) {
        (true, Some(topic)) => format!("No posts in `{topic}` yet."),
        (true, None) => "No posts yet.".to_owned(),
        (false, _) => {
            let lines: Vec<String> = posts.iter().map(|(topic, post)| post.line(topic)).collect();
            lines.join("\n")
        }
    };
    text.push_str("\n\n");
    text.push_str(&board.claims_text());
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("ho-blackboard"),
            ToolResult::success(text.into()),
        );
    }
}

/// Sends `text` to `agent` with its next request; it wakes no idle agent.
fn send_note(commands: &mut Commands, agent: Entity, text: &str) {
    let note = Deliver::new(agent, text, DeliveryMode::Inform);
    commands
        .trigger(note.with_author(rig_harness::rig_core::conversation::Author::plugin("board")));
}

/// A despawned agent's claims go.
fn release_on_despawn(gone: On<Remove<Agent>>, ids: Query<&AgentId>, mut board: ResMut<Board>) {
    let Ok(id) = ids.get(gone.entity) else {
        return;
    };
    let holds = |claim: &Claim| claim.holder.agent.as_deref() == Some(id.0.as_str());
    if board.claims.iter().any(holds) {
        board.claims.retain(|claim| !holds(claim));
    }
}

/// `/board` and `/board post <topic> <text>`.
fn board_command(
    In(args): In<CommandArgs>,
    mut board: ResMut<Board>,
    panels: Query<(Entity, Has<TuiPanel>), With<BoardPanel>>,
    agents: Query<Entity, With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let line = args.args.trim();
    if line.is_empty() {
        for (panel, shown) in &panels {
            if shown {
                commands.entity(panel).remove::<TuiPanel>();
            } else {
                let placement = Placement::Right(Constraint::Length(PANEL_WIDTH));
                commands.entity(panel).insert(TuiPanel::new(placement));
            }
        }
        return;
    }
    let Some(rest) = line.strip_prefix("post") else {
        notices.write(Notice::error(
            args.agent,
            "/board takes nothing, or `post <topic> <text>`",
        ));
        return;
    };
    let Some((topic, text)) = rest.trim().split_once(char::is_whitespace) else {
        notices.write(Notice::error(
            args.agent,
            "/board post needs a topic and a text",
        ));
        return;
    };
    let user = Author {
        agent: None,
        label: "user".to_owned(),
    };
    let request = PostArgs {
        topic: topic.to_owned(),
        text: text.to_owned(),
        claim: None,
        release: None,
    };
    match board.post(&user, &request) {
        Ok((post, said)) => {
            let text = note(topic, &post);
            for agent in &agents {
                send_note(&mut commands, agent, &text);
            }
            notices.write(Notice::info(args.agent, said));
        }
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
        }
    }
}

/// The entity of the board's panel, which has a [`TuiPanel`] while it is
/// shown.
#[derive(Component)]
struct BoardPanel;

fn spawn_panel(mut commands: Commands) {
    commands.spawn((Name::new("board panel"), BoardPanel));
}

/// A frame for a changed board; the view does not watch it.
fn redraw(mut redraw: MessageWriter<RequestRedraw>) {
    redraw.write(RequestRedraw);
}

fn draw_panel(
    mut panels: Query<&mut PanelCanvas, With<BoardPanel>>,
    board: Res<Board>,
    theme: Res<Theme>,
) {
    let mut lines: Vec<Line> = Vec::new();
    for topic in &board.topics {
        lines.push(Line::from(topic.name.clone()).bold());
        lines.extend(topic.posts.iter().map(|post| {
            let claim = post.claimed.as_ref().map(|key| format!(" [claims {key}]"));
            format!(
                "#{} {}: {}{}",
                post.id,
                post.author.label,
                post.text,
                claim.unwrap_or_default()
            )
            .into()
        }));
    }
    if lines.is_empty() {
        lines.push("The board is empty.".into());
    }
    lines.push(Line::default());
    lines.extend(
        board
            .claims_text()
            .lines()
            .map(|line| Line::from(line.to_owned())),
    );
    for mut canvas in &mut panels {
        canvas.render(
            Paragraph::new(lines.clone())
                .wrap(Wrap { trim: false })
                .block(theme.panel("board")),
        );
    }
}

#[cfg(test)]
mod tests;
