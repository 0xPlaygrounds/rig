use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_reflect::prelude::ReflectDefault;
use rig_core::completion::Message;
use rig_core::conversation::{AsideId, Author};
use rig_memory::{MemoryStore, NoRewrite, fork};

use super::{FOLLOWS, ReflectFollows};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, Conversation};
use crate::inbox::{Deliver, DeliveryMode};
use crate::session_log::tests::{edit, push_message};
use crate::session_log::{SessionLogPlugin, SessionStore};
use crate::stage::{AgentPass, Stage};

/// A plugin's mode, kept with the line it was set on.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Follows, Default)]
struct Mode(String);

fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    app
}

fn agent(app: &mut App) -> Entity {
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    let agent = agents.iter(app.world()).next();
    assert!(agent.is_some(), "an agent");
    agent.unwrap_or(Entity::PLACEHOLDER)
}

fn mode(app: &App, agent: Entity) -> Option<String> {
    app.world().get::<Mode>(agent).map(|mode| mode.0.clone())
}

fn set_mode(app: &mut App, agent: Entity, mode: &str) {
    app.world_mut()
        .entity_mut(agent)
        .insert(Mode(mode.to_owned()));
    app.update();
}

fn say(app: &mut App, agent: Entity, text: &str) {
    push_message(app.world_mut(), agent, Message::user(text));
    app.update();
}

/// The kept branch titled `title`.
fn branch(app: &App, agent: Entity, title: &str) -> AsideId {
    let conversation = app.world().get::<Conversation>(agent);
    let kept = conversation.map(|c| c.branches()).unwrap_or_default();
    let found = kept.into_iter().find(|kept| kept.branch.title == title);
    assert!(found.is_some(), "a branch {title}");
    found.map_or(AsideId::default(), |kept| kept.id)
}

fn switch(app: &mut App, agent: Entity, title: &str) {
    let id = branch(app, agent, title);
    edit(app.world_mut(), agent, move |conversation| {
        assert!(
            conversation
                .switch_branch(id, Author::plugin("tree"))
                .is_ok()
        );
    });
    app.update();
}

fn branch_off(app: &mut App, agent: Entity, at: usize, title: &'static str) {
    edit(app.world_mut(), agent, move |conversation| {
        assert!(
            conversation
                .branch_off(at, title, Author::plugin("tree"))
                .is_ok()
        );
    });
    app.update();
}

/// A value set right after a split at the end stays with the new line: going
/// back gives the old line's value, going forward the new one's, round trip
/// after round trip; a restore gives the active line's.
#[test]
fn a_value_set_after_a_split_at_the_end_stays_with_its_line() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    branch_off(&mut app, agent, 1, "other");
    assert_eq!(
        mode(&app, agent).as_deref(),
        Some("plan"),
        "as of the split"
    );
    set_mode(&mut app, agent, "build");
    for _ in 0..2 {
        switch(&mut app, agent, "main");
        assert_eq!(mode(&app, agent).as_deref(), Some("plan"));
        switch(&mut app, agent, "other");
        assert_eq!(mode(&app, agent).as_deref(), Some("build"));
    }
    drop(app);
    let mut again = self::app(&store);
    let agent = self::agent(&mut again);
    assert_eq!(mode(&again, agent).as_deref(), Some("build"));
}

/// A split at an earlier entry starts without the value set after it, and
/// its own value is anchored at its first entry, so a later split there
/// carries it; the line it left keeps its own.
#[test]
fn a_value_follows_a_split_mid_line_and_is_anchored_at_its_first_entry() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    say(&mut app, agent, "two");
    set_mode(&mut app, agent, "plan");
    branch_off(&mut app, agent, 1, "other");
    assert_eq!(mode(&app, agent), None, "not set on this line");
    set_mode(&mut app, agent, "build");
    say(&mut app, agent, "three");
    let conversation = app.world().get::<Conversation>(agent);
    let anchored = conversation.is_some_and(|c| {
        let last = c.entries().next_back().map(|entry| entry.id);
        c.line_values().is_none()
            && c.asides()
                .any(|(_, a)| a.kind == FOLLOWS && a.after == last)
    });
    assert!(anchored, "{conversation:?}");
    branch_off(&mut app, agent, 2, "deeper");
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
    switch(&mut app, agent, "main");
    assert_eq!(mode(&app, agent).as_deref(), Some("plan"));
    switch(&mut app, agent, "other");
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
}

/// A fork of the session as it was before a message gives the agent the
/// value its line had then.
#[test]
fn a_fork_gives_the_value_as_of_its_cut() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    say(&mut app, agent, "two");
    set_mode(&mut app, agent, "build");
    let id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    let two = app.world().get::<Conversation>(agent).and_then(|c| c.id(1));
    drop(app);
    let copy = MemoryStore::default();
    let cut = two.map(|two| (id.0.as_str(), two));
    assert!(fork(&store, &copy, cut, &NoRewrite).is_ok());
    let mut forked = self::app(&copy);
    let agent = self::agent(&mut forked);
    assert_eq!(mode(&forked, agent).as_deref(), Some("plan"));
}

/// An edit that takes out the entries values followed keeps one value per
/// key, the newest, after what stands for them now.
#[test]
fn values_of_entries_a_splice_took_out_collapse_into_one() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    say(&mut app, agent, "two");
    set_mode(&mut app, agent, "build");
    edit(app.world_mut(), agent, |conversation| {
        let summary = rig_core::conversation::Entry::new(Message::user("summary"));
        conversation.splice(..2, [summary], Author::plugin("compaction"));
    });
    app.update();
    let conversation = app.world().get::<Conversation>(agent);
    let follows: Vec<_> = conversation
        .map(|c| {
            let summary = c.id(0);
            c.asides()
                .filter(|(_, aside)| aside.kind == FOLLOWS)
                .map(|(_, aside)| (aside.after == summary, aside.value.get("value").cloned()))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(follows, [(true, Some("build".into()))]);
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
    branch_off(&mut app, agent, 1, "other");
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
}

/// A value changed in the frame a line changes is kept, not taken back by
/// the line's older value.
#[test]
fn a_value_changed_in_the_frame_of_a_split_is_kept() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    app.world_mut()
        .entity_mut(agent)
        .insert(Mode("plan".to_owned()));
    branch_off(&mut app, agent, 1, "other");
    assert_eq!(mode(&app, agent).as_deref(), Some("plan"));
    app.update();
    assert_eq!(mode(&app, agent).as_deref(), Some("plan"));
}

/// What the probe in Gate saw, and whether it still looks.
#[derive(Resource, Default)]
struct Probe {
    armed: bool,
    seen: Vec<String>,
}

/// Reads the agent's mode once in Gate and changes it from what it read,
/// as a plugin starting work on a message does.
fn probe(mut probe: ResMut<Probe>, mut modes: Query<&mut Mode, With<Agent>>) {
    if !std::mem::take(&mut probe.armed) {
        return;
    }
    for mut mode in &mut modes {
        probe.seen.push(mode.0.clone());
        mode.0.push('+');
    }
}

/// A line changed before the agent loop gives the agent its value before
/// the loop's next stage, so a change made there starts from the line's
/// value and stays with that line, not with the line it left.
#[test]
fn a_line_changed_before_the_agent_loop_gives_its_value_to_the_loop() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    app.init_resource::<Probe>()
        .add_systems(AgentPass, probe.in_set(Stage::Gate));
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    branch_off(&mut app, agent, 1, "other");
    set_mode(&mut app, agent, "build");
    say(&mut app, agent, "two");
    let main = branch(&app, agent, "main");
    edit(app.world_mut(), agent, move |conversation| {
        assert!(conversation.switch_branch(main, Author::user()).is_ok());
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "three", DeliveryMode::Inform));
    app.world_mut().resource_mut::<Probe>().armed = true;
    app.update();
    let seen = app.world().resource::<Probe>().seen.clone();
    assert_eq!(seen, ["plan"], "the line switched to");
    assert_eq!(mode(&app, agent).as_deref(), Some("plan+"));
    switch(&mut app, agent, "other");
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
    switch(&mut app, agent, "main");
    assert_eq!(mode(&app, agent).as_deref(), Some("plan+"));
}

/// The line a system in `PreUpdate` switches the agents to, once, as a key
/// or a slash command does.
#[derive(Resource, Default)]
struct SwitchTo(Option<AsideId>);

fn switch_early(mut to: ResMut<SwitchTo>, mut agents: Query<&mut Conversation, With<Agent>>) {
    if let Some(line) = to.0.take() {
        for mut conversation in &mut agents {
            assert!(conversation.switch_branch(line, Author::user()).is_ok());
        }
    }
}

/// A line switched before `Update`, with no turn running, gives the agent
/// its value before `Update`: a plugin system there reads the line's value,
/// and its change stays with that line.
#[test]
fn a_line_switched_before_update_gives_its_value_to_update() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    app.init_resource::<Probe>()
        .init_resource::<SwitchTo>()
        .add_systems(PreUpdate, switch_early)
        .add_systems(Update, probe);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    branch_off(&mut app, agent, 1, "other");
    set_mode(&mut app, agent, "build");
    say(&mut app, agent, "two");
    let main = branch(&app, agent, "main");
    app.world_mut().resource_mut::<SwitchTo>().0 = Some(main);
    app.world_mut().resource_mut::<Probe>().armed = true;
    app.update();
    let seen = app.world().resource::<Probe>().seen.clone();
    assert_eq!(seen, ["plan"], "the line switched to");
    switch(&mut app, agent, "other");
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
    switch(&mut app, agent, "main");
    assert_eq!(mode(&app, agent).as_deref(), Some("plan+"));
}
