use std::fs;

use crate::session_picker::SessionEntry;
use crate::sessions::SessionsPlugin;
use rig_core::completion::{Message, Usage};
use rig_core::conversation::MessageId;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{app_in, commit_message, first_agent, scratch_home};
use rig_memory::Anchor;

use super::*;
use crate::telemetry::usage::UsagePlugin;

/// What the session spent is in its `/resume` line, from its listing,
/// as the cost or, when no call was priced, the tokens, said to be spent.
#[test]
fn what_the_session_spent_is_in_its_resume_line() {
    let home = scratch_home("rig-telemetry-resume");
    let id = SessionId::generate();
    let (mut app, _wakes) = app_in(&home, &id);
    app.add_plugins((AgentPlugin, SessionLogPlugin, SessionsPlugin, UsagePlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let said = commit_message(app.world_mut(), agent, Message::user("Fix the parser"));
    assert_eq!(said, Ok(()));
    let mut usage = Usage::default();
    usage.input_tokens = Some(1_200);
    usage.output_tokens = Some(300);
    let mut spent = app.world_mut().resource_mut::<SessionUsage>();
    spent.0.record(None, None, &usage);
    app.update();

    let listed = SessionEntry::read(&home.session(&id), &StoreOpener::default());
    let labels = listed.as_ref().map(|entry| &entry.meta.labels);
    let spent = labels.and_then(|labels| labels.get(SPENT));
    assert_eq!(
        spent.map(String::as_str),
        Some("1.5k tokens spent"),
        "{labels:?}"
    );
    let line = listed.map(|entry| entry.label()).unwrap_or_default();
    assert!(line.contains("  · 1.5k tokens spent"), "{line}");
    fs::remove_dir_all(home.root()).ok();
}

/// The context label says the main agent's context, `ctx ?` while it is
/// stale, and follows a new main agent, not the agent shown.
#[test]
fn the_context_label_is_the_main_agents() {
    let mut app = App::new();
    app.add_plugins(AgentPlugin).init_resource::<SessionUsage>();
    add(&mut app);
    let context = |tokens: u64| {
        LastUsage(Some(Anchor {
            usage: Usage::new().input_tokens(tokens),
            through: MessageId::default(),
        }))
    };
    let world = app.world_mut();
    world.spawn((Agent, AgentId("b".to_owned()), context(4_700)));
    app.update();
    let label = |app: &mut App| {
        let world = app.world_mut();
        let mut labels = world.query_filtered::<&ResumeLabel, With<Context>>();
        let label = labels.single(world).map(|label| label.text.clone());
        label.unwrap_or_default()
    };
    assert_eq!(label(&mut app), "ctx 4.7k");
    let world = app.world_mut();
    let main = world
        .spawn((Agent, AgentId("a".to_owned()), context(12_000)))
        .id();
    app.update();
    assert_eq!(label(&mut app), "ctx 12k");
    // Showing another agent leaves it.
    let world = app.world_mut();
    let other = world
        .spawn((Agent, AgentId("c".to_owned()), context(900)))
        .id();
    app.update();
    assert_eq!(label(&mut app), "ctx 12k");
    app.insert_resource(ShownAgent(Some(other)));
    app.update();
    assert_eq!(label(&mut app), "ctx 12k");
    // After a compaction the figure is stale until a reply reports one.
    app.world_mut().entity_mut(main).insert(ContextStale {});
    app.update();
    assert_eq!(label(&mut app), "ctx ?");
    app.world_mut().entity_mut(main).remove::<ContextStale>();
    app.update();
    assert_eq!(label(&mut app), "ctx 12k");
    assert_eq!(context_label(None, false), "");
    assert_eq!(context_label(None, true), "ctx ?");
}
