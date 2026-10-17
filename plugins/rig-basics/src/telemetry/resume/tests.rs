use std::fs;

use crate::sessions::SessionsPlugin;
use rig_core::completion::{Message, Usage};
use rig_harness::harness_protocol::SessionId;
use rig_harness::sessions::SessionEntry;
use rig_harness_test_support::{app_in, commit_message, first_agent, scratch_home};

use super::*;
use crate::telemetry::UsagePlugin;

/// What the session spent is in its `/resume` line, from its listing,
/// as the cost or, when no call was priced, the tokens.
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
    assert_eq!(spent.map(String::as_str), Some("1.5k tokens"), "{labels:?}");
    let line = listed.map(|entry| entry.label()).unwrap_or_default();
    assert!(line.contains("  · 1.5k tokens"), "{line}");
    fs::remove_dir_all(home.root()).ok();
}
