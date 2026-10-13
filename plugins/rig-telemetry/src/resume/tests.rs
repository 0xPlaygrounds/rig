use std::fs;
use std::path::Path;

use rig_core::completion::{Message, Usage};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{app_in, commit_message, first_agent, scratch_home};
use rig_sessions::{SessionsPlugin, list};

use super::*;
use crate::UsagePlugin;

/// What the session spent is in its `/resume` line, from its listing,
/// as the cost or, when no call was priced, the tokens.
#[test]
fn what_the_session_spent_is_in_its_resume_line() {
    let home = scratch_home("rig-telemetry-resume");
    let id = SessionId::generate();
    let (mut app, _wakes) = app_in(&home, &id);
    app.add_plugins((AgentPlugin, JournalPlugin, SessionsPlugin, UsagePlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let said = commit_message(app.world_mut(), agent, Message::user("Fix the parser"));
    assert_eq!(said, Ok(()));
    let mut usage = Usage::default();
    usage.input_tokens = Some(1_200);
    usage.output_tokens = Some(300);
    let mut spent = app.world_mut().resource_mut::<SessionSpending>();
    spent.0.record(None, None, &usage);
    app.update();

    let listed = list(&home, &SessionStores::default(), Path::new("/nowhere"));
    let labels: Vec<_> = listed
        .iter()
        .map(|entry| entry.meta.labels.clone())
        .collect();
    let spent = labels.first().and_then(|labels| labels.get(SPENT).cloned());
    assert_eq!(spent.as_deref(), Some("1.5k tokens"), "{labels:?}");
    fs::remove_dir_all(home.root()).ok();
}
