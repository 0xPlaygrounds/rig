use std::collections::BTreeMap;

use serde_json::json;

use crate::{CURRENT, MODEL, SavedAgent, Session, TITLE};

fn agent(id: &str, parent: Option<&str>, retired: bool) -> SavedAgent {
    SavedAgent {
        id: id.to_owned(),
        parent: parent.map(str::to_owned),
        retired,
        ..SavedAgent::default()
    }
}

/// Without a current agent, the main agent is the first by id that
/// nothing spawned, of those a restore brings back; its model and title
/// read by their keys, and a value of another shape is an error, not a
/// silent `None`.
#[test]
fn the_main_agent_and_its_saved_values_read_without_the_host() {
    let mut main = agent("b", None, false);
    main.saved = BTreeMap::from([
        (MODEL.to_owned(), json!("openai/gpt-5")),
        (TITLE.to_owned(), json!(["not", "a", "string"])),
    ]);
    let session = Session {
        agents: vec![
            agent("0-child", Some("b"), false),
            agent("a-gone", None, true),
            agent("a-orphan", Some("a-gone"), false),
            main,
            agent("c", None, false),
        ],
        ..Session::default()
    };
    let found = session.main_agent();
    assert_eq!(found.map(|agent| agent.id.as_str()), Some("b"));
    assert_eq!(
        found.and_then(SavedAgent::model).as_deref(),
        Some("openai/gpt-5")
    );
    assert_eq!(found.and_then(SavedAgent::title), None);
    let title = found.and_then(|agent| agent.value::<String>(TITLE));
    assert!(title.is_some_and(|title| title.is_err()));
    assert!(Session::default().main_agent().is_none());
}

/// The main agent is the current one the session names, though another
/// sorts first; a current agent a restore leaves out, a subagent or an
/// unknown id names none.
#[test]
fn the_main_agent_is_the_current_one() {
    let main = |current: &str| {
        let session = Session {
            agents: vec![
                agent("a-first", None, false),
                agent("0-gone", None, true),
                agent("child", Some("a-first"), false),
                agent("b-fork", None, false),
            ],
            saved: BTreeMap::from([(CURRENT.to_owned(), json!(current))]),
            ..Session::default()
        };
        session.main_agent().map(|agent| agent.id.clone())
    };
    assert_eq!(main("b-fork").as_deref(), Some("b-fork"));
    for none in ["0-gone", "child", "unknown"] {
        assert_eq!(main(none).as_deref(), Some("a-first"), "{none}");
    }
}
