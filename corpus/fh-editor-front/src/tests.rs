use std::time::Instant;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, connect as connect_model, first_agent, log_ended,
    log_tool_outputs, reply, scratch_home,
};
use serde_json::json;

use super::*;

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(EditorFrontPlugin);
}

/// A fake editor: what it was sent, and what it sends back.
struct FakeEditor {
    link: EditorLink,
    got: Vec<ToEditor>,
}

impl FakeEditor {
    fn send(&self, message: &FromEditor) {
        if let Ok(line) = serde_json::to_string(message) {
            self.link.send.unbounded_send(line).ok();
        }
    }

    /// What it was sent since last time.
    fn drain(&mut self) -> Vec<ToEditor> {
        let new: Vec<ToEditor> = self
            .link
            .receive
            .try_iter()
            .filter_map(|line| serde_json::from_str(&line).ok())
            .collect();
        self.got.extend(new.iter().cloned());
        new
    }

    fn streamed(&self) -> String {
        self.got
            .iter()
            .filter_map(|message| match message {
                ToEditor::Delta { kind, text, .. } if kind == "text" => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reviews(&self) -> Vec<&str> {
        self.got
            .iter()
            .filter_map(|message| match message {
                ToEditor::Review { path, .. } => Some(path.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// Two editors on one session, `a.txt` open in the first with the unsaved
/// text `draft…`: the agent reads the buffer, its edit of `a.txt` is
/// reviewed (hunk 1 accepted, hunk 2 rejected; the second editor's late
/// answer ignored), its write of `b.txt`, which no editor has open, goes
/// to disk unreviewed, and both editors see its answer stream.
#[test]
fn open_buffers_win_and_edits_are_reviewed() {
    let dir = scratch_home("fh-editor-front").root().to_owned();
    std::fs::create_dir_all(&dir).ok();
    let (a, b) = (dir.join("a.txt"), dir.join("b.txt"));
    std::fs::write(&a, "disk\n").ok();
    let (a_path, b_path) = (a.display().to_string(), b.display().to_string());
    let edits = json!([
        { "old_text": "draft", "new_text": "DRAFT" },
        { "old_text": "end", "new_text": "END" },
    ]);
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": a_path })),
        calls("e1", "edit", json!({ "path": a_path, "edits": edits })),
        calls("w1", "write", json!({ "path": b_path, "content": "new\n" })),
        reply("Done."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((
        AgentPlugin,
        JournalPlugin,
        ReadTool,
        EditTool,
        WriteTool,
        EditorFrontPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended)
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect_model(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    let mut first = FakeEditor {
        link: connect(app.world_mut()),
        got: Vec::new(),
    };
    let mut second = FakeEditor {
        link: connect(app.world_mut()),
        got: Vec::new(),
    };
    let draft = "draft\nkeep\nend\n";
    first.send(&FromEditor::Buffer {
        path: a_path.clone(),
        text: draft.to_owned(),
    });
    first.send(&FromEditor::Send {
        text: "Shout the first line.".to_owned(),
        selection: None,
    });

    let mut disk_while_reviewed = None;
    let mut answered_late = false;
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(10)
        && app.world().resource::<Ended>().0.is_empty()
    {
        app.update();
        wakes.recv_timeout(Duration::from_millis(50)).ok();
        for message in first.drain() {
            match message {
                ToEditor::Review { id, .. } => {
                    disk_while_reviewed = std::fs::read_to_string(&a).ok();
                    let accept = vec![true, false];
                    first.send(&FromEditor::Review { id, accept });
                }
                ToEditor::DiagnosticsRequest { id, .. } => {
                    let items = vec!["1:1 DRAFT: unknown word".to_owned()];
                    first.send(&FromEditor::Diagnostics { id, items });
                }
                _ => {}
            }
        }
        for message in second.drain() {
            // Answered once the first editor's answer closed it: ignored.
            if let (ToEditor::ReviewClosed { id }, false) = (message, answered_late) {
                answered_late = true;
                let accept = vec![true, true];
                second.send(&FromEditor::Review { id, accept });
            }
        }
    }

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    assert!(output("r1").contains("draft"), "{}", output("r1"));
    assert!(!output("r1").contains("disk"), "{}", output("r1"));
    let edited = output("e1");
    assert!(
        edited.starts_with("partially applied: hunk 2 rejected"),
        "{edited}"
    );
    assert!(edited.contains("DRAFT: unknown word"), "{edited}");
    assert_eq!(disk_while_reviewed.as_deref(), Some("disk\n"));
    assert_eq!(std::fs::read_to_string(&a).ok().as_deref(), Some("disk\n"));
    let buffer = app
        .world()
        .resource::<Buffers>()
        .0
        .get(&resolve(None, &a_path))
        .cloned();
    assert_eq!(
        buffer.map(|buffer| buffer.text).as_deref(),
        Some("DRAFT\nkeep\nend\n")
    );
    assert_eq!(std::fs::read_to_string(&b).ok().as_deref(), Some("new\n"));
    assert_eq!(first.reviews(), vec![a_path.as_str()]);
    assert_eq!(second.reviews(), vec![a_path.as_str()]);
    assert!(answered_late);
    assert_eq!(first.streamed(), "Done.");
    assert_eq!(second.streamed(), "Done.");
}
