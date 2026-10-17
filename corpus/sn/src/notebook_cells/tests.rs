use std::path::Path;

use rig_basics::coding_tools::{EditTool, ReadTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home,
};
use serde_json::{Value, json};

use super::*;

/// The spec's fixture, as Jupyter writes it: indent 1, a markdown cell, a
/// code cell with a stream output, a code cell with an image and an
/// error, and metadata of its own.
const FIXTURE: &str = r##"{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "intro",
   "metadata": {},
   "source": [
    "# Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "load",
   "metadata": {
    "tags": ["setup"]
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1\n"
     ]
    }
   ],
   "source": [
    "print(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "plot",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "ValueError",
     "evalue": "bad axis",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m: bad axis"
     ]
    }
   ],
   "source": [
    "plot(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "team": {
   "owner": "analysis"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"##;

/// The notebook's cells, parsed.
fn cells(notebook: &Value) -> Vec<Value> {
    notebook
        .get("cells")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The text of `path`, parsed.
fn parsed(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    serde_json::from_str(&text).unwrap_or(Value::Null)
}

/// The lines `before` and `after` share at their start and at their end.
fn common(before: &str, after: &str) -> (usize, usize) {
    let (old, new): (Vec<&str>, Vec<&str>) = (before.lines().collect(), after.lines().collect());
    let head = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let tail = old
        .iter()
        .rev()
        .zip(new.iter().rev())
        .take(old.len().min(new.len()) - head)
        .take_while(|(a, b)| a == b)
        .count();
    (head, tail)
}

#[test]
fn a_notebook_is_read_as_cells_and_edited_one_cell_at_a_time() {
    let dir = scratch_home("notebook-cells").root().to_owned();
    std::fs::create_dir_all(&dir).ok();
    let notebook = dir.join("analysis.ipynb");
    std::fs::write(&notebook, FIXTURE).ok();
    let original = parsed(&notebook);

    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "read", json!({ "path": "analysis.ipynb" })),
        reply("Read it."),
        calls(
            "c2",
            "notebook_edit",
            json!({ "path": "analysis.ipynb", "cell": 1, "action": "replace", "source": "print(2)" }),
        ),
        reply("Replaced."),
        calls(
            "c3",
            "notebook_edit",
            json!({ "path": "analysis.ipynb", "cell": 2, "action": "insert_after",
                    "source": "## Notes", "cell_type": "markdown" }),
        ),
        reply("Inserted."),
        calls(
            "c4",
            "edit",
            json!({ "path": "analysis.ipynb",
                    "edits": [{ "old_text": "print(2)", "new_text": "print(3)" }] }),
        ),
        reply("Refused."),
        calls(
            "c5",
            "notebook_edit",
            json!({ "path": "new.ipynb", "cell": -1, "action": "insert_after", "source": "x = 1" }),
        ),
        reply("Created."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ReadTool,
        EditTool,
        NotebookCellsPlugin,
    ))
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        app.world_mut()
            .entity_mut(agent)
            .insert((connection, WorkDir(dir.clone())));
    }
    let turn = |app: &mut App, ask: &str, ended: usize| -> String {
        app.world_mut()
            .trigger(Deliver::new(agent, ask, DeliveryMode::Steer));
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() >= ended
        });
        let outputs = &app.world().resource::<ToolOutputs>().0;
        outputs
            .get(&format!("c{ended}"))
            .cloned()
            .unwrap_or_default()
    };

    // 1. The read shows cells, not JSON.
    let read = turn(&mut app, "Read the notebook.", 1);
    assert!(read.contains("[notebook: 3 cells]"), "{read}");
    assert!(read.contains("[2] code (exec 8)"), "{read}");
    assert!(read.contains("[image/png, "), "{read}");
    assert!(read.contains("ValueError: bad axis"), "{read}");
    assert!(
        !read.contains("iVBORw0KGgo") && !read.contains('\u{1b}'),
        "{read}"
    );

    // 2. Only cell 1 changes, in the parsed notebook and in the text.
    let replaced = turn(&mut app, "Print 2.", 2);
    let text = std::fs::read_to_string(&notebook).unwrap_or_default();
    let after = parsed(&notebook);
    let (old, new) = (cells(&original), cells(&after));
    assert_eq!(new.len(), 3, "{replaced}");
    assert_eq!(
        new.get(1).and_then(|cell| cell.get("source")),
        Some(&json!(["print(2)"]))
    );
    assert_eq!(
        new.get(1).and_then(|cell| cell.get("outputs")),
        Some(&json!([]))
    );
    assert_eq!(
        new.get(1).and_then(|cell| cell.get("execution_count")),
        Some(&Value::Null)
    );
    assert_eq!(
        new.get(1).and_then(|cell| cell.get("id")),
        Some(&json!("load"))
    );
    assert_eq!((new.first(), new.get(2)), (old.first(), old.get(2)));
    assert_eq!(after.get("metadata"), original.get("metadata"));
    let openings: Vec<usize> = (FIXTURE.lines().enumerate())
        .filter_map(|(number, line)| (line == "  {").then_some(number))
        .collect();
    let (head, tail) = common(FIXTURE, &text);
    let changed_end = FIXTURE.lines().count() - tail;
    assert!(
        openings.get(1).is_some_and(|start| head >= *start)
            && openings.get(2).is_some_and(|end| changed_end <= *end),
        "lines {head}..{changed_end} changed:\n{text}"
    );
    assert!(text.ends_with("}\n"), "{text}");

    // 3. A new markdown cell after cell 2, with an id of its own.
    turn(&mut app, "Add notes.", 3);
    let new = cells(&parsed(&notebook));
    let ids: Vec<&str> = new
        .iter()
        .filter_map(|cell| cell.get("id").and_then(Value::as_str))
        .collect();
    assert_eq!(new.len(), 4);
    assert_eq!(
        new.get(3).and_then(|cell| cell.get("cell_type")),
        Some(&json!("markdown"))
    );
    assert_eq!(ids.len(), 4);
    assert!(
        ids.get(3)
            .is_some_and(|id| !["intro", "load", "plot"].contains(id))
    );

    // 4. `edit` refuses the notebook and leaves it as it was.
    let before = std::fs::read_to_string(&notebook).unwrap_or_default();
    let refused = turn(&mut app, "Print 3.", 4);
    assert!(refused.contains("notebook_edit"), "{refused}");
    assert_eq!(
        std::fs::read_to_string(&notebook).unwrap_or_default(),
        before
    );

    // 5. A missing notebook is created with its one cell.
    turn(&mut app, "Start a notebook.", 5);
    let created = parsed(&dir.join("new.ipynb"));
    assert_eq!(created.get("nbformat"), Some(&json!(4)));
    assert_eq!(cells(&created).len(), 1);
    assert_eq!(created.pointer("/cells/0/source"), Some(&json!(["x = 1"])));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(NotebookCellsPlugin);
}
