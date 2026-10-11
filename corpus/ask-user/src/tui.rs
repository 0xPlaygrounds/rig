//! The terminal view's side: a picker of the question's choices, and how a
//! call is drawn.

use rig_harness::prelude::*;
use rig_tui::{AppToolRenderersExt, PickItem, PickRequest, RESULT_LINES};

use super::AskedUser;

pub(super) fn build(app: &mut App) {
    app.add_message::<PickRequest>()
        .add_observer(pick)
        .add_tool_renderer("ask_user", |call| {
            let question = call.argument("question").unwrap_or_default();
            let mut lines = vec![call.header("Ask", question)];
            lines.extend(call.result_lines(RESULT_LINES));
            lines
        });
}

/// Opens a picker of the choices, each running `/answer #id <choice>`, and
/// "Other…" for free text; a question without choices only says it waits
/// for the user's words.
fn pick(
    asked: On<AskedUser>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let question = &asked.question;
    if question.choices.is_empty() {
        let text = format!("{} Type your answer and press Enter.", question.text);
        notices.write(Notice::info(None, text));
        return;
    }
    let id = question.id;
    let mut items: Vec<PickItem> = question
        .choices
        .iter()
        .map(|choice| PickItem {
            label: choice.clone(),
            command: format!("answer #{id} {choice}"),
        })
        .collect();
    if question.free_text {
        items.push(PickItem {
            label: "Other…".to_owned(),
            command: format!("answer #{id}"),
        });
    }
    picks.write(PickRequest {
        agent: question.agent,
        title: question.text.clone(),
        items,
        selected: 0,
    });
}
