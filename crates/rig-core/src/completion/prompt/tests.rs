use super::*;
use crate::message::Message;

/// `text` renders exactly what the kernel's string rendering sent before
/// the prompt became data: the trimmed role, then each non-blank section
/// trimmed between its tags, a blank line apart, with nothing before the
/// first block when the role is empty.
#[test]
fn text_renders_the_role_and_tagged_sections() {
    let prompt = SystemPrompt {
        role: "  You are rig.\n".into(),
        sections: vec![
            Section::new("tool_rules", "- Read before you edit.\n- Run the tests."),
            Section::new("blank", " \n "),
            Section::new("project_instructions", "\nUse nextest.\n"),
        ],
    };
    assert_eq!(
        prompt.text(),
        "You are rig.\n\n<tool_rules>\n- Read before you edit.\n- Run the tests.\n</tool_rules>\
         \n\n<project_instructions>\nUse nextest.\n</project_instructions>"
    );

    let sections_only = SystemPrompt {
        role: String::new(),
        sections: vec![Section::new("environment", "Monday")],
    };
    assert_eq!(
        sections_only.text(),
        "<environment>\nMonday\n</environment>"
    );
    assert_eq!(SystemPrompt::default().text(), "");

    let request = crate::completion::CompletionRequest::new("hi").with_system(&prompt);
    assert_eq!(
        request.chat_history.first(),
        Some(&Message::system(prompt.text()).into())
    );
}
