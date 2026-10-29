use std::path::Path;

use rig_harness::rig_core::completion::message::{
    AssistantMessage, ToolFunction, ToolResultContent,
};
use rig_harness::rig_core::conversation::Author;
use rig_harness::rig_core::message::ToolName;
use serde_json::json;

use super::*;

/// A text block.
fn text(text: &str) -> acp::ContentBlock {
    acp::ContentBlock::from(text.to_owned())
}

/// The text parts of `parts`, and how many attachments they have.
fn shown(parts: &[MessagePart]) -> (Vec<String>, usize) {
    let texts = parts.iter().filter_map(|part| match part {
        MessagePart::Text(text) => Some(text.clone()),
        MessagePart::Attachment(_) => None,
    });
    let attached = parts
        .iter()
        .filter(|part| matches!(part, MessagePart::Attachment(_)));
    (texts.collect(), attached.count())
}

#[test]
fn a_prompt_takes_text_links_text_resources_and_images_and_refuses_the_rest_whole() {
    let png = "iVBORw0KGgo=";
    let embedded = |contents| acp::ContentBlock::Resource(acp::EmbeddedResource::new(contents));
    let text_file = acp::TextResourceContents::new("fn main() {}", "file:///p/main.rs");
    let json_blob = acp::BlobResourceContents::new("eyJhIjogMX0=", "file:///p/a.json")
        .mime_type("application/json".to_owned());
    let image_blob =
        acp::BlobResourceContents::new(png, "file:///p/a.png").mime_type("image/png".to_owned());
    let blocks = [
        text("Look at"),
        acp::ContentBlock::ResourceLink(acp::ResourceLink::new("lib.rs", "file:///p/lib.rs")),
        embedded(acp::EmbeddedResourceResource::TextResourceContents(
            text_file,
        )),
        embedded(acp::EmbeddedResourceResource::BlobResourceContents(
            json_blob,
        )),
        acp::ContentBlock::Image(acp::ImageContent::new(png, "image/png")),
        embedded(acp::EmbeddedResourceResource::BlobResourceContents(
            image_blob,
        )),
    ];
    let parts = prompt(&blocks);
    let (texts, attached) = shown(parts.as_deref().unwrap_or_default());
    assert_eq!(
        texts,
        [
            "Look at",
            "[lib.rs](file:///p/lib.rs)",
            "file:///p/main.rs:\n```\nfn main() {}\n```",
            "file:///p/a.json:\n```\n{\"a\": 1}\n```",
        ]
    );
    assert_eq!(attached, 2);
    // Audio, or a binary blob, refuses the prompt, the text with it.
    let binary = acp::BlobResourceContents::new("AAEC", "file:///p/a.bin")
        .mime_type("application/octet-stream".to_owned());
    let audio = acp::ContentBlock::Audio(acp::AudioContent::new("AAEC", "audio/wav"));
    for refused in [
        audio,
        embedded(acp::EmbeddedResourceResource::BlobResourceContents(binary)),
    ] {
        assert!(prompt(&[text("Hear this"), refused]).is_err());
    }
    assert!(prompt(&[text("   ")]).is_err());
}

#[test]
fn a_compacted_conversation_replays_what_its_summary_replaced_in_place_of_the_summary() {
    let read = ToolName::new("read").ok();
    let Some(read) = read else { return };
    let call = ToolCall::from_wire("c1", ToolFunction::new(read, json!({ "path": "a.txt" })));
    let result = call.result(vec![ToolResultContent::text("a text")]);
    let asked = AssistantContent::ToolCall(call);
    let mut conversation = Conversation::from_messages([
        message::Message::user("Read a.txt."),
        message::Message::Assistant(AssistantMessage::new(vec![asked])),
        message::Message::User {
            content: vec![UserContent::ToolResult(result)],
        },
        message::Message::assistant("It says a text."),
    ]);
    let originals: Vec<Entry> = (0..4).filter_map(|at| conversation.entry(at)).collect();
    let summary = Entry::new(message::Message::user("Summary.")).by(Author::plugin("compact"));
    let spliced = conversation.splice(0..4, [summary], Author::plugin("compact"));
    conversation.push(message::Message::assistant("After it."));
    let Some(&summary) = spliced.changed.first() else {
        return;
    };
    let replaced = BTreeMap::from([(summary, originals)]);
    let updates = replay(&conversation, &replaced, Path::new("/p"));
    let shown: Vec<String> = updates
        .iter()
        .map(|update| match update {
            acp::SessionUpdate::UserMessageChunk(chunk) => format!("user {:?}", chunk.content),
            acp::SessionUpdate::AgentMessageChunk(chunk) => format!("agent {:?}", chunk.content),
            acp::SessionUpdate::ToolCall(call) => {
                format!(
                    "call {} {:?} {:?}",
                    call.tool_call_id, call.kind, call.locations
                )
            }
            acp::SessionUpdate::ToolCallUpdate(update) => {
                format!("done {} {:?}", update.tool_call_id, update.fields.status)
            }
            other => format!("{other:?}"),
        })
        .collect();
    let user = |said: &str| format!("user {:?}", text(said));
    let agent = |said: &str| format!("agent {:?}", text(said));
    let location = vec![acp::ToolCallLocation::new(Path::new("/p").join("a.txt"))];
    assert_eq!(
        shown,
        [
            user("Read a.txt."),
            format!("call c1 {:?} {location:?}", acp::ToolKind::Read),
            format!("done c1 {:?}", Some(acp::ToolCallStatus::Completed)),
            agent("It says a text."),
            agent("After it."),
        ]
    );
}

#[test]
fn a_location_is_a_local_file_made_absolute_and_none_for_another_host_or_home() {
    let at = |path: &str| {
        let located = locations(&json!({ "path": path }), Path::new("/p"));
        located
            .into_iter()
            .map(|location| location.path)
            .collect::<Vec<_>>()
    };
    assert_eq!(at("src/lib.rs"), [Path::new("/p").join("src/lib.rs")]);
    assert_eq!(at("/etc/hosts"), [Path::new("/etc/hosts")]);
    assert_eq!(at("C:\\repo\\a.rs").len(), 1);
    assert!(at("bench:~/note.txt").is_empty());
    assert!(at("~/note.txt").is_empty());
}

#[test]
fn rigs_file_tools_have_their_acp_kinds_and_a_move_both_its_files() {
    for (tool, expected) in [
        ("read", acp::ToolKind::Read),
        ("list", acp::ToolKind::Read),
        ("edit", acp::ToolKind::Edit),
        ("write", acp::ToolKind::Edit),
        ("delete", acp::ToolKind::Delete),
        ("move", acp::ToolKind::Move),
        ("search", acp::ToolKind::Search),
        ("shell", acp::ToolKind::Execute),
        ("files__add", acp::ToolKind::Other),
    ] {
        assert_eq!(kind(tool), expected, "{tool}");
    }
    let moved = locations(
        &json!({ "from": "a.txt", "to": "/q/b.txt" }),
        Path::new("/p"),
    );
    let moved: Vec<_> = moved.into_iter().map(|location| location.path).collect();
    assert_eq!(
        moved,
        [
            Path::new("/p").join("a.txt"),
            Path::new("/q/b.txt").to_owned()
        ]
    );
}
