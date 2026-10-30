use std::path::{Path, PathBuf};

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

/// `path` made absolute, as a location names it.
fn absolute(path: impl AsRef<Path>) -> PathBuf {
    std::path::absolute(path.as_ref()).unwrap_or_else(|_| path.as_ref().to_owned())
}

/// A read of `path` under the call id `id`, and its result `said`.
fn read(id: &str, path: &str, said: &str) -> Option<[message::Message; 2]> {
    let read = ToolName::new("read").ok()?;
    let call = ToolCall::from_wire(id, ToolFunction::new(read, json!({ "path": path })));
    let result = call.result(vec![ToolResultContent::text(said)]);
    Some([
        message::Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )])),
        message::Message::User {
            content: vec![UserContent::ToolResult(result)],
        },
    ])
}

#[test]
fn a_compacted_conversation_replays_what_its_summary_replaced_in_place_of_the_summary() {
    let (Some([asked, answered]), Some(again)) =
        (read("c1", "a.txt", "a text"), read("c1", "b.txt", "b text"))
    else {
        return;
    };
    let mut conversation = Conversation::from_messages([
        message::Message::user("Read a.txt."),
        asked,
        answered,
        message::Message::assistant("It says a text."),
    ]);
    let originals: Vec<Entry> = (0..4).filter_map(|at| conversation.entry(at)).collect();
    let summary = Entry::new(message::Message::user("Summary.")).by(Author::plugin("compact"));
    let spliced = conversation.splice(0..4, [summary], Author::plugin("compact"));
    conversation.push(message::Message::assistant("After it."));
    let skill = Entry::new(message::Message::user("<skill>Read b.txt.</skill>"))
        .by(Author::user().titled("skill b"));
    conversation.push(skill);
    conversation.push(Entry::new(message::Message::user("Rig's own note.")).by(Author::rig()));
    // A provider may use a call's id again in a later turn.
    for message in again {
        conversation.push(message);
    }
    let (Some(&summary), Some(first)) = (
        spliced.changed.first(),
        originals.get(1).and_then(|asked| asked.id),
    ) else {
        return;
    };
    let first = call_id("main", first, 0);
    // The id a live call of the last turn has.
    let second = call_id_in("main", &conversation, &CallId::from_wire("c1"), |_| false);
    let second = second.unwrap_or_default();
    assert_ne!(first, second);
    let replaced = BTreeMap::from([(summary, originals)]);
    let updates = replay("main", &conversation, &replaced, Path::new("/p"));
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
                let said = update.fields.raw_output.clone().unwrap_or_default();
                format!("done {} {said}", update.tool_call_id)
            }
            other => format!("{other:?}"),
        })
        .collect();
    let user = |said: &str| format!("user {:?}", text(said));
    let agent = |said: &str| format!("agent {:?}", text(said));
    let location = |path| {
        vec![acp::ToolCallLocation::new(absolute(
            Path::new("/p").join(path),
        ))]
    };
    assert_eq!(
        shown,
        [
            user("Read a.txt."),
            format!(
                "call {first} {:?} {:?}",
                acp::ToolKind::Read,
                location("a.txt")
            ),
            format!("done {first} \"a text\""),
            agent("It says a text."),
            agent("After it."),
            user("<skill>Read b.txt.</skill>"),
            format!(
                "call {second} {:?} {:?}",
                acp::ToolKind::Read,
                location("b.txt")
            ),
            format!("done {second} \"b text\""),
        ]
    );
}

#[test]
fn a_call_is_known_by_where_it_sits_whatever_the_providers_id() {
    let mut messages = Vec::new();
    for (id, said) in [("c1", "one"), ("c1#2", "two"), ("c1", "three")] {
        let Some(pair) = read(id, "a.txt", said) else {
            return;
        };
        messages.extend(pair);
    }
    // Two calls of one reply, with one id, as rig lets a reply repeat it.
    let (Some(read), Some(list)) = (ToolName::new("read").ok(), ToolName::new("list").ok()) else {
        return;
    };
    let both = [
        ToolCall::from_wire("c9", ToolFunction::new(read, json!({ "path": "b.txt" }))),
        ToolCall::from_wire("c9", ToolFunction::new(list, json!({ "path": "." }))),
    ];
    let results = both.iter().zip(["four", "five"]).map(|(call, said)| {
        UserContent::ToolResult(call.result(vec![ToolResultContent::text(said)]))
    });
    let results = results.collect();
    let asked = both.into_iter().map(AssistantContent::ToolCall).collect();
    messages.push(message::Message::Assistant(AssistantMessage::new(asked)));
    messages.push(message::Message::User { content: results });
    let conversation = Conversation::from_messages(messages);
    let updates = replay("main", &conversation, &BTreeMap::new(), Path::new("/p"));
    let done: Vec<_> = updates
        .into_iter()
        .filter_map(|update| match update {
            acp::SessionUpdate::ToolCallUpdate(update) => {
                let said = update.fields.raw_output.unwrap_or_default();
                Some(format!("{} {said}", update.tool_call_id))
            }
            _ => None,
        })
        .collect();
    let at = |index, n, said| {
        let id = conversation.id(index).map(|id| call_id("main", id, n));
        format!("{} {said}", id.unwrap_or_default())
    };
    assert_eq!(
        done,
        [
            at(0, 0, "\"one\""),
            at(2, 0, "\"two\""),
            at(4, 0, "\"three\""),
            at(6, 0, "\"four\""),
            at(6, 1, "\"five\""),
        ]
    );
    // Live calls find the same ids: each the first not taken yet.
    let repeated = CallId::from_wire("c9");
    let first = call_id_in("main", &conversation, &repeated, |_| false);
    let (first_id, second_id) = (at(6, 0, ""), at(6, 1, ""));
    assert_eq!(first.as_deref(), Some(first_id.trim_end()));
    let second = call_id_in("main", &conversation, &repeated, |id| {
        id == first_id.trim_end()
    });
    assert_eq!(second.as_deref(), Some(second_id.trim_end()));
}

#[test]
fn an_embedded_resource_is_fenced_longer_than_its_own_fences() {
    let fence = "`".repeat(3);
    let text = format!("# Notes\n{fence}sh\nls\n{fence}");
    let longer = "`".repeat(4);
    assert_eq!(
        fenced("file:///p/notes.md", &text),
        format!("file:///p/notes.md:\n{longer}\n{text}\n{longer}")
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
    assert_eq!(
        at("src/lib.rs"),
        [absolute(Path::new("/p").join("src/lib.rs"))]
    );
    assert_eq!(at("/etc/hosts"), [absolute("/etc/hosts")]);
    assert_eq!(at("C:\\repo\\a.rs").len(), 1);
    // Drive-relative on Windows, a file name elsewhere.
    let drive = at("C:fixture.txt");
    assert!(
        drive.len() == 1 && drive.iter().all(|path| path.is_absolute()),
        "{drive:?}"
    );
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
            absolute(Path::new("/p").join("a.txt")),
            absolute("/q/b.txt")
        ]
    );
}
