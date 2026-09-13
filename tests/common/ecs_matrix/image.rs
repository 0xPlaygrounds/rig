//! Image-bearing user messages beside the shared matrix's effect-log
//! oracle: one committed fixture, the user content each cell sends, and
//! the facts both interpreters must show on top of record parity — the
//! image's bytes (or URL), media type and options verbatim in every
//! request and in committed history, user content in its given order,
//! the image only where the user put it, and the add tool run once.
#![allow(dead_code, reason = "the image matrix runs on four of the six wires")]

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::effect::{EffectKind, Outcome};
use rig::effect_log::EffectLog;
use rig::message::{AssistantContent, DocumentSourceKind, ImageMediaType, Message, UserContent};

use super::cells::Cell;

/// `tests/data/red_square.png`: 64×64 RGB, a 32×32 solid red square on
/// white, 141 bytes.
pub(crate) const FIXTURE_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/red_square.png");
pub(crate) const FIXTURE_LEN: usize = 141;
pub(crate) const FIXTURE_SHA256: &str =
    "face83b5c536c3ee99512975104576e77636617e9b6736f2a8d16774f84a50cc";
/// The same bytes, public and immutable: the fixture at the commit that
/// added it.
pub(crate) const FIXTURE_URL: &str = "https://raw.githubusercontent.com/0xPlaygrounds/rig/b06bf77efa015ee5637b46bbdbcab07cfe57d4e0/tests/data/red_square.png";

pub(crate) const IMAGE_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
pub(crate) const COLOR_PROMPT: &str =
    "Describe the dominant non-background color of this image in one short sentence.";
pub(crate) const MIXED_BEFORE: &str = "Marker one: the same image follows twice.";
pub(crate) const MIXED_BETWEEN: &str = "Marker two: here it is again.";
pub(crate) const MIXED_AFTER: &str =
    "Marker three: describe the dominant non-background color of the image in one short sentence.";
pub(crate) const IMAGE_TOOL_PREAMBLE: &str =
    "You are a concise assistant with an add tool. Use the tool when asked, then answer directly.";
pub(crate) const IMAGE_TOOL_PROMPT: &str = "Look at the image, then use the add tool to add 17 and 25. Reply in one short sentence giving the image's dominant non-background color and the sum.";
pub(crate) const FOLLOWUP_PROMPT: &str =
    "What shape did the image I sent earlier contain? Answer in one short sentence.";

/// What the cell's user message holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageCase {
    /// The image, then the prompt.
    Text,
    /// Text, the image, text, the same image, the prompt.
    MixedOrder,
    /// The image, then a prompt that asks for the add tool.
    Tool,
    /// [`Self::Text`], then a text-only second prompt about the image.
    Followup,
}

/// How the image is carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageSource {
    /// The fixture's bytes, base64, with its media type.
    Inline,
    /// [`FIXTURE_URL`], with its media type.
    Url,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageCell {
    pub case: ImageCase,
    pub source: ImageSource,
}

impl ImageCell {
    /// How many times the user message carries the image.
    fn occurrences(self) -> usize {
        match self.case {
            ImageCase::MixedOrder => 2,
            _ => 1,
        }
    }
}

pub(crate) fn fixture_bytes() -> Vec<u8> {
    let bytes = std::fs::read(FIXTURE_PATH).expect("the fixture is committed");
    assert_eq!(bytes.len(), FIXTURE_LEN, "the fixture's length");
    assert_eq!(sha256(&bytes), FIXTURE_SHA256, "the fixture's hash");
    bytes
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The image part the cell sends.
fn part(source: ImageSource) -> UserContent {
    match source {
        ImageSource::Inline => UserContent::image_base64(
            BASE64_STANDARD.encode(fixture_bytes()),
            Some(ImageMediaType::PNG),
            None,
        ),
        ImageSource::Url => UserContent::image_url(FIXTURE_URL, Some(ImageMediaType::PNG), None),
    }
}

/// The user content the cell sends, in order; `prompt` is the program's
/// prompt (the text beside the image).
pub(crate) fn user_content(image: ImageCell, prompt: &str) -> Vec<UserContent> {
    match image.case {
        ImageCase::Text | ImageCase::Tool | ImageCase::Followup => {
            vec![part(image.source), UserContent::text(prompt)]
        }
        ImageCase::MixedOrder => vec![
            UserContent::text(MIXED_BEFORE),
            part(image.source),
            UserContent::text(MIXED_BETWEEN),
            part(image.source),
            UserContent::text(prompt),
        ],
    }
}

/// The cell's first prompt as the message both interpreters send.
pub(crate) fn prompt_message(image: ImageCell, prompt: &str) -> Message {
    Message::User {
        content: user_content(image, prompt),
    }
}

/// Every image part of `message`, decoded to what it carries.
fn images(message: &Message) -> Vec<&rig::message::Image> {
    match message {
        Message::User { content } => content
            .iter()
            .filter_map(|part| match part {
                UserContent::Image(image) => Some(image),
                _ => None,
            })
            .collect(),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Image(image) => Some(image),
                _ => None,
            })
            .collect(),
        Message::System { .. } => Vec::new(),
    }
}

/// The image is the fixture: its decoded bytes (or its URL), its media
/// type, no options.
fn assert_image(image: &rig::message::Image, source: ImageSource, what: &str) {
    match (&image.data, source) {
        (DocumentSourceKind::Base64(data), ImageSource::Inline) => {
            let bytes = BASE64_STANDARD.decode(data).expect("the image is base64");
            assert_eq!(bytes.len(), FIXTURE_LEN, "{what}: the image's length");
            assert_eq!(sha256(&bytes), FIXTURE_SHA256, "{what}: the image's bytes");
        }
        (DocumentSourceKind::Url(url), ImageSource::Url) => {
            assert_eq!(url, FIXTURE_URL, "{what}: the image's URL");
        }
        (data, _) => panic!("{what}: the image's source is {data:?}, not {source:?}"),
    }
    assert_eq!(
        image.media_type,
        Some(ImageMediaType::PNG),
        "{what}: the media type"
    );
    assert_eq!(image.detail, None, "{what}: no detail was asked");
    assert_eq!(
        image.additional_params, None,
        "{what}: no options were asked"
    );
}

/// The messages a request or a committed history holds, checked as the
/// cell's: the first user message is exactly the content sent (order,
/// text markers, every image), no later message carries an image, and
/// the cell's second prompt is the text it was.
fn assert_messages(cell: &Cell, image: ImageCell, messages: &[Message], what: &str) {
    let first = messages
        .iter()
        .find(|message| matches!(message, Message::User { .. }))
        .unwrap_or_else(|| panic!("{what}: a user message"));
    let expected = prompt_message(image, cell.program.prompt);
    assert_eq!(
        *first, expected,
        "{what}: the image-bearing user message, verbatim"
    );
    let carried = images(first);
    assert_eq!(
        carried.len(),
        image.occurrences(),
        "{what}: the image count"
    );
    for (n, carried) in carried.iter().enumerate() {
        assert_image(carried, image.source, &format!("{what}: image {n}"));
    }
    let mut users = messages
        .iter()
        .filter(|message| matches!(message, Message::User { .. }));
    users.next();
    for (n, later) in users.enumerate() {
        assert!(
            images(later).is_empty(),
            "{what}: user message {} carries no image: {later:?}",
            n + 1
        );
    }
    for message in messages
        .iter()
        .filter(|message| matches!(message, Message::Assistant { .. }))
    {
        assert!(
            images(message).is_empty(),
            "{what}: the assistant sent no image: {message:?}"
        );
    }
}

fn completion_requests(log: &EffectLog) -> Vec<&rig::completion::CompletionRequest> {
    log.records
        .iter()
        .filter_map(|record| match &record.kind {
            EffectKind::Completion { request, .. } => Some(request),
            _ => None,
        })
        .collect()
}

/// Every request of the log carries the cell's image message as its
/// first user message and nothing else carries an image; the tool cell
/// ran `add` exactly once, to 42, and its second request saw the call
/// and the result after the image; the follow-up's second run saw the
/// image once, in the first user message, and its text prompt after.
pub(crate) fn assert_log(cell: &Cell, log: &EffectLog) {
    let Some(image) = cell.image else { return };
    let requests = completion_requests(log);
    assert!(!requests.is_empty(), "{}: a completion", cell.name);
    for (n, request) in requests.iter().enumerate() {
        assert_messages(
            cell,
            image,
            &request.chat_history,
            &format!("{}: request {n}", cell.name),
        );
    }
    let tools: Vec<_> = log
        .records
        .iter()
        .filter_map(|record| match (&record.kind, &record.outcome) {
            (EffectKind::ToolCall { name, args }, Ok(Outcome::ToolResult { result })) => {
                Some((name, args, result))
            }
            (EffectKind::ToolCall { .. }, outcome) => {
                panic!("{}: the tool answered: {outcome:?}", cell.name)
            }
            _ => None,
        })
        .collect();
    match image.case {
        ImageCase::Tool => {
            assert_eq!(tools.len(), 1, "{}: add runs once", cell.name);
            let (name, args, result) = tools[0];
            assert_eq!(name, "add");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(args).expect("the call's arguments"),
                serde_json::json!({"x": 17, "y": 25}),
                "{}: the add call",
                cell.name
            );
            assert_eq!(result.output().render(), "42", "{}: the sum", cell.name);
            assert_eq!(requests.len(), 2, "{}: two completions", cell.name);
            let second = &requests[1].chat_history;
            let call_id = second
                .iter()
                .find_map(|message| match message {
                    Message::Assistant { content, .. } => {
                        content.iter().find_map(|part| match part {
                            AssistantContent::ToolCall(call) => Some(call.id.clone()),
                            _ => None,
                        })
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{}: the second request carries the call", cell.name));
            let answered = second.iter().any(|message| {
                match message {
                Message::User { content } => content.iter().any(|part| {
                    matches!(part, UserContent::ToolResult(result) if result.call == call_id)
                }),
                _ => false,
            }
            });
            assert!(
                answered,
                "{}: the second request carries the result",
                cell.name
            );
        }
        ImageCase::Followup => {
            assert!(tools.is_empty());
            assert_eq!(requests.len(), 2, "{}: two completions", cell.name);
            let second = &requests[1].chat_history;
            let last_user = second
                .iter()
                .rev()
                .find(|message| matches!(message, Message::User { .. }))
                .expect("the second prompt");
            assert_eq!(
                *last_user,
                Message::user(cell.program.second_prompt.expect("the follow-up")),
                "{}: the second prompt is text only",
                cell.name
            );
            assert!(
                second
                    .iter()
                    .filter(|message| matches!(message, Message::User { .. }))
                    .count()
                    == 2,
                "{}: the second request holds both prompts",
                cell.name
            );
        }
        ImageCase::Text | ImageCase::MixedOrder => {
            assert!(tools.is_empty());
            assert_eq!(requests.len(), 1, "{}: one completion", cell.name);
        }
    }
    assert_answer(cell, &super::corpus::golden_answer(log));
}

/// The committed transcript of a run, checked as the requests were. The
/// follow-up's second run owns only its own turn — its prompt and its
/// answer; the loaded conversation is the request's history, not the
/// run's transcript — so a transcript that opens with the text prompt is
/// the second run's, and carries no image at all.
pub(crate) fn assert_history(cell: &Cell, history: &[Message], what: &str) {
    let Some(image) = cell.image else { return };
    let second = cell.program.second_prompt.map(Message::user);
    let first_user = history
        .iter()
        .find(|message| matches!(message, Message::User { .. }));
    if image.case == ImageCase::Followup && first_user == second.as_ref() {
        for message in history {
            assert!(
                images(message).is_empty(),
                "{what}: the second run's own transcript re-sends no image: {message:?}"
            );
        }
    } else {
        assert_messages(cell, image, history, what);
    }
    assert!(
        history
            .iter()
            .any(|message| matches!(message, Message::Assistant { .. })),
        "{what}: the answer is committed"
    );
}

/// A tolerant grounding check: the model saw a red thing (with the tool,
/// the sum; asked later about the shape, a square). The regression oracle
/// is the golden, not this phrase.
pub(crate) fn assert_answer(cell: &Cell, answer: &str) {
    let Some(image) = cell.image else { return };
    let lower = answer.to_ascii_lowercase();
    if image.case == ImageCase::Followup {
        assert!(
            lower.contains("square") || lower.contains("red"),
            "{}: the answer recalls the image: {answer:?}",
            cell.name
        );
        return;
    }
    assert!(
        lower.contains("red"),
        "{}: the answer names the color: {answer:?}",
        cell.name
    );
    if image.case == ImageCase::Tool {
        assert!(
            lower.contains("42"),
            "{}: the answer states the sum: {answer:?}",
            cell.name
        );
    }
}
