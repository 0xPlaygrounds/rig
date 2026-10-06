use serde_json::json;

use super::{Citation, Source, SourceLocation, Span, attach, resolve};
use crate::message::{AssistantContent, Text};
use crate::wire::{SpanUnit, WireCitation, WireSpan};

fn url(url: &str) -> Source {
    Source::new(SourceLocation::Url {
        url: url.to_owned(),
    })
}

fn span(start: u64, end: u64, unit: SpanUnit) -> WireSpan {
    WireSpan::new(start, end, unit)
}

/// "é" is two UTF-8 bytes and one UTF-16 unit; "𝄞" is four bytes and two
/// UTF-16 units.
const TEXT: &str = "é 𝄞 Dock Seven";

#[test]
fn each_unit_resolves_to_the_same_bytes() {
    let bytes = TEXT.find("Dock").expect("the text holds it");
    let expected = Span {
        start: bytes,
        end: bytes + "Dock Seven".len(),
    };
    for wire in [
        span(8, 18, SpanUnit::Bytes),
        span(4, 14, SpanUnit::Chars),
        span(5, 15, SpanUnit::Utf16),
    ] {
        assert_eq!(
            resolve(TEXT, &wire.clone().quoted("Dock Seven")),
            Ok(expected),
            "{wire:?}"
        );
    }
}

#[test]
fn a_span_that_splits_a_character_or_overruns_does_not_resolve() {
    for wire in [
        span(1, 3, SpanUnit::Bytes),
        span(3, 4, SpanUnit::Utf16),
        span(0, 99, SpanUnit::Chars),
        span(0, 99, SpanUnit::Utf16),
        span(4, 2, SpanUnit::Chars),
        span(0, u64::MAX, SpanUnit::Bytes),
    ] {
        assert!(resolve(TEXT, &wire).is_err(), "{wire:?}");
    }
}

#[test]
fn a_span_must_cover_the_text_the_provider_quoted() {
    // Character offsets read as bytes land elsewhere: the quote catches it.
    let read_as_bytes = span(4, 14, SpanUnit::Bytes).quoted("Dock Seven");
    assert!(resolve(TEXT, &read_as_bytes).is_err());
}

#[test]
fn attach_keeps_whole_block_citations_and_drops_unresolved_ones() {
    let mut text = Text::new(TEXT);
    attach(
        &mut text,
        Vec::new(),
        vec![
            WireCitation::new(None, vec![url("https://a.example")]),
            WireCitation::new(
                Some(span(4, 14, SpanUnit::Chars).quoted("Dock Seven")),
                vec![url("https://b.example")],
            ),
            WireCitation::new(
                Some(span(4, 14, SpanUnit::Chars).quoted("Dock Eight")),
                vec![url("https://c.example")],
            ),
        ],
        "test",
        0,
    );
    let cited: Vec<_> = text
        .citations()
        .iter()
        .map(|citation| text.cited(citation))
        .collect();
    assert_eq!(cited, [Some(TEXT), Some("Dock Seven")]);
}

#[test]
fn an_edited_text_reads_no_citations() {
    let text = Text::new("Dock Seven");
    let citation = Citation::new([url("https://a.example")]);
    let mut text = text.with_citations([citation]);
    assert_eq!(text.citations().len(), 1);
    text.text.push('!');
    assert!(text.citations().is_empty());
    text.text.pop();
    assert_eq!(
        text.citations().len(),
        1,
        "the original text reads them again"
    );
    text.clear_citations();
    assert!(text.citations().is_empty());
}

#[test]
fn a_span_is_made_only_on_character_boundaries() {
    let text = Text::new(TEXT);
    assert!(text.span(0..2).is_some());
    assert!(text.span(0..1).is_none());
    let (start, end) = (2, 0);
    assert!(text.span(start..end).is_none());
    assert!(text.span(0..99).is_none());

    // A span made for another text is dropped when it does not fit.
    let other = Text::new("a longer text than the first").span(20..28);
    let mut citation = Citation::new([url("https://a.example")]);
    citation.span = other;
    assert!(
        Text::new("short")
            .with_citations([citation])
            .citations()
            .is_empty()
    );
}

#[test]
fn citations_never_enter_the_fingerprint() {
    let plain = AssistantContent::Text(Text::new("Dock Seven"));
    let cited = AssistantContent::Text(
        Text::new("Dock Seven").with_citations([Citation::new([url("https://a.example")])]),
    );
    assert_eq!(plain.fingerprint(), cited.fingerprint());
}

#[test]
fn text_without_citations_serializes_as_before() {
    let block = AssistantContent::Text(Text::new("hi"));
    assert_eq!(
        serde_json::to_value(&block).expect("serializes"),
        json!({"type": "text", "text": "hi"})
    );
}

#[test]
fn citations_round_trip_and_unreadable_ones_load_as_none() {
    let text = Text::new("Dock Seven");
    let mut citation = Citation::new([url("https://a.example").title("Docks")]);
    citation.span = text.span(0..4);
    let text = text.with_citations([citation]);
    let stored = serde_json::to_value(&text).expect("serializes");
    let loaded: Text = serde_json::from_value(stored.clone()).expect("loads");
    assert_eq!(loaded, text);
    assert_eq!(loaded.citations().len(), 1);

    let mut malformed = stored.clone();
    malformed["citations"] = json!({"list": "not a list"});
    let loaded: Text = serde_json::from_value(malformed).expect("loads");
    assert!(loaded.citations().is_empty());
    assert_eq!(loaded.text, "Dock Seven");

    // A stored span that does not fit its text is dropped on load.
    let mut overrun = stored;
    overrun["citations"]["list"][0]["span"] = json!({"start": 0, "end": 99});
    let loaded: Text = serde_json::from_value(overrun).expect("loads");
    assert!(loaded.citations().is_empty());
}
