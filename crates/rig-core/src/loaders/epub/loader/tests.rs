use crate::loaders::epub::text_processors::XmlProcessingError;
use crate::loaders::epub::{RawTextProcessor, StripXmlProcessor, TextProcessor};
use crate::loaders::test_fixtures::{fixture_glob, fixture_path};

use super::EpubFileLoader;

#[test]
fn test_epub_loader_with_errors() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();
    let actual = loader
        .load_with_path()
        .ignore_errors()
        .by_chapter()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);

    let (_, chapters) = &actual[0];
    assert_eq!(chapters.len(), 3);

    for chapter in chapters {
        assert!(chapter.1.is_ok());
    }
}

#[test]
fn test_epub_loader_with_ignoring_errors() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();
    let actual = loader
        .load_with_path()
        .ignore_errors()
        .by_chapter()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);

    let (_, chapters) = &actual[0];
    assert_eq!(chapters.len(), 3);
}

#[test]
fn test_single_file() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();

    let actual = loader
        .read()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);
}

#[test]
fn test_single_file_with_path() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();

    let actual = loader
        .read_with_path()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);

    let (path, _) = &actual[0];
    assert_eq!(path, &fixture_path("dummy.epub"));
}

fn assert_stripped_xml(xml: &str, expected: &str) {
    let actual = StripXmlProcessor::process(xml);
    assert!(
        matches!(&actual, Ok(text) if text == expected),
        "input {xml:?}: expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn stripped_xml_preserves_entity_text() {
    assert_stripped_xml("<p>A&amp;B&#x21;</p>", "A&B!");
}

#[test]
fn stripped_xml_resolves_predefined_and_adjacent_references_once() {
    for (xml, expected) in [
        ("<p>&lt;&gt;&amp;&apos;&quot;</p>", "<>&'\""),
        ("<p>&amp;&#38;&#x26;</p>", "&&&"),
        ("<p>&amp;lt;&amp;#33;</p>", "&lt;&#33;"),
        ("&amp;&#33;", "&!"),
    ] {
        assert_stripped_xml(xml, expected);
    }
}

#[test]
fn stripped_xml_resolves_decimal_and_hex_unicode_references() {
    for (xml, expected) in [
        ("<p>&#33;&#233;&#128512;</p>", "!é😀"),
        ("<p>&#x21;&#xE9;&#x1F600;</p>", "!é😀"),
        ("<p>é&#128512;終</p>", "é😀終"),
    ] {
        assert_stripped_xml(xml, expected);
    }
}

#[test]
fn stripped_xml_preserves_whitespace_around_references() {
    for (xml, expected) in [
        ("<p> &amp; </p>", " & "),
        ("<p>&amp; \t\n &lt;</p>", "& \t\n <"),
        ("<p>A &amp; B&#33; </p>", "A & B! "),
        ("<p>&#32;A&#x20;</p>", " A "),
        ("<p>A&#9;&#xA;B</p>", "A\t\nB"),
        ("<p> \t\n </p>", ""),
        ("<p> &#32;&#x9; </p>", ""),
    ] {
        assert_stripped_xml(xml, expected);
    }
}

#[test]
fn stripped_xml_preserves_cdata_literals_and_boundaries() {
    for (xml, expected) in [
        (
            "<p><![CDATA[&amp;&#33;&unknown;]]></p>",
            "&amp;&#33;&unknown;",
        ),
        ("<p>A<![CDATA[B]]>C</p>", "A B C"),
        ("<p>A&amp;<![CDATA[B]]>&lt;C</p>", "A& B <C"),
        ("<p><![CDATA[A]]><![CDATA[B]]></p>", "A B"),
        ("<p>A<![CDATA[ ]]>&amp;B</p>", "A &B"),
        ("<p><![CDATA[A]]> &amp; </p>", "A  & "),
    ] {
        assert_stripped_xml(xml, expected);
    }
}

#[test]
fn stripped_xml_preserves_markup_boundaries() {
    for (xml, expected) in [
        ("<p>A<b>B</b>C</p><p>D</p>", "ABCD"),
        ("<p>A&amp;<b>B&#33;</b>C</p>", "A&B!C"),
        ("<p>A<br/>&amp;B</p>", "A&B"),
        ("<p>A<!-- comment -->&amp;B</p>", "A&B"),
        ("<p>A<?note value?>&amp;B</p>", "A&B"),
        ("<p>A<b> </b>&amp;B</p>", "A&B"),
    ] {
        assert_stripped_xml(xml, expected);
    }
}

#[test]
fn stripped_xml_rejects_unknown_references() {
    for name in ["unknown", "nbsp", "AMP", ""] {
        let xml = format!("<p>A&{name};B</p>");
        let actual = StripXmlProcessor::process(&xml);
        assert!(
            matches!(
                &actual,
                Err(XmlProcessingError::Xml(quick_xml::Error::Escape(
                    quick_xml::escape::EscapeError::UnrecognizedEntity(_, entity)
                ))) if entity == name
            ),
            "input {xml:?}: expected an unknown entity error, got {actual:?}"
        );
    }
}

#[test]
fn stripped_xml_rejects_invalid_character_references() {
    for reference in [
        "&#;",
        "&#x;",
        "&#xyz;",
        "&#12z;",
        "&#+33;",
        "&#-1;",
        "&#0;",
        "&#x0;",
        "&#55296;",
        "&#xD800;",
        "&#1114112;",
        "&#x110000;",
        "&#99999999999999999999;",
    ] {
        let xml = format!("<p>A{reference}B</p>");
        let actual = StripXmlProcessor::process(&xml);
        assert!(
            matches!(
                &actual,
                Err(XmlProcessingError::Xml(quick_xml::Error::Escape(
                    quick_xml::escape::EscapeError::InvalidCharRef(_)
                )))
            ),
            "input {xml:?}: expected an invalid character reference error, got {actual:?}"
        );
    }
}

#[test]
fn stripped_xml_rejects_unterminated_references() {
    for xml in ["<p>A&amp</p>", "A&#33", "A&"] {
        let actual = StripXmlProcessor::process(xml);
        assert!(actual.is_err(), "input {xml:?}: got {actual:?}");
    }
}
