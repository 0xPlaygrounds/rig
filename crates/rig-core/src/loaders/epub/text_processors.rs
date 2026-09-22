use std::{convert::Infallible, error::Error};

use quick_xml::Reader;
use quick_xml::escape::{resolve_xml_entity, unescape_with};
use quick_xml::events::Event;

pub trait TextProcessor {
    type Error: Error + 'static;

    fn process(text: &str) -> Result<String, Self::Error>;
}

pub struct RawTextProcessor;

impl TextProcessor for RawTextProcessor {
    type Error = Infallible;

    fn process(text: &str) -> Result<String, Self::Error> {
        Ok(text.to_string())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum XmlProcessingError {
    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("Failed to unescape XML entity: {0}")]
    Encoding(#[from] quick_xml::encoding::EncodingError),

    #[error("Invalid UTF-8 sequence: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

/// Strip markup while resolving predefined XML entities and numeric character references.
/// Unknown references are errors; DTD entity declarations are not expanded.
pub struct StripXmlProcessor;

impl TextProcessor for StripXmlProcessor {
    type Error = XmlProcessingError;

    fn process(xml: &str) -> Result<String, Self::Error> {
        let mut reader = Reader::from_str(xml.trim());

        let mut result = String::with_capacity(xml.len() / 2);
        let mut last_was_text = false;
        let mut text = String::new();

        loop {
            let event = reader.read_event()?;
            match &event {
                Event::Text(e) => {
                    text.push_str(&e.decode()?);
                    continue;
                }
                Event::GeneralRef(e) => {
                    let reference = format!("&{};", e.decode()?);
                    text.push_str(
                        &unescape_with(&reference, resolve_xml_entity)
                            .map_err(quick_xml::Error::from)?,
                    );
                    continue;
                }
                _ => {}
            }

            // References split a single text node into events, so defer whitespace
            // filtering until the whole node is available, including spaces around refs.
            if !text.trim().is_empty() {
                if last_was_text {
                    result.push(' ');
                }
                result.push_str(&text);
                last_was_text = true;
            }
            text.clear();

            // CDATA stays literal; markup resets text adjacency.
            match event {
                Event::CData(e) => {
                    let text = String::from_utf8(e.into_inner().into_owned())?;
                    if !text.trim().is_empty() {
                        if last_was_text {
                            result.push(' ');
                        }
                        result.push_str(&text);
                        last_was_text = true;
                    }
                }
                Event::Eof => break,
                _ => {
                    last_was_text = false;
                }
            }
        }

        Ok(result)
    }
}
