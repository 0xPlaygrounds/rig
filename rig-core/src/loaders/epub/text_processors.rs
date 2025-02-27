use std::{convert::Infallible, error::Error};

use quick_xml::events::Event;
use quick_xml::Reader;

// ================================================================
// Implementing TextProcessor trait for post-processing epubs
// ================================================================

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
    Unescape(#[from] quick_xml::events::attributes::AttrError),

    #[error("Invalid UTF-8 sequence: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

pub struct StripXmlProcessor;

impl TextProcessor for StripXmlProcessor {
    type Error = XmlProcessingError;

    fn process(xml: &str) -> Result<String, Self::Error> {
        let mut reader = Reader::from_str(xml.trim());

        let mut result = String::with_capacity(xml.len() / 2); // Rough estimate
        let mut last_was_text = false;

        loop {
            match reader.read_event()? {
                Event::Text(e) => {
                    let text = e.unescape()?.into_owned();
                    if !text.trim().is_empty() {
                        if last_was_text {
                            result.push(' ');
                        }
                        result.push_str(&text);
                        last_was_text = true;
                    }
                }
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
