//! The Files API: upload media once and refer to it by URI. A file's URI
//! goes in a message as [`DocumentSourceKind::FileId`](crate::message::DocumentSourceKind::FileId),
//! and Gemini deletes files after 48 hours.
//!
//! ```no_run
//! use rig_core::providers::gemini::Gemini;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let files = Gemini::from_env()?.files();
//! let file = files
//!     .upload(std::fs::read("report.pdf")?, "application/pdf", Some("report"))
//!     .await?;
//! println!("{:?}", file.uri);
//! # Ok(())
//! # }
//! ```

use serde::Deserialize;

use super::api::{self, Recognized};
use crate::error::{EncodeError, ProviderError};
use crate::operation::FileStore;
use crate::providers::internal::wire::{classify_or, classify_untyped_line};
use crate::providers::internal::with_query_pairs;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

const FILES_PATH: &str = "/v1beta/files";

/// Gemini caps a page of `files` at 100.
const MAX_PAGE_SIZE: usize = 100;

/// The boundary between an upload's metadata and its bytes.
const BOUNDARY: &str = "rig-gemini-upload-4f2c8a";

/// One Files API verb.
#[derive(Debug)]
pub enum FileRequest {
    /// Upload `bytes` of `mime_type` in one request; answers with the file.
    Upload {
        /// The file's bytes.
        bytes: Vec<u8>,
        /// Its IANA media type.
        mime_type: String,
        /// A name for listings.
        display_name: Option<String>,
    },
    /// Read a file by name, `files/<id>` or `<id>`.
    Get(String),
    /// One page of the listing, after `page_token` when continuing.
    List {
        /// The previous page's cursor.
        page_token: Option<String>,
    },
    /// Delete a file by name.
    Delete(String),
}

/// A file, a page of files, or an empty acknowledgement.
#[derive(Clone, Debug, Default)]
pub enum FileReply {
    /// `upload` and `get`: the file.
    File(Box<api::File>),
    /// `list`: one page.
    Page(api::ListFilesResponse),
    /// `delete`, or an empty listing.
    #[default]
    Acknowledged,
}

impl FileReply {
    /// The file an upload or read returned.
    pub fn file(self) -> Result<api::File, ProviderError> {
        match self {
            Self::File(file) => Ok(*file),
            _ => Err(ProviderError::Response(
                "the Files API reply carried no file".to_owned(),
            )),
        }
    }

    /// The cursor of the next page, when there is one.
    pub fn next_page_token(&self) -> Option<String> {
        match self {
            Self::Page(page) => page
                .next_page_token
                .clone()
                .filter(|token| !token.is_empty()),
            _ => None,
        }
    }

    /// The files of one listing page.
    pub fn entries(self) -> Result<Vec<api::File>, ProviderError> {
        match self {
            Self::Page(page) => Ok(page.files),
            Self::Acknowledged => Ok(Vec::new()),
            Self::File(_) => Err(ProviderError::Response(
                "the Files API reply carried a file, not a listing page".to_owned(),
            )),
        }
    }
}

/// The Files API wire.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Files {
    /// The key and the API root.
    pub provider: super::GeminiConfig,
    /// Requested entries per listing page.
    pub page_size: usize,
}

impl Files {
    /// The wire over `provider`.
    pub fn new(provider: super::GeminiConfig) -> Self {
        Self {
            provider,
            page_size: MAX_PAGE_SIZE,
        }
    }
}

impl super::GeminiConfig {
    /// The Files API wire.
    pub(crate) fn files(&self) -> Files {
        Files::new(self.clone())
    }
}

/// `/v1beta/files/<id>` from a name, refusing anything that could retarget
/// the path.
pub(crate) fn file_path(name: &str) -> Result<String, EncodeError> {
    let id = name.strip_prefix("files/").unwrap_or(name);
    let is_id_char = |ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_');
    if id.is_empty() || !id.chars().all(is_id_char) {
        return Err(EncodeError::request(format!(
            "`{name}` is not a file name `files/<id>`"
        )));
    }
    Ok(format!("{FILES_PATH}/{id}"))
}

/// A multipart/related upload body: the file's metadata, then its bytes.
fn upload_body(
    bytes: &[u8],
    mime_type: &str,
    display_name: Option<String>,
) -> Result<Vec<u8>, EncodeError> {
    let metadata = api::CreateFileRequest {
        file: Some(api::File {
            display_name,
            mime_type: Some(mime_type.to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut body = Vec::with_capacity(bytes.len() + 512);
    body.extend_from_slice(
        format!("--{BOUNDARY}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(&serde_json::to_vec(&metadata)?);
    body.extend_from_slice(
        format!("\r\n--{BOUNDARY}\r\nContent-Type: {mime_type}\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    Ok(body)
}

impl Wire for Files {
    type Op = FileStore;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = FilesDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    fn encode(&self, request: FileRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = match request {
            FileRequest::Upload {
                bytes,
                mime_type,
                display_name,
            } => {
                let path = with_query_pairs("/upload/v1beta/files", &[("uploadType", "multipart")]);
                http::Request::post(self.provider.uri(&path))
                    .header(
                        "Content-Type",
                        format!("multipart/related; boundary={BOUNDARY}"),
                    )
                    .body(Body::Bytes(upload_body(&bytes, &mime_type, display_name)?))?
            }
            FileRequest::Get(name) => {
                http::Request::get(self.provider.uri(&file_path(&name)?)).body(Body::empty())?
            }
            FileRequest::List { page_token } => {
                let page_size = self.page_size.to_string();
                let mut pairs = vec![("pageSize", page_size.as_str())];
                if let Some(token) = page_token.as_deref() {
                    pairs.push(("pageToken", token));
                }
                http::Request::get(self.provider.uri(&with_query_pairs(FILES_PATH, &pairs)))
                    .body(Body::empty())?
            }
            FileRequest::Delete(name) => {
                http::Request::delete(self.provider.uri(&file_path(&name)?)).body(Body::empty())?
            }
        };
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        FilesDecoder
    }
}

/// Decodes one Files API reply.
pub struct FilesDecoder;

impl<'id> Decoder<'id, FileStore> for FilesDecoder {
    type Event = FileReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<FileReply> {
        let body = frame.as_str();
        classify_or(&body, as_page, |data| {
            classify_or(data, as_created, |data| {
                classify_or(data, as_acknowledgement, as_file)
            })
        })
    }

    fn decode(
        &mut self,
        reply: FileReply,
        out: Out<'id, FileStore>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }

    /// A reply with no body at all is an acknowledgement.
    fn eof(&mut self, out: Out<'id, FileStore>) -> Result<Flow, ProviderError> {
        Ok(out.end(FileReply::Acknowledged))
    }
}

fn as_page(data: &str) -> WireEvent<FileReply> {
    classify_untyped_line::<Recognized<api::ListFilesResponse>>(data.as_bytes())
        .map(|Recognized(page)| FileReply::Page(page))
}

fn as_created(data: &str) -> WireEvent<FileReply> {
    #[derive(Deserialize)]
    struct Created {
        file: api::File,
    }
    classify_untyped_line::<Created>(data.as_bytes())
        .map(|created| FileReply::File(Box::new(created.file)))
}

fn as_acknowledgement(data: &str) -> WireEvent<FileReply> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Acknowledgement {}
    classify_untyped_line::<Acknowledgement>(data.as_bytes()).map(|_| FileReply::Acknowledged)
}

fn as_file(data: &str) -> WireEvent<FileReply> {
    classify_untyped_line::<Recognized<api::File>>(data.as_bytes())
        .map(|Recognized(file)| FileReply::File(Box::new(file)))
}

#[cfg(test)]
mod tests;
