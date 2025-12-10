use bytes::Bytes;
use mime::Mime;
use std::borrow::Cow;

/// A generic multipart form part that can represent text or binary data
#[derive(Clone, Debug)]
pub struct Part {
    name: String,
    content: PartContent,
    filename: Option<String>,
    content_type: Option<Mime>,
}

#[derive(Clone, Debug)]
enum PartContent {
    Text(String),
    Binary(Bytes),
}

impl Part {
    /// Create a text part
    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: PartContent::Text(value.into()),
            filename: None,
            content_type: None,
        }
    }

    /// Create a binary part (e.g., file upload)
    pub fn bytes(name: impl Into<String>, data: impl Into<Bytes>) -> Self {
        Self {
            name: name.into(),
            content: PartContent::Binary(data.into()),
            filename: None,
            content_type: None,
        }
    }

    /// Set the filename for this part
    pub fn filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Set the content type for this part
    pub fn content_type(mut self, content_type: Mime) -> Self {
        self.content_type = Some(content_type);
        self
    }

    /// Get the part name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the filename if set
    pub fn get_filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    /// Get the content type if set
    pub fn get_content_type(&self) -> Option<&Mime> {
        self.content_type.as_ref()
    }
}

/// Generic multipart form data container
#[derive(Clone, Debug, Default)]
pub struct MultipartForm {
    parts: Vec<Part>,
    boundary: Option<String>,
}

impl MultipartForm {
    /// Create a new empty multipart form
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a part to the form
    pub fn part(mut self, part: Part) -> Self {
        self.parts.push(part);
        self
    }

    /// Add a text field
    pub fn text(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.part(Part::text(name, value))
    }

    /// Add a file/binary field
    pub fn file(
        self,
        name: impl Into<String>,
        filename: impl Into<String>,
        content_type: Mime,
        data: impl Into<Bytes>,
    ) -> Self {
        self.part(
            Part::bytes(name, data)
                .filename(filename)
                .content_type(content_type),
        )
    }

    /// Set a custom boundary (optional, one will be generated if not set)
    pub fn boundary(mut self, boundary: impl Into<String>) -> Self {
        self.boundary = Some(boundary.into());
        self
    }

    /// Get the parts
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Generate a boundary string
    fn generate_boundary() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("----boundary{}", timestamp)
    }

    /// Get or generate boundary
    fn get_boundary(&self) -> Cow<'_, str> {
        match &self.boundary {
            Some(b) => Cow::Borrowed(b),
            None => Cow::Owned(Self::generate_boundary()),
        }
    }

    /// Encode the multipart form to bytes with the given boundary
    pub fn encode(&self) -> (String, Bytes) {
        let boundary = self.get_boundary();
        let mut body = Vec::new();

        for part in &self.parts {
            body.extend_from_slice(b"--");
            body.extend_from_slice(boundary.as_bytes());
            body.extend_from_slice(b"\r\n");

            // Content-Disposition header
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"");
            body.extend_from_slice(part.name.as_bytes());
            body.extend_from_slice(b"\"");

            if let Some(filename) = &part.filename {
                body.extend_from_slice(b"; filename=\"");
                body.extend_from_slice(filename.as_bytes());
                body.extend_from_slice(b"\"");
            }
            body.extend_from_slice(b"\r\n");

            // Content-Type header if specified
            if let Some(content_type) = &part.content_type {
                body.extend_from_slice(b"Content-Type: ");
                body.extend_from_slice(content_type.as_ref().as_bytes());
                body.extend_from_slice(b"\r\n");
            }

            body.extend_from_slice(b"\r\n");

            // Content
            match &part.content {
                PartContent::Text(text) => body.extend_from_slice(text.as_bytes()),
                PartContent::Binary(bytes) => body.extend_from_slice(bytes),
            }

            body.extend_from_slice(b"\r\n");
        }

        // Final boundary
        body.extend_from_slice(b"--");
        body.extend_from_slice(boundary.as_bytes());
        body.extend_from_slice(b"--\r\n");

        (boundary.into_owned(), Bytes::from(body))
    }
}

impl From<MultipartForm> for reqwest::multipart::Form {
    fn from(value: MultipartForm) -> Self {
        let mut form = reqwest::multipart::Form::new();

        for part in value.parts {
            match part.content {
                PartContent::Text(text) => {
                    form = form.text(part.name, text);
                }
                PartContent::Binary(bytes) => {
                    let mut req_part = reqwest::multipart::Part::bytes(bytes.to_vec());

                    if let Some(filename) = part.filename {
                        req_part = req_part.file_name(filename);
                    }
                    if let Some(content_type) = part.content_type {
                        req_part = req_part.mime_str(content_type.as_ref()).unwrap();
                    }

                    form = form.part(part.name, req_part);
                }
            }
        }

        form
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multipart_encoding() {
        let form = MultipartForm::new()
            .text("field1", "value1")
            .text("field2", "value2");

        let (boundary, body) = form.encode();
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("field1"));
        assert!(body_str.contains("value1"));
        assert!(body_str.contains(&boundary));
    }

    #[test]
    fn test_file_part() {
        let form = MultipartForm::new().file(
            "upload",
            "test.txt",
            "text/plain".parse().unwrap(),
            Bytes::from("file contents"),
        );

        let (_, body) = form.encode();
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("filename=\"test.txt\""));
        assert!(body_str.contains("Content-Type: text/plain"));
        assert!(body_str.contains("file contents"));
    }
}
