use std::str;

/// A line decoder that handles incrementally reading lines from text.
/// Ported from JavaScript implementation.
pub struct LineDecoder {
    buffer: Vec<u8>,
    carriage_return_index: Option<usize>,
}

impl Default for LineDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LineDecoder {
    /// Create a new LineDecoder
    pub fn new() -> Self {
        LineDecoder {
            buffer: Vec::new(),
            carriage_return_index: None,
        }
    }

    /// Decode a chunk of data into lines
    pub fn decode(&mut self, chunk: &[u8]) -> Vec<String> {
        if chunk.is_empty() {
            return Vec::new();
        }

        // Append the new chunk to the buffer
        self.buffer.extend_from_slice(chunk);

        let mut lines = Vec::new();

        // Process lines while we can find newlines
        while let Some(pattern_index) = find_newline_index(&self.buffer, self.carriage_return_index)
        {
            if pattern_index.carriage && self.carriage_return_index.is_none() {
                // Skip until we either get a corresponding `\n`, a new `\r` or nothing
                self.carriage_return_index = Some(pattern_index.index);
                continue;
            }

            // We got double \r or \rtext\n
            if let Some(cr_index) = self.carriage_return_index {
                if pattern_index.index != cr_index + 1 || pattern_index.carriage {
                    if cr_index > 0 {
                        let line = decode_text(&self.buffer[0..cr_index - 1]);
                        lines.push(line);
                    } else {
                        // Handle edge case for carriage return at beginning
                        lines.push(String::new());
                    }

                    if cr_index < self.buffer.len() {
                        self.buffer = self.buffer[cr_index..].to_vec();
                    } else {
                        self.buffer.clear();
                    }
                    self.carriage_return_index = None;
                    continue;
                }
            }

            let end_index = if self.carriage_return_index.is_some() {
                pattern_index.preceding - 1
            } else {
                pattern_index.preceding
            };

            if end_index > 0 {
                let line = decode_text(&self.buffer[0..end_index]);
                lines.push(line);
            } else {
                lines.push(String::new());
            }

            if pattern_index.index < self.buffer.len() {
                self.buffer = self.buffer[pattern_index.index..].to_vec();
            } else {
                self.buffer.clear();
            }
            self.carriage_return_index = None;
        }

        lines
    }

    /// Flush any remaining data in the buffer
    pub fn flush(&mut self) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        self.decode("\n".as_bytes())
    }
}

/// Helper structure for newline index information
struct NewlineIndex {
    preceding: usize,
    index: usize,
    carriage: bool,
}

/// Find the index of the next newline character in the buffer
fn find_newline_index(buffer: &[u8], start_index: Option<usize>) -> Option<NewlineIndex> {
    const NEWLINE: u8 = 0x0a; // \n
    const CARRIAGE: u8 = 0x0d; // \r

    let start = start_index.unwrap_or(0);

    for (i, &byte) in buffer.iter().enumerate().skip(start) {
        if byte == NEWLINE {
            return Some(NewlineIndex {
                preceding: i,
                index: i + 1,
                carriage: false,
            });
        }

        if byte == CARRIAGE {
            return Some(NewlineIndex {
                preceding: i,
                index: i + 1,
                carriage: true,
            });
        }
    }

    None
}

/// Find the index after a double newline pattern in the buffer
pub fn find_double_newline_index(buffer: &[u8]) -> isize {
    const NEWLINE: u8 = 0x0a; // \n
    const CARRIAGE: u8 = 0x0d; // \r

    for i in 0..buffer.len().saturating_sub(1) {
        // Check for \n\n pattern
        if buffer[i] == NEWLINE && buffer[i + 1] == NEWLINE {
            return (i + 2) as isize;
        }

        // Check for \r\r pattern
        if buffer[i] == CARRIAGE && buffer[i + 1] == CARRIAGE {
            return (i + 2) as isize;
        }

        // Check for \r\n\r\n pattern
        if i + 3 < buffer.len()
            && buffer[i] == CARRIAGE
            && buffer[i + 1] == NEWLINE
            && buffer[i + 2] == CARRIAGE
            && buffer[i + 3] == NEWLINE
        {
            return (i + 4) as isize;
        }
    }

    -1
}

/// Decode a byte slice into a UTF-8 string
fn decode_text(bytes: &[u8]) -> String {
    match str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            // Handle invalid UTF-8 by replacing invalid sequences
            String::from_utf8_lossy(bytes).to_string()
        }
    }
}

/// Decode multiple chunks of data, with an option to flush
pub fn decode_chunks(chunks: &[&[u8]], flush: bool) -> Vec<String> {
    let mut decoder = LineDecoder::new();
    let mut lines = Vec::new();

    for chunk in chunks {
        lines.extend(decoder.decode(chunk));
    }

    if flush {
        lines.extend(decoder.flush());
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_string_chunks(chunks: &[&str], flush: bool) -> Vec<String> {
        let byte_chunks: Vec<&[u8]> = chunks.iter().map(|s| s.as_bytes()).collect();
        decode_chunks(&byte_chunks, flush)
    }

    #[test]
    fn test_basic() {
        // baz is not included because the line hasn't ended yet
        assert_eq!(
            decode_string_chunks(&["foo", " bar\nbaz"], false),
            vec!["foo bar"]
        );
    }

    #[test]
    fn test_basic_with_cr() {
        assert_eq!(
            decode_string_chunks(&["foo", " bar\r\nbaz"], false),
            vec!["foo bar"]
        );
        assert_eq!(
            decode_string_chunks(&["foo", " bar\r\nbaz"], true),
            vec!["foo bar", "baz"]
        );
    }

    #[test]
    fn test_trailing_new_lines() {
        assert_eq!(
            decode_string_chunks(&["foo", " bar", "baz\n", "thing\n"], false),
            vec!["foo barbaz", "thing"]
        );
    }

    #[test]
    fn test_trailing_new_lines_with_cr() {
        assert_eq!(
            decode_string_chunks(&["foo", " bar", "baz\r\n", "thing\r\n"], false),
            vec!["foo barbaz", "thing"]
        );
    }

    #[test]
    fn test_escaped_new_lines() {
        assert_eq!(
            decode_string_chunks(&["foo", " bar\\nbaz\n"], false),
            vec!["foo bar\\nbaz"]
        );
    }

    #[test]
    fn test_escaped_new_lines_with_cr() {
        assert_eq!(
            decode_string_chunks(&["foo", " bar\\r\\nbaz\n"], false),
            vec!["foo bar\\r\\nbaz"]
        );
    }

    #[test]
    fn test_cr_and_lf_split_across_chunks() {
        assert_eq!(
            decode_string_chunks(&["foo\r", "\n", "bar"], true),
            vec!["foo", "bar"]
        );
    }

    #[test]
    fn test_single_cr() {
        assert_eq!(
            decode_string_chunks(&["foo\r", "bar"], true),
            vec!["foo", "bar"]
        );
    }

    #[test]
    fn test_double_cr() {
        assert_eq!(
            decode_string_chunks(&["foo\r", "bar\r"], true),
            vec!["foo", "bar"]
        );
        assert_eq!(
            decode_string_chunks(&["foo\r", "\r", "bar"], true),
            vec!["foo", "", "bar"]
        );
        // implementation detail that we don't yield the single \r line until a new \r or \n is encountered
        assert_eq!(
            decode_string_chunks(&["foo\r", "\r", "bar"], false),
            vec!["foo"]
        );
    }

    #[test]
    fn test_double_cr_then_crlf() {
        assert_eq!(
            decode_string_chunks(&["foo\r", "\r", "\r", "\n", "bar", "\n"], false),
            vec!["foo", "", "", "bar"]
        );
        assert_eq!(
            decode_string_chunks(&["foo\n", "\n", "\n", "bar", "\n"], false),
            vec!["foo", "", "", "bar"]
        );
    }

    #[test]
    fn test_double_newline() {
        assert_eq!(
            decode_string_chunks(&["foo\n\nbar"], true),
            vec!["foo", "", "bar"]
        );
        assert_eq!(
            decode_string_chunks(&["foo", "\n", "\nbar"], true),
            vec!["foo", "", "bar"]
        );
        assert_eq!(
            decode_string_chunks(&["foo\n", "\n", "bar"], true),
            vec!["foo", "", "bar"]
        );
        assert_eq!(
            decode_string_chunks(&["foo", "\n", "\n", "bar"], true),
            vec!["foo", "", "bar"]
        );
    }

    #[test]
    fn test_multi_byte_characters_across_chunks() {
        let mut decoder = LineDecoder::new();

        // bytes taken from the string 'известни' and arbitrarily split
        // so that some multi-byte characters span multiple chunks
        assert_eq!(decoder.decode(&[0xd0]), Vec::<String>::new());
        assert_eq!(
            decoder.decode(&[0xb8, 0xd0, 0xb7, 0xd0]),
            Vec::<String>::new()
        );
        assert_eq!(
            decoder.decode(&[0xb2, 0xd0, 0xb5, 0xd1, 0x81, 0xd1, 0x82, 0xd0, 0xbd, 0xd0, 0xb8]),
            Vec::<String>::new()
        );

        let decoded = decoder.decode(&[0xa]);
        assert_eq!(decoded, vec!["известни"]);
    }

    #[test]
    fn test_flushing_trailing_newlines() {
        assert_eq!(
            decode_string_chunks(&["foo\n", "\nbar"], true),
            vec!["foo", "", "bar"]
        );
    }

    #[test]
    fn test_flushing_empty_buffer() {
        assert_eq!(decode_string_chunks(&[], true), Vec::<String>::new());
    }

    #[test]
    fn test_find_double_newline_index() {
        // Test \n\n patterns
        assert_eq!(find_double_newline_index("foo\n\nbar".as_bytes()), 5);
        assert_eq!(find_double_newline_index("\n\nbar".as_bytes()), 2);
        assert_eq!(find_double_newline_index("foo\n\n".as_bytes()), 5);
        assert_eq!(find_double_newline_index("\n\n".as_bytes()), 2);

        // Test \r\r patterns
        assert_eq!(find_double_newline_index("foo\r\rbar".as_bytes()), 5);
        assert_eq!(find_double_newline_index("\r\rbar".as_bytes()), 2);
        assert_eq!(find_double_newline_index("foo\r\r".as_bytes()), 5);
        assert_eq!(find_double_newline_index("\r\r".as_bytes()), 2);

        // Test \r\n\r\n patterns
        assert_eq!(find_double_newline_index("foo\r\n\r\nbar".as_bytes()), 7);
        assert_eq!(find_double_newline_index("\r\n\r\nbar".as_bytes()), 4);
        assert_eq!(find_double_newline_index("foo\r\n\r\n".as_bytes()), 7);
        assert_eq!(find_double_newline_index("\r\n\r\n".as_bytes()), 4);

        // Test not found cases
        assert_eq!(find_double_newline_index("foo\nbar".as_bytes()), -1);
        assert_eq!(find_double_newline_index("foo\rbar".as_bytes()), -1);
        assert_eq!(find_double_newline_index("foo\r\nbar".as_bytes()), -1);
        assert_eq!(find_double_newline_index("".as_bytes()), -1);

        // Test incomplete patterns
        assert_eq!(find_double_newline_index("foo\r\n\r".as_bytes()), -1);
        assert_eq!(find_double_newline_index("foo\r\n".as_bytes()), -1);
    }
}
