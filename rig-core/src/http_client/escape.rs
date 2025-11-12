use std::fmt;
use std::str;

/// A helper struct to escape bytes for logging.
pub(crate) struct Escape<'a>(&'a [u8]);

impl<'a> Escape<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Escape(bytes)
    }
}

impl fmt::Debug for Escape<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // For valid UTF-8 strings, output directly for better readability
        if let Ok(s) = str::from_utf8(self.0) {
            return write!(f, "{}", s);
        }
        write!(f, "{}", self)
    }
}

impl fmt::Display for Escape<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &c in self.0 {
            match c {
                b'\n' => write!(f, "\\n")?,
                b'\r' => write!(f, "\\r")?,
                b'\t' => write!(f, "\\t")?,
                b'\\' => write!(f, "\\\\")?,
                b'"' => write!(f, "\\\"")?,
                b'\0' => write!(f, "\\0")?,
                // ASCII printable (0x20-0x7e, excluding space which is 0x20)
                c if (0x20..0x7f).contains(&c) => write!(f, "{}", c as char)?,
                // Non-printable bytes
                c => write!(f, "\\x{c:02x}")?,
            }
        }
        Ok(())
    }
}
