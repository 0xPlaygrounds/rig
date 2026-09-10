//! Bounded framing diagnostics beside the SSE decoder; never stores frame data.

/// Counts bytes after the most recent SSE event delimiter, across body chunks.
#[derive(Default)]
pub(crate) struct SseTail {
    pending: usize,
    line_has_bytes: bool,
    after_cr: bool,
    // The optional leading UTF-8 BOM is ignored by SSE. No other data is retained.
    bom_prefix: usize,
    prefix_done: bool,
}

impl SseTail {
    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        const BOM: [u8; 3] = [0xef, 0xbb, 0xbf];
        for &byte in bytes {
            if !self.prefix_done {
                if BOM.get(self.bom_prefix) == Some(&byte) {
                    self.bom_prefix += 1;
                    if self.bom_prefix == BOM.len() {
                        self.prefix_done = true;
                        self.bom_prefix = 0;
                    }
                    continue;
                }
                self.prefix_done = true;
                for &prefix in BOM.iter().take(self.bom_prefix) {
                    self.byte(prefix);
                }
                self.bom_prefix = 0;
            }
            self.byte(byte);
        }
    }

    fn byte(&mut self, byte: u8) {
        if self.after_cr && byte == b'\n' {
            self.after_cr = false;
            // The LF belongs to the preceding CR, including when that CR
            // finished a blank delimiter line at the end of another chunk.
            if self.pending != 0 {
                self.pending = self.pending.saturating_add(1);
            }
            return;
        }
        self.after_cr = false;
        self.pending = self.pending.saturating_add(1);
        if byte == b'\r' || byte == b'\n' {
            if !self.line_has_bytes {
                self.pending = 0;
            }
            self.line_has_bytes = false;
            self.after_cr = byte == b'\r';
        } else {
            self.line_has_bytes = true;
        }
    }

    /// Raw byte count since the last blank line (saturated at usize::MAX).
    pub(crate) fn pending(&self) -> usize {
        self.pending.saturating_add(self.bom_prefix)
    }
}

#[cfg(test)]
mod tests;
