//! Computes the request attestation over compact JSON with a zeroed billing checksum.

use std::ops::Range;

use xxhash_rust::xxh64::Xxh64;

fn find(bytes: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    bytes
        .get(start..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| start + offset)
}

fn field_range(bytes: &[u8], mut start: usize, mut end: usize, lower: usize) -> Range<usize> {
    if bytes.get(end) == Some(&b',') {
        end += 1;
    } else if start > lower && bytes.get(start - 1) == Some(&b',') {
        start -= 1;
    }
    start..end
}

fn array_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 1usize;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, byte) in bytes.get(start..)?.iter().enumerate() {
        if escaped {
            escaped = false;
        } else if quoted {
            match byte {
                b'\\' => escaped = true,
                b'"' => quoted = false,
                _ => (),
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(start + offset + 1);
                    }
                }
                _ => (),
            }
        }
    }
    None
}

fn next_excluded(bytes: &[u8], start: usize) -> Option<Range<usize>> {
    let fallbacks = b"\"fallbacks\":[";
    let fallback = find(bytes, fallbacks, start).and_then(|at| {
        array_end(bytes, at + fallbacks.len()).map(|end| field_range(bytes, at, end, start))
    });
    let credit = b"\"fallback_credit_token\":\"";
    let credit = find(bytes, credit, start).and_then(|at| {
        find(bytes, b"\"", at + credit.len()).map(|end| field_range(bytes, at, end + 1, start))
    });
    let tokens = b"\"max_tokens\":";
    let mut search = start;
    let tokens = loop {
        let Some(at) = find(bytes, tokens, search) else {
            break None;
        };
        search = at + tokens.len();
        let digits = bytes
            .get(search..)
            .into_iter()
            .flatten()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if digits > 0 {
            break Some(field_range(bytes, at, search + digits, start));
        }
    };
    [fallback, credit, tokens]
        .into_iter()
        .flatten()
        .min_by_key(|range| range.start)
}

/// Hashes original compact JSON bytes with `cch=00000` already substituted.
pub(crate) fn compute(bytes: &[u8]) -> String {
    let mut hash = Xxh64::new(super::CCH_SEED);
    let model_key = b"\"model\":\"";
    let mut start = 0;
    while start < bytes.len() {
        let excluded = next_excluded(bytes, start);
        let model = find(bytes, model_key, start);
        if let Some(range) = excluded.filter(|range| model.is_none_or(|at| range.start <= at)) {
            if let Some(part) = bytes.get(start..range.start) {
                hash.update(part);
            }
            start = range.end;
        } else if let Some((at, end)) =
            model.and_then(|at| find(bytes, b"\"", at + model_key.len()).map(|end| (at, end)))
        {
            if let Some(part) = bytes.get(start..at + model_key.len()) {
                hash.update(part);
            }
            start = end;
        } else {
            if let Some(part) = bytes.get(start..) {
                hash.update(part);
            }
            break;
        }
    }
    format!("{:05x}", hash.digest() & 0xfffff)
}

#[cfg(test)]
mod tests;
