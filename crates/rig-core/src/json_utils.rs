use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use std::cmp::Ordering;
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::LazyLock;

/// `skip_serializing_if` helper: serde requires a `fn(&bool) -> bool`, so the
/// trivially-copy lint does not apply here.
#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn is_false(value: &bool) -> bool {
    !value
}

/// The numeric order of two JSON numbers, exact: integers compare as
/// integers and an integer against a float rounds neither, so `5` equals
/// `5.0` but `2^53 + 1` does not equal `2^53` as a float. With serde_json's
/// `arbitrary_precision`, numbers compare by their literal's decimal value.
pub(crate) fn compare_numbers(a: &serde_json::Number, b: &serde_json::Number) -> Option<Ordering> {
    if holds_literals() {
        return Some(compare_literals(&a.to_string(), &b.to_string()));
    }
    let integer = |n: &serde_json::Number| {
        n.as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
    };
    match (integer(a), integer(b)) {
        (Some(a), Some(b)) => Some(a.cmp(&b)),
        (Some(a), None) => compare_integer_float(a, b.as_f64()?),
        (None, Some(b)) => compare_integer_float(b, a.as_f64()?).map(Ordering::reverse),
        (None, None) => a.as_f64()?.partial_cmp(&b.as_f64()?),
    }
}

/// Whether a [`serde_json::Number`] keeps the literal it was read from, as
/// it does with `arbitrary_precision`, which a dependent crate can turn on.
fn holds_literals() -> bool {
    static HOLDS: LazyLock<bool> = LazyLock::new(|| {
        serde_json::from_str::<serde_json::Number>("1.50").is_ok_and(|n| n.to_string() == "1.50")
    });
    *HOLDS
}

/// A JSON number literal as `sign * 0.digits * 10^(exponent + shift)`.
/// `digits` has no leading or trailing zero, so one value has one form. The
/// exponent stays text, since a literal's can be any length.
struct Literal<'a> {
    sign: i8,
    exponent_sign: i128,
    exponent: &'a str,
    shift: i128,
    digits: String,
}

impl<'a> Literal<'a> {
    fn new(text: &'a str) -> Self {
        let (sign, text) = text.strip_prefix('-').map_or((1, text), |text| (-1, text));
        let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, ""));
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let digits = format!("{whole}{fraction}");
        let digits = digits.trim_start_matches('0');
        let (exponent_sign, exponent) = match exponent.strip_prefix('-') {
            Some(exponent) => (-1, exponent),
            None => (1, exponent.trim_start_matches('+')),
        };
        let zero = digits.is_empty();
        Self {
            sign: if zero { 0 } else { sign },
            exponent_sign,
            exponent: if zero { "" } else { exponent },
            shift: if zero {
                0
            } else {
                digits.len() as i128 - fraction.len() as i128
            },
            digits: digits.trim_end_matches('0').to_owned(),
        }
    }

    /// This exponent's signed decimal digits, left-padded to `width`.
    fn exponent_digits(&self, width: usize) -> impl Iterator<Item = i128> {
        let padding = std::iter::repeat_n(0, width.saturating_sub(self.exponent.len()));
        let digits = self.exponent.chars().filter_map(|digit| digit.to_digit(10));
        padding
            .chain(digits)
            .map(|digit| i128::from(digit) * self.exponent_sign)
    }

    /// This exponent minus `other`'s, held at a size no `shift` can reach
    /// once past it: each further digit only moves it further away.
    fn exponent_gap(&self, other: &Self) -> i128 {
        const FAR: i128 = 1_000_000_000_000_000_000_000_000_000_000_000_000;
        let width = self.exponent.len().max(other.exponent.len());
        let digits = self
            .exponent_digits(width)
            .zip(other.exponent_digits(width));
        digits.fold(0, |gap, (a, b)| (gap * 10 + a - b).clamp(-FAR, FAR))
    }
}

/// Two JSON number literals in the order of their exact decimal values.
pub(crate) fn compare_literals(a: &str, b: &str) -> Ordering {
    let (a, b) = (Literal::new(a), Literal::new(b));
    a.sign.cmp(&b.sign).then_with(|| {
        let power = a.exponent_gap(&b) + a.shift - b.shift;
        let magnitude = power.cmp(&0).then_with(|| a.digits.cmp(&b.digits));
        if a.sign < 0 {
            magnitude.reverse()
        } else {
            magnitude
        }
    })
}

/// `integer` against `float` without rounding. Inside `i128`'s range a
/// float's whole part converts exactly. The bound is a literal because
/// `i128::MAX as f64` rounds up to it.
fn compare_integer_float(integer: i128, float: f64) -> Option<Ordering> {
    const LIMIT: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    if float < -LIMIT {
        return Some(Ordering::Greater);
    }
    if float >= LIMIT {
        return Some(Ordering::Less);
    }
    match integer.cmp(&(float.trunc() as i128)) {
        Ordering::Equal => 0.0_f64.partial_cmp(&float.fract()),
        unequal => Some(unequal),
    }
}

/// `value` with lexicographically sorted object keys at every depth,
/// independent of serde_json's `preserve_order` feature. Array order is retained.
pub fn canonical(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            let entries = entries
                .into_iter()
                .map(|(key, value)| (key, canonical(value)));
            serde_json::Value::Object(entries.collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonical).collect())
        }
        other => other,
    }
}

/// Renders [`canonical`] JSON compactly.
pub fn to_canonical_string(value: &serde_json::Value) -> String {
    canonical(value.clone()).to_string()
}

/// Removes `key` from `map`, returning its value or `None` if absent.
/// Remaining entries keep their relative iteration order with or without
/// serde_json's `preserve_order` feature.
pub fn remove_preserving_order(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<serde_json::Value> {
    let mut removed = None;
    map.retain(|candidate, value| {
        if candidate == key {
            removed = Some(value.take());
            false
        } else {
            true
        }
    });
    removed
}

pub fn merge(a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Object(mut a_map), serde_json::Value::Object(b_map)) => {
            b_map.into_iter().for_each(|(key, value)| {
                a_map.insert(key, value);
            });
            serde_json::Value::Object(a_map)
        }
        (a, _) => a,
    }
}

/// Applies a request builder's `additional_params` call: `None` clears, and a
/// value merges over the parameters earlier calls set. Merging combines JSON
/// objects key by key; earlier parameters that are not an object are kept.
pub(crate) fn merge_params(
    existing: Option<serde_json::Value>,
    params: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    Some(match (existing, params?) {
        (Some(existing), params) => merge(existing, params),
        (None, params) => params,
    })
}

// Callers require the image or audio feature.
#[cfg_attr(not(any(feature = "image", feature = "audio")), allow(dead_code))]
pub fn merge_inplace(a: &mut serde_json::Value, b: serde_json::Value) {
    if let (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) = (a, b) {
        b_map.into_iter().for_each(|(key, value)| {
            a_map.insert(key, value);
        });
    }
}

/// Normalize a provider-wire field that may contain encoded JSON in a string.
///
/// This deliberately unwraps [`serde_json::Value::String`] and is only for
/// provider decoding, before a value enters Rig's canonical message model.
pub fn value_to_json_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Parse tool arguments from a streamed string payload.
/// Some providers emit an empty string for parameterless tool calls; normalize that to `{}`.
pub fn parse_tool_arguments(arguments: &str) -> serde_json::Result<serde_json::Value> {
    if arguments.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }

    serde_json::from_str(arguments)
}

/// Whether `arguments` are a whole JSON object ([`parse_tool_arguments`]).
/// Only a text that ends in `}` (or says nothing) can be one, so no other
/// is parsed.
pub fn is_whole_object(arguments: &str) -> bool {
    let end = arguments.trim_end();
    (end.is_empty() || end.ends_with('}'))
        && matches!(
            parse_tool_arguments(arguments),
            Ok(serde_json::Value::Object(_))
        )
}

/// The longest object a cut-off JSON text `text` still states: open strings,
/// arrays and objects are closed and a dangling key or separator dropped, as
/// pi's streaming JSON parse does. `None` when no prefix states an object.
pub fn parse_partial_object(text: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    // Bounds the work on a large cut-off input: the last valid cut is
    // almost always near its end.
    const MAX_ATTEMPTS: usize = 256;
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    // Prefix lengths a value just ended at, with the containers open there.
    let mut cuts: Vec<(usize, Vec<u8>, bool)> = Vec::new();
    let bytes = text.as_bytes();
    for (at, byte) in bytes.iter().enumerate() {
        if in_string {
            match (escaped, *byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => {
                    in_string = false;
                    cuts.push((at + 1, stack.clone(), false));
                }
                (false, _) => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                stack.push(*byte);
                cuts.push((at + 1, stack.clone(), false));
            }
            b'}' | b']' => {
                stack.pop();
                cuts.push((at + 1, stack.clone(), false));
            }
            b if b.is_ascii_alphanumeric() || *b == b'.' || *b == b'-' || *b == b'+' => {
                let next = bytes.get(at + 1).copied();
                if !next.is_some_and(|n| {
                    n.is_ascii_alphanumeric() || n == b'.' || n == b'-' || n == b'+'
                }) {
                    cuts.push((at + 1, stack.clone(), false));
                }
            }
            _ => {}
        }
    }
    if in_string {
        let at = text.len() - usize::from(escaped);
        cuts.push((at, stack.clone(), true));
    }
    cuts.iter()
        .rev()
        .take(MAX_ATTEMPTS)
        .find_map(|(at, open, quote)| {
            let mut candidate = text.get(..*at)?.to_owned();
            if *quote {
                candidate.push('"');
            }
            for container in open.iter().rev() {
                candidate.push(if *container == b'{' { '}' } else { ']' });
            }
            match serde_json::from_str(&candidate) {
                Ok(serde_json::Value::Object(object)) => Some(object),
                _ => None,
            }
        })
}

/// The best object a tool call's argument text states so far, for showing a
/// call while it streams. It never fails: complete JSON parses as is, text
/// with raw control characters or invalid escapes is repaired first, a
/// cut-off prefix keeps the longest object it states
/// ([`parse_partial_object`]), and anything else is an empty object, as is
/// a top level that is not an object.
///
/// It is for display only. A call's arguments are what its end states, and
/// only those are validated and executed.
///
/// ```
/// use rig_core::streaming::parse_partial_arguments;
///
/// let partial = parse_partial_arguments(r#"{"path": "notes.md", "content": "Hel"#);
/// assert_eq!(partial["path"], "notes.md");
/// assert_eq!(partial["content"], "Hel");
/// ```
pub fn parse_partial_arguments(text: &str) -> serde_json::Map<String, serde_json::Value> {
    if text.trim().is_empty() {
        return serde_json::Map::new();
    }
    let repaired = repair_json(text);
    let complete = serde_json::from_str(text)
        .ok()
        .or_else(|| serde_json::from_str(&repaired).ok());
    if let Some(value) = complete {
        return match value {
            serde_json::Value::Object(object) => object,
            _ => serde_json::Map::new(),
        };
    }
    let cut = without_partial_escape(&repaired);
    parse_partial_object(cut)
        .or_else(|| parse_partial_object(text))
        .unwrap_or_default()
}

/// `text` with raw control characters inside strings escaped and each
/// backslash that starts no valid escape doubled, as models sometimes write
/// them.
fn repair_json(text: &str) -> String {
    let mut repaired = String::with_capacity(text.len());
    let mut in_string = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if !in_string {
            in_string = c == '"';
            repaired.push(c);
            continue;
        }
        match c {
            '"' => {
                in_string = false;
                repaired.push(c);
            }
            '\\' => match chars.peek() {
                Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') => {
                    repaired.push(c);
                    if let Some(next) = chars.next() {
                        repaired.push(next);
                    }
                }
                // A trailing backslash may begin an escape the next fragment ends.
                None => repaired.push(c),
                Some(_) => repaired.push_str("\\\\"),
            },
            '\n' => repaired.push_str("\\n"),
            '\r' => repaired.push_str("\\r"),
            '\t' => repaired.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write as _;
                // Writing to a `String` cannot fail.
                let _ = write!(repaired, "\\u{:04x}", u32::from(c));
            }
            c => repaired.push(c),
        }
    }
    repaired
}

/// `text` without a `\u` escape its end cuts short, which would otherwise
/// cost the whole string it ends.
fn without_partial_escape(text: &str) -> &str {
    let bytes = text.as_bytes();
    let tail = bytes.len().saturating_sub(5);
    let Some(at) = (tail..bytes.len())
        .rev()
        .find(|at| bytes.get(*at..*at + 2) == Some(b"\\u"))
    else {
        return text;
    };
    let digits = bytes.get(at + 2..).unwrap_or_default();
    // A preceding backslash run of odd length escapes this backslash itself.
    let escaping = bytes
        .get(..at)
        .unwrap_or_default()
        .iter()
        .rev()
        .take_while(|byte| **byte == b'\\')
        .count();
    if digits.len() < 4 && digits.iter().all(u8::is_ascii_hexdigit) && escaping % 2 == 0 {
        text.get(..at).unwrap_or(text)
    } else {
        text
    }
}

/// Deserializes a string or object as one item, a sequence as multiple items,
/// and null as an empty list. Strings use [`FromStr`]; objects use [`Deserialize`].
pub fn string_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible>,
    D: Deserializer<'de>,
{
    struct StringOrVec<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrVec<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string, sequence, or null")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            let item = FromStr::from_str(value).map_err(de::Error::custom)?;
            Ok(vec![item])
        }

        fn visit_seq<A>(self, seq: A) -> Result<Vec<T>, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        fn visit_map<M>(self, map: M) -> Result<Vec<T>, M::Error>
        where
            M: MapAccess<'de>,
        {
            let item = Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))?;
            Ok(vec![item])
        }

        fn visit_none<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }

        fn visit_unit<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }
    }

    deserializer.deserialize_any(StringOrVec(PhantomData))
}

/// Deserializes `T`, mapping an explicit `null` to `T::default()`.
///
/// Driven through `deserialize_option` rather than `deserialize_any`; the two
/// agree only for self-describing formats, which is all rig decodes (JSON).
pub fn null_or_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + Default,
    D: Deserializer<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
mod tests;

/// Lenient reads of a provider reply: each accessor gives `None`, or an
/// empty slice, for a field that is missing or of another type, and never
/// fails. Decoders read replies through it, so a field no block or finish
/// needs can never fail a reply.
///
/// ```
/// use rig_core::json_utils::Lenient;
/// use serde_json::json;
///
/// let reply = json!({"id": "r1", "usage": {"input_tokens": "7"}, "output": null});
/// assert_eq!(reply.str("id"), Some("r1"));
/// assert_eq!(reply.at("/usage/input_tokens").and_then(Lenient::as_u64_lenient), Some(7));
/// assert!(reply.arr("output").is_empty());
/// ```
pub trait Lenient {
    /// The string at `key`.
    fn str(&self, key: &str) -> Option<&str>;
    /// The unsigned integer at `key`, from a number or a numeric string.
    fn u64(&self, key: &str) -> Option<u64>;
    /// The integer at `key`, from a number or a numeric string.
    fn i64(&self, key: &str) -> Option<i64>;
    /// The number at `key`, from a number or a numeric string.
    fn f64(&self, key: &str) -> Option<f64>;
    /// The boolean at `key`.
    fn bool(&self, key: &str) -> Option<bool>;
    /// The object at `key`.
    fn obj(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>>;
    /// The array at `key`, empty when absent or of another type.
    fn arr(&self, key: &str) -> &[serde_json::Value];
    /// The value at a JSON pointer, `None` for `null`.
    fn at(&self, pointer: &str) -> Option<&serde_json::Value>;
    /// This value as an unsigned integer, from a number or a numeric string.
    fn as_u64_lenient(&self) -> Option<u64>;
}

impl Lenient for serde_json::Value {
    fn str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64_lenient()
    }

    fn i64(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            serde_json::Value::Number(number) => number.as_i64(),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        }
    }

    fn f64(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            serde_json::Value::Number(number) => number.as_f64(),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        }
    }

    fn bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    fn obj(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.get(key)?.as_object()
    }

    fn arr(&self, key: &str) -> &[serde_json::Value] {
        self.get(key)
            .and_then(serde_json::Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    fn at(&self, pointer: &str) -> Option<&serde_json::Value> {
        self.pointer(pointer).filter(|value| !value.is_null())
    }

    fn as_u64_lenient(&self) -> Option<u64> {
        match self {
            serde_json::Value::Number(number) => number.as_u64().or_else(|| {
                number
                    .as_f64()
                    .filter(|float| *float >= 0.0 && float.fract() == 0.0)
                    .map(|float| float as u64)
            }),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        }
    }
}
