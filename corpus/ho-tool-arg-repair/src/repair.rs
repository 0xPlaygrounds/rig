//! Repairing broken argument JSON, and fitting values to a tool's schema.
//! Only what is unambiguous is repaired: a value cut off before it began,
//! an unterminated string or a bare word is left broken.

use serde_json::{Map, Number, Value};

/// What was repaired, by kind, each once.
pub(crate) type Kinds = Vec<&'static str>;

/// `raw`, the argument text a model sent that is no JSON object, read as
/// an object, with the kinds of repair that took; `None` when it cannot be
/// read unambiguously.
pub(crate) fn parse_broken(raw: &str) -> Option<(Map<String, Value>, Kinds)> {
    let mut kinds = Kinds::new();
    // An object sent as a JSON string, itself broken.
    let text = match serde_json::from_str::<Value>(raw) {
        Ok(Value::String(inner)) => {
            note(&mut kinds, "string-wrapped");
            inner
        }
        _ => raw.to_owned(),
    };
    let mut parser = Parser {
        chars: text.chars().collect(),
        at: 0,
        kinds,
    };
    let value = parser.value()?;
    parser.skip_space();
    if parser.peek().is_some() {
        return None;
    }
    match value {
        Value::Object(object) => Some((object, parser.kinds)),
        _ => None,
    }
}

/// Fits `arguments` to the JSON schema `schema` where a value has the
/// wrong type and the fit is unambiguous, noting each kind in `kinds`.
pub(crate) fn coerce(arguments: &mut Map<String, Value>, schema: &Value, kinds: &mut Kinds) {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    for (name, value) in arguments.iter_mut() {
        if let Some(property) = properties.get(name) {
            coerce_value(value, property, kinds);
        }
    }
}

/// The types a schema allows: its `type`, one or a list.
fn types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(one)) => vec![one.as_str()],
        Some(Value::Array(many)) => many.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

/// Whether `value` is of the JSON schema type `kind`.
fn is(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn coerce_value(value: &mut Value, schema: &Value, kinds: &mut Kinds) {
    let allowed = types(schema);
    if let Value::Object(inner) = value {
        coerce(inner, schema, kinds);
    }
    if let (Value::Array(items), Some(item)) = (&mut *value, schema.get("items")) {
        for one in items.iter_mut() {
            coerce_value(one, item, kinds);
        }
    }
    if allowed.is_empty() || allowed.iter().any(|kind| is(value, kind)) {
        return;
    }
    let fitted = allowed.iter().find_map(|kind| fit(value, kind, schema));
    if let Some((fitted, kind)) = fitted {
        *value = fitted;
        note(kinds, kind);
    }
}

/// `value` as the type `kind`, when that reading is the only one.
fn fit(value: &Value, kind: &str, schema: &Value) -> Option<(Value, &'static str)> {
    let text = value.as_str()?.trim();
    match kind {
        "integer" => {
            let number = text
                .parse::<i64>()
                .map(Number::from)
                .or_else(|_| text.parse::<u64>().map(Number::from))
                .ok()?;
            Some((Value::Number(number), "string-to-integer"))
        }
        "number" => {
            let number: Number = serde_json::from_str(text).ok()?;
            Some((Value::Number(number), "string-to-number"))
        }
        "boolean" => match text {
            "true" => Some((Value::Bool(true), "string-to-boolean")),
            "false" => Some((Value::Bool(false), "string-to-boolean")),
            _ => None,
        },
        "array" => {
            let strings = schema
                .get("items")
                .is_some_and(|items| types(items) == ["string"]);
            strings.then(|| (Value::Array(vec![value.clone()]), "string-to-array"))
        }
        _ => None,
    }
}

/// A one-line picture of the arguments a tool whose schema is `schema`
/// takes: `{"path": <string>, "count": <integer>}`.
pub(crate) fn summary(schema: &Value) -> String {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return "{}".to_owned();
    };
    let fields: Vec<String> = properties
        .iter()
        .map(|(name, property)| {
            let kinds = types(property);
            let kind = if kinds.is_empty() {
                "any".to_owned()
            } else {
                kinds.join("|")
            };
            format!("\"{name}\": <{kind}>")
        })
        .collect();
    format!("{{{}}}", fields.join(", "))
}

fn note(kinds: &mut Kinds, kind: &'static str) {
    if !kinds.contains(&kind) {
        kinds.push(kind);
    }
}

/// A lenient JSON reader: JSON, plus what cheap models get wrong.
struct Parser {
    chars: Vec<char>,
    at: usize,
    kinds: Kinds,
}

/// How a container went on after a member.
enum After {
    /// Another member follows; whether a comma came first.
    Member(bool),
    /// The container is closed, by its bracket or by the end of the text.
    Closed,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let next = self.peek();
        self.at += usize::from(next.is_some());
        next
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.at += 1;
        }
    }

    fn note(&mut self, kind: &'static str) {
        note(&mut self.kinds, kind);
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_space();
        match self.peek()? {
            '{' => self.object(),
            '[' => self.array(),
            quote @ ('"' | '\'') => self.string(quote).map(Value::String),
            '-' | '0'..='9' => self.number(),
            _ => match self.word()?.as_str() {
                "true" => Some(Value::Bool(true)),
                "false" => Some(Value::Bool(false)),
                "null" => Some(Value::Null),
                python => {
                    let value = match python {
                        "True" => Value::Bool(true),
                        "False" => Value::Bool(false),
                        "None" => Value::Null,
                        _ => return None,
                    };
                    self.note("python-literals");
                    Some(value)
                }
            },
        }
    }

    /// After a member of a container closed by `close`: a comma and the
    /// next member, the close (a comma before it is a trailing comma), or
    /// the end of the text, which closes it.
    fn after(&mut self, close: char) -> Option<After> {
        self.skip_space();
        let comma = self.peek() == Some(',');
        if comma {
            self.at += 1;
            self.skip_space();
        }
        match self.peek() {
            None => {
                if comma {
                    self.note("trailing-comma");
                }
                self.note("unclosed");
                Some(After::Closed)
            }
            Some(next) if next == close => {
                self.at += 1;
                if comma {
                    self.note("trailing-comma");
                }
                Some(After::Closed)
            }
            Some(_) => Some(After::Member(comma)),
        }
    }

    fn object(&mut self) -> Option<Value> {
        self.bump();
        let mut object = Map::new();
        let mut open = self.after('}')?;
        while let After::Member(comma) = open {
            if comma == object.is_empty() {
                return None;
            }
            let key = match self.peek()? {
                quote @ ('"' | '\'') => self.string(quote)?,
                _ => {
                    let key = self.word()?;
                    self.note("unquoted-keys");
                    key
                }
            };
            self.skip_space();
            if self.bump()? != ':' {
                return None;
            }
            let value = self.value()?;
            object.insert(key, value);
            open = self.after('}')?;
        }
        Some(Value::Object(object))
    }

    fn array(&mut self) -> Option<Value> {
        self.bump();
        let mut items = Vec::new();
        let mut open = self.after(']')?;
        while let After::Member(comma) = open {
            if comma == items.is_empty() {
                return None;
            }
            items.push(self.value()?);
            open = self.after(']')?;
        }
        Some(Value::Array(items))
    }

    /// A string in `quote`s; `None` when it never ends.
    fn string(&mut self, quote: char) -> Option<String> {
        self.bump();
        if quote == '\'' {
            self.note("single-quotes");
        }
        let mut text = String::new();
        loop {
            match self.bump()? {
                c if c == quote => return Some(text),
                '\\' => text.push(self.escape(quote)?),
                c if c.is_control() => {
                    self.note("raw-control-characters");
                    text.push(c);
                }
                c => text.push(c),
            }
        }
    }

    fn escape(&mut self, quote: char) -> Option<char> {
        Some(match self.bump()? {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{8}',
            'f' => '\u{c}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => {
                let hex: String = (0..4).filter_map(|_| self.bump()).collect();
                let code = u32::from_str_radix(&hex, 16).ok()?;
                char::from_u32(code)?
            }
            c if c == quote => c,
            _ => return None,
        })
    }

    fn number(&mut self) -> Option<Value> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
        {
            self.at += 1;
        }
        let text: String = self.chars.get(start..self.at)?.iter().collect();
        serde_json::from_str::<Number>(&text)
            .ok()
            .map(Value::Number)
    }

    /// A bare word: an unquoted key or a literal.
    fn word(&mut self) -> Option<String> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '$' | '-'))
        {
            self.at += 1;
        }
        let word: String = self.chars.get(start..self.at)?.iter().collect();
        (!word.is_empty()).then_some(word)
    }
}
