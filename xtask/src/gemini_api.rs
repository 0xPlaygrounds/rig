//! `cargo xtask gemini-api [--fetch]`: generate rig-core's Gemini mirror.
//!
//! Reads Google's pinned v1beta discovery document at
//! `crates/rig-core/src/providers/gemini/api/discovery.json` and writes
//! `api/generated.rs`: one struct per schema, the settings templates, and one
//! enum per enumerated field. `--fetch` refreshes the pinned document first.
//! Output is formatted with `rustfmt`, so a second run changes nothing.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Map, Value};

/// Where the pinned discovery document and the generated mirror live.
const API_DIR: &str = "crates/rig-core/src/providers/gemini/api";

/// Google's public discovery endpoint for the v1beta surface.
const DISCOVERY_URL: &str =
    "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta";

/// Schemas whose discovery names read poorly as Rust types.
const SCHEMA_RENAMES: &[(&str, &str)] = &[("V1mainMediaResolution", "PartMediaResolution")];

/// Inline enums named by hand. The discovery document names none, and a
/// field-derived name (`Category`, `Level`) would depend on schema order.
const ENUM_NAMES: &[((&str, &str), &str)] = &[
    (("SafetySetting", "category"), "HarmCategory"),
    (("SafetyRating", "category"), "HarmCategory"),
    (("SafetySetting", "threshold"), "HarmBlockThreshold"),
    (("SafetyRating", "probability"), "HarmProbability"),
    (("V1mainMediaResolution", "level"), "MediaResolutionLevel"),
    (("FunctionCallingConfig", "mode"), "FunctionCallingMode"),
    (("ExecutableCode", "language"), "ExecutableCodeLanguage"),
    (("CodeExecutionResult", "outcome"), "CodeExecutionOutcome"),
];

/// A settings template: a mirror minus the fields rig owns. Removed fields
/// stay in `FIELDS`, so `Unmodeled` cannot reintroduce them.
struct Template {
    name: &'static str,
    source: &'static str,
    remove: &'static [&'static str],
    retype: &'static [(&'static str, &'static str)],
}

const TEMPLATES: &[Template] = &[
    Template {
        name: "RequestSettings",
        source: "GenerateContentRequest",
        remove: &["contents", "systemInstruction", "model", "cachedContent"],
        retype: &[
            ("generationConfig", "GenerationSettings"),
            ("tools", "HostedTool"),
            ("toolConfig", "ToolConfigSettings"),
        ],
    },
    Template {
        name: "GenerationSettings",
        source: "GenerationConfig",
        remove: &[
            "temperature",
            "maxOutputTokens",
            "responseMimeType",
            "responseSchema",
            "responseJsonSchema",
            "_responseJsonSchema",
        ],
        retype: &[],
    },
    Template {
        name: "HostedTool",
        source: "Tool",
        remove: &["functionDeclarations"],
        retype: &[],
    },
    Template {
        name: "ToolConfigSettings",
        source: "ToolConfig",
        remove: &["functionCallingConfig", "includeServerSideToolInvocations"],
        retype: &[],
    },
];

/// Template fields rendered as plain structs, skipped when default.
const INLINE_TEMPLATE_FIELDS: &[&str] = &["GenerationSettings", "ToolConfigSettings"];

const RUST_KEYWORDS: &[&str] = &[
    "type", "ref", "enum", "struct", "fn", "match", "mod", "use", "self", "move", "loop", "in",
    "where", "impl", "trait", "async", "await", "dyn", "box", "override", "final", "abstract",
    "macro", "yield", "static", "const", "crate", "super", "as", "break", "continue", "else",
    "extern", "false", "true", "for", "if", "let", "mut", "pub", "return", "unsafe", "while",
    "try",
];

pub(crate) const USAGE: &str = "\
  gemini-api [--fetch]        regenerate rig-core's Gemini mirror from the pinned
                              discovery document; --fetch refreshes it first
";

/// Run the task.
pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<(), String> {
    let fetch = match args.as_slice() {
        [] => false,
        [flag] if flag == "--fetch" => true,
        other => return Err(format!("unexpected gemini-api arguments {other:?}")),
    };
    let dir = root.join(API_DIR);
    let discovery = dir.join("discovery.json");
    if fetch {
        fetch_discovery(&discovery)?;
    }
    let text = std::fs::read_to_string(&discovery)
        .map_err(|error| format!("reading {}: {error}", discovery.display()))?;
    let doc: Value = serde_json::from_str(&text)
        .map_err(|error| format!("parsing {}: {error}", discovery.display()))?;
    let generator = Generator::new(&doc)?;
    let source = generator.render()?;
    let formatted = rustfmt(root, &source)?;
    let target = dir.join("generated.rs");
    let unchanged = std::fs::read_to_string(&target).is_ok_and(|current| current == formatted);
    if !unchanged {
        std::fs::write(&target, formatted)
            .map_err(|error| format!("writing {}: {error}", target.display()))?;
    }
    println!(
        "{} structs, {} templates, {} enums{}",
        generator.schemas.len(),
        TEMPLATES.len(),
        generator.enums.len(),
        if unchanged { " (unchanged)" } else { "" }
    );
    Ok(())
}

/// Replace the pinned discovery document with Google's current one.
fn fetch_discovery(target: &Path) -> Result<(), String> {
    let output = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            DISCOVERY_URL,
        ])
        .output()
        .map_err(|error| format!("running curl: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "fetching {DISCOVERY_URL}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let doc: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("the fetched discovery document is not JSON: {error}"))?;
    let mut pretty = serde_json::to_string_pretty(&doc).map_err(|error| error.to_string())?;
    pretty.push('\n');
    std::fs::write(target, pretty).map_err(|error| format!("writing {}: {error}", target.display()))
}

/// Format generated source the way `cargo fmt` would.
fn rustfmt(root: &Path, source: &str) -> Result<String, String> {
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("running rustfmt: {error}"))?;
    child
        .stdin
        .take()
        .ok_or("rustfmt has no stdin")?
        .write_all(source.as_bytes())
        .map_err(|error| format!("writing to rustfmt: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("waiting for rustfmt: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustfmt rejected the generated source: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
}

/// How a field is held in its struct.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Opt,
    Vec,
    Map,
    Inline,
}

struct Generator {
    revision: String,
    /// Schemas by (renamed) name, with `$ref`s renamed too.
    schemas: BTreeMap<String, Value>,
    /// Renamed name to discovery name.
    original: HashMap<String, String>,
    /// Enum name to its wire values.
    enums: BTreeMap<String, Vec<String>>,
    /// (schema, property) to enum name.
    enum_of: HashMap<(String, String), String>,
    /// Schema to the schemas its fields reference.
    edges: HashMap<String, BTreeSet<String>>,
}

impl Generator {
    fn new(doc: &Value) -> Result<Self, String> {
        let raw = doc
            .get("schemas")
            .and_then(Value::as_object)
            .ok_or("the discovery document has no schemas")?;
        let rename = |name: &str| -> String {
            SCHEMA_RENAMES
                .iter()
                .find(|(from, _)| *from == name)
                .map_or_else(|| name.to_owned(), |(_, to)| (*to).to_owned())
        };
        let mut schemas = BTreeMap::new();
        for (name, schema) in raw {
            let mut schema = schema.clone();
            rename_refs(&mut schema, &rename);
            schemas.insert(rename(name), schema);
        }
        let original = SCHEMA_RENAMES
            .iter()
            .map(|(from, to)| ((*to).to_owned(), (*from).to_owned()))
            .collect();
        let mut generator = Self {
            revision: doc
                .get("revision")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned(),
            schemas,
            original,
            enums: BTreeMap::new(),
            enum_of: HashMap::new(),
            edges: HashMap::new(),
        };
        generator.name_enums()?;
        generator.collect_edges();
        Ok(generator)
    }

    fn name_enums(&mut self) -> Result<(), String> {
        let mut found = Vec::new();
        for (schema, body) in &self.schemas {
            for (property, prop) in properties(body) {
                let values = prop
                    .get("enum")
                    .or_else(|| prop.get("items").and_then(|items| items.get("enum")))
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .filter(|values| !values.is_empty());
                if let Some(values) = values {
                    found.push((schema.clone(), property.clone(), values));
                }
            }
        }
        for (schema, property, values) in found {
            let original = self
                .original
                .get(&schema)
                .cloned()
                .unwrap_or(schema.clone());
            let named = ENUM_NAMES
                .iter()
                .find(|((owner, field), _)| *owner == original && *field == property)
                .map(|(_, name)| (*name).to_owned());
            let name = match named {
                Some(name) => {
                    if let Some(existing) = self.enums.get(&name)
                        && *existing != values
                    {
                        return Err(format!("{name} is named for two different value sets"));
                    }
                    self.enums.insert(name.clone(), values);
                    name
                }
                None => self.enum_for(&schema, &property, values),
            };
            self.enum_of.insert((schema, property), name);
        }
        Ok(())
    }

    /// Name an enum after its field, qualified by its owner when two fields
    /// share a name but not a value set.
    fn enum_for(&mut self, schema: &str, property: &str, values: Vec<String>) -> String {
        let mut first = property.chars();
        let head: String = first
            .next()
            .map(|c| c.to_uppercase().collect())
            .unwrap_or_default();
        let base = pascal(&format!("{head}{}", first.as_str().trim_start_matches('_')));
        for candidate in [base.clone(), format!("{schema}{base}")] {
            if self.schemas.contains_key(&candidate) {
                continue;
            }
            match self.enums.get(&candidate) {
                Some(existing) if *existing != values => {}
                _ => {
                    self.enums.insert(candidate.clone(), values);
                    return candidate;
                }
            }
        }
        let name = format!("{schema}{base}");
        self.enums.insert(name.clone(), values);
        name
    }

    fn collect_edges(&mut self) {
        for (schema, body) in &self.schemas {
            let mut targets = BTreeSet::new();
            for (_, prop) in properties(body) {
                refs(prop, &mut targets);
            }
            self.edges.insert(schema.clone(), targets);
        }
    }

    fn reaches(&self, from: &str, to: &str) -> bool {
        let mut seen = BTreeSet::new();
        let mut stack = vec![from.to_owned()];
        while let Some(node) = stack.pop() {
            if node == to {
                return true;
            }
            if !seen.insert(node.clone()) {
                continue;
            }
            if let Some(next) = self.edges.get(&node) {
                stack.extend(next.iter().cloned());
            }
        }
        false
    }

    fn rust_type(
        &self,
        owner: &str,
        property: &str,
        prop: &Value,
        retype: Option<&str>,
    ) -> Result<(String, Kind), String> {
        let array = prop.get("type").and_then(Value::as_str) == Some("array");
        if let Some(retype) = retype {
            if array {
                return Ok((format!("Vec<{retype}>"), Kind::Vec));
            }
            if INLINE_TEMPLATE_FIELDS.contains(&retype) {
                return Ok((retype.to_owned(), Kind::Inline));
            }
            return Ok((format!("Option<{retype}>"), Kind::Opt));
        }
        if array {
            let items = prop
                .get("items")
                .ok_or_else(|| format!("{owner}.{property} has no items"))?;
            let inner = self.element_type(owner, property, items, true)?;
            return Ok((format!("Vec<{inner}>"), Kind::Vec));
        }
        let inner = self.element_type(owner, property, prop, false)?;
        Ok((format!("Option<{inner}>"), Kind::Opt))
    }

    fn element_type(
        &self,
        owner: &str,
        property: &str,
        prop: &Value,
        in_vec: bool,
    ) -> Result<String, String> {
        if let Some(target) = prop.get("$ref").and_then(Value::as_str) {
            if !in_vec && self.reaches(target, owner) {
                return Ok(format!("Box<{target}>"));
            }
            return Ok(target.to_owned());
        }
        if prop.get("enum").is_some() {
            return self
                .enum_of
                .get(&(owner.to_owned(), property.to_owned()))
                .cloned()
                .ok_or_else(|| format!("{owner}.{property} has an unnamed enum"));
        }
        let format = prop.get("format").and_then(Value::as_str);
        Ok(match prop.get("type").and_then(Value::as_str) {
            Some("string") => "String".to_owned(),
            Some("integer") => match format {
                Some("int32") => "i32",
                Some("uint32") => "u32",
                _ => "i64",
            }
            .to_owned(),
            Some("number") => "f64".to_owned(),
            Some("boolean") => "bool".to_owned(),
            Some("any") => "serde_json::Value".to_owned(),
            Some("object") => match prop
                .get("additionalProperties")
                .and_then(|additional| additional.get("$ref"))
                .and_then(Value::as_str)
            {
                Some(target) => format!("std::collections::BTreeMap<String, {target}>"),
                None => "serde_json::Map<String, serde_json::Value>".to_owned(),
            },
            _ => return Err(format!("unhandled type {owner}.{property}: {prop}")),
        })
    }

    fn render(&self) -> Result<String, String> {
        let mut out = String::new();
        out.push_str(
            "// @generated by `cargo xtask gemini-api` from discovery.json. Do not edit.\n",
        );
        let _ = writeln!(
            out,
            "// Discovery revision {}, {} schemas.\n",
            self.revision,
            self.schemas.len()
        );
        out.push_str("#![allow(clippy::doc_markdown, clippy::large_enum_variant, rustdoc::bare_urls, rustdoc::broken_intra_doc_links, rustdoc::invalid_html_tags)]\n\n");
        out.push_str("use serde::{Deserialize, Serialize};\n\nuse super::{Mirrored, Unmodeled, mirror_enum};\n\n");
        let empty = Map::new();
        for (name, schema) in &self.schemas {
            let props = schema
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or(&empty);
            let all: Vec<&String> = props.keys().collect();
            out.push_str(
                &self.emit_struct(
                    name,
                    props.iter().collect(),
                    name,
                    &all,
                    &[],
                    schema
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?,
            );
        }
        for template in TEMPLATES {
            let schema = self
                .schemas
                .get(template.source)
                .ok_or_else(|| format!("template source {} is missing", template.source))?;
            let props = schema
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or(&empty);
            for removed in template.remove {
                if !props.contains_key(*removed) {
                    return Err(format!(
                        "{} has no field {removed} to remove",
                        template.source
                    ));
                }
            }
            let kept = props
                .iter()
                .filter(|(field, _)| !template.remove.contains(&field.as_str()))
                .collect();
            let all: Vec<&String> = props.keys().collect();
            let description = format!(
                "`{}` without the fields rig owns ({}).",
                template.source,
                template.remove.join(", ")
            );
            out.push_str(&self.emit_struct(
                template.name,
                kept,
                template.source,
                &all,
                template.retype,
                &description,
            )?);
        }
        for (name, values) in &self.enums {
            out.push_str(&emit_enum(name, values));
        }
        Ok(out)
    }

    fn emit_struct(
        &self,
        name: &str,
        props: Vec<(&String, &Value)>,
        source: &str,
        all_fields: &[&String],
        retypes: &[(&str, &str)],
        description: &str,
    ) -> Result<String, String> {
        let retype_of = |field: &str| {
            retypes
                .iter()
                .find(|(from, _)| *from == field)
                .map(|(_, to)| *to)
        };
        let mut out = doc(description, "");
        out.push_str("#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]\n");
        let _ = writeln!(out, "pub struct {name} {{");
        let mut taken = BTreeSet::new();
        let mut idents = BTreeMap::new();
        for (property, prop) in &props {
            let ident = field_ident(property, &taken);
            taken.insert(ident.clone());
            idents.insert((*property).clone(), ident.clone());
            let (ty, kind) = self.rust_type(source, property, prop, retype_of(property))?;
            out.push_str(&doc(
                prop.get("description")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                "    ",
            ));
            let mut attrs = vec![format!("rename = \"{property}\"")];
            attrs.push(match kind {
                Kind::Opt => "default, skip_serializing_if = \"Option::is_none\"".to_owned(),
                Kind::Vec => "default, deserialize_with = \"super::null_as_default\", skip_serializing_if = \"Vec::is_empty\"".to_owned(),
                Kind::Inline | Kind::Map => {
                    format!("default, skip_serializing_if = \"{ty}::is_empty\"")
                }
            });
            let _ = writeln!(out, "    #[serde({})]", attrs.join(", "));
            let _ = writeln!(out, "    pub {ident}: {ty},");
        }
        out.push_str("    /// Fields this mirror does not type yet.\n");
        out.push_str("    #[serde(flatten)]\n");
        out.push_str("    pub unmodeled: Unmodeled<Self>,\n");
        out.push_str("}\n\n");

        let mut fields: Vec<&str> = all_fields.iter().map(|field| field.as_str()).collect();
        fields.sort_unstable();
        let fields = fields
            .iter()
            .map(|field| format!("\"{field}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let mut visits = String::new();
        for (property, prop) in &props {
            let Some(ident) = idents.get(*property) else {
                continue;
            };
            let array = prop.get("type").and_then(Value::as_str) == Some("array");
            let (target, kind) = match retype_of(property) {
                Some(retype) => (
                    Some(retype.to_owned()),
                    if array {
                        Kind::Vec
                    } else if INLINE_TEMPLATE_FIELDS.contains(&retype) {
                        Kind::Inline
                    } else {
                        Kind::Opt
                    },
                ),
                None => (
                    ref_target(prop),
                    if array {
                        Kind::Vec
                    } else if prop.get("additionalProperties").is_some() {
                        Kind::Map
                    } else {
                        Kind::Opt
                    },
                ),
            };
            if target.is_none() {
                continue;
            }
            let sub = format!("&format!(\"{{path}}.{property}\")");
            let _ = match kind {
                Kind::Opt => writeln!(
                    visits,
                    "        if let Some(value) = &self.{ident} {{ value.unmodeled_fields({sub}, out); }}"
                ),
                Kind::Inline => {
                    writeln!(visits, "        self.{ident}.unmodeled_fields({sub}, out);")
                }
                Kind::Vec => writeln!(
                    visits,
                    "        for (index, value) in self.{ident}.iter().enumerate() {{ value.unmodeled_fields(&format!(\"{{path}}.{property}[{{index}}]\"), out); }}"
                ),
                Kind::Map => writeln!(
                    visits,
                    "        for (key, value) in self.{ident}.iter().flatten() {{ value.unmodeled_fields(&format!(\"{{path}}.{property}.{{key}}\"), out); }}"
                ),
            };
        }
        let _ = writeln!(out, "impl Mirrored for {name} {{");
        let _ = writeln!(out, "    const NAME: &'static str = \"{name}\";");
        let _ = writeln!(
            out,
            "    const FIELDS: &'static [&'static str] = &[{fields}];"
        );
        out.push_str("    fn unmodeled_fields(&self, path: &str, out: &mut Vec<String>) {\n");
        out.push_str(
            "        out.extend(self.unmodeled.keys().map(|key| format!(\"{path}.{key}\")));\n",
        );
        out.push_str(&visits);
        out.push_str("    }\n}\n\n");
        if INLINE_TEMPLATE_FIELDS.contains(&name) {
            let _ = write!(
                out,
                "impl {name} {{\n    /// Whether nothing is set.\n    pub fn is_empty(&self) -> bool {{\n        *self == Self::default()\n    }}\n}}\n\n"
            );
        }
        Ok(out)
    }
}

fn properties(schema: &Value) -> impl Iterator<Item = (&String, &Value)> {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
}

fn rename_refs(node: &mut Value, rename: &impl Fn(&str) -> String) {
    match node {
        Value::Object(map) => {
            if let Some(Value::String(target)) = map.get_mut("$ref") {
                *target = rename(target);
            }
            for value in map.values_mut() {
                rename_refs(value, rename);
            }
        }
        Value::Array(values) => {
            for value in values {
                rename_refs(value, rename);
            }
        }
        _ => {}
    }
}

fn refs(prop: &Value, out: &mut BTreeSet<String>) {
    if let Some(target) = prop.get("$ref").and_then(Value::as_str) {
        out.insert(target.to_owned());
    }
    if let Some(items) = prop.get("items") {
        refs(items, out);
    }
    if let Some(additional) = prop
        .get("additionalProperties")
        .filter(|value| value.is_object())
    {
        refs(additional, out);
    }
}

fn ref_target(prop: &Value) -> Option<String> {
    prop.get("$ref")
        .or_else(|| prop.get("items").and_then(|items| items.get("$ref")))
        .or_else(|| {
            prop.get("additionalProperties")
                .and_then(|additional| additional.get("$ref"))
        })
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// A doc comment. A bare code fence is marked `text`, since rustdoc would
/// compile it as a Rust doctest.
fn doc(text: &str, indent: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut fenced = false;
    for line in text.replace('\r', "").split('\n') {
        let line = if line.trim_start().starts_with("```") {
            let opening = !fenced;
            fenced = !fenced;
            if opening && line.trim() == "```" {
                "```text"
            } else {
                line
            }
        } else {
            line
        };
        let _ = writeln!(out, "{}", format!("{indent}/// {line}").trim_end());
    }
    out
}

fn snake(name: &str) -> String {
    let name = name.trim_start_matches('_');
    let mut out = String::new();
    let mut previous: Option<char> = None;
    for c in name.chars() {
        if c.is_ascii_uppercase()
            && previous.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit())
        {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
        previous = Some(c);
    }
    out
}

fn field_ident(wire: &str, taken: &BTreeSet<String>) -> String {
    let mut ident = snake(wire);
    if wire.starts_with('_') {
        ident = format!("legacy_{ident}");
    }
    if RUST_KEYWORDS.contains(&ident.as_str()) {
        ident = format!("r#{ident}");
    }
    while taken.contains(&ident) {
        ident.push('_');
    }
    ident
}

fn pascal(name: &str) -> String {
    name.split(|c: char| c == '_' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default()
        })
        .collect()
}

fn variant_ident(value: &str, prefix: &str) -> String {
    let value = match value.strip_prefix(prefix) {
        Some(rest) if !prefix.is_empty() && !rest.is_empty() => rest,
        _ => value,
    };
    let mut ident: String = value
        .split(|c: char| c == '_' || c == '-' || c == '.' || c.is_whitespace())
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|first| {
                    first
                        .to_uppercase()
                        .chain(chars.flat_map(char::to_lowercase))
                        .collect::<String>()
                })
                .unwrap_or_default()
        })
        .collect();
    if !ident.chars().next().is_some_and(char::is_alphabetic) {
        ident = format!("V{ident}");
    }
    if ident == "Unknown" {
        ident = "UnknownValue".to_owned();
    }
    ident
}

/// The `_`-separated words every value starts with, keeping at least one
/// word in each variant.
fn common_prefix(values: &[String]) -> String {
    if values.len() < 2 {
        return String::new();
    }
    let parts: Vec<Vec<&str>> = values
        .iter()
        .map(|value| value.split('_').collect())
        .collect();
    let mut prefix = Vec::new();
    for index in 0.. {
        let words: Option<Vec<&str>> = parts.iter().map(|part| part.get(index).copied()).collect();
        match words {
            Some(words) if words.windows(2).all(|pair| pair.first() == pair.last()) => {
                match words.first() {
                    Some(word) => prefix.push(*word),
                    None => break,
                }
            }
            _ => break,
        }
    }
    if parts.iter().any(|part| part.len() == prefix.len()) {
        prefix.pop();
    }
    if prefix.is_empty() {
        String::new()
    } else {
        format!("{}_", prefix.join("_"))
    }
}

fn emit_enum(name: &str, values: &[String]) -> String {
    let prefix = common_prefix(values);
    let mut taken = BTreeSet::new();
    let mut body = String::new();
    for value in values {
        let mut ident = variant_ident(value, &prefix);
        while taken.contains(&ident) {
            ident.push('_');
        }
        taken.insert(ident.clone());
        let _ = writeln!(body, "    {ident} => \"{value}\",");
    }
    format!("mirror_enum! {{\n    {name} {{\n{body}    }}\n}}\n\n")
}

#[cfg(test)]
mod tests;
