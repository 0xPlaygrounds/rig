//! Every recorded Gemini REST body through the generated `gemini::api` mirror.
//! Each body must decode, leave no field `unmodeled`, and re-serialize to the
//! same JSON. Interactions bodies have a hand-maintained mirror and are skipped.
#![allow(clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use rig_core::providers::gemini::api::{self, Mirrored};
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[derive(Default)]
struct Report {
    bodies: usize,
    failures: Vec<String>,
}

impl Report {
    fn check<T: DeserializeOwned + Serialize + Mirrored>(&mut self, origin: &str, text: &str) {
        let Ok(original) = serde_json::from_str::<Value>(text) else {
            return;
        };
        self.bodies += 1;
        let mut deserializer = serde_json::Deserializer::from_str(text);
        let typed: T = match serde_path_to_error::deserialize(&mut deserializer) {
            Ok(typed) => typed,
            Err(error) => {
                self.failures
                    .push(format!("{origin}: {} does not decode: {error}", T::NAME));
                return;
            }
        };
        let mut unmodeled = Vec::new();
        typed.unmodeled_fields(T::NAME, &mut unmodeled);
        for field in unmodeled {
            self.failures.push(format!("{origin}: unmodeled {field}"));
        }
        let reserialized = serde_json::to_value(&typed).expect("a mirror serializes");
        let mut diffs = Vec::new();
        diff(T::NAME, &original, &reserialized, &mut diffs);
        for change in diffs {
            self.failures.push(format!("{origin}: {change}"));
        }
    }

    fn exchange(
        &mut self,
        origin: &str,
        path: &str,
        request: Option<&str>,
        response: Option<&str>,
    ) {
        let generate = path.contains(":generateContent");
        let stream = path.contains(":streamGenerateContent");
        if generate || stream {
            if let Some(request) = request {
                self.check::<api::GenerateContentRequest>(origin, request);
            }
            match response {
                Some(body) if stream => {
                    for event in body.lines().filter_map(|line| line.strip_prefix("data:")) {
                        self.check::<api::GenerateContentResponse>(origin, event.trim());
                    }
                }
                Some(body) => self.check::<api::GenerateContentResponse>(origin, body),
                None => {}
            }
        } else if path.contains(":countTokens") {
            if let Some(body) = request {
                self.check::<api::CountTokensRequest>(origin, body);
            }
            if let Some(body) = response {
                self.check::<api::CountTokensResponse>(origin, body);
            }
        } else if path.contains(":batchEmbedContents") {
            if let Some(body) = request {
                self.check::<api::BatchEmbedContentsRequest>(origin, body);
            }
            if let Some(body) = response {
                self.check::<api::BatchEmbedContentsResponse>(origin, body);
            }
        } else if path.contains(":embedContent") {
            if let Some(body) = request {
                self.check::<api::EmbedContentRequest>(origin, body);
            }
            if let Some(body) = response {
                self.check::<api::EmbedContentResponse>(origin, body);
            }
        } else if path.contains("/cachedContents") {
            if let Some(body) = request {
                self.check::<api::CachedContent>(origin, body);
            }
            match response {
                Some(body) if path.ends_with("/cachedContents") && request.is_none() => {
                    self.check::<api::ListCachedContentsResponse>(origin, body)
                }
                Some(body) => self.check::<api::CachedContent>(origin, body),
                None => {}
            }
        } else if path.ends_with("/models")
            && let Some(body) = response
        {
            self.check::<api::ListModelsResponse>(origin, body);
        }
    }
}

/// Every difference between the recorded JSON and the mirror's re-serialization.
fn diff(path: &str, original: &Value, reserialized: &Value, out: &mut Vec<String>) {
    match (original, reserialized) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in a {
                let sub = format!("{path}.{key}");
                match b.get(key) {
                    Some(other) => diff(&sub, value, other, out),
                    None => out.push(format!("dropped {sub}")),
                }
            }
            for key in b.keys().filter(|key| !a.contains_key(*key)) {
                out.push(format!("added {path}.{key}"));
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (index, (x, y)) in a.iter().zip(b).enumerate() {
                diff(&format!("{path}[{index}]"), x, y, out);
            }
        }
        (Value::Number(a), Value::Number(b)) if a.as_f64() == b.as_f64() => {}
        (a, b) if a == b => {}
        _ => out.push(format!(
            "changed {path}: {original:.80} -> {reserialized:.80}"
        )),
    }
}

fn yaml_str<'a>(value: &'a serde_yaml::Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .try_fold(value, |node, key| node.get(*key))?
        .as_str()
        .filter(|text| !text.is_empty())
}

fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("a readable fixture directory") {
        let path = entry.expect("a readable entry").path();
        if path.is_dir() {
            walk(&path, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "yaml")
        {
            files.push(path);
        }
    }
}

#[test]
fn recorded_corpus_round_trips_through_the_mirror() {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/cassettes/gemini"
    ));
    let mut files = Vec::new();
    walk(root, &mut files);
    files.sort();

    let mut report = Report::default();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("a readable fixture");
        let origin = file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string();
        for document in serde_yaml::Deserializer::from_str(&text) {
            let document = serde_yaml::Value::deserialize(document).expect("a YAML document");
            let Some(path) = yaml_str(&document, &["when", "path"]) else {
                continue;
            };
            if path.contains("/interactions") {
                continue;
            }
            let status = document
                .get("then")
                .and_then(|then| then.get("status"))
                .and_then(serde_yaml::Value::as_u64)
                .unwrap_or(0);
            let response =
                yaml_str(&document, &["then", "body"]).filter(|_| (200..300).contains(&status));
            report.exchange(
                &origin,
                path,
                yaml_str(&document, &["when", "body"]),
                response,
            );
        }
    }

    assert!(
        report.bodies > 0,
        "no Gemini body was found under {}",
        root.display()
    );
    assert!(
        report.failures.is_empty(),
        "{} of {} bodies do not round-trip:\n{}",
        report.failures.len(),
        report.bodies,
        report.failures.join("\n")
    );
}
