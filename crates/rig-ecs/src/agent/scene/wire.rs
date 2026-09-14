//! Scene-only binary pooling across graph, effects, stream logs and extensions.
//! Transport DTOs stay ordinary values in memory. The envelope escapes reserved
//! object shapes, so arbitrary provider/extension JSON round-trips unchanged.

use std::collections::BTreeMap;

use base64::{Engine, prelude::BASE64_STANDARD};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{WorldScene, WorldSceneData};
use crate::agent::content::binary::{
    BinaryAssets, BinaryEncoding, BinaryId, BinaryRecord, PartSource,
};
use rig_core::message::DocumentSourceKind;

mod bounded;

const FORMAT: &str = "rig-ecs/world/2";
const BINARY: &str = "$rig_binary";
const OBJECT: &str = "$rig_object";
/// Maximum serialized input and expanded scene representation, excluding caller-owned input.
const MAX_BYTES: usize = 512 * 1024 * 1024;
const MAX_NODES: usize = 1_000_000;
const MAX_DEPTH: usize = 64;

type SpellingIndex = BTreeMap<[u8; 32], PartSource>;

/// Bound caller-constructed graph data before component decoding or world allocation.
pub(super) fn validate_graph(scene: &super::RunScene) -> Result<(), String> {
    fn value_node(value: &Value, budget: &mut Budget, depth: usize) -> Result<(), String> {
        budget.take(value.as_str().map_or(8, str::len), depth)?;
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    budget.take(key.len(), depth)?;
                    value_node(child, budget, depth + 1)?;
                }
            }
            Value::Array(items) => {
                for child in items {
                    value_node(child, budget, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut budget = Budget::new();
    for binary in &scene.binaries {
        budget.take(binary.data.len(), 0)?;
    }
    for entity in &scene.entities {
        budget.take(8, 0)?;
        for (name, component) in &entity.components {
            budget.take(name.len(), 1)?;
            value_node(component, &mut budget, 2)?;
        }
        for (name, target) in &entity.relations {
            budget.take(name.len(), 1)?;
            if let super::Target::Handler { key } = target {
                budget.take(key.as_str().len(), 2)?;
            }
        }
    }
    Ok(())
}

/// Typed handles stay compact in scene JSON, but content validation rebuilds DTOs.
/// Charge every occurrence before resolving any of them, including shared handles.
pub(super) fn validate_content_expansion(world: &mut bevy_ecs::world::World) -> Result<(), String> {
    use crate::agent::content::parts::{AudioPart, DocumentPart, ImagePart, VideoPart};
    let mut parts = world.query::<(
        Option<&ImagePart>,
        Option<&AudioPart>,
        Option<&VideoPart>,
        Option<&DocumentPart>,
    )>();
    let assets = world.resource::<BinaryAssets>();
    let mut budget = Budget::new();
    for (image, audio, video, document) in parts.iter(world) {
        for source in [
            image.map(|part| &part.source),
            audio.map(|part| &part.source),
            video.map(|part| &part.source),
            document.map(|part| &part.source),
        ]
        .into_iter()
        .flatten()
        {
            budget.take(
                assets
                    .resolved_len(source)
                    .map_err(|error| error.to_string())?,
                0,
            )?;
        }
    }
    Ok(())
}

fn problem(message: &str) -> String {
    message.to_owned()
}

struct Budget {
    bytes: usize,
    nodes: usize,
}
impl Budget {
    fn new() -> Self {
        Self {
            bytes: MAX_BYTES,
            nodes: MAX_NODES,
        }
    }
    fn take(&mut self, bytes: usize, depth: usize) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Err(problem("scene depth limit exceeded"));
        }
        self.nodes = self
            .nodes
            .checked_sub(1)
            .ok_or_else(|| problem("scene node limit exceeded"))?;
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .ok_or_else(|| problem("expanded scene byte limit exceeded"))?;
        Ok(())
    }
}

fn raw(value: &Value) -> Option<Vec<u8>> {
    value
        .as_array()?
        .iter()
        .map(|value| value.as_u64().and_then(|n| u8::try_from(n).ok()))
        .collect()
}

fn discover(
    value: &Value,
    assets: &mut BinaryAssets,
    spellings: &mut SpellingIndex,
    depth: usize,
    budget: &mut Budget,
) -> Result<(), String> {
    budget.take(value.as_str().map_or(8, str::len), depth)?;
    match value {
        Value::Object(map) => {
            if map.len() == 2 {
                let source = match (map.get("type").and_then(Value::as_str), map.get("value")) {
                    (Some("base64"), Some(Value::String(value))) => {
                        Some(DocumentSourceKind::Base64(value.clone()))
                    }
                    (Some("raw"), Some(value)) => raw(value).map(DocumentSourceKind::Raw),
                    _ => None,
                };
                if let Some(source) = source {
                    let spelling = match &source {
                        DocumentSourceKind::Base64(value) => {
                            Some(Sha256::digest(value.as_bytes()).into())
                        }
                        _ => None,
                    };
                    match assets.intern(source) {
                        Ok(reference) => {
                            if let Some(key) = spelling {
                                spellings.insert(key, reference);
                            }
                        }
                        // Unknown application JSON can resemble a source without
                        // being valid base64. Preserve it as literal data.
                        Err(crate::agent::content::binary::BinaryError::Base64) => {}
                        Err(error) => return Err(error.to_string()),
                    }
                }
            }
            for (key, value) in map {
                budget.take(key.len(), depth)?;
                discover(value, assets, spellings, depth + 1, budget)?;
            }
        }
        Value::Array(items) => {
            for value in items {
                discover(value, assets, spellings, depth + 1, budget)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn marker(reference: PartSource) -> Result<Value, String> {
    Ok(
        serde_json::json!({ BINARY: serde_json::to_value(reference).map_err(|_|problem("binary reference serialization failed"))? }),
    )
}

fn compress(
    value: Value,
    assets: &BinaryAssets,
    spellings: &SpellingIndex,
) -> Result<Value, String> {
    match value {
        Value::String(value) => {
            let key: [u8; 32] = Sha256::digest(value.as_bytes()).into();
            if let Some(reference) = spellings.get(&key) {
                return marker(reference.clone());
            }
            Ok(Value::String(value))
        }
        Value::Array(items) => {
            let array = Value::Array(items);
            if let Some(bytes) = raw(&array) {
                let id = BinaryId::of(&bytes);
                if assets.get(id).is_ok_and(|stored| stored == bytes) {
                    return marker(PartSource::Binary {
                        id,
                        encoding: BinaryEncoding::Raw,
                    });
                }
            }
            let Value::Array(items) = array else {
                return Err(problem("invalid scene array"));
            };
            Ok(Value::Array(
                items
                    .into_iter()
                    .map(|item| compress(item, assets, spellings))
                    .collect::<Result<_, _>>()?,
            ))
        }
        Value::Object(map) => {
            let escaped = map.len() == 1 && (map.contains_key(BINARY) || map.contains_key(OBJECT));
            let map = map
                .into_iter()
                .map(|(key, value)| Ok((key, compress(value, assets, spellings)?)))
                .collect::<Result<serde_json::Map<_, _>, String>>()?;
            if escaped {
                Ok(serde_json::json!({OBJECT:map}))
            } else {
                Ok(Value::Object(map))
            }
        }
        value => Ok(value),
    }
}

fn charge_reference(
    reference: &PartSource,
    assets: &BinaryAssets,
    budget: &mut Budget,
    depth: usize,
) -> Result<(), String> {
    let PartSource::Binary { id, encoding } = reference else {
        return Err(problem("scene reference must name binary content"));
    };
    assets
        .resolved_len(reference)
        .map_err(|error| error.to_string())?;
    let bytes = assets.get(*id).map_err(|error| error.to_string())?.len();
    let expanded = match encoding {
        BinaryEncoding::Raw => bytes.checked_mul(4),
        BinaryEncoding::Base64 { .. } => bytes.checked_add(2).map(|n| n / 3 * 4),
    }
    .ok_or_else(|| problem("scene expansion overflow"))?;
    budget.take(expanded, depth)?;
    if matches!(encoding, BinaryEncoding::Raw) {
        budget.nodes = budget
            .nodes
            .checked_sub(bytes)
            .ok_or_else(|| problem("scene node limit exceeded"))?;
    }
    Ok(())
}

/// Run the expansion accounting without allocating expanded strings or arrays.
fn check_expansion(
    value: &Value,
    assets: &BinaryAssets,
    budget: &mut Budget,
    depth: usize,
) -> Result<(), String> {
    budget.take(value.as_str().map_or(8, str::len), depth)?;
    match value {
        Value::Object(map) => {
            let map = if map.len() == 1 {
                if let Some(reference) = map.get(BINARY) {
                    let reference: PartSource = serde_json::from_value(reference.clone())
                        .map_err(|_| problem("invalid binary scene reference"))?;
                    return charge_reference(&reference, assets, budget, depth);
                }
                if let Some(literal) = map.get(OBJECT) {
                    literal
                        .as_object()
                        .ok_or_else(|| problem("invalid escaped scene object"))?
                } else {
                    map
                }
            } else {
                map
            };
            for (key, value) in map {
                budget.take(key.len(), depth)?;
                check_expansion(value, assets, budget, depth + 1)?;
            }
        }
        Value::Array(items) => {
            for value in items {
                check_expansion(value, assets, budget, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn expand(
    value: Value,
    assets: &BinaryAssets,
    budget: &mut Budget,
    depth: usize,
) -> Result<Value, String> {
    budget.take(value.as_str().map_or(8, str::len), depth)?;
    match value {
        Value::Object(mut map) => {
            if map.len() == 1 {
                if let Some(reference) = map.remove(BINARY) {
                    let reference: PartSource = serde_json::from_value(reference)
                        .map_err(|_| problem("invalid binary scene reference"))?;
                    charge_reference(&reference, assets, budget, depth)?;
                    return match assets.resolve(&reference).map_err(|e| e.to_string())? {
                        DocumentSourceKind::Base64(value) => Ok(Value::String(value)),
                        DocumentSourceKind::Raw(bytes) => Ok(serde_json::json!(bytes)),
                        _ => Err(problem("invalid resolved scene reference")),
                    };
                }
                if let Some(literal) = map.remove(OBJECT) {
                    let Value::Object(map) = literal else {
                        return Err(problem("invalid escaped scene object"));
                    };
                    return expand_object(map, assets, budget, depth);
                }
            }
            expand_object(map, assets, budget, depth)
        }
        Value::Array(items) => Ok(Value::Array(
            items
                .into_iter()
                .map(|value| expand(value, assets, budget, depth + 1))
                .collect::<Result<_, _>>()?,
        )),
        value => Ok(value),
    }
}

fn expand_object(
    map: serde_json::Map<String, Value>,
    assets: &BinaryAssets,
    budget: &mut Budget,
    depth: usize,
) -> Result<Value, String> {
    let mut result = serde_json::Map::new();
    for (key, value) in map {
        budget.take(key.len(), depth)?;
        result.insert(key, expand(value, assets, budget, depth + 1)?);
    }
    Ok(Value::Object(result))
}

fn table(value: &mut Value) -> Result<&mut serde_json::Map<String, Value>, String> {
    value
        .get_mut("graph")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| problem("scene graph missing"))
}

impl Serialize for WorldScene {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let mut value = WorldSceneData::serialize(self, serde_json::value::Serializer)
            .map_err(S::Error::custom)?;
        table(&mut value)
            .map_err(S::Error::custom)?
            .remove("binaries");
        let mut assets = BinaryAssets::default()
            .merged(&self.graph.binaries)
            .map_err(S::Error::custom)?;
        let mut spellings = SpellingIndex::new();
        for (id, bytes) in assets.iter() {
            let spelling = BASE64_STANDARD.encode(bytes);
            spellings.insert(
                Sha256::digest(spelling.as_bytes()).into(),
                PartSource::Binary {
                    id,
                    encoding: BinaryEncoding::Base64 {
                        padding: (spelling.len() - spelling.trim_end_matches('=').len()) as u8,
                        last_symbol: None,
                    },
                },
            );
        }
        discover(&value, &mut assets, &mut spellings, 0, &mut Budget::new())
            .map_err(S::Error::custom)?;
        let mut value = compress(value, &assets, &spellings).map_err(S::Error::custom)?;
        check_expansion(&value, &assets, &mut Budget::new(), 0).map_err(S::Error::custom)?;
        table(&mut value).map_err(S::Error::custom)?.insert(
            "binaries".into(),
            serde_json::to_value(assets.snapshot()).map_err(S::Error::custom)?,
        );
        value
            .as_object_mut()
            .ok_or_else(|| S::Error::custom("scene object missing"))?
            .insert("format".into(), Value::String(FORMAT.into()));
        bounded::validate(&value).map_err(S::Error::custom)?;
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WorldScene {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let mut value = bounded::deserialize(deserializer)?;
        if value.as_object_mut().and_then(|map| map.remove("format"))
            != Some(Value::String(FORMAT.into()))
        {
            return Err(D::Error::custom("unsupported world scene format"));
        }
        let records = table(&mut value)
            .map_err(D::Error::custom)?
            .remove("binaries")
            .ok_or_else(|| D::Error::custom("binary table missing"))?;
        let records: Vec<BinaryRecord> =
            serde_json::from_value(records).map_err(D::Error::custom)?;
        let assets = BinaryAssets::default()
            .merged(&records)
            .map_err(D::Error::custom)?;
        let mut value = expand(value, &assets, &mut Budget::new(), 0).map_err(D::Error::custom)?;
        table(&mut value).map_err(D::Error::custom)?.insert(
            "binaries".into(),
            serde_json::to_value(records).map_err(D::Error::custom)?,
        );
        WorldSceneData::deserialize(value).map_err(D::Error::custom)
    }
}

impl WorldScene {
    /// Parse a scene with a bounded input buffer and bounded binary expansion.
    /// Loading the parsed data still validates graph references before mutation.
    pub fn from_json(bytes: &[u8]) -> Result<Self, rig_core::error::ErrorReport> {
        if bytes.len() > MAX_BYTES {
            return Err(super::extension_error("scene input byte limit exceeded"));
        }
        serde_json::from_slice(bytes)
            .map_err(|_| super::extension_error("invalid or oversized world scene"))
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "controlled validation fixtures"
)]
mod expansion_tests;
