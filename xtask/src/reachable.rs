//! Where a type can actually be *named* from outside the crate.
//!
//! rustdoc reports each item's **definition** path, which is not always a path
//! a caller may write: rig-core repeatedly declares a private module and
//! re-exports its contents (`mod audio_generation; pub use
//! audio_generation::*;`), so the definition path
//! `providers::azure::audio_generation::AudioGenerationModel` names a private
//! module and does not compile from another crate.
//!
//! So the generator walks the public module tree from the crate root, following
//! public modules and both kinds of re-export, and records every path each item
//! can be named by. The alias tree then uses the shortest one, which is also the
//! one a human would have written.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Every publicly nameable path for each item id, keyed by id.
pub type PublicPaths = BTreeMap<String, BTreeSet<Vec<String>>>;

/// Walk the crate's public surface from `root`, collecting the paths by which
/// each item can be named.
pub fn public_paths(json: &Value) -> PublicPaths {
    let Some(index) = json.get("index").and_then(Value::as_object) else {
        return PublicPaths::new();
    };
    let root = json
        .get("root")
        .map(|root| root.to_string().trim_matches('"').to_string())
        .unwrap_or_default();
    let crate_name = index
        .get(&root)
        .and_then(|item| item.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("rig_core")
        .to_string();

    let mut paths: PublicPaths = PublicPaths::new();
    let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
    let mut visited: BTreeSet<(String, usize)> = BTreeSet::new();
    queue.push_back((root, vec![crate_name]));

    while let Some((id, prefix)) = queue.pop_front() {
        // A module reachable by several routes is walked once per depth, which
        // keeps the shortest route without looping on cyclic re-exports.
        if !visited.insert((id.clone(), prefix.len())) {
            continue;
        }
        let Some(module) = index
            .get(&id)
            .and_then(|item| item.get("inner"))
            .and_then(|inner| inner.get("module"))
        else {
            continue;
        };
        let Some(items) = module.get("items").and_then(Value::as_array) else {
            continue;
        };

        for child in items.iter().filter_map(Value::as_u64) {
            let child = child.to_string();
            let Some(item) = index.get(&child) else {
                continue;
            };
            if !is_public(item) {
                continue;
            }
            let Some(inner) = item.get("inner") else {
                continue;
            };

            // A glob re-export lifts the target module's public items into
            // *this* module's path, which is exactly how the private
            // `audio_generation` modules surface their types.
            if let Some(use_item) = inner.get("use") {
                let target = use_item
                    .get("id")
                    .and_then(Value::as_u64)
                    .map(|id| id.to_string());
                let Some(target) = target else { continue };
                if use_item.get("is_glob").and_then(Value::as_bool) == Some(true) {
                    queue.push_back((target, prefix.clone()));
                } else if let Some(name) = use_item.get("name").and_then(Value::as_str) {
                    let mut path = prefix.clone();
                    path.push(name.to_string());
                    record(&mut paths, &target, path.clone());
                    // A re-exported module is walkable at its new path.
                    queue.push_back((target, path));
                }
                continue;
            }

            let Some(name) = item.get("name").and_then(Value::as_str) else {
                continue;
            };
            let mut path = prefix.clone();
            path.push(name.to_string());

            if inner.get("module").is_some() {
                queue.push_back((child.clone(), path.clone()));
            }
            record(&mut paths, &child, path);
        }
    }

    paths
}

fn record(paths: &mut PublicPaths, id: &str, path: Vec<String>) {
    paths.entry(id.to_string()).or_default().insert(path);
}

/// rustdoc reports visibility as `"public"`, or as a struct for restricted
/// visibility. Only the former can be named from another crate.
fn is_public(item: &Value) -> bool {
    item.get("visibility").and_then(Value::as_str) == Some("public")
}

/// The shortest publicly nameable path for `id`, if any.
pub fn shortest(paths: &PublicPaths, id: &str) -> Option<Vec<String>> {
    paths
        .get(id)?
        .iter()
        .min_by_key(|path| (path.len(), (*path).clone()))
        .cloned()
}
