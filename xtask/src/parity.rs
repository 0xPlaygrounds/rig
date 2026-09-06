//! Parity inventory and execution evidence. This tooling never runs a legacy
//! interpreter on behalf of the native counterpart.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) mod batch;
pub(crate) mod manifest;
pub(crate) mod queue;
pub(crate) mod review;
pub(crate) mod targets;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

fn put(value: &mut Value, keys: &[&str], data: Value) -> Result<()> {
    let (last, parents) = keys.split_last().ok_or("empty JSON field path")?;
    let mut parent = value;
    for key in parents {
        parent = parent.get_mut(*key).ok_or("missing JSON parent")?;
    }
    parent
        .as_object_mut()
        .ok_or("expected JSON object")?
        .insert((*last).into(), data);
    Ok(())
}
fn bytes(value: &Value) -> Result<Vec<u8>> {
    let mut data = serde_json::to_vec_pretty(value)?;
    data.push(b'\n');
    Ok(data)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {field}").into())
}
fn array<'a>(value: &'a Value, field: &str) -> Result<&'a Vec<Value>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {field}").into())
}
fn strings(value: &Value, field: &str) -> Result<Vec<String>> {
    array(value, field)?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("non-string in {field}").into())
        })
        .collect()
}
fn write(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, bytes(value)?)?;
    Ok(())
}
fn store_raw(directory: &Path, data: &[u8], extension: &str) -> Result<String> {
    fs::create_dir_all(directory)?;
    let name = format!("{}.{}", hash(data), extension);
    let path = directory.join(&name);
    if path.exists() && fs::read(&path)? != data {
        return Err("content-addressed artifact collision".into());
    }
    fs::write(path, data)?;
    Ok(name)
}
fn store(directory: &Path, value: &Value) -> Result<String> {
    store_raw(directory, &bytes(value)?, "json")
}
fn capture(root: &Path, args: &[&str]) -> Result<String> {
    let (program, args) = args.split_first().ok_or("empty command")?;
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?)
}
fn checked(root: &Path, name: &str) -> Result<PathBuf> {
    let path = root.join(name).canonicalize()?;
    if !path.starts_with(root.canonicalize()?) || !path.is_file() {
        return Err(format!("missing or escaping file: {name}").into());
    }
    Ok(path)
}
fn source_index(root: &Path) -> Result<Value> {
    let listing = capture(
        root,
        &[
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
    )?;
    let mut sources = BTreeMap::new();
    for name in listing.split('\0').filter(|name| !name.is_empty()) {
        let path = root.join(name);
        let suffix = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.starts_with("tests/ecs_parity/evidence/")
            || name.contains("__pycache__")
            || !path.is_file()
        {
            continue;
        }
        if matches!(
            suffix,
            "rs" | "toml" | "lock" | "yaml" | "yml" | "json" | "py" | "sh"
        ) || name.starts_with(".cargo/")
            || name.starts_with(".config/")
        {
            sources.insert(name, hash(&fs::read(checked(root, name)?)?));
        }
    }
    Ok(serde_json::to_value(sources)?)
}
