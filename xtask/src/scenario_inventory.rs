//! Source evidence reconciled with an actual nextest test listing.
//!
//! This command discovers candidates, not proven coverage. Source-only tests,
//! macro-expanded tests without a direct source match, and helper-mediated
//! cassette calls remain explicit audit obligations in the output.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use quote::ToTokens;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use syn::{
    Expr, Item, Lit, Meta,
    spanned::Spanned,
    visit::{self, Visit},
};

mod generated;

#[cfg(test)]
mod tests;

pub(crate) fn run(
    root: &Path,
    listing: &Path,
    registration_map: Option<&Path>,
) -> Result<(), String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let root = root.as_path();
    let registrations = generated::load(root, registration_map)?;
    let listing_text = read(listing)?;
    let list: Value = serde_json::from_str(&listing_text).map_err(|e| e.to_string())?;
    let suites = list
        .get("rust-suites")
        .and_then(Value::as_object)
        .ok_or("expected nextest JSON rust-suites object")?;
    if suites.is_empty() {
        return Err("empty compiled suite list".into());
    }
    let mut tests = Vec::new();
    let mut unresolved = Vec::new();
    let mut macros = Vec::new();
    let mut missing_suites = Vec::new();
    let mut helpers = Vec::new();
    let mut imports = Vec::new();
    let mut binaries = Vec::new();
    for entry in std::fs::read_dir(root.join("tests/providers")).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir()
            && let Some(name) = path.file_name().and_then(|s| s.to_str())
            && root.join("tests").join(format!("{name}.rs")).is_file()
        {
            binaries.push(name.to_owned());
        }
    }
    binaries.sort();
    for suite in registrations
        .get("suites")
        .and_then(Value::as_array)
        .ok_or("missing generated suites")?
    {
        let binary = generated::string(suite, "binary")?;
        if !binaries.iter().any(|b| b == binary) {
            return Err(format!(
                "generated suite targets undiscovered binary {binary}"
            ));
        }
    }
    for binary in &binaries {
        let suite = suites
            .values()
            .find(|s| s["package-name"] == "rig" && s["binary-name"] == *binary);
        if suite.is_none() {
            missing_suites.push(binary.clone());
        }
        let source = root.join("tests").join(format!("{binary}.rs"));
        let mut found = BTreeMap::new();
        let mut walker = Walker {
            root,
            tests: &mut found,
            macros: &mut macros,
            binary,
            source_hash: None,
            helpers: &mut helpers,
            imports: &mut imports,
        };
        walker.file(&source, &root.join("tests"), "", &[])?;
        generated::apply(binary, &macros, &mut found, &registrations)?;
        let empty = serde_json::Map::new();
        let compiled = match suite {
            Some(suite) => suite["testcases"].as_object().ok_or("missing testcases")?,
            None => &empty,
        };
        for (name, mut evidence) in found {
            let fields = evidence
                .as_object_mut()
                .ok_or("source evidence is not an object")?;
            fields.insert("binary".into(), json!(binary));
            fields.insert("test".into(), json!(name));
            fields.insert("compiled".into(), json!(compiled.contains_key(&name)));
            fields.insert(
                "filter_match".into(),
                compiled
                    .get(&name)
                    .map(|v| v["filter-match"].clone())
                    .unwrap_or(Value::Null),
            );
            fields.insert(
                "suite_status".into(),
                suite.map(|v| v["status"].clone()).unwrap_or(Value::Null),
            );
            fields.insert("execution_result".into(), Value::Null);
            fields.insert(
                "ignored".into(),
                compiled
                    .get(&name)
                    .map(|v| v["ignored"].clone())
                    .unwrap_or(Value::Null),
            );
            fields
                .entry("classification")
                .or_insert_with(|| json!("unclassified"));
            fields.insert("status".into(), json!("unported"));
            tests.push(evidence);
        }
        for name in compiled.keys() {
            if !tests
                .iter()
                .any(|v| v["binary"] == *binary && v["test"] == *name)
            {
                unresolved.push(json!({"binary": binary, "test": name,
                    "reason": "compiled test has unresolved source registration; inspect macros and conditional test attributes"}));
            }
        }
    }
    if tests.is_empty() {
        return Err("no provider source tests discovered".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": 3,
            "generated_registration_map": registrations,
            "listing_sha256": format!("{:x}", Sha256::digest(listing_text.as_bytes())),
            "source_revision": git_output(root, &["rev-parse", "HEAD"]),
            "source_status": git_output(root, &["status", "--porcelain", "--untracked-files=all"]),
            "evidence_kind": "discovery_not_coverage",
            "scope": "root rig provider test targets; excludes companion crates and supplemental rig-agent unit tests",
            "configuration": "caller must record the exact nextest command alongside this report",
            "tests": tests,
            "unresolved_compiled_tests": unresolved,
            "unlisted_provider_targets": missing_suites,
            "source_macros": macros,
            "source_helpers": helpers,
            "source_imports": imports,
            "helper_resolution": "unresolved; use scoped definitions, imports, and call arguments to audit transitive obligations",
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

struct Walker<'a> {
    root: &'a Path,
    tests: &'a mut BTreeMap<String, Value>,
    macros: &'a mut Vec<Value>,
    binary: &'a str,
    source_hash: Option<String>,
    helpers: &'a mut Vec<Value>,
    imports: &'a mut Vec<Value>,
}

impl Walker<'_> {
    fn file(
        &mut self,
        source: &Path,
        directory: &Path,
        module: &str,
        gates: &[String],
    ) -> Result<(), String> {
        let canonical = source
            .canonicalize()
            .map_err(|e| format!("{}: {e}", source.display()))?;
        let text = read(&canonical)?;
        let file = syn::parse_file(&text).map_err(|e| format!("{}: {e}", canonical.display()))?;
        let mut gates = gates.to_vec();
        gates.extend(attributes(&file.attrs));
        let previous = self
            .source_hash
            .replace(format!("{:x}", Sha256::digest(text.as_bytes())));
        let result = self.items(
            &file.items,
            &canonical,
            directory,
            canonical.parent().ok_or("source without parent")?,
            module,
            &gates,
        );
        self.source_hash = previous;
        result
    }

    fn items(
        &mut self,
        items: &[Item],
        source: &Path,
        directory: &Path,
        path_directory: &Path,
        module: &str,
        gates: &[String],
    ) -> Result<(), String> {
        for item in items {
            match item {
                Item::Mod(child) => {
                    let mut child_gates = gates.to_vec();
                    child_gates.extend(attributes(&child.attrs));
                    let name = qualify(module, &child.ident.to_string());
                    if let Some((_, items)) = &child.content {
                        let child_directory = match explicit_module_path(child)? {
                            Some(path) => path_directory.join(path),
                            None => directory.join(child.ident.to_string()),
                        };
                        self.items(
                            items,
                            source,
                            &child_directory,
                            &child_directory,
                            &name,
                            &child_gates,
                        )?;
                    } else {
                        let path = module_file(source, directory, path_directory, child)?;
                        let dir = if path.file_name().is_some_and(|n| n == "mod.rs") {
                            path.parent().ok_or("module without parent")?.to_path_buf()
                        } else {
                            path.with_extension("")
                        };
                        self.file(&path, &dir, &name, &child_gates)?;
                    }
                }
                Item::Fn(function) => {
                    let mut details = Calls::default();
                    details.visit_block(&function.block);
                    let mut conditions = gates.to_vec();
                    conditions.extend(attributes(&function.attrs));
                    let name = qualify(module, &function.sig.ident.to_string());
                    let evidence = json!({
                        "source": source.strip_prefix(self.root).map_err(|e| e.to_string())?.to_string_lossy(),
                        "source_sha256": self.source_hash,
                        "module": module,
                        "binary": self.binary,
                        "function": name,
                        "parameters": function.sig.inputs.to_token_stream().to_string(),
                        "body": function.block.to_token_stream().to_string(),
                        "line": function.sig.ident.span().start().line,
                        "attributes": conditions,
                        "calls": details.calls,
                        "method_calls": details.methods,
                        "assertions": details.assertions,
                    });
                    if !is_test(&function.attrs) {
                        self.helpers.push(evidence);
                        continue;
                    }
                    // Same function name can exist under mutually exclusive
                    // cfg branches. Surface the ambiguity rather than overwrite.
                    if self.tests.insert(name.clone(), evidence).is_some() {
                        return Err(format!(
                            "duplicate source test {name}; resolve configuration-specific definitions"
                        ));
                    }
                }
                Item::Use(import) => {
                    let mut conditions = gates.to_vec();
                    conditions.extend(attributes(&import.attrs));
                    self.imports.push(json!({
                        "source": source.strip_prefix(self.root).map_err(|e| e.to_string())?.to_string_lossy(),
                        "source_sha256": self.source_hash,
                        "binary": self.binary, "module": module,
                        "line": import.span().start().line,
                        "attributes": conditions,
                        "import": import.to_token_stream().to_string(),
                    }));
                }
                Item::Macro(invocation) => {
                    let mut conditions = gates.to_vec();
                    conditions.extend(attributes(&invocation.attrs));
                    self.macros.push(json!({
                    "source": source.strip_prefix(self.root).map_err(|e| e.to_string())?.to_string_lossy(),
                    "module": module, "line": invocation.span().start().line,
                    "macro": invocation.mac.path.to_token_stream().to_string(),
                    "tokens": invocation.mac.tokens.to_string(),
                    "definition_name": invocation.ident.as_ref().map(ToString::to_string),
                    "source_sha256": self.source_hash,
                    "binary": self.binary,
                    "attributes": conditions,
                }));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn qualify(module: &str, name: &str) -> String {
    if module.is_empty() {
        name.into()
    } else {
        format!("{module}::{name}")
    }
}

fn attributes(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| {
            a.path().is_ident("cfg") || a.path().is_ident("cfg_attr") || a.path().is_ident("ignore")
        })
        .map(|a| a.to_token_stream().to_string())
        .collect()
}

fn is_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| test_meta(&a.meta))
}

fn test_meta(meta: &Meta) -> bool {
    if meta.path().is_ident("cfg_attr") {
        if let Meta::List(list) = meta {
            return list
                .parse_args_with(
                    syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                )
                .is_ok_and(|args| args.iter().skip(1).any(test_meta));
        }
        return false;
    }
    meta.path()
        .segments
        .last()
        .is_some_and(|s| s.ident == "test" || s.ident == "rstest" || s.ident == "test_case")
}

fn explicit_module_path(child: &syn::ItemMod) -> Result<Option<String>, String> {
    for attribute in &child.attrs {
        if attribute.path().is_ident("path") {
            if let Meta::NameValue(value) = &attribute.meta
                && let Expr::Lit(value) = &value.value
                && let Lit::Str(path) = &value.lit
            {
                return Ok(Some(path.value()));
            }
            return Err("nonliteral module path".into());
        }
    }
    Ok(None)
}

fn module_file(
    source: &Path,
    directory: &Path,
    path_directory: &Path,
    child: &syn::ItemMod,
) -> Result<PathBuf, String> {
    if let Some(path) = explicit_module_path(child)? {
        return Ok(path_directory.join(path));
    }
    let direct = directory.join(format!("{}.rs", child.ident));
    let nested = directory.join(child.ident.to_string()).join("mod.rs");
    match (direct.is_file(), nested.is_file()) {
        (true, false) => Ok(direct),
        (false, true) => Ok(nested),
        _ => Err(format!(
            "ambiguous or missing module {} from {}",
            child.ident,
            source.display()
        )),
    }
}

#[derive(Default)]
struct Calls {
    calls: Vec<Value>,
    methods: Vec<Value>,
    assertions: Vec<Value>,
}

impl<'ast> Visit<'ast> for Calls {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.methods
            .push(json!({"method": call.method.to_string(), "line": call.span().start().line}));
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let Expr::Path(path) = &*call.func {
            self.calls
                .push(json!({"function": path.path.to_token_stream().to_string(),
                "line": call.span().start().line,
                "first_argument": call.args.first().map(|v| v.to_token_stream().to_string()),
                "arguments": call.args.iter().map(|v| v.to_token_stream().to_string()).collect::<Vec<_>>()}));
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident.to_string().starts_with("assert") || s.ident == "ensure")
        {
            self.assertions.push(json!({"line": mac.span().start().line,
                "assertion": mac.to_token_stream().to_string()}));
        }
        visit::visit_macro(self, mac);
    }
}
