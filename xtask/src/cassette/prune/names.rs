//! What the source tree names: for every test of a provider target, the
//! fixtures it records or replays, the fixtures it reads otherwise and the
//! effect goldens it names, and every fixture or golden something else
//! names.
//!
//! A test owns a fixture when it passes the scenario literal to a cassette
//! wrapper or a matrix row starts with it, as `cassette owner` reads it. Any
//! other string literal in the test or its row that resolves to a fixture is
//! a read. A literal outside every test of a provider file (a constant, a
//! helper) is read by every test of that file, or, in a file with no test,
//! protects what it names. A literal anywhere else in the workspace (another
//! target, a shared driver, a unit test) protects what it names. The scan
//! over-approximates: a name never read keeps its file, never the reverse.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ItemFn, ItemMod};

use crate::cassette::owner;

/// A test of a provider target: its nextest binary id and its name.
pub(crate) type TestId = (String, String);

/// One effect golden, by its stem.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Golden {
    pub(crate) name: String,
}

impl Golden {
    /// The golden's files, relative to the workspace root.
    pub(crate) fn files(&self) -> Vec<String> {
        vec![format!("{EFFECTS}/{}.effects.json", self.name)]
    }

    /// The golden as the manifest names it.
    pub(crate) fn label(&self) -> String {
        self.name.clone()
    }
}

/// The effect goldens, relative to the workspace root.
pub(crate) const EFFECTS: &str = "crates/rig-cassette/fixtures/effects";

/// What one provider test names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Named {
    /// Fixtures it records or replays.
    pub(crate) owns: BTreeSet<String>,
    /// Fixtures its own body or row names otherwise.
    pub(crate) reads: BTreeSet<String>,
    /// Fixtures its file names outside every test.
    pub(crate) file_reads: BTreeSet<String>,
    pub(crate) goldens: BTreeSet<Golden>,
    /// The source file, relative to the workspace root, and its module path.
    pub(crate) file: String,
    pub(crate) module: String,
}

/// Everything the source tree names.
#[derive(Debug, Default)]
pub(crate) struct Names {
    pub(crate) tests: BTreeMap<TestId, Named>,
    /// Fixtures named outside every provider test, with the first file.
    pub(crate) fixtures: BTreeMap<String, String>,
    /// Goldens named outside every provider test, with the first file.
    pub(crate) goldens: BTreeMap<Golden, String>,
    /// Provider files another target compiles too (`#[path]`): their tests
    /// run there as well, so taking one out takes out both.
    pub(crate) shared_files: BTreeSet<String>,
}

/// The fixtures and goldens that exist, for resolving literals.
#[derive(Debug, Default)]
pub(crate) struct Corpus {
    /// `<provider>/<scenario>.yaml`.
    pub(crate) fixtures: BTreeSet<String>,
    pub(crate) goldens: BTreeSet<Golden>,
    /// Every fixture by its scenario (`<scenario>.yaml` below the provider).
    by_scenario: BTreeMap<String, Vec<String>>,
}

impl Corpus {
    pub(crate) fn new(fixtures: BTreeSet<String>, goldens: BTreeSet<Golden>) -> Self {
        let mut by_scenario: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for fixture in &fixtures {
            if let Some((_, scenario)) = fixture.split_once('/') {
                by_scenario
                    .entry(scenario.to_owned())
                    .or_default()
                    .push(fixture.clone());
            }
        }
        Self {
            fixtures,
            goldens,
            by_scenario,
        }
    }
}

/// The directories whose Rust sources the scan reads.
const SOURCE_DIRS: &[&str] = &["crates", "examples", "src", "test-support", "tests"];

const PROVIDERS: &str = "crates/rig-cassette/tests/providers";

/// Sources that name scenarios only to look their replies up in the reply
/// bank, which keeps a deleted fixture's replies: they protect nothing.
const BANK_READERS: &[&str] = &[
    "crates/rig-cassette/tests/runtime.rs",
    "crates/rig-cassette/tests/runtime/",
];

pub(crate) fn scan(root: &Path, corpus: &Corpus) -> Result<Names, String> {
    let mut names = Names::default();
    let providers = root.join(PROVIDERS);
    let mut modules = BTreeMap::new();
    for entry in std::fs::read_dir(&providers)
        .map_err(|error| format!("{}: {error}", providers.display()))?
    {
        let dir = entry
            .map_err(|error| format!("{}: {error}", providers.display()))?
            .path();
        if let Some(provider) = dir.file_name().and_then(|name| name.to_str())
            && dir.is_dir()
        {
            modules.extend(owner::module_map(&dir.join("mod.rs"), provider));
        }
    }
    for dir in SOURCE_DIRS {
        let dir = root.join(dir);
        if !dir.is_dir() {
            continue;
        }
        for file in crate::support::files_under(&dir, Some("rs"))? {
            let relative = file
                .strip_prefix(root)
                .unwrap_or(&file)
                .to_string_lossy()
                .replace('\\', "/");
            if relative.split('/').any(|part| part == "target") {
                continue;
            }
            let source = std::fs::read_to_string(&file)
                .map_err(|error| format!("{}: {error}", file.display()))?;
            if !source.contains('"') {
                continue;
            }
            if BANK_READERS
                .iter()
                .any(|reader| relative.starts_with(reader))
            {
                names.shared_files.extend(shared_files(&source));
                continue;
            }
            let provider = provider_of(&relative);
            let module = provider.map(|provider| {
                modules
                    .get(&file)
                    .cloned()
                    .unwrap_or_else(|| fallback_module(&providers, &file, provider))
            });
            let found = file_names(&source, module.as_deref())
                .map_err(|error| format!("{relative}: {error}"))?;
            names.add(&relative, provider, module.as_deref(), found, corpus);
        }
    }
    Ok(names)
}

/// The provider whose tests a workspace-relative file holds.
fn provider_of(relative: &str) -> Option<&str> {
    relative
        .strip_prefix(PROVIDERS)?
        .strip_prefix('/')?
        .split_once('/')
        .map(|(provider, _)| provider)
}

fn fallback_module(providers: &Path, file: &Path, provider: &str) -> String {
    let relative: PathBuf = file
        .strip_prefix(providers.join(provider))
        .unwrap_or(file)
        .with_extension("");
    let mut parts = vec![provider.to_owned()];
    parts.extend(
        relative
            .components()
            .map(|part| part.as_os_str().to_string_lossy().into_owned()),
    );
    if parts.last().is_some_and(|last| last == "mod") {
        parts.pop();
    }
    parts.join("::")
}

/// The string literals of one file, by where they sit.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct FileNames {
    /// Per test (its qualified name): the literals it owns and every literal
    /// in it.
    pub(crate) tests: BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)>,
    /// Literals outside every test.
    pub(crate) outside: BTreeSet<String>,
}

/// Read one file's literals. `module` is the file's module path inside a
/// provider target, `None` for a file outside the provider tests, whose
/// literals are all outside.
pub(crate) fn file_names(source: &str, module: Option<&str>) -> Result<FileNames, String> {
    let file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut finder = Finder {
        modules: module.map(|module| vec![module.to_owned()]),
        test: None,
        found: FileNames::default(),
    };
    finder.visit_file(&file);
    Ok(finder.found)
}

struct Finder {
    /// `None` outside the provider tests: nothing there is a test.
    modules: Option<Vec<String>>,
    /// The enclosing test's qualified name.
    test: Option<String>,
    found: FileNames,
}

impl Finder {
    fn qualified(&self, name: &str) -> Option<String> {
        self.modules
            .as_ref()
            .map(|modules| format!("{}::{name}", modules.join("::")))
    }

    fn literal(&mut self, value: String, owned: bool) {
        match &self.test {
            Some(test) => {
                let entry = self.found.tests.entry(test.clone()).or_default();
                if owned {
                    entry.0.insert(value.clone());
                }
                entry.1.insert(value);
            }
            None => {
                self.found.outside.insert(value);
            }
        }
    }

    /// Item-level macro rows: `name: (...)` or `name: … => "golden"`, each
    /// a test of its own.
    fn rows(&mut self, tokens: &TokenStream) {
        let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
        for segment in trees.split(|tree| matches!(tree, TokenTree::Punct(p) if p.as_char() == ';'))
        {
            let mut rest = segment;
            while let [TokenTree::Punct(hash), TokenTree::Group(group), tail @ ..] = rest
                && hash.as_char() == '#'
                && group.delimiter() == Delimiter::Bracket
            {
                rest = tail;
            }
            let row = match rest {
                [TokenTree::Ident(name), TokenTree::Punct(colon), body @ ..]
                    if colon.as_char() == ':'
                        && !matches!(body.first(), Some(TokenTree::Punct(p)) if p.as_char() == ':')
                        && is_row_body(body) =>
                {
                    self.qualified(&name.to_string()).map(|test| (test, body))
                }
                _ => None,
            };
            match row {
                Some((test, body)) => {
                    let owned = match body.first() {
                        Some(TokenTree::Group(group))
                            if group.delimiter() == Delimiter::Parenthesis =>
                        {
                            group
                                .stream()
                                .into_iter()
                                .next()
                                .and_then(|tree| string(&tree))
                        }
                        _ => None,
                    };
                    self.found.tests.entry(test.clone()).or_default();
                    let previous = self.test.replace(test);
                    if let Some(owned) = owned {
                        self.literal(owned, true);
                    }
                    for value in strings(body.iter().cloned().collect()) {
                        self.literal(value, false);
                    }
                    self.test = previous;
                }
                None => {
                    for value in strings(segment.iter().cloned().collect()) {
                        self.literal(value, false);
                    }
                }
            }
        }
    }
}

/// The value of a string literal token.
fn string(tree: &TokenTree) -> Option<String> {
    let TokenTree::Literal(literal) = tree else {
        return None;
    };
    syn::parse_str::<syn::LitStr>(&literal.to_string())
        .ok()
        .map(|literal| literal.value())
}

/// Every string literal in `tokens`, nested groups included.
fn strings(tokens: TokenStream) -> Vec<String> {
    let mut out = Vec::new();
    for tree in tokens {
        match &tree {
            TokenTree::Group(group) => out.extend(strings(group.stream())),
            other => out.extend(string(other)),
        }
    }
    out
}

fn is_test_fn(node: &ItemFn) -> bool {
    node.attrs.iter().any(|attr| {
        attr.path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    })
}

/// The scenario a wrapper's first argument names, as `cassette owner`
/// reads it: a literal, or the first argument of a call or the receiver of
/// a method chain around one.
fn named_scenario(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(literal),
            ..
        }) => Some(literal.value()),
        Expr::Call(call) => call.args.first().and_then(named_scenario),
        Expr::MethodCall(call) => named_scenario(&call.receiver),
        Expr::Paren(paren) => named_scenario(&paren.expr),
        _ => None,
    }
}

impl<'ast> Visit<'ast> for Finder {
    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        // Doc comments name scenarios in prose; they read nothing.
        if !node.path().is_ident("doc") {
            visit::visit_attribute(self, node);
        }
    }

    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        let inline = node.content.is_some();
        if inline && let Some(modules) = &mut self.modules {
            modules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
        if inline && let Some(modules) = &mut self.modules {
            modules.pop();
        }
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let test = (self.test.is_none() && is_test_fn(node))
            .then(|| self.qualified(&node.sig.ident.to_string()))
            .flatten();
        let entered = test.is_some();
        if let Some(test) = test {
            self.found.tests.entry(test.clone()).or_default();
            self.test = Some(test);
        }
        visit::visit_item_fn(self, node);
        if entered {
            self.test = None;
        }
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        let wrapper = match node.func.as_ref() {
            Expr::Path(path) => path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            _ => None,
        };
        if self.test.is_some()
            && wrapper.is_some_and(|name| name.starts_with("with_") && name.contains("cassette"))
            && let Some(scenario) = node.args.first().and_then(named_scenario)
        {
            self.literal(scenario, true);
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_lit_str(&mut self, node: &'ast syn::LitStr) {
        self.literal(node.value(), false);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if self.test.is_some() || self.modules.is_none() {
            for value in strings(node.tokens.clone()) {
                self.literal(value, false);
            }
        } else {
            self.rows(&node.tokens);
        }
        visit::visit_macro(self, node);
    }
}

impl Names {
    fn add(
        &mut self,
        relative: &str,
        provider: Option<&str>,
        module: Option<&str>,
        found: FileNames,
        corpus: &Corpus,
    ) {
        let Some(provider) = provider else {
            for value in &found.outside {
                if let Some(shared) = shared_file(value) {
                    self.shared_files.insert(shared);
                    continue;
                }
                if let Some(golden) =
                    golden_path(value).filter(|golden| corpus.goldens.contains(golden))
                {
                    self.goldens
                        .entry(golden)
                        .or_insert_with(|| relative.to_owned());
                    continue;
                }
                for fixture in resolve_fixture(value, None, corpus) {
                    self.fixtures
                        .entry(fixture)
                        .or_insert_with(|| relative.to_owned());
                }
                let agent = Golden {
                    name: value.clone(),
                };
                if corpus.goldens.contains(&agent) {
                    let golden = agent;
                    self.goldens
                        .entry(golden)
                        .or_insert_with(|| relative.to_owned());
                }
            }
            return;
        };
        let binary = format!("rig-cassette::{provider}");
        let golden = |value: &str| {
            if let Some(golden) = golden_path(value) {
                return corpus.goldens.contains(&golden).then_some(golden);
            }
            let agent = Golden {
                name: value.to_owned(),
            };
            corpus.goldens.contains(&agent).then_some(agent)
        };
        if found.tests.is_empty() {
            for value in &found.outside {
                for fixture in resolve_fixture(value, Some(provider), corpus) {
                    self.fixtures
                        .entry(fixture)
                        .or_insert_with(|| relative.to_owned());
                }
                if let Some(golden) = golden(value) {
                    self.goldens
                        .entry(golden)
                        .or_insert_with(|| relative.to_owned());
                }
            }
            return;
        }
        for (test, (owned, literals)) in found.tests {
            let named = self.tests.entry((binary.clone(), test)).or_default();
            relative.clone_into(&mut named.file);
            module.unwrap_or_default().clone_into(&mut named.module);
            for value in &owned {
                named
                    .owns
                    .extend(resolve_fixture(value, Some(provider), corpus));
            }
            for value in &literals {
                for fixture in resolve_fixture(value, Some(provider), corpus) {
                    if !named.owns.contains(&fixture) {
                        named.reads.insert(fixture);
                    }
                }
                named.goldens.extend(golden(value));
            }
            for value in &found.outside {
                for fixture in resolve_fixture(value, Some(provider), corpus) {
                    if !named.owns.contains(&fixture) && !named.reads.contains(&fixture) {
                        named.file_reads.insert(fixture);
                    }
                }
                named.goldens.extend(golden(value));
            }
        }
    }
}

/// The provider file a `#[path]` literal outside the provider tests
/// compiles into another target.
fn shared_file(literal: &str) -> Option<String> {
    let (_, tail) = literal.rsplit_once("providers/")?;
    literal
        .ends_with(".rs")
        .then(|| format!("{PROVIDERS}/{tail}"))
}

/// Every provider file a source compiles through `#[path]`.
fn shared_files(source: &str) -> BTreeSet<String> {
    source
        .split("#[path = \"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .filter_map(shared_file)
        .collect()
}

/// The golden a path literal names: a file `<name>.effects.json`.
fn golden_path(literal: &str) -> Option<Golden> {
    let file = literal.rsplit_once('/').map_or(literal, |(_, file)| file);
    let name = file.strip_suffix(".effects.json")?;
    Some(Golden {
        name: name.to_owned(),
    })
}

/// The fixtures a literal names: a path ending in one under
/// `fixtures/cassettes/`, a `<provider>/<scenario>[.yaml]`, or, read in a
/// provider's tests, a `<scenario>[.yaml]` of that provider. Outside the
/// provider tests a bare `<scenario>[.yaml]` names it for every provider.
pub(crate) fn resolve_fixture(
    literal: &str,
    provider: Option<&str>,
    corpus: &Corpus,
) -> Vec<String> {
    if literal.is_empty() || literal.contains(char::is_whitespace) {
        return Vec::new();
    }
    if let Some((_, rest)) = literal.split_once("fixtures/cassettes/") {
        let rest = rest.trim_start_matches('/');
        return [rest.to_owned(), format!("{rest}.yaml")]
            .into_iter()
            .filter(|fixture| corpus.fixtures.contains(fixture))
            .take(1)
            .collect();
    }
    let scenario = literal.strip_suffix(".yaml").unwrap_or(literal);
    let mut found = BTreeSet::new();
    let direct = format!("{scenario}.yaml");
    if corpus.fixtures.contains(&direct) {
        found.insert(direct.clone());
    }
    match provider {
        Some(provider) => {
            let own = format!("{provider}/{direct}");
            if corpus.fixtures.contains(&own) {
                found.insert(own);
            }
        }
        None => {
            if let Some(fixtures) = corpus.by_scenario.get(&direct) {
                found.extend(fixtures.iter().cloned());
            }
        }
    }
    found.into_iter().collect()
}

/// Whether a segment's body after `name:` is a matrix row: a parenthesized
/// row, or a value with `=> "golden"`; a header (`wrapper: a, run: b` or
/// `family: f`) is neither.
pub(crate) fn is_row_body<T: std::borrow::Borrow<TokenTree>>(body: &[T]) -> bool {
    let punct =
        |tree: &T, ch: char| matches!(tree.borrow(), TokenTree::Punct(p) if p.as_char() == ch);
    let commas = body.iter().any(|tree| punct(tree, ','));
    let parenthesized = body.first().is_some_and(|tree| {
        matches!(tree.borrow(), TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis)
    });
    let arrow = body
        .windows(2)
        .any(|pair| matches!(pair, [a, b] if punct(a, '=') && punct(b, '>')));
    !commas && (parenthesized || arrow)
}
