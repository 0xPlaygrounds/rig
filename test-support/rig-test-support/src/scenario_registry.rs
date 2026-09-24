//! Discover cassette references from direct calls and supported matrix rows.

use crate::matrix_registry;
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprLit, ItemFn, Lit};

/// Malformed Rust or a scenario declaration that cannot be checked statically.
#[derive(Debug, thiserror::Error)]
pub enum ScenarioError {
    /// The source is not valid Rust syntax.
    #[error(transparent)]
    Parse(#[from] syn::Error),
    /// A wrapper or supported matrix has an invalid scenario declaration.
    #[error("{0}")]
    Invalid(String),
}

/// Collect recorded scenarios, excluding ignored functions and matrix rows.
/// Direct calls retain support for literal `CassetteSpec` builder expressions.
pub fn cassette_scenarios(
    source: &str,
    wrapper_names: &[&'static str],
) -> Result<Vec<String>, ScenarioError> {
    Ok(cassette_scenario_sites(source, wrapper_names)?
        .into_iter()
        .map(|site| site.scenario)
        .collect())
}

/// One recorded scenario and the call that records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenarioSite {
    /// The scenario literal.
    pub scenario: String,
    /// The cassette wrapper the call goes through.
    pub wrapper: String,
    /// Account failures the spec itself declares
    /// (`.expects_account_failure(AccountFailure::Auth)` gives `"Auth"`).
    pub declared: Vec<String>,
}

/// Like [`cassette_scenarios`], keeping each scenario's wrapper and the
/// account failures its spec declares.
pub fn cassette_scenario_sites(
    source: &str,
    wrapper_names: &[&'static str],
) -> Result<Vec<ScenarioSite>, ScenarioError> {
    let syntax = syn::parse_file(source)?;
    let mut visitor = CassetteScenarioVisitor {
        wrapper_names,
        scenarios: Vec::new(),
        failures: Vec::new(),
    };
    visitor.visit_file(&syntax);
    if visitor.failures.is_empty() {
        Ok(visitor.scenarios)
    } else {
        Err(ScenarioError::Invalid(visitor.failures.join("\n")))
    }
}

/// Every free function and impl method in `source`, in source order, with
/// the account failures it declares on its cassette session: a call to
/// `expect_account_failure(AccountFailure::X)` declares `X`, and a call to
/// `bogus_api_key()` declares `Auth`. A function that declares nothing is
/// listed with none, so a caller can tell a same-named helper that declares
/// nothing from one that is absent.
pub fn declaring_functions(source: &str) -> Result<Vec<(String, Vec<String>)>, ScenarioError> {
    let syntax = syn::parse_file(source)?;
    let mut visitor = DeclaringVisitor::default();
    visitor.visit_file(&syntax);
    Ok(visitor.functions)
}

#[derive(Default)]
struct DeclaringVisitor {
    functions: Vec<(String, Vec<String>)>,
    current: Option<(String, Vec<String>)>,
}

impl DeclaringVisitor {
    fn enter(&mut self, name: &syn::Ident) -> Option<(String, Vec<String>)> {
        self.current.replace((name.to_string(), Vec::new()))
    }

    fn leave(&mut self, outer: Option<(String, Vec<String>)>) {
        if let Some(function) = std::mem::replace(&mut self.current, outer) {
            self.functions.push(function);
        }
    }
}

impl<'ast> Visit<'ast> for DeclaringVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let outer = self.enter(&node.sig.ident);
        visit::visit_item_fn(self, node);
        self.leave(outer);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let outer = self.enter(&node.sig.ident);
        visit::visit_impl_item_fn(self, node);
        self.leave(outer);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let declared = match node.method.to_string().as_str() {
            "expect_account_failure" => node.args.first().and_then(last_path_segment),
            "bogus_api_key" => Some("Auth".to_owned()),
            _ => None,
        };
        if let (Some(declared), Some((_, list))) = (declared, self.current.as_mut())
            && !list.contains(&declared)
        {
            list.push(declared);
        }
        visit::visit_expr_method_call(self, node);
    }
}

/// The account failures a call from `caller` to a helper declares, given
/// every definition of that helper name as (file, declared failures): the
/// definitions in the calling file when there are any, else all of them,
/// provided they agree. Same-named helpers are never merged. Errors name the
/// files of the disagreeing definitions: the call cannot be resolved
/// without a module path.
pub fn wrapper_declarations<'a>(
    definitions: &'a [(std::path::PathBuf, std::collections::BTreeSet<String>)],
    caller: &std::path::Path,
) -> Result<std::collections::BTreeSet<String>, Vec<&'a std::path::Path>> {
    let local: Vec<_> = definitions
        .iter()
        .filter(|(file, _)| file == caller)
        .collect();
    let candidates: Vec<_> = if local.is_empty() {
        definitions.iter().collect()
    } else {
        local
    };
    let mut distinct = candidates.iter().map(|(_, declared)| declared);
    match distinct.next() {
        None => Ok(std::collections::BTreeSet::new()),
        Some(first) if distinct.all(|declared| declared == first) => Ok(first.clone()),
        Some(_) => Err(candidates.iter().map(|(file, _)| file.as_path()).collect()),
    }
}

fn last_path_segment(expr: &Expr) -> Option<String> {
    let Expr::Path(path) = expr else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

/// Account failures a `CassetteSpec` builder chain declares.
fn spec_declarations(expr: &Expr) -> Vec<String> {
    match expr {
        Expr::MethodCall(call) => {
            let mut declared = spec_declarations(&call.receiver);
            if call.method == "expects_account_failure"
                && let Some(kind) = call.args.first().and_then(last_path_segment)
            {
                declared.push(kind);
            }
            declared
        }
        Expr::Paren(paren) => spec_declarations(&paren.expr),
        _ => Vec::new(),
    }
}

struct CassetteScenarioVisitor<'a> {
    wrapper_names: &'a [&'static str],
    scenarios: Vec<ScenarioSite>,
    failures: Vec<String>,
}

impl<'ast, 'a> Visit<'ast> for CassetteScenarioVisitor<'a> {
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "golden_matrix")
        {
            match syn::parse2::<matrix_registry::GoldenMatrix>(node.tokens.clone()) {
                Ok(matrix) => {
                    let wrapper = matrix
                        .wrapper
                        .segments
                        .last()
                        .expect("wrapper path")
                        .ident
                        .to_string();
                    if self.wrapper_names.contains(&wrapper.as_str()) {
                        self.scenarios.extend(
                            matrix
                                .rows
                                .into_iter()
                                .filter(|row| !row.ignored)
                                .map(|row| ScenarioSite {
                                    scenario: row.scenario.value(),
                                    wrapper: wrapper.clone(),
                                    declared: Vec::new(),
                                }),
                        );
                    }
                }
                Err(error) => self.failures.push(error.to_string()),
            }
        }
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "native_matrix")
        {
            match syn::parse2::<matrix_registry::NativeMatrix>(node.tokens.clone()) {
                Ok(matrix) => {
                    let wrapper = matrix
                        .wrapper
                        .segments
                        .last()
                        .expect("wrapper path")
                        .ident
                        .to_string();
                    if self.wrapper_names.contains(&wrapper.as_str()) {
                        self.scenarios.extend(
                            matrix
                                .rows
                                .into_iter()
                                .filter(|row| !row.ignored)
                                .map(|row| ScenarioSite {
                                    scenario: row.scenario.value(),
                                    wrapper: wrapper.clone(),
                                    declared: Vec::new(),
                                }),
                        );
                    }
                }
                Err(error) => self.failures.push(error.to_string()),
            }
        }
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "resume_matrix")
        {
            match syn::parse2::<matrix_registry::ResumeMatrix>(node.tokens.clone()) {
                Ok(matrix) => {
                    let wrapper = matrix
                        .wrapper
                        .segments
                        .last()
                        .expect("wrapper path")
                        .ident
                        .to_string();
                    if self.wrapper_names.contains(&wrapper.as_str()) {
                        self.scenarios.extend(
                            matrix
                                .rows
                                .into_iter()
                                .filter(|row| !row.ignored)
                                .map(|row| ScenarioSite {
                                    scenario: row.scenario.value(),
                                    wrapper: wrapper.clone(),
                                    declared: Vec::new(),
                                }),
                        );
                    }
                }
                Err(error) => self.failures.push(error.to_string()),
            }
        }
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "case_matrix")
        {
            match syn::parse2::<matrix_registry::CaseMatrix>(node.tokens.clone()) {
                Ok(matrix) => {
                    let Some(wrapper) = matrix.wrapper else {
                        return;
                    };
                    let wrapper = wrapper
                        .segments
                        .last()
                        .expect("wrapper path")
                        .ident
                        .to_string();
                    if self.wrapper_names.contains(&wrapper.as_str()) {
                        self.scenarios.extend(
                            matrix.rows.into_iter().filter(|(_, ignored)| !ignored).map(
                                |(scenario, _)| ScenarioSite {
                                    scenario: scenario.value(),
                                    wrapper: wrapper.clone(),
                                    declared: Vec::new(),
                                },
                            ),
                        );
                    }
                }
                Err(error) => self.failures.push(error.to_string()),
            }
        }
        visit::visit_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        // A `#[ignore]`d test documents that its cassette isn't recorded yet
        // (e.g. no provider API key available to record with); don't require
        // a file for scenarios it references.
        if node.attrs.iter().any(|attr| attr.path().is_ident("ignore")) {
            return;
        }

        visit::visit_item_fn(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(wrapper_name) = cassette_wrapper_name(node)
            && self.wrapper_names.contains(&wrapper_name.as_str())
        {
            match node.args.first() {
                Some(expr) => match cassette_scenario_value(expr) {
                    Some(scenario) => self.scenarios.push(ScenarioSite {
                        scenario,
                        wrapper: wrapper_name.clone(),
                        declared: spec_declarations(expr),
                    }),
                    None => self.failures.push(format!(
                        "calls {wrapper_name} without a string-literal cassette scenario"
                    )),
                },
                _ => self.failures.push(format!(
                    "calls {wrapper_name} without a string-literal cassette scenario"
                )),
            }
        }

        visit::visit_expr_call(self, node);
    }
}

fn cassette_scenario_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(scenario),
            ..
        }) => Some(scenario.value()),
        Expr::Call(call) if is_cassette_spec_new(call) => call.args.first().and_then(|expr| {
            let Expr::Lit(ExprLit {
                lit: Lit::Str(scenario),
                ..
            }) = expr
            else {
                return None;
            };

            Some(scenario.value())
        }),
        Expr::MethodCall(method_call) => cassette_scenario_value(&method_call.receiver),
        Expr::Paren(paren) => cassette_scenario_value(&paren.expr),
        _ => None,
    }
}

fn is_cassette_spec_new(call: &ExprCall) -> bool {
    let Expr::Path(path) = call.func.as_ref() else {
        return false;
    };

    let mut segments = path.path.segments.iter().rev();
    matches!(
        (segments.next(), segments.next()),
        (Some(method), Some(receiver))
            if method.ident == "new" && receiver.ident == "CassetteSpec"
    )
}

fn cassette_wrapper_name(node: &ExprCall) -> Option<String> {
    let Expr::Path(path) = node.func.as_ref() else {
        return None;
    };

    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

#[cfg(test)]
mod tests;
