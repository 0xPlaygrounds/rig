//! Scoped nested tool dispatch for composite tools and code runtimes.

use std::{collections::BTreeSet, sync::Arc};

use crate::{
    tool::{ToolCallExtensions, ToolExecutionResult, ToolFailure},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{RunContext, ToolCallContext};

/// Limits and allowlist for nested dispatch.
#[derive(Debug, Clone)]
pub struct ScopedExecutionPolicy {
    /// Allowed tool names; `None` allows every registered tool except recursion.
    pub allowlist: Option<BTreeSet<String>>,
    /// Maximum nested call depth.
    pub max_depth: usize,
}

impl Default for ScopedExecutionPolicy {
    fn default() -> Self {
        Self {
            allowlist: None,
            max_depth: 8,
        }
    }
}

pub(crate) trait NestedDispatchDyn: WasmCompatSend + WasmCompatSync {
    fn dispatch<'a>(
        &'a self,
        tool_name: &'a str,
        args: &'a str,
        extensions: &'a ToolCallExtensions,
        call_context: &'a ToolCallContext,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult>;
}

/// Cloneable executor automatically attached to each top-level tool call.
#[derive(Clone)]
pub struct ScopedToolExecutor {
    dispatcher: Arc<dyn NestedDispatchDyn>,
    inherited_extensions: ToolCallExtensions,
    parent: ToolCallContext,
    active_tools: Arc<Vec<String>>,
    policy: ScopedExecutionPolicy,
}

impl std::fmt::Debug for ScopedToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedToolExecutor")
            .field("parent", &self.parent.internal_call_id)
            .field("depth", &self.parent.depth)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl ScopedToolExecutor {
    pub(crate) fn new(
        dispatcher: Arc<dyn NestedDispatchDyn>,
        inherited_extensions: ToolCallExtensions,
        parent: ToolCallContext,
        active_tool: String,
    ) -> Self {
        Self {
            dispatcher,
            inherited_extensions,
            parent,
            active_tools: Arc::new(vec![active_tool]),
            policy: ScopedExecutionPolicy::default(),
        }
    }

    /// Narrow the tools available to nested calls.
    pub fn with_allowlist(mut self, allowlist: impl IntoIterator<Item = String>) -> Self {
        self.policy.allowlist = Some(allowlist.into_iter().collect());
        self
    }

    /// Set the maximum nested depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.policy.max_depth = max_depth;
        self
    }

    /// Parent run context inherited by nested calls.
    pub fn run_context(&self) -> &RunContext {
        &self.parent.run
    }

    /// Dispatch a nested call with inherited extensions, a fresh internal ID,
    /// parent correlation, cancellation, allowlist, depth, and recursion guards.
    pub async fn call(&self, tool_name: &str, args: &serde_json::Value) -> ToolExecutionResult {
        if let Some(reason) = self.parent.run.control().cancellation_reason() {
            return ToolExecutionResult::failed(reason, ToolFailure::cancelled(reason));
        }
        if self.parent.depth.saturating_add(1) > self.policy.max_depth {
            return ToolExecutionResult::failed(
                "nested tool depth exceeded",
                ToolFailure::permission_denied("nested tool depth exceeded"),
            );
        }
        if self
            .policy
            .allowlist
            .as_ref()
            .is_some_and(|allowlist| !allowlist.contains(tool_name))
        {
            return ToolExecutionResult::failed(
                format!("nested tool `{tool_name}` is not allowed"),
                ToolFailure::permission_denied("nested tool is not in the scoped allowlist"),
            );
        }
        if self.active_tools.iter().any(|active| active == tool_name) {
            return ToolExecutionResult::failed(
                format!("recursive nested call to `{tool_name}` rejected"),
                ToolFailure::invalid_args("recursive nested tool call"),
            );
        }

        let internal_call_id = crate::id::generate();
        let call_context = ToolCallContext {
            run: self.parent.run.clone(),
            internal_call_id,
            provider_call_id: None,
            parent_internal_call_id: Some(self.parent.internal_call_id.clone()),
            depth: self.parent.depth.saturating_add(1),
        };
        let mut extensions = self.inherited_extensions.clone();
        extensions.insert(call_context.clone());
        extensions.insert(call_context.run.clone());
        let mut active_tools = (*self.active_tools).clone();
        active_tools.push(tool_name.to_owned());
        extensions.insert(Self {
            dispatcher: self.dispatcher.clone(),
            inherited_extensions: extensions.clone(),
            parent: call_context.clone(),
            active_tools: Arc::new(active_tools),
            policy: self.policy.clone(),
        });

        self.dispatcher
            .dispatch(
                tool_name,
                &crate::json_utils::value_to_json_string(args),
                &extensions,
                &call_context,
            )
            .await
    }
}
