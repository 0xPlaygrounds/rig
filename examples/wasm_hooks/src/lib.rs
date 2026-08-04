//! Browser-wasm hook state without making the stored hook worker-affine.
//!
//! **Stored configuration is shareable; execution may remain worker-local.**
//! The [`HookEntry`] below has a zero-capture callback, so the record remains
//! `Send + Sync`. Its invocation future accesses the JavaScript [`Array`] from
//! `thread_local!`, on the worker that owns the handle.

use std::cell::RefCell;

use js_sys::Array;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use wasm_bindgen::prelude::*;

thread_local! {
    /// `Array` is a JavaScript handle and must stay on the worker that created
    /// it. `RefCell` is sound here because each worker owns its own slot.
    static BROWSER_EVENTS: RefCell<Array> = RefCell::new(Array::new());
}

/// Builds a hook that records model-call events in worker-local JavaScript
/// state.
///
/// Add the returned record with `AgentBuilder::add_hook`. Do not move the
/// `Array` into the closure or pass it to `HookEntry::with_state`: retained
/// callbacks and state are `Send + Sync` on every target. Accessing it from
/// inside the invocation future keeps the JavaScript handle worker-local.
pub fn browser_audit_hook() -> HookEntry {
    HookEntry::new("browser-audit", |event| async move {
        if let HookEvent::BeforeModelCall { turn, .. } = event {
            BROWSER_EVENTS.with(|events| {
                events
                    .borrow()
                    .push(&JsValue::from_str(&format!("before-model-call:{turn}")));
            });
        }
        HookDecision::Continue
    })
}

/// Returns the worker-local JavaScript event array for display or inspection.
#[wasm_bindgen]
pub fn browser_events() -> Array {
    BROWSER_EVENTS.with(|events| events.borrow().clone())
}

/// Clears the worker-local JavaScript event array.
#[wasm_bindgen]
pub fn clear_browser_events() {
    BROWSER_EVENTS.with(|events| events.borrow().set_length(0));
}

#[cfg(all(test, target_arch = "wasm32", target_os = "unknown"))]
mod wasm_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use js_sys::Promise;
    use rig::agent::CompletionCallAction;
    use rig::hooks::{HookDecision, HookEntry, Hooks};
    use rig::message::Message;
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;
    use wasm_bindgen_test::wasm_bindgen_test;

    thread_local! {
        static TRACE: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    }

    fn reset_trace() {
        TRACE.with(|trace| trace.borrow_mut().clear());
    }

    fn record(label: &'static str) {
        TRACE.with(|trace| trace.borrow_mut().push(label));
    }

    fn trace() -> Vec<&'static str> {
        TRACE.with(|trace| trace.borrow().clone())
    }

    #[wasm_bindgen_test(async)]
    async fn async_hook_resolves_after_javascript_promise_with_rc_state() {
        let hook = HookEntry::new("async", |_event| {
            let worker_local = Rc::new(String::from("resolved"));
            async move {
                let resolved = JsFuture::from(Promise::resolve(&JsValue::UNDEFINED))
                    .await
                    .is_ok();
                if resolved {
                    HookDecision::CompletionCall(CompletionCallAction::stop(worker_local.as_str()))
                } else {
                    HookDecision::Continue
                }
            }
        });
        let prompt = Message::user("hello");
        let action = Hooks::new()
            .with(hook)
            .dispatch_completion_call(1, &prompt, &[])
            .await;

        assert_eq!(action, CompletionCallAction::stop("resolved"));
    }

    #[wasm_bindgen_test(async)]
    async fn hooks_run_in_registration_order() {
        reset_trace();
        let hooks = Hooks::new()
            .with(HookEntry::sync("first", |_| {
                record("first");
                HookDecision::Continue
            }))
            .with(HookEntry::sync("second", |_| {
                record("second");
                HookDecision::Continue
            }))
            .with(HookEntry::sync("third", |_| {
                record("third");
                HookDecision::Continue
            }));
        let prompt = Message::user("hello");

        let action = hooks.dispatch_completion_call(1, &prompt, &[]).await;

        assert_eq!(action, CompletionCallAction::Continue);
        assert_eq!(trace(), vec!["first", "second", "third"]);
    }

    #[wasm_bindgen_test(async)]
    async fn terminal_hook_short_circuits_later_entries() {
        reset_trace();
        let hooks = Hooks::new()
            .with(HookEntry::sync("continue", |_| {
                record("continue");
                HookDecision::Continue
            }))
            .with(HookEntry::sync("stop", |_| {
                record("stop");
                HookDecision::CompletionCall(CompletionCallAction::stop("halt"))
            }))
            .with(HookEntry::sync("unreachable", |_| {
                record("unreachable");
                HookDecision::Continue
            }));
        let prompt = Message::user("hello");

        let action = hooks.dispatch_completion_call(1, &prompt, &[]).await;

        assert_eq!(action, CompletionCallAction::stop("halt"));
        assert_eq!(trace(), vec!["continue", "stop"]);
    }
}
