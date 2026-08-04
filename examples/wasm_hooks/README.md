# Browser-wasm hooks

This package is a copy-pasteable hook that keeps a JavaScript `Array` in
worker-local storage. Its source is compiled by the workspace checks and by the
executed wasm test job:

- [`src/lib.rs`](src/lib.rs) contains the complete hook.
- The same file contains three `wasm-bindgen-test` tests for Promise-backed
  async resolution, registration ordering, and terminal short-circuiting.

**Stored configuration is shareable; execution may remain worker-local.** The
`HookEntry` callback captures nothing and therefore remains `Send + Sync`; its
returned future accesses the JavaScript handle through `thread_local!` on the
worker that owns it. Directly capturing the `Array`, putting it in `Rc` or
`RefCell`, or passing it to `HookEntry::with_state` would make retained hook
configuration worker-affine and is intentionally rejected.

For a standalone application, the relevant dependencies are:

```toml
[dependencies]
js-sys = "0.3.95"
rig = { version = "0.41", default-features = false, features = ["agent"] }
wasm-bindgen = "0.2.118"
```

Run the executable wasm coverage from the repository root with:

```sh
wasm-pack test --node examples/wasm_hooks
```

Node is intentional: these tests exercise `wasm32-unknown-unknown`, a real
JavaScript Promise/microtask boundary, worker-local Rust state, and Rig's hook
folds, but no DOM API. A headless browser would add startup cost without
covering another behavior. The example's JavaScript `Array` path is compiled by
the same command.
