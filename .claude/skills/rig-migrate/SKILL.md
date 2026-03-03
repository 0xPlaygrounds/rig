---
name: rig-migrate
description: >
  Help migrate Rig code between versions. Use when upgrading rig-core or
  companion crates, resolving breaking changes, or updating deprecated APIs.
  Detects the current version from Cargo.toml automatically.
argument-hint: "[target-version]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Edit
  - Write
  - Bash
---

# Rig Migration Assistant

Current project Rig version (auto-detected):
```
!`grep -E '^rig-core|^rig ' Cargo.toml 2>/dev/null || grep -rE 'rig-core\s*=' Cargo.toml */Cargo.toml 2>/dev/null | head -5 || echo "rig-core version not found in Cargo.toml"`
```

Latest Rig release:
```
!`cargo search rig-core --limit 1 2>/dev/null || echo "Could not fetch latest version. Check https://crates.io/crates/rig-core"`
```

## Migration Workflow

1. **Detect**: Compare current version against target version.
2. **Audit**: Search for deprecated patterns and breaking API usages.
3. **Plan**: List all files and changes required.
4. **Migrate**: Apply changes systematically.
5. **Validate**: Run `cargo fmt`, `cargo clippy --all-targets --all-features`, `cargo test`.

## Common Migration Patterns

### Send/Sync to WasmCompat (introduced in 0.5+)

All trait bounds must use WASM-compatible variants:

```rust
// Before
pub trait MyTrait: Send + Sync {
    fn method(&self) -> impl Future<Output = ()> + Send;
}

// After
use rig::{WasmCompatSend, WasmCompatSync};

pub trait MyTrait: WasmCompatSend + WasmCompatSync {
    fn method(&self) -> impl Future<Output = ()> + WasmCompatSend;
}
```

**Search pattern**: `grep -rn ': Send\b\|+ Send\b\|: Sync\b\|+ Sync\b' --include='*.rs'`

### String Error Types to Proper Enums

```rust
// Before
fn process() -> Result<(), String> { ... }

// After
#[derive(Debug, thiserror::Error)]
enum ProcessError {
    #[error("Parse failed: {0}")]
    Parse(#[from] serde_json::Error),
}

fn process() -> Result<(), ProcessError> { ... }
```

**Search pattern**: `grep -rn 'Result<.*,\s*String>' --include='*.rs'` (results should be manually verified to avoid false positives)

### Provider API Updates

When providers update their APIs, Rig's type definitions change. Check the CHANGELOG for specific field additions/removals.

**Typical changes**:
- New fields added to request/response structs
- Model constant renames (e.g., `GPT_4` -> `GPT_4O`)
- New capability declarations

### CompletionRequest Model Override (new)

`CompletionRequest` now has an optional `model` field:

```rust
// When constructing CompletionRequest manually, include the field:
let request = CompletionRequest {
    model: None,  // or Some("model-override".to_string())
    preamble: None,
    chat_history: OneOrMany::one("Hello".into()),
    // ... rest of fields
};
```

## Migration Checklist

Use this checklist when migrating:

- [ ] Update `rig-core` version in all `Cargo.toml` files
- [ ] Update companion crate versions (`rig-mongodb`, `rig-lancedb`, etc.)
- [ ] Search for deprecated API patterns
- [ ] Replace `Send`/`Sync` with `WasmCompatSend`/`WasmCompatSync`
- [ ] Replace `String` error types with proper error enums
- [ ] Remove `.unwrap()` / `.expect()` on fallible operations
- [ ] Update model constants if renamed
- [ ] Run `cargo fmt && cargo clippy --all-targets --all-features && cargo test`
- [ ] Verify examples still compile
