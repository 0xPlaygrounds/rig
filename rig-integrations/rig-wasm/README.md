# rig-wasm: Rig, but it's WASM!
WASM bindings to Rig.

To install, run the following command:

```bash
npm i rig-wasm
```

Examples can be found in the `examples` folder.

Please note that this package is extremely experimental at the moment and as such, you may find that there are DX papercuts here and there differing in severity. Opening a ticket for any issues you find would be hugely helpful.

Like the original `rig` crates, there will also likely be breaking changes between minor versions. Migration paths will be provided between each version if required.

## Missing for feature parity
- Pipelines
- Custom completion models/agents
- Azure

## Implementation checklist
Providers:
  - [x] Nearly every model provider from the original `rig-core` package
  - [ ] Azure

Vector store integrations:
  - [x] Qdrant
  - [x] In-memory vector store (for testing/small datasets)
  - [ ] Everyone else

Agents:
  - [x] Prompting
  - [x] Tool usage (and related TS definitions)
  - [x] Multi-turn
  - [x] Dynamic context
  - [x] Documents
  - [x] Options builder (for idiomatic DX)

Embedding:
  - [x] Embed
  - [ ] Embeddings builder
  - [x] JS side plugin adapter

## Folder architecture
- `examples`: A list of examples (entirely in TS) that use `rig-wasm`.
- `pkg`: The JavaScript package.
  - `src`: Some handwritten code. Primarily used for re-exporting and each file also serves as a module entrypoint for module hygiene.
  - `src/providers`: Generally imports from the generated `rig_wasm` module. Split into modules to avoid issues with everything being in one module and to potentially allow for better code splitting/tree shaking.
  - `src/vector_stores`: Hand-written implementations for vector stores (because the original Rust vector stores don't compile to WASM).
  - `src/*`: Generally, interfaces and utility functions that you may find quite useful.
- `src`: The Rust code.
- `build.rs`: A build script. Primarily used to regenerate all providers based on the contents of `rig-core/src/providers` as their `wasm-bindgen` counterparts.

## Development
- Use `just build-wasm` (which will activate the relevant command from the core justfile at the top level of the project).
- Go into `rig-wasm/pkg` and use `npm run build` which will run the Rollup pipeline as well as copying some WASM files over.
- Try some of the examples! Or do some development work.

## Current caveats
- No pipelines (it'll be at JS speed anyway... so we can probably impl this later)
- No in-memory vector store (it requires some duck typing and general type trickery to produce values that actually satisfy the criteria)
