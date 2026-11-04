//! The integration tests of rig-derive, one module per suite, linked into one
//! binary. The trybuild cases and the dependency-rename fixtures resolve their
//! paths from the crate root.

mod calculator;
mod context_value;
mod custom_name;
mod dependency_rename;
mod embed_trait_dispatch;
mod embed_tuple;
mod portable_tool;
mod required_defaults;
mod schemars_schema;
mod serve;
mod tool_binding_hygiene;
mod tool_context;
mod typed_error;
mod ui;
mod visibility;
