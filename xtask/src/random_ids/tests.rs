use super::*;

#[test]
fn guard_detects_generator_calls_imports_and_function_references() {
    let root = std::env::temp_dir().join(format!("xtask-generator-guard-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("probe.rs"), "use rig_core::id::generate;\nlet a = rig_core::id::generate();\nlet b = value.unwrap_or_else(rig_core::id::generate);\n").unwrap();
    let found = generator_offenders(&root, &root);
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(found.unwrap().len(), 3);
}
