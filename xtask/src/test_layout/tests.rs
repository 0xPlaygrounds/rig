use super::*;

fn gated(attrs: &str) -> bool {
    let file: syn::File =
        syn::parse_str(&format!("{attrs}\nmod tests {{}}")).expect("test source parses");
    let Some(Item::Mod(module)) = file.items.first() else {
        panic!("expected a module item");
    };
    is_test_gated(module)
}

#[test]
fn test_gate_recognizes_only_test_only_predicates() {
    assert!(gated("#[cfg(test)]"));
    assert!(gated("#[allow(dead_code)]\n#[cfg(test)]"));
    assert!(gated("#[cfg(all(test, not(target_family = \"wasm\")))]"));
    assert!(gated(
        "#[cfg(all(test, not(all(target_arch = \"wasm32\", target_os = \"unknown\"))))]"
    ));
    assert!(gated("#[cfg(all(feature = \"pdf\", test))]"));
    assert!(gated("#[cfg(all(any(test, test), feature = \"x\"))]"));
    assert!(gated("#[cfg(any(test))]"));

    assert!(!gated("#[cfg(any(test, debug_assertions))]"));
    assert!(!gated("#[cfg(not(test))]"));
    assert!(!gated("#[cfg(all(not(test), feature = \"x\"))]"));
    assert!(!gated("#[cfg(feature = \"test\")]"));
    assert!(!gated("#[cfg_attr(test, derive(Debug))]"));
    assert!(!gated("#[cfg(target_family = \"wasm\")]"));
    assert!(!gated(""));
}

#[test]
fn inline_bodies_are_reported_and_declarations_are_not() {
    let source = r#"
        pub fn shipped() {}

        #[cfg(test)]
        mod declared;

        #[cfg(test)]
        mod inline_one { use super::*; }

        mod outer {
            #[cfg(all(test, feature = "x"))]
            mod inline_nested {}
            #[cfg(test)]
            mod nested_declared;
        }

        #[cfg(feature = "x")]
        mod feature_only { #[test] fn not_gated_by_cfg_test() {} }
    "#;
    let file = syn::parse_file(source).expect("parses");
    let mut offenders = Vec::new();
    collect_inline_test_modules(Path::new("x.rs"), &file.items, &mut offenders);
    assert_eq!(
        offenders,
        vec![
            "x.rs:8: `mod inline_one` has an inline body".to_owned(),
            "x.rs:12: `mod inline_nested` has an inline body".to_owned(),
        ]
    );
}

#[test]
fn child_candidates_follow_rustc_resolution() {
    let mod_rs = child_candidates(Path::new("src/foo/mod.rs"), "tests");
    assert_eq!(
        mod_rs,
        vec![
            PathBuf::from("src/foo/tests.rs"),
            PathBuf::from("src/foo/tests/mod.rs")
        ]
    );
    let lib_rs = child_candidates(Path::new("src/lib.rs"), "tests");
    assert_eq!(lib_rs.first(), Some(&PathBuf::from("src/tests.rs")));
    let plain = child_candidates(Path::new("src/foo.rs"), "tests");
    assert_eq!(plain.first(), Some(&PathBuf::from("src/foo/tests.rs")));
}

/// Files declared under a gate are test-only, transitively through their own
/// declarations, and an inline test module inside one of them is not an
/// offender.
#[test]
fn test_only_files_are_transitive_and_exempt() {
    let root = std::env::temp_dir().join(format!(
        "rig-xtask-test-layout-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    let src = root.join("src");
    std::fs::create_dir_all(src.join("foo").join("tests")).expect("temp tree");
    let write = |rel: &str, body: &str| {
        std::fs::write(src.join(rel), body).expect("write temp file");
    };
    write("lib.rs", "mod foo;\n");
    write("foo.rs", "pub fn f() {}\n#[cfg(test)]\nmod tests;\n");
    write(
        "foo/tests.rs",
        "mod support;\n#[cfg(test)]\nmod inner { }\n",
    );
    write("foo/tests/support.rs", "");

    let parsed: Vec<(PathBuf, syn::File)> =
        ["lib.rs", "foo.rs", "foo/tests.rs", "foo/tests/support.rs"]
            .iter()
            .map(|rel| {
                let path = src.join(rel);
                let text = std::fs::read_to_string(&path).expect("read");
                (path, syn::parse_file(&text).expect("parse"))
            })
            .collect();

    let test_only = test_only_files(&parsed);
    assert!(test_only.contains(&src.join("foo/tests.rs")));
    assert!(test_only.contains(&src.join("foo/tests/support.rs")));
    assert!(!test_only.contains(&src.join("foo.rs")));
    assert!(!test_only.contains(&src.join("lib.rs")));

    let mut offenders = Vec::new();
    for (path, file) in &parsed {
        if !test_only.contains(path) {
            collect_inline_test_modules(path, &file.items, &mut offenders);
        }
    }
    assert!(offenders.is_empty(), "{offenders:?}");

    std::fs::remove_dir_all(&root).expect("cleanup");
}
