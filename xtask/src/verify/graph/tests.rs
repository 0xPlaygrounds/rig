use super::*;
use serde_json::json;

fn dep(name: &str, kind: Option<&str>, optional: bool, features: &[&str]) -> Value {
    json!({
        "name": name,
        "kind": kind,
        "optional": optional,
        "uses_default_features": true,
        "features": features,
        "path": format!("/repo/{name}"),
    })
}

fn package(name: &str, deps: Vec<Value>, features: Value) -> Value {
    json!({
        "name": name,
        "manifest_path": format!("/repo/{name}/Cargo.toml"),
        "dependencies": deps,
        "features": features,
        "targets": [{"kind": ["lib"]}],
    })
}

/// `core` <- `store` <- (optional) `facade` <- `service` (by feature),
/// `example` (with `store`), `plain` (default features); `dev` dev-depends on
/// the facade and `downstream` depends on `dev`.
fn graph() -> Graph {
    Graph::from_metadata(&json!({"packages": [
        package("core", vec![], json!({"extra": []})),
        package("store", vec![dep("core", None, false, &[])], json!({"fast": []})),
        package("other", vec![dep("core", None, false, &[])], json!({})),
        package(
            "facade",
            vec![
                dep("core", None, false, &[]),
                dep("store", None, true, &[]),
                dep("other", None, true, &[]),
            ],
            json!({
                "default": [],
                "store": ["dep:store"],
                "other": ["dep:other"],
                "pdf": ["core/extra"],
                "speed": ["store?/fast"],
            }),
        ),
        package(
            "service",
            vec![dep("facade", None, false, &[])],
            json!({"store": ["facade/store"], "other": ["facade/other"]}),
        ),
        package("example", vec![dep("facade", None, false, &["store"])], json!({})),
        package("plain", vec![dep("facade", None, false, &[])], json!({})),
        package("dev", vec![dep("facade", Some("dev"), false, &[])], json!({})),
        package("downstream", vec![dep("dev", None, false, &[])], json!({})),
    ]}))
    .unwrap()
}

fn observers(changed: &[&str]) -> BTreeMap<String, Features> {
    let g = graph();
    let changed: BTreeSet<usize> = changed.iter().map(|n| g.index(n).unwrap()).collect();
    let dependents = g.dependents(&changed);
    (0..g.packages.len())
        .filter_map(|p| {
            g.observing(p, &changed, &dependents)
                .map(|f| (g.name(p).to_owned(), f))
        })
        .collect()
}

fn with(features: &[&str]) -> Features {
    Features::Default(features.iter().map(|f| (*f).to_owned()).collect())
}

#[test]
fn a_leaf_change_reaches_only_the_features_that_observe_it() {
    let seen = observers(&["store"]);
    let expected: BTreeMap<String, Features> = [
        ("store", Features::All),
        ("facade", with(&["store"])),
        ("service", with(&["store"])),
        ("example", with(&[])),
    ]
    .into_iter()
    .map(|(n, f)| (n.to_owned(), f))
    .collect();
    assert_eq!(seen, expected);
}

#[test]
fn a_dev_dependency_edge_is_one_hop_and_never_propagates() {
    let seen = observers(&["core"]);
    assert!(seen.contains_key("dev"), "{seen:?}");
    assert!(!seen.contains_key("downstream"), "{seen:?}");
}

#[test]
fn a_core_change_runs_its_consumers_with_every_feature_that_reaches_it() {
    let seen = observers(&["core"]);
    assert_eq!(seen.get("facade"), Some(&with(&["other", "pdf", "store"])));
    assert_eq!(seen.get("plain"), Some(&with(&[])));
    assert_eq!(seen.get("service"), Some(&with(&["other", "store"])));
}

#[test]
fn a_weak_feature_applies_only_once_its_dependency_is_on() {
    let g = graph();
    let facade = g.index("facade").unwrap();
    let store = g.index("store").unwrap();
    let alone = g.resolve(facade, &with(&["speed"]), false);
    assert!(!alone.contains_key(&store));
    let both = g.resolve(facade, &with(&["speed", "store"]), false);
    assert!(both.get(&store).is_some_and(|f| f.contains("fast")));
}

#[test]
fn a_relative_path_into_another_package_is_a_read() {
    let dir = std::env::temp_dir().join(format!("rig-graph-reads-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("core/src")).unwrap();
    std::fs::write(
        dir.join("core/src/lib.rs"),
        "const A: &str = \"../store/fixtures\"; const B: &str = \"../core/src\"; \
         const C: &[u8] = include_bytes!(\"../../store/data.bin\");",
    )
    .unwrap();
    let mut metadata = json!({"packages": [
        package("core", vec![], json!({})),
        package("store", vec![], json!({})),
    ]});
    for p in metadata["packages"].as_array_mut().unwrap() {
        let name = p["name"].as_str().unwrap().to_owned();
        p["manifest_path"] = json!(dir.join(&name).join("Cargo.toml"));
    }
    let g = Graph::from_metadata(&metadata).unwrap();
    let reads = g.reads(&dir, &["core/src/lib.rs".to_owned()]);
    std::fs::remove_dir_all(&dir).ok();
    let read = |prefix: &str, embedded| Read {
        prefix: prefix.to_owned(),
        package: 0,
        embedded,
    };
    assert_eq!(
        reads,
        vec![read("store/fixtures", false), read("store/data.bin", true)]
    );
}

#[test]
fn a_literal_is_found_whatever_quotes_and_comments_come_before_it() {
    let text = r####"
        // "../not/a/read" in a comment
        const A: &str = "an escaped \" quote";
        const B: &str = r#"a raw " string"#;
        const C: &str = "../store/fixtures";
        const D: &[u8] = include_bytes!("../../store/data.bin");
    "####;
    assert_eq!(
        relative_literals(text),
        vec![
            ("../store/fixtures".to_owned(), false),
            ("../../store/data.bin".to_owned(), true),
        ]
    );
}
