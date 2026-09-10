use assert_fs::prelude::{FileTouch, FileWriteStr, PathChild};

use super::FileLoader;

#[test]
fn test_file_loader() {
    let temp = assert_fs::TempDir::new().expect("Failed to create temp dir");
    let foo_file = temp.child("foo.txt");
    let bar_file = temp.child("bar.txt");

    foo_file.touch().expect("Failed to create foo.txt");
    bar_file.touch().expect("Failed to create bar.txt");

    foo_file.write_str("foo").expect("Failed to write to foo");
    bar_file.write_str("bar").expect("Failed to write to bar");

    let glob = temp.path().to_string_lossy().to_string() + "/*.txt";

    let loader = FileLoader::with_glob(&glob).unwrap();
    let mut actual = loader
        .ignore_errors()
        .read()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();
    let mut expected = vec!["foo".to_string(), "bar".to_string()];

    actual.sort();
    expected.sort();

    assert!(!actual.is_empty());
    assert!(expected == actual);
}

#[test]
fn test_file_loader_bytes() {
    let temp = assert_fs::TempDir::new().expect("Failed to create temp dir");
    let foo_file = temp.child("foo.txt");
    let bar_file = temp.child("bar.txt");

    foo_file.touch().expect("Failed to create foo.txt");
    bar_file.touch().expect("Failed to create bar.txt");

    foo_file.write_str("foo").expect("Failed to write to foo");
    bar_file.write_str("bar").expect("Failed to write to bar");

    let foo_bytes = std::fs::read(foo_file.path()).unwrap();
    let bar_bytes = std::fs::read(bar_file.path()).unwrap();

    let loader = FileLoader::from_bytes_multi(vec![foo_bytes, bar_bytes]);
    let mut actual = loader
        .read()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();
    let mut expected = vec!["foo".to_string(), "bar".to_string()];

    actual.sort();
    expected.sort();

    assert!(!actual.is_empty());
    assert!(expected == actual);
}
