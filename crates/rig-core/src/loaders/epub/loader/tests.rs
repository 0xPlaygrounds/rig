use crate::loaders::epub::RawTextProcessor;
use crate::loaders::test_fixtures::{fixture_glob, fixture_path};

use super::EpubFileLoader;

#[test]
fn test_epub_loader_with_errors() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();
    let actual = loader
        .load_with_path()
        .ignore_errors()
        .by_chapter()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);

    let (_, chapters) = &actual[0];
    assert_eq!(chapters.len(), 3);

    for chapter in chapters {
        assert!(chapter.1.is_ok());
    }
}

#[test]
fn test_epub_loader_with_ignoring_errors() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();
    let actual = loader
        .load_with_path()
        .ignore_errors()
        .by_chapter()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);

    let (_, chapters) = &actual[0];
    assert_eq!(chapters.len(), 3);
}

#[test]
fn test_single_file() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();

    let actual = loader
        .read()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);
}

#[test]
fn test_single_file_with_path() {
    let glob = fixture_glob("*.epub");
    let loader = EpubFileLoader::<_, RawTextProcessor>::with_glob(&glob).unwrap();

    let actual = loader
        .read_with_path()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);

    let (path, _) = &actual[0];
    assert_eq!(path, &fixture_path("dummy.epub"));
}
