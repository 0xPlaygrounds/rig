use crate::loaders::test_fixtures::{fixture_glob, fixture_path};

use super::PdfFileLoader;

#[test]
fn test_pdf_loader() {
    let glob = fixture_glob("*.pdf");
    let loader = PdfFileLoader::with_glob(&glob).unwrap();
    let actual = loader
        .load_with_path()
        .ignore_errors()
        .by_page()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    let mut actual = actual
        .into_iter()
        .map(|result| {
            let (path, pages) = result;
            pages.iter().for_each(|(page_no, content)| {
                println!("{path:?} Page {page_no}: {content:?}");
            });
            (path, pages)
        })
        .collect::<Vec<_>>();

    let mut expected = vec![
        (
            fixture_path("dummy.pdf"),
            vec![(0, "Test\nPDF\nDocument\n".to_string())],
        ),
        (
            fixture_path("file-id-verifiers.pdf"),
            vec![
                (0, "rig-file-id-page-one-verifier-3a91\n".to_string()),
                (1, "rig-file-id-page-two-verifier-8c27\n".to_string()),
                (2, "rig-file-id-page-three-verifier-f54e\n".to_string()),
            ],
        ),
        (
            fixture_path("pages.pdf"),
            vec![
                (0, "Page\n1\n".to_string()),
                (1, "Page\n2\n".to_string()),
                (2, "Page\n3\n".to_string()),
            ],
        ),
    ];

    actual.sort();
    expected.sort();

    assert!(!actual.is_empty());
    assert!(expected == actual);
}

#[test]
fn test_pdf_loader_bytes() {
    // this should never fail!
    let bytes = std::fs::read(fixture_path("dummy.pdf")).unwrap();

    let loader = PdfFileLoader::from_bytes(bytes);

    let actual = loader
        .load()
        .ignore_errors()
        .by_page()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 1);
    assert_eq!(actual, vec!["Test\nPDF\nDocument\n".to_string()]);

    // this should never fail!
    let bytes = std::fs::read(fixture_path("pages.pdf")).unwrap();

    let loader = PdfFileLoader::from_bytes(bytes);

    let actual = loader
        .load()
        .ignore_errors()
        .by_page()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(actual.len(), 3);
    assert_eq!(
        actual,
        vec![
            "Page\n1\n".to_string(),
            "Page\n2\n".to_string(),
            "Page\n3\n".to_string(),
        ]
    );
}

#[test]
fn test_pdf_loader_bytes_multi() {
    let dummy = std::fs::read(fixture_path("dummy.pdf")).unwrap();
    let pages = std::fs::read(fixture_path("pages.pdf")).unwrap();

    let loader = PdfFileLoader::from_bytes_multi(vec![dummy, pages]);

    let actual = loader
        .load()
        .ignore_errors()
        .by_page()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(
        actual,
        vec![
            "Test\nPDF\nDocument\n".to_string(),
            "Page\n1\n".to_string(),
            "Page\n2\n".to_string(),
            "Page\n3\n".to_string(),
        ]
    );
}
