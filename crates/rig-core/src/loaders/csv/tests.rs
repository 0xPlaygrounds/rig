use assert_fs::prelude::{FileWriteStr, PathChild};

use super::CsvFileLoader;

#[test]
fn renders_header_value_lines_per_row_and_per_file() {
    let csv = b"name,motto\nAda,\"count, then \"\"compute\"\"\"\nGrace,\"line one\nline two\"\n";

    let rows = CsvFileLoader::from_bytes(csv.to_vec())
        .rows()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![vec![
            "name: Ada\nmotto: count, then \"compute\"".to_string(),
            "name: Grace\nmotto: line one\nline two".to_string(),
        ]]
    );

    let documents = CsvFileLoader::from_bytes(csv.to_vec())
        .read()
        .ignore_errors()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        documents,
        vec![
            "name: Ada\nmotto: count, then \"compute\"\n\nname: Grace\nmotto: line one\nline two"
                .to_string()
        ]
    );
}

#[test]
fn delimiter_headerless_and_flexible_input() {
    let rows = CsvFileLoader::from_bytes(b"Ada\t1815\nGrace\t1906\n".to_vec())
        .delimiter(b'\t')
        .has_headers(false)
        .rows()
        .ignore_errors()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec!["Ada\n1815".to_string(), "Grace\n1906".to_string()]
    );

    let rows = CsvFileLoader::from_bytes(b"name,born\nAda\nGrace,1906,admiral\n".to_vec())
        .flexible(true)
        .rows()
        .ignore_errors()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            "name: Ada".to_string(),
            "name: Grace\nborn: 1906\nadmiral".to_string()
        ]
    );
}

#[test]
fn directory_yields_each_file_and_its_parse_errors() {
    let temp = assert_fs::TempDir::new().unwrap();
    temp.child("good.csv")
        .write_str("id,name\n1,Ada\n")
        .unwrap();
    temp.child("ragged.csv")
        .write_str("id,name\n1,Ada\n2\n")
        .unwrap();

    let mut results = CsvFileLoader::with_dir(temp.path().to_str().unwrap())
        .unwrap()
        .rows_with_path()
        .into_iter()
        .map(|result| result.map_err(|error| error.to_string()))
        .collect::<Vec<_>>();
    results.sort_by_key(|result| result.is_err());

    let [good, ragged] = results.as_slice() else {
        panic!("expected two results, got {results:?}");
    };
    assert_eq!(
        good.as_ref().ok(),
        Some(&(
            temp.path().join("good.csv"),
            vec!["id: 1\nname: Ada".to_string()]
        ))
    );
    assert!(
        ragged
            .as_ref()
            .is_err_and(|error| error.starts_with("CSV error")),
        "{ragged:?}"
    );
}
