use assert_fs::prelude::{FileWriteStr, PathChild};

use super::CsvFileLoader;

#[test]
fn from_bytes_header_value_format() {
    let csv = b"name,age,city\nAlice,30,Paris\nBob,25,London".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv)
        .read()
        .ignore_errors()
        .into_iter()
        .collect();

    assert!(results.len() == 1);
    let text = &results[0];
    assert!(text.contains("name: Alice"));
    assert!(text.contains("age: 30"));
    assert!(text.contains("city: Paris"));
    assert!(text.contains("name: Bob"));
    assert!(text.contains("age: 25"));
    assert!(text.contains("city: London"));
}

#[test]
fn rows_separated_by_blank_line() {
    let csv = b"x,y\n1,2\n3,4".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv)
        .read()
        .ignore_errors()
        .into_iter()
        .collect();

    assert!(results.len() == 1);
    assert!(
        results[0].contains("\n\n"),
        "rows should be separated by a blank line"
    );
}

#[test]
fn empty_csv_returns_empty_string() {
    let csv = b"name,age\n".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv)
        .read()
        .ignore_errors()
        .into_iter()
        .collect();

    assert!(results.len() == 1);
    assert!(results[0].is_empty());
}

#[test]
fn headers_only_no_rows() {
    let csv = b"a,b,c".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv)
        .read()
        .ignore_errors()
        .into_iter()
        .collect();

    assert!(results.len() == 1);
    assert!(results[0].is_empty());
}

#[test]
fn missing_headers_yields_error() {
    let csv = b"".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv).read().into_iter().collect();

    assert!(results.len() == 1);
    assert!(results[0].is_err());
}

#[test]
fn with_dir_reads_csv_files() {
    let temp = assert_fs::TempDir::new().map_err(|e| format!("{e}"));
    let temp = match temp {
        Ok(t) => t,
        Err(_) => return,
    };

    let file = temp.child("data.csv");
    if file.write_str("color,count\nred,5\nblue,3").is_err() {
        return;
    }

    let loader = match CsvFileLoader::with_dir(temp.path().to_str().unwrap_or_default()) {
        Ok(l) => l,
        Err(_) => return,
    };

    let results: Vec<_> = loader.read().ignore_errors().into_iter().collect();

    assert!(results.len() == 1);
    assert!(results[0].contains("color: red"));
    assert!(results[0].contains("count: 5"));
}

#[test]
fn read_with_path_includes_source() {
    let csv = b"k,v\nfoo,bar".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv)
        .read_with_path()
        .ignore_errors()
        .into_iter()
        .collect();

    assert!(results.len() == 1);
    let (path, text) = &results[0];
    assert!(path.to_str().is_some_and(|p| p == "<memory>"));
    assert!(text.contains("k: foo"));
    assert!(text.contains("v: bar"));
}

#[test]
fn fields_with_commas_and_quotes() {
    let csv = b"name,bio\n\"Smith, John\",\"He said \"\"hello\"\"\"\n".to_vec();

    let results: Vec<_> = CsvFileLoader::from_bytes(csv)
        .read()
        .ignore_errors()
        .into_iter()
        .collect();

    assert!(results.len() == 1);
    assert!(results[0].contains("name: Smith, John"));
    assert!(results[0].contains("bio: He said \"hello\""));
}
