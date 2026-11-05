use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, GenericListArray,
    Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, MapArray,
    RecordBatch, StringArray, StructArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    builder::{
        FixedSizeBinaryBuilder, FixedSizeListBuilder, Int32Builder, ListBuilder, MapBuilder,
        StringBuilder, StringDictionaryBuilder, StringRunBuilder, UnionBuilder,
    },
    types::{Float64Type, Int8Type, Int16Type, Int32Type},
};
use lancedb::arrow::arrow_schema::{DataType, Field, Fields};
use serde_json::json;

use crate::utils::deserializer::record_batch_to_json;

fn fixed_list_actors() -> FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(StringBuilder::new(), 2);
    builder.values().append_value("Johnny Depp");
    builder.values().append_value("Cate Blanchet");
    builder.append(true);
    builder.values().append_value("Meryl Streep");
    builder.values().append_value("Scarlett Johansson");
    builder.append(true);
    builder.values().append_value("Brad Pitt");
    builder.values().append_value("Natalie Portman");
    builder.append(true);

    builder.finish()
}

fn name_list() -> GenericListArray<i32> {
    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.values().append_value("Alice");
    builder.values().append_value("Bob");
    builder.append(true);
    builder.values().append_value("Charlie");
    builder.append(true);
    builder.values().append_value("David");
    builder.values().append_value("Eve");
    builder.values().append_value("Frank");
    builder.append(true);
    builder.finish()
}

fn nested_list_of_animals() -> GenericListArray<i32> {
    // [ [ [ "Dog", "Cat" ], ["Mouse"] ], [ [ "Giraffe" ], ["Cow", "Pig"] ], [ [ "Sloth" ], ["Ant", "Monkey"] ] ]
    let mut builder = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
    builder
        .values()
        .append_value(vec![Some("Dog"), Some("Cat")]);
    builder.values().append_value(vec![Some("Mouse")]);
    builder.append(true);
    builder.values().append_value(vec![Some("Giraffe")]);
    builder
        .values()
        .append_value(vec![Some("Cow"), Some("Pig")]);
    builder.append(true);
    builder.values().append_value(vec![Some("Sloth")]);
    builder
        .values()
        .append_value(vec![Some("Ant"), Some("Monkey")]);
    builder.append(true);
    builder.finish()
}

fn movie_struct() -> StructArray {
    StructArray::from(vec![
        (
            Arc::new(Field::new("name", DataType::Utf8, false)),
            Arc::new(StringArray::from(vec![
                "Pulp Fiction",
                "The Shawshank Redemption",
                "La La Land",
            ])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("year", DataType::UInt32, false)),
            Arc::new(UInt32Array::from(vec![1999, 2026, 1745])) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                "actors",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Utf8, true)), 2),
                false,
            )),
            Arc::new(fixed_list_actors()) as ArrayRef,
        ),
    ])
}

#[tokio::test]
async fn test_primitive_deserialization() {
    let string = Arc::new(StringArray::from_iter_values(vec!["Marty", "Tony"])) as ArrayRef;
    let large_string =
        Arc::new(LargeStringArray::from_iter_values(vec!["Jerry", "Freddy"])) as ArrayRef;
    let binary = Arc::new(BinaryArray::from_iter_values(vec![b"hello", b"world"])) as ArrayRef;
    let large_binary =
        Arc::new(LargeBinaryArray::from_iter_values(vec![b"abc", b"def"])) as ArrayRef;
    let float_32 = Arc::new(Float32Array::from_iter_values(vec![0.0, 1.0])) as ArrayRef;
    let float_64 = Arc::new(Float64Array::from_iter_values(vec![0.0, 1.0])) as ArrayRef;
    let int_8 = Arc::new(Int8Array::from_iter_values(vec![0, -1])) as ArrayRef;
    let int_16 = Arc::new(Int16Array::from_iter_values(vec![-0, 1])) as ArrayRef;
    let int_32 = Arc::new(Int32Array::from_iter_values(vec![0, -1])) as ArrayRef;
    let int_64 = Arc::new(Int64Array::from_iter_values(vec![-0, 1])) as ArrayRef;
    let uint_8 = Arc::new(UInt8Array::from_iter_values(vec![0, 1])) as ArrayRef;
    let uint_16 = Arc::new(UInt16Array::from_iter_values(vec![0, 1])) as ArrayRef;
    let uint_32 = Arc::new(UInt32Array::from_iter_values(vec![0, 1])) as ArrayRef;
    let uint_64 = Arc::new(UInt64Array::from_iter_values(vec![0, 1])) as ArrayRef;

    let record_batch = RecordBatch::try_from_iter(vec![
        ("float_32", float_32),
        ("float_64", float_64),
        ("int_8", int_8),
        ("int_16", int_16),
        ("int_32", int_32),
        ("int_64", int_64),
        ("uint_8", uint_8),
        ("uint_16", uint_16),
        ("uint_32", uint_32),
        ("uint_64", uint_64),
        ("string", string),
        ("large_string", large_string),
        ("large_binary", large_binary),
        ("binary", binary),
    ])
    .unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![
            json!({
                "binary": [
                    104,
                    101,
                    108,
                    108,
                    111
                ],
                "float_32": 0.0,
                "float_64": 0.0,
                "int_16": 0,
                "int_32": 0,
                "int_64": 0,
                "int_8": 0,
                "large_binary": [
                    97,
                    98,
                    99
                ],
                "large_string": "Jerry",
                "string": "Marty",
                "uint_16": 0,
                "uint_32": 0,
                "uint_64": 0,
                "uint_8": 0
            }),
            json!({
                "binary": [
                    119,
                    111,
                    114,
                    108,
                    100
                ],
                "float_32": 1.0,
                "float_64": 1.0,
                "int_16": 1,
                "int_32": -1,
                "int_64": 1,
                "int_8": -1,
                "large_binary": [
                    100,
                    101,
                    102
                ],
                "large_string": "Freddy",
                "string": "Tony",
                "uint_16": 1,
                "uint_32": 1,
                "uint_64": 1,
                "uint_8": 1
            })
        ]
    );
}

#[tokio::test]
async fn test_dictionary_deserialization() {
    let dictionary_values = StringArray::from(vec![None, Some("abc"), Some("def")]);

    let mut builder =
        StringDictionaryBuilder::<Int8Type>::new_with_dictionary(3, &dictionary_values).unwrap();
    builder.append("def").unwrap();
    builder.append_null();
    builder.append("abc").unwrap();

    let dictionary_array = builder.finish();

    let record_batch =
        RecordBatch::try_from_iter(vec![("some_dict", Arc::new(dictionary_array) as ArrayRef)])
            .unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![
            json!({
                "some_dict": {
                    "2": ""
                }
            }),
            json!({
                "some_dict": {
                    "0": "abc"
                }
            }),
            json!({
                "some_dict": {
                    "1": "def"
                }
            })
        ]
    );
}

#[tokio::test]
async fn test_union_deserialization() {
    let mut builder = UnionBuilder::new_dense();
    builder.append::<Int32Type>("type_a", 1).unwrap();
    builder.append::<Float64Type>("type_b", 3.0).unwrap();
    builder.append::<Int32Type>("type_a", 4).unwrap();
    let union = builder.build().unwrap();

    let record_batch =
        RecordBatch::try_from_iter(vec![("some_union", Arc::new(union) as ArrayRef)]).unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![
            json!({
                "some_union": [
                    1
                ]
            }),
            json!({
                "some_union": [
                    3.0
                ]
            }),
            json!({
                "some_union": [
                    4
                ]
            })
        ]
    );
}

#[tokio::test]
async fn test_run_end_deserialization() {
    let mut builder = StringRunBuilder::<Int16Type>::new();

    // The builder builds the dictionary value by value
    builder.append_value("abc");
    builder.append_null();
    builder.extend([Some("def"), Some("def"), Some("abc")]);
    let array = builder.finish();

    let record_batch =
        RecordBatch::try_from_iter(vec![("some_run_end", Arc::new(array) as ArrayRef)]).unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![
            json!({
                "some_run_end": "abc"
            }),
            json!({
                "some_run_end": ""
            }),
            json!({
                "some_run_end": "def"
            }),
            json!({
                "some_run_end": "def"
            }),
            json!({
                "some_run_end": "abc"
            })
        ]
    );
}

#[tokio::test]
async fn test_map_deserialization() {
    let record_batch = RecordBatch::try_from_iter(vec![(
        "map_col",
        Arc::new(
            MapArray::new_from_strings(
                vec!["tarentino", "darabont", "chazelle"].into_iter(),
                &movie_struct(),
                &[0, 1, 2],
            )
            .unwrap(),
        ) as ArrayRef,
    )])
    .unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![
            json!({
                "map_col": {
                    "tarentino": {
                        "actors": [
                            "Johnny Depp",
                            "Cate Blanchet"
                        ],
                        "name": "Pulp Fiction",
                        "year": 1999
                    }
                }
            }),
            json!({
                "map_col": {
                    "darabont": {
                        "actors": [
                            "Meryl Streep",
                            "Scarlett Johansson"
                        ],
                        "name": "The Shawshank Redemption",
                        "year": 2026
                    }
                }
            })
        ]
    );
}

fn map_record_batch(rows: &[(&str, &[(&str, i32)])]) -> RecordBatch {
    let mut maps = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    let mut ids = Vec::new();
    for (id, entries) in rows {
        ids.push(*id);
        for (key, value) in *entries {
            maps.keys().append_value(key);
            maps.values().append_value(*value);
        }
        maps.append(true).unwrap();
    }
    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(StringArray::from(ids)) as ArrayRef),
        ("attributes", Arc::new(maps.finish()) as ArrayRef),
    ])
    .unwrap()
}

#[test]
fn map_entries_stay_with_their_document() {
    let batch = map_record_batch(&[
        ("empty-first", &[]),
        ("first", &[("owner", 11), ("priority", 7)]),
        ("empty-middle", &[]),
        ("second", &[("owner", 22)]),
        ("empty-last", &[]),
    ]);

    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            json!({"id": "empty-first", "attributes": {}}),
            json!({"id": "first", "attributes": {"owner": 11, "priority": 7}}),
            json!({"id": "empty-middle", "attributes": {}}),
            json!({"id": "second", "attributes": {"owner": 22}}),
            json!({"id": "empty-last", "attributes": {}}),
        ],
    );
}

#[test]
fn sliced_maps_use_their_entry_offsets() {
    let batch = map_record_batch(&[
        ("prefix", &[("owner", 99), ("priority", 8)]),
        ("empty", &[]),
        ("selected", &[("owner", 11), ("priority", 7)]),
        ("suffix", &[("owner", 22)]),
    ])
    .slice(1, 2);

    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            json!({"id": "empty", "attributes": {}}),
            json!({"id": "selected", "attributes": {"owner": 11, "priority": 7}}),
        ],
    );
}

#[test]
fn empty_map_batch_has_no_rows() {
    assert!(
        record_batch_to_json(&map_record_batch(&[]))
            .unwrap()
            .is_empty()
    );
}

fn fixed_binary_record_batch(row_count: u8) -> RecordBatch {
    let mut fingerprints = FixedSizeBinaryBuilder::new(16);
    for row in 0..row_count {
        fingerprints.append_value([row; 16]).unwrap();
    }
    RecordBatch::try_from_iter(vec![
        (
            "id",
            Arc::new(UInt8Array::from_iter_values(0..row_count)) as ArrayRef,
        ),
        ("fingerprint", Arc::new(fingerprints.finish()) as ArrayRef),
    ])
    .unwrap()
}

fn assert_fixed_binary_rows(batch: &RecordBatch, expected_ids: std::ops::Range<u8>) {
    let expected: Vec<_> = expected_ids
        .map(|id| json!({"id": id, "fingerprint": vec![id; 16]}))
        .collect();
    assert_eq!(record_batch_to_json(batch).unwrap(), expected);
}

#[test]
fn fixed_binary_with_fewer_rows_than_byte_width() {
    assert_fixed_binary_rows(&fixed_binary_record_batch(1), 0..1);
}

#[test]
fn fixed_binary_with_more_rows_than_byte_width() {
    assert_fixed_binary_rows(&fixed_binary_record_batch(17), 0..17);
}

#[test]
fn fixed_binary_with_rows_equal_to_byte_width() {
    assert_fixed_binary_rows(&fixed_binary_record_batch(16), 0..16);
}

#[test]
fn sliced_fixed_binary_preserves_every_row() {
    assert_fixed_binary_rows(&fixed_binary_record_batch(19).slice(1, 17), 1..18);
}

#[test]
fn empty_fixed_binary_batch_has_no_rows() {
    assert_fixed_binary_rows(&fixed_binary_record_batch(0), 0..0);
}

#[tokio::test]
async fn test_recursion() {
    let id_values = StringArray::from(vec!["id1", "id2", "id3"]);
    let age_values = Float32Array::from(vec![25.0, 30.5, 22.1]);
    let struct_array = StructArray::from(vec![
        (
            Arc::new(Field::new("id", DataType::Utf8, false)),
            Arc::new(id_values) as ArrayRef,
        ),
        (
            Arc::new(Field::new("age", DataType::Float32, false)),
            Arc::new(age_values) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                "names",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            )),
            Arc::new(name_list()) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                "favorite_animals",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    true,
                ))),
                false,
            )),
            Arc::new(nested_list_of_animals()) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                "favorite_movie",
                DataType::Struct(Fields::from_iter(vec![
                    Field::new("name", DataType::Utf8, false),
                    Field::new("year", DataType::UInt32, false),
                    Field::new(
                        "actors",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Utf8, true)),
                            2,
                        ),
                        false,
                    ),
                ])),
                false,
            )),
            Arc::new(movie_struct()) as ArrayRef,
        ),
    ]);

    let record_batch =
        RecordBatch::try_from_iter(vec![("employees", Arc::new(struct_array) as ArrayRef)])
            .unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![
            json!({
                "employees": {
                    "age": 25.0,
                    "favorite_animals": [
                        [
                            "Dog",
                            "Cat"
                        ],
                        [
                            "Mouse"
                        ]
                    ],
                    "favorite_movie": {
                        "actors": [
                            "Johnny Depp",
                            "Cate Blanchet"
                        ],
                        "name": "Pulp Fiction",
                        "year": 1999
                    },
                    "id": "id1",
                    "names": [
                        "Alice",
                        "Bob"
                    ]
                }
            }),
            json!({
                "employees": {
                    "age": 30.5,
                    "favorite_animals": [
                        [
                            "Giraffe"
                        ],
                        [
                            "Cow",
                            "Pig"
                        ]
                    ],
                    "favorite_movie": {
                        "actors": [
                            "Meryl Streep",
                            "Scarlett Johansson"
                        ],
                        "name": "The Shawshank Redemption",
                        "year": 2026
                    },
                    "id": "id2",
                    "names": [
                        "Charlie"
                    ]
                }
            }),
            json!({
                "employees": {
                    "age": 22.100000381469727,
                    "favorite_animals": [
                        [
                            "Sloth"
                        ],
                        [
                            "Ant",
                            "Monkey"
                        ]
                    ],
                    "favorite_movie": {
                        "actors": [
                            "Brad Pitt",
                            "Natalie Portman"
                        ],
                        "name": "La La Land",
                        "year": 1745
                    },
                    "id": "id3",
                    "names": [
                        "David",
                        "Eve",
                        "Frank"
                    ]
                }
            })
        ]
    );
}
