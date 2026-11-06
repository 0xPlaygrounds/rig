use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, Decimal128Array, DictionaryArray, FixedSizeBinaryArray,
    FixedSizeListArray, Float32Array, Float64Array, GenericListArray, Int8Array, Int16Array,
    Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, MapArray, RecordBatch, RunArray,
    StringArray, StructArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    builder::{
        Decimal128Builder, FixedSizeBinaryBuilder, FixedSizeListBuilder, Int32Builder, ListBuilder,
        MapBuilder, NullBuilder, StringBuilder, StringDictionaryBuilder, StringRunBuilder,
        StructBuilder, UnionBuilder,
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
            json!({ "some_dict": "def" }),
            json!({ "some_dict": null }),
            json!({ "some_dict": "abc" }),
        ]
    );

    let no_values =
        RunArray::<Int16Type>::try_new(&Int16Array::from(vec![1]), &StringArray::from(vec!["x"]))
            .unwrap()
            .slice(0, 0);
    let null_key = DictionaryArray::new(Int8Array::from(vec![None]), Arc::new(no_values));
    let record_batch =
        RecordBatch::try_from_iter(vec![("d", Arc::new(null_key) as ArrayRef)]).unwrap();

    assert_eq!(
        record_batch_to_json(&record_batch).unwrap(),
        vec![json!({ "d": null })]
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
                "some_run_end": null
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

    assert_eq!(
        record_batch_to_json(&record_batch.slice(1, 2)).unwrap(),
        vec![
            json!({ "some_run_end": null }),
            json!({ "some_run_end": "def" }),
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

#[test]
fn map_entries_stay_with_their_row() {
    let mut maps = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    let rows: [&[(&str, i32)]; 4] = [
        &[("skipped", 0)],
        &[],
        &[("owner", 11), ("priority", 7)],
        &[("owner", 22)],
    ];
    for entries in rows {
        for (key, value) in entries {
            maps.keys().append_value(key);
            maps.values().append_value(*value);
        }
        maps.append(true).unwrap();
    }
    let batch =
        RecordBatch::try_from_iter(vec![("attributes", Arc::new(maps.finish()) as ArrayRef)])
            .unwrap()
            .slice(1, 3);

    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            json!({ "attributes": {} }),
            json!({ "attributes": { "owner": 11, "priority": 7 } }),
            json!({ "attributes": { "owner": 22 } }),
        ]
    );

    let mut null_cells = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for (key, value) in [("k0", None), ("k1", Some(4)), ("k2", None)] {
        null_cells.keys().append_value(key);
        null_cells.values().append_option(value);
    }
    null_cells.append(true).unwrap();
    let batch = RecordBatch::try_from_iter(vec![(
        "attributes",
        Arc::new(null_cells.finish()) as ArrayRef,
    )])
    .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![json!({ "attributes": { "k0": null, "k1": 4, "k2": null } })]
    );

    let mut null_values = MapBuilder::new(None, StringBuilder::new(), NullBuilder::new());
    for key in ["a", "b"] {
        null_values.keys().append_value(key);
        null_values.values().append_null();
    }
    null_values.append(true).unwrap();
    let batch = RecordBatch::try_from_iter(vec![(
        "attributes",
        Arc::new(null_values.finish()) as ArrayRef,
    )])
    .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![json!({ "attributes": { "a": null, "b": null } })]
    );

    let mut dictionaries = MapBuilder::new(
        None,
        StringDictionaryBuilder::<Int32Type>::new(),
        StringDictionaryBuilder::<Int32Type>::new(),
    );
    let rows: [&[(&str, Option<&str>)]; 5] = [
        &[("k0", Some("a")), ("k1", Some("b"))],
        &[],
        &[("k0", Some("a")), ("k1", None), ("k2", Some("b"))],
        &[("k1", Some("a"))],
        &[],
    ];
    for entries in rows {
        for (key, value) in entries {
            dictionaries.keys().append_value(key);
            match value {
                Some(value) => dictionaries.values().append_value(value),
                None => dictionaries.values().append_null(),
            }
        }
        dictionaries.append(true).unwrap();
    }
    let batch = RecordBatch::try_from_iter(vec![(
        "attributes",
        Arc::new(dictionaries.finish()) as ArrayRef,
    )])
    .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            json!({ "attributes": { "k0": "a", "k1": "b" } }),
            json!({ "attributes": {} }),
            json!({ "attributes": { "k0": "a", "k1": null, "k2": "b" } }),
            json!({ "attributes": { "k1": "a" } }),
            json!({ "attributes": {} }),
        ]
    );

    let mut all_null = MapBuilder::new(
        None,
        StringBuilder::new(),
        StringDictionaryBuilder::<Int32Type>::new(),
    );
    all_null.keys().append_value("a");
    all_null.values().append_null();
    all_null.append(true).unwrap();
    let batch = RecordBatch::try_from_iter(vec![(
        "attributes",
        Arc::new(all_null.finish()) as ArrayRef,
    )])
    .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![json!({ "attributes": { "a": null } })]
    );

    let mut lists = MapBuilder::new(
        None,
        StringBuilder::new(),
        ListBuilder::new(StringDictionaryBuilder::<Int32Type>::new()),
    );
    let rows: [&[(&str, &[&str])]; 3] = [
        &[("k0", &["a", "b"]), ("k1", &["a"])],
        &[],
        &[("k2", &[]), ("k3", &["b", "a", "b"]), ("k4", &[])],
    ];
    for entries in rows {
        for (key, items) in entries {
            lists.keys().append_value(key);
            for item in *items {
                lists.values().values().append_value(item);
            }
            lists.values().append(true);
        }
        lists.append(true).unwrap();
    }
    lists.append(false).unwrap();
    let batch =
        RecordBatch::try_from_iter(vec![("attributes", Arc::new(lists.finish()) as ArrayRef)])
            .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            json!({ "attributes": { "k0": ["a", "b"], "k1": ["a"] } }),
            json!({ "attributes": {} }),
            json!({ "attributes": { "k2": [], "k3": ["b", "a", "b"], "k4": [] } }),
            json!({ "attributes": null }),
        ]
    );

    let mut structs = MapBuilder::new(
        None,
        StringBuilder::new(),
        StructBuilder::new(
            vec![Field::new_dictionary(
                "d",
                DataType::Int32,
                DataType::Utf8,
                true,
            )],
            vec![Box::new(StringDictionaryBuilder::<Int32Type>::new())],
        ),
    );
    for (key, value) in [("k0", "a"), ("k1", "b"), ("k2", "a")] {
        structs.keys().append_value(key);
        structs
            .values()
            .field_builder::<StringDictionaryBuilder<Int32Type>>(0)
            .unwrap()
            .append_value(value);
        structs.values().append(true);
    }
    structs.append(true).unwrap();
    let batch =
        RecordBatch::try_from_iter(vec![("attributes", Arc::new(structs.finish()) as ArrayRef)])
            .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![json!({
            "attributes": { "k0": { "d": "a" }, "k1": { "d": "b" }, "k2": { "d": "a" } }
        })]
    );
}

#[test]
fn fixed_size_binary_has_one_value_per_row() {
    let mut fingerprints = FixedSizeBinaryBuilder::new(2);
    for row in 0..3u8 {
        fingerprints.append_value([row; 2]).unwrap();
    }
    let batch = RecordBatch::try_from_iter(vec![(
        "fingerprint",
        Arc::new(fingerprints.finish()) as ArrayRef,
    )])
    .unwrap();

    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        (0..3u8)
            .map(|row| json!({ "fingerprint": [row, row] }))
            .collect::<Vec<_>>()
    );

    // The slots under these nulls hold another row's value, as Lance can write them.
    let nulls = Int32Array::from(vec![Some(1), None, Some(1)])
        .nulls()
        .cloned();
    let mut tags = ListBuilder::new(Int32Builder::new());
    tags.append_value([Some(1), None]);
    tags.append_null();
    tags.append_value([]);
    let details = StructArray::new(
        Fields::from(vec![Field::new("n", DataType::Int32, true)]),
        vec![Arc::new(Int32Array::from(vec![1, 1, 1]))],
        nulls.clone(),
    );
    let batch = RecordBatch::try_from_iter(vec![
        (
            "fingerprint",
            Arc::new(FixedSizeBinaryArray::new(
                2,
                vec![4u8, 5, 4, 5, 4, 5].into(),
                nulls.clone(),
            )) as ArrayRef,
        ),
        (
            "count",
            Arc::new(Int32Array::new(vec![7, 7, 7].into(), nulls)) as ArrayRef,
        ),
        (
            "name",
            Arc::new(StringArray::from(vec![Some("a"), None, Some("a")])) as ArrayRef,
        ),
        ("tags", Arc::new(tags.finish()) as ArrayRef),
        ("details", Arc::new(details) as ArrayRef),
    ])
    .unwrap();

    let mut rows = vec![
        json!({
            "fingerprint": [4, 5], "count": 7, "name": "a",
            "tags": [1, null], "details": { "n": 1 },
        }),
        json!({
            "fingerprint": null, "count": null, "name": null, "tags": null, "details": null,
        }),
        json!({
            "fingerprint": [4, 5], "count": 7, "name": "a", "tags": [], "details": { "n": 1 },
        }),
    ];
    assert_eq!(record_batch_to_json(&batch).unwrap(), rows);
    assert_eq!(
        record_batch_to_json(&batch.slice(1, 2)).unwrap(),
        rows.split_off(1)
    );

    // JSON can't hold this value, but the cell is null, so it must not be converted.
    let nulls = Int32Array::from(vec![None]).nulls().cloned();
    let amounts = Decimal128Array::new(vec![10_i128.pow(25)].into(), nulls)
        .with_precision_and_scale(38, 0)
        .unwrap();
    let batch = RecordBatch::try_from_iter(vec![("d", Arc::new(amounts) as ArrayRef)]).unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![json!({ "d": null })]
    );

    // The same value as a valid child of a null parent row.
    let decimal = DataType::Decimal128(38, 0);
    let too_large = 10_i128.pow(25);
    let children: ArrayRef = Arc::new(
        Decimal128Array::from(vec![too_large, 7, too_large])
            .with_precision_and_scale(38, 0)
            .unwrap(),
    );
    let nulls = Int32Array::from(vec![None, Some(1), None]).nulls().cloned();
    let fixed = FixedSizeListArray::new(
        Arc::new(Field::new("item", decimal.clone(), true)),
        1,
        children.clone(),
        nulls.clone(),
    );
    let details = StructArray::new(
        Fields::from(vec![Field::new("n", decimal.clone(), true)]),
        vec![children],
        nulls,
    );
    let decimals = || Decimal128Builder::new().with_data_type(decimal.clone());
    let mut list = ListBuilder::new(decimals());
    let mut map = MapBuilder::new(None, StringBuilder::new(), decimals());
    for (value, is_valid) in [(too_large, false), (7, true), (too_large, false)] {
        list.values().append_value(value);
        list.append(is_valid);
        map.keys().append_value("k");
        map.values().append_value(value);
        map.append(is_valid).unwrap();
    }
    let batch = RecordBatch::try_from_iter(vec![
        ("fixed", Arc::new(fixed) as ArrayRef),
        ("details", Arc::new(details) as ArrayRef),
        ("list", Arc::new(list.finish()) as ArrayRef),
        ("map", Arc::new(map.finish()) as ArrayRef),
    ])
    .unwrap();
    let null_row = json!({ "fixed": null, "details": null, "list": null, "map": null });
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            null_row.clone(),
            json!({ "fixed": [7], "details": { "n": 7 }, "list": [7], "map": { "k": 7 } }),
            null_row,
        ]
    );

    // Encoded children share their values, so a null parent row must also skip those.
    let shared = Decimal128Array::from(vec![too_large, 7])
        .with_precision_and_scale(38, 0)
        .unwrap();
    let runs = RunArray::<Int16Type>::try_new(&Int16Array::from(vec![1, 2]), &shared).unwrap();
    let keys = DictionaryArray::new(Int8Array::from(vec![0, 1]), Arc::new(shared));
    let nulls = Int32Array::from(vec![None, Some(1)]).nulls().cloned();
    let parent = |child: ArrayRef| -> ArrayRef {
        let item = Field::new("item", child.data_type().clone(), true);
        Arc::new(FixedSizeListArray::new(
            Arc::new(item),
            1,
            child,
            nulls.clone(),
        ))
    };
    let batch = RecordBatch::try_from_iter(vec![
        ("runs", parent(Arc::new(runs))),
        ("keys", parent(Arc::new(keys))),
    ])
    .unwrap();
    assert_eq!(
        record_batch_to_json(&batch).unwrap(),
        vec![
            json!({ "runs": null, "keys": null }),
            json!({ "runs": [7], "keys": [7] }),
        ]
    );
}
