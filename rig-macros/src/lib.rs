enum Kind {
    Single,
    Many,
}

trait Embeddable {
    type Kind;
    fn embeddable(&self);
}

#[cfg(test)]
mod tests {
    use super::Embeddable;
    use rig_macros_derive::Embedding;

    #[derive(Embedding)]
    struct MyStruct {
        id: String,
        #[embed]
        name: String,
    }

    #[test]
    fn test_macro() {
        let my_struct = MyStruct {
            id: "1".to_string(),
            name: "John".to_string(),
        };

        my_struct.embeddable();

        assert!(false)
    }
}
