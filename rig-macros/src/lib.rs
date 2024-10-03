enum Kind {
    Single,
    Many,
}

trait Embeddable {
    type Kind;
    fn embeddable(&self) -> Vec<String>;
}

#[cfg(test)]
mod tests {
    use super::{Embeddable, Kind};
    use rig_macros_derive::Embedding;

    impl Embeddable for usize {
        type Kind = Kind;

        fn embeddable(&self) -> Vec<String> {
            vec![self.to_string()]
        }
    }

    impl Embeddable for String {
        type Kind = Kind;

        fn embeddable(&self) -> Vec<String> {
            vec![self.clone()]
        }
    }

    #[derive(Embedding)]
    struct MyStruct {
        #[embed]
        id: usize,
        #[embed]
        name: String,
    }

    #[test]
    fn test_macro() {
        let my_struct = MyStruct {
            id: 1,
            name: "John".to_string(),
        };

        println!("{:?}", my_struct.embeddable());

        assert!(false)
    }
}
