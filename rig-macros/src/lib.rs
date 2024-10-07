enum Kind {
    Single,
    Many,
}

trait Embeddable {
    type Kind;
    fn embeddable(&self) -> Vec<String>;
}

#[derive(serde::Serialize)]
pub struct JobStruct {
    job_title: String,
    company: String,
}

mod something {
    use super::JobStruct;

    pub fn embeddable(input: &JobStruct) -> Vec<String> {
        vec![serde_json::to_string(input).unwrap()]
    }
}

#[cfg(test)]
mod tests {
    use crate::JobStruct;

    use super::{Embeddable, Kind};
    use rig_macros_derive::Embedding;

    impl Embeddable for usize {
        type Kind = Kind;

        fn embeddable(&self) -> Vec<String> {
            vec![self.to_string()]
        }
    }

    #[derive(Embedding)]
    struct SomeStruct {
        #[embed]
        id: usize,
        name: String,
        #[embed(embed_with = "super::something")]
        job: JobStruct,
    }

    #[test]
    fn test_macro() {
        let job_struct = JobStruct {
            job_title: "developer".to_string(),
            company: "playgrounds".to_string(),
        };
        let some_struct = SomeStruct {
            id: 1,
            name: "John".to_string(),
            job: job_struct,
        };

        println!("{:?}", some_struct.embeddable());

        assert!(false)
    }
}
