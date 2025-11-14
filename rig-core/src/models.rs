#[macro_export]
macro_rules! models {
    (
    $(#[$enum_meta:meta])*
    pub enum $name:ident {
        $(
            $(#[$variant_meta:meta])*
            $variant:ident => $string:literal
        ),* $(,)?
    }
) => {
    #[non_exhaustive]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    $(#[$enum_meta])*
    pub enum $name {
        $(
            $(#[$variant_meta])*
            $variant,
        )*
    }

    impl std::fmt::Display for $name {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str((*self).into())
        }
    }

    impl From<$name> for &'static str {
        fn from(value: $name) -> Self {
            match value {
                $(
                    $name::$variant => $string,
                )*
            }
        }
    }

    impl TryFrom<String> for $name {
        type Error = String;

        fn try_from(value: String) -> Result<Self, Self::Error> {
            match value.as_str() {
                $(
                    $string => Ok($name::$variant),
                )*
                _ => Err(format!("Invalid model '{value}'"))
            }
        }
    }
};
}
