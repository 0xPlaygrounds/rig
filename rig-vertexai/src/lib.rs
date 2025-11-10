pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
pub use completion::{
    GEMINI_1_5_FLASH, GEMINI_1_5_FLASH_LATEST, GEMINI_1_5_PRO, GEMINI_1_5_PRO_LATEST,
    GEMINI_2_0_FLASH_EXP, GEMINI_2_5_FLASH, GEMINI_2_5_FLASH_LITE, GEMINI_2_5_PRO,
};
