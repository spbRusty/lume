//! LLM utilities and backends.

#![warn(missing_docs)]

pub mod chat_template;
pub mod ollama;
pub mod sampling;

pub use chat_template::render_chatml;
pub use ollama::OllamaBackend;
pub use sampling::to_ollama_options;
