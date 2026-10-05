//! Error types for the Lume agent harness.

use std::io;

use thiserror::Error;

/// Result type alias for Lume operations.
pub type Result<T, E = LumeError> = std::result::Result<T, E>;

/// Errors that can occur in Lume.
#[derive(Debug, Error)]
pub enum LumeError {
    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// Serialization/deserialization error.
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    /// HTTP error.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// TOML error.
    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Protocol error.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Tool not found.
    #[error("tool not found: {0}")]
    ToolNotFound(String),

    /// Tool execution failed.
    #[error("tool failed: {name} - {message}")]
    ToolFailed {
        /// Tool name.
        name: String,
        /// Failure message.
        message: String,
    },

    /// Model unavailable.
    #[error("model unavailable: {0}")]
    ModelUnavailable(String),

    /// Budget exceeded.
    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),

    /// Not implemented.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}
