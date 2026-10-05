//! Lume core types and traits.

#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod model;
pub mod sampling;
pub mod tool;
pub mod types;

pub use config::{LumeConfig, McpServerConfig, McpTransport};
pub use error::{LumeError, Result};
pub use model::{Model, ModelTier};
pub use sampling::{SamplingParams, StopReason};
pub use tool::Tool;
pub use types::{
    ChatChunk, ChatRequest, FinishReason, Message, Role, SamplingParams as TypesSamplingParams,
    ToolCall, ToolSpec, Usage,
};
