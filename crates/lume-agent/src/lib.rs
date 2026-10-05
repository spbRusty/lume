//! Agent implementation.

#![warn(missing_docs)]

pub mod context;
pub mod r#loop;
pub mod parsing;
pub mod tools;

pub use context::Conversation;
pub use r#loop::{Agent, AgentConfig, AgentOutcome, ReActAgent};
pub use lume_mcp::ToolRegistry;
pub use parsing::{parse_tool_calls, strip_tool_calls};
pub use tools::{EditFileTool, GrepTool, ListDirTool, ReadFileTool, ShellTool, WriteFileTool};
