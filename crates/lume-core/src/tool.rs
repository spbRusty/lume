//! Tool trait definition.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::ToolSpec;

/// Tool trait for executable tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get tool specification.
    fn spec(&self) -> ToolSpec;

    /// Call the tool with arguments.
    async fn call(&self, args: serde_json::Value) -> Result<String>;
}
