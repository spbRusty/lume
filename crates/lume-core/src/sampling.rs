//! Sampling parameters and related types.

pub use crate::types::SamplingParams;

/// Stop reason (not used in core types as specified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Natural stop.
    Stop,
    /// Tool use.
    ToolUse,
    /// Length limit reached.
    Length,
}
