//! Agent execution loop.
//!
//! The loop is the reason this crate exists. It repeatedly asks the model what to do
//! next, runs whatever tools it asked for, feeds the results back, and stops when the
//! model answers without requesting tools.
//!
//! Message history is passed to the backend as structured [`Message`] values rather
//! than as a pre-rendered prompt string. That is deliberate: the ollama backend posts
//! to `/api/chat`, and the server applies the model's own chat template. Rendering
//! ChatML here as well would apply the template twice and corrupt the prompt. The
//! renderer in `lume-llm` exists for backends that take a raw prompt instead.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use lume_core::error::{LumeError, Result};
use lume_core::model::Model;
use lume_core::types::{ChatRequest, Message, Role, SamplingParams};
use lume_mcp::ToolRegistry;

use crate::context::Conversation;
use crate::parsing::{parse_tool_calls, strip_tool_calls};

/// Agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of model turns before the run is abandoned.
    pub max_iterations: usize,
    /// Budget for a single tool result, in bytes. Longer output is truncated on a
    /// character boundary and marked, so the model can tell it was clipped.
    pub max_tool_output_bytes: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 24,
            max_tool_output_bytes: 8192,
        }
    }
}

/// Agent outcome.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The model's final answer, with tool-call JSON stripped out.
    pub final_text: String,
    /// Number of model turns taken.
    pub iterations: usize,
    /// Tool calls executed, formatted as `name(arguments)` for logging.
    pub tool_calls: Vec<String>,
}

/// Agent trait.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run agent with prompt.
    async fn run(&self, prompt: &str, tools: &ToolRegistry) -> Result<AgentOutcome>;
}

/// ReAct agent.
pub struct ReActAgent {
    model: Arc<dyn Model>,
    conversation: Mutex<Conversation>,
    config: AgentConfig,
}

impl ReActAgent {
    /// Create new agent.
    pub fn new(model: Arc<dyn Model>, conversation: Conversation, config: AgentConfig) -> Self {
        Self {
            model,
            conversation: Mutex::new(conversation),
            config,
        }
    }

    /// The model this agent talks to.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

#[async_trait]
impl Agent for ReActAgent {
    async fn run(&self, prompt: &str, tools: &ToolRegistry) -> Result<AgentOutcome> {
        let mut iterations = 0usize;
        let mut tool_calls_made: Vec<String> = Vec::new();

        if !prompt.trim().is_empty() {
            let mut conv = self.conversation.lock().await;
            conv.push(Message {
                role: Role::User,
                content: prompt.to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            });
        }

        loop {
            if iterations >= self.config.max_iterations {
                warn!(
                    max_iterations = self.config.max_iterations,
                    "agent hit the iteration ceiling"
                );
                return Err(LumeError::BudgetExceeded(format!(
                    "agent exceeded {} iterations",
                    self.config.max_iterations
                )));
            }

            // Keep the history inside the model's window before spending a turn on it.
            {
                let mut conv = self.conversation.lock().await;
                if conv.compact_to_fit() {
                    debug!(
                        messages = conv.len(),
                        tokens = conv.estimate_tokens(),
                        "compacted conversation history"
                    );
                }
            }

            iterations += 1;

            let request = {
                let conv = self.conversation.lock().await;
                ChatRequest {
                    model: self.model.name().to_string(),
                    messages: conv.messages(),
                    tools: tools.specs(),
                    params: SamplingParams::default(),
                }
            };

            let reply = self.model.chat(request).await?;
            let calls = parse_tool_calls(&reply);

            if calls.is_empty() {
                // Nothing parsed as a tool call, so there is nothing to strip: running
                // the stripper here would silently delete JSON the model legitimately
                // put in its answer, because it keys off a "name" field rather than
                // off a successful parse.
                let final_text = reply.trim().to_string();
                let final_text = if is_effectively_empty(&final_text) {
                    "the model stopped without producing an answer".to_string()
                } else {
                    final_text
                };
                let mut conv = self.conversation.lock().await;
                conv.push(Message {
                    role: Role::Assistant,
                    content: final_text.clone(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                });
                debug!(iterations, "agent finished without requesting tools");
                return Ok(AgentOutcome {
                    final_text,
                    iterations,
                    tool_calls: tool_calls_made,
                });
            }

            debug!(iterations, count = calls.len(), "model requested tools");

            // Record the assistant turn with its tool calls attached, so the next model
            // turn can see what it asked for alongside what came back.
            {
                let mut conv = self.conversation.lock().await;
                conv.push(Message {
                    role: Role::Assistant,
                    content: strip_tool_calls(&reply).trim().to_string(),
                    tool_calls: calls.clone(),
                    tool_call_id: None,
                    name: None,
                });
            }

            for call in &calls {
                tool_calls_made.push(format!("{}({})", call.name, call.arguments));

                // A failing tool must not kill the run: the model is told what went
                // wrong and gets a chance to adapt on the next turn.
                let content = match tools.dispatch(call).await {
                    Ok(output) => truncate_to_bytes(&output, self.config.max_tool_output_bytes),
                    Err(err) => {
                        warn!(tool = %call.name, error = %err, "tool call failed");
                        format!("tool error: {err}")
                    }
                };

                let mut conv = self.conversation.lock().await;
                conv.push(Message {
                    role: Role::Tool,
                    content,
                    tool_calls: vec![],
                    tool_call_id: Some(call.id.clone()),
                    name: Some(call.name.clone()),
                });
            }
        }
    }
}

/// Truncate `input` to at most `max_bytes` bytes without splitting a UTF-8 sequence.
///
/// Returns the input untouched when it already fits. A truncation marker is appended so
/// the model can tell clipped output from complete output.
pub(crate) fn truncate_to_bytes(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_string();
    }
    const MARKER: &str = "\n…[truncated]";

    // `char_indices` yields byte offsets of character starts, so the last offset that
    // fits the budget is always a valid slice boundary.
    let cut_at = |budget: usize| {
        input
            .char_indices()
            .map(|(idx, _)| idx)
            .take_while(|idx| *idx <= budget)
            .last()
            .unwrap_or(0)
    };

    // Too small for the marker to be honest about clipping, so hard cut without it.
    if max_bytes <= MARKER.len() {
        return input[..cut_at(max_bytes)].to_string();
    }

    let cut = cut_at(max_bytes - MARKER.len());
    let mut out = String::with_capacity(max_bytes);
    out.push_str(&input[..cut]);
    out.push_str(MARKER);
    out
}

fn is_effectively_empty(text: &str) -> bool {
    text.replace("```json", "")
        .replace("```", "")
        .chars()
        .all(|c| !c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_input_is_returned_verbatim() {
        assert_eq!(truncate_to_bytes("hello", 100), "hello");
    }

    #[test]
    fn truncation_never_splits_a_utf8_sequence() {
        // Cyrillic is two bytes per char, so a naive byte slice would panic here.
        let input = "привет мир".repeat(20);
        let out = truncate_to_bytes(&input, 32);
        assert!(out.len() <= 32, "got {} bytes", out.len());
        assert!(out.ends_with("[truncated]"));
        // Would panic if we had sliced mid-character.
        assert!(out.contains("привет") || out.len() <= 32);
    }

    #[test]
    fn truncation_of_multibyte_input_is_lossless_up_to_the_cut() {
        let input = "あ".repeat(50); // three bytes each
        let out = truncate_to_bytes(&input, 30);
        assert!(out.len() <= 30);
        assert!(out.ends_with("\n…[truncated]"));
    }

    #[test]
    fn tiny_budget_still_returns_a_valid_string() {
        let out = truncate_to_bytes("abcdefghij", 2);
        assert!(out.len() <= 2);
    }

    #[test]
    fn budget_larger_than_input_is_a_noop() {
        let out = truncate_to_bytes("abc", 9999);
        assert_eq!(out, "abc");
    }

    #[test]
    fn an_empty_fence_is_not_treated_as_an_answer() {
        assert!(is_effectively_empty("```json\n\n```"));
        assert!(is_effectively_empty("   \n "));
        assert!(is_effectively_empty(""));
        assert!(!is_effectively_empty("done"));
        assert!(!is_effectively_empty("42 files written"));
    }

    #[tokio::test]
    async fn a_blank_tool_name_ends_the_run_instead_of_dispatching() {
        struct OneShotModel(String);

        #[async_trait]
        impl Model for OneShotModel {
            fn name(&self) -> &str {
                "one-shot"
            }

            async fn chat(&self, _req: ChatRequest) -> Result<String> {
                Ok(self.0.clone())
            }
        }

        let agent = ReActAgent::new(
            Arc::new(OneShotModel(
                "```json\n{\"name\": \"\", \"arguments\": {\"path\": \"x\"}}\n```".to_string(),
            )),
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("do the thing", &ToolRegistry::new())
            .await
            .expect("a blank name must not fail the run");

        assert!(
            outcome.tool_calls.is_empty(),
            "nothing should be dispatched"
        );
        assert_eq!(outcome.iterations, 1, "the run must stop on the first turn");
        assert!(
            outcome.final_text.contains("\"name\": \"\""),
            "the raw reply must survive into final_text, got {:?}",
            outcome.final_text
        );
    }
}
