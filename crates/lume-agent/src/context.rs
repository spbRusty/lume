//! Conversation context management.

use lume_core::types::{Message, Role};

/// Conversation context.
///
/// Holds the message history plus the window budget it must fit inside. Compaction
/// notes are tracked separately from the user's system prompt so that repeated
/// compaction never degrades the instructions themselves.
#[derive(Debug, Clone)]
pub struct Conversation {
    messages: Vec<Message>,
    system: Option<String>,
    compaction_note: Option<String>,
    context_window: usize,
}

impl Conversation {
    /// Create new conversation.
    pub fn new(context_window: usize) -> Self {
        Self {
            messages: Vec::new(),
            system: None,
            compaction_note: None,
            context_window,
        }
    }

    /// Set system message.
    pub fn set_system(&mut self, s: impl Into<String>) {
        self.system = Some(s.into());
    }

    /// The system prompt, if one was set.
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// The configured context window in tokens.
    pub fn context_window(&self) -> usize {
        self.context_window
    }

    /// Push a message.
    pub fn push(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Get message count.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get messages (including system if present).
    pub fn messages(&self) -> Vec<Message> {
        let mut out = Vec::new();
        if let Some(ref s) = self.system {
            out.push(system_message(s.clone()));
        }
        if let Some(ref n) = self.compaction_note {
            out.push(system_message(n.clone()));
        }
        out.extend(self.messages.iter().cloned());
        out
    }

    /// Estimate tokens using heuristic (chars / 4).
    pub fn estimate_tokens(&self) -> usize {
        let mut total = 0usize;
        if let Some(ref s) = self.system {
            total += s.chars().count();
        }
        if let Some(ref n) = self.compaction_note {
            total += n.chars().count();
        }
        for m in &self.messages {
            total += m.content.chars().count();
            for tc in &m.tool_calls {
                total += tc.name.chars().count();
                total += tc.arguments.to_string().chars().count();
            }
        }
        total / 4
    }

    /// Drop all but the last `keep_recent` messages, recording what was lost.
    ///
    /// The user's system prompt is never modified. Dropped-message bookkeeping goes
    /// into a separate compaction note, so calling this repeatedly accumulates the
    /// history of what was pruned instead of corrupting the instructions.
    ///
    /// Returns `true` when at least one message was dropped.
    pub fn compact(&mut self, keep_recent: usize) -> bool {
        if self.messages.len() <= keep_recent {
            return false;
        }
        let dropped = self.messages.len() - keep_recent;
        let start = self.messages.len() - keep_recent;
        self.messages.drain(..start);

        let note = format!("[compacted: {dropped} earlier message(s) dropped from history]");
        self.compaction_note = Some(match self.compaction_note.take() {
            Some(prev) => format!("{prev}\n{note}"),
            None => note,
        });
        true
    }

    /// Compact repeatedly until the estimated token count fits the context window.
    ///
    /// Each pass halves the number of retained messages. Returns `true` if anything was
    /// dropped. If the window cannot be met even with an empty history — for example a
    /// very large system prompt — this gives up rather than looping forever.
    pub fn compact_to_fit(&mut self) -> bool {
        if self.estimate_tokens() <= self.context_window {
            return false;
        }
        let mut compacted = false;
        let mut keep = self.messages.len();
        while self.estimate_tokens() > self.context_window {
            keep /= 2;
            if !self.compact(keep) {
                break;
            }
            compacted = true;
            if keep == 0 {
                break;
            }
        }
        compacted
    }
}

fn system_message(content: String) -> Message {
    Message {
        role: Role::System,
        content,
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: text.to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn successive_compactions_leave_system_prompt_untouched() {
        let mut c = Conversation::new(4096);
        c.set_system("You are a careful engineer.");
        let original = c.system().expect("system set").to_string();

        for i in 0..20 {
            c.push(user(&format!("message {i}")));
        }

        assert!(c.compact(4));
        assert!(c.compact(2));
        assert!(c.compact(1));

        assert_eq!(
            c.system(),
            Some("You are a careful engineer."),
            "system prompt must survive repeated compaction byte-identically"
        );
        assert_eq!(original, "You are a careful engineer.");
    }

    #[test]
    fn compact_keeps_the_most_recent_messages() {
        let mut c = Conversation::new(4096);
        for i in 0..10 {
            c.push(user(&format!("m{i}")));
        }
        assert!(c.compact(3));
        assert_eq!(c.len(), 3);

        let rendered = c.messages();
        let tail: Vec<&str> = rendered
            .iter()
            .rev()
            .take(3)
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(tail, ["m9", "m8", "m7"]);
    }

    #[test]
    fn compact_reports_false_when_nothing_to_drop() {
        let mut c = Conversation::new(4096);
        c.push(user("only one"));
        assert!(!c.compact(5));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn compaction_note_accumulates_instead_of_overwriting() {
        let mut c = Conversation::new(4096);
        for i in 0..10 {
            c.push(user(&format!("m{i}")));
        }
        c.compact(4);
        let after_first = c.messages().len();
        c.compact(2);
        let after_second = c.messages().len();

        // No system prompt was set here, so `messages()` is the compaction note plus
        // the retained history: 1 + 4 after the first prune, 1 + 2 after the second.
        assert_eq!(after_first, 5);
        assert_eq!(after_second, 3);

        let rendered = c.messages();
        let notes: Vec<&str> = rendered
            .iter()
            .filter(|m| m.role == Role::System && m.content.starts_with("[compacted"))
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(notes.len(), 1, "repeated pruning merges into one note");
        assert!(
            notes[0].contains("6 earlier"),
            "first prune recorded: {}",
            notes[0]
        );
        assert!(
            notes[0].contains("2 earlier"),
            "second prune recorded: {}",
            notes[0]
        );
    }

    #[test]
    fn compact_to_fit_is_a_noop_when_already_inside_window() {
        let mut c = Conversation::new(8192);
        c.push(user("short"));
        assert!(!c.compact_to_fit());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn compact_to_fit_shrinks_history_to_meet_the_window() {
        // A tiny window forces real compaction.
        let mut c = Conversation::new(40);
        for i in 0..40 {
            c.push(user(&format!("message number {i} with some padding text")));
        }
        let before = c.estimate_tokens();
        assert!(before > 40, "precondition: over budget");

        assert!(c.compact_to_fit());
        assert!(
            c.estimate_tokens() <= 40 || c.is_empty(),
            "expected to fit the window, got {} tokens from {} messages",
            c.estimate_tokens(),
            c.len()
        );
    }

    #[test]
    fn compact_to_fit_gives_up_when_window_is_unreachable() {
        // A system prompt larger than the whole window: must terminate, not hang.
        let mut c = Conversation::new(10);
        c.set_system("s".repeat(400));
        c.push(user("hello"));
        let _ = c.compact_to_fit();
        assert_eq!(
            c.len(),
            0,
            "history should be emptied in a last-ditch attempt"
        );
    }

    #[test]
    fn estimate_tokens_grows_with_content() {
        let mut c = Conversation::new(4096);
        assert_eq!(c.estimate_tokens(), 0);
        c.push(user(&"x".repeat(400)));
        assert_eq!(c.estimate_tokens(), 100);
    }
}
