//! `lume chat`: one interactive session driven by one agent.
//!
//! The orchestrator is the wrong tool here, and the reason is worth writing down
//! because it is the whole point of this module. `Orchestrator::execute` builds a
//! fresh [`ReActAgent`] per subtask and returns the last subtask's text, so every
//! call starts with an empty [`Conversation`]. A chat session is the opposite of
//! that: turn two has to know what turn one did, so the agent is built once, here,
//! and the same instance serves every line the user types. The conversation lives
//! inside the agent's `Mutex<Conversation>`, which is why holding onto one agent is
//! enough to accumulate history.
//!
//! Everything that does not need a model is a small pure function: line
//! classification and the footer are testable without stdin, and the read loop takes
//! its "run one turn" step as a callback so the whole loop can be driven from a
//! script.
//!
//! Two properties of the loop are load-bearing:
//!
//! * **EOF ends the session.** `read_line` returns `Ok(0)` forever once the stream
//!   is exhausted. A loop that only checked `line.trim().is_empty()` would treat
//!   that as an empty line and spin at full speed forever, which is exactly what a
//!   piped `printf` or a closed heredoc produces. The byte count is checked
//!   explicitly.
//! * **A failed turn does not end the session.** A model error is reported and the
//!   prompt comes back; the history stays exactly as it was.

use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};

use lume_agent::{Agent, AgentConfig, AgentOutcome, Conversation, ReActAgent, ToolRegistry};
use lume_core::config::LumeConfig;
use lume_core::types::SamplingParams;
use lume_llm::OllamaBackend;
use lume_mcp::McpServerConnection;

/// Drawn before every read.
const PROMPT: &str = "you";

/// What `/help` prints. Kept next to the prompt so the two cannot drift apart.
const HELP: &str = "\
commands:
  /help            print this list
  /exit, /quit     leave the session (Ctrl-D or EOF does the same)
anything else is sent to the model";

/// A slash command the loop answers itself.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// Leave the session.
    Exit,
    /// List the commands.
    Help,
}

/// What one line of input means.
#[derive(Debug, PartialEq, Eq)]
enum Line<'a> {
    /// Nothing but whitespace: never sent to the model, never an error either.
    Blank,
    /// A slash command this session knows.
    Command(Command),
    /// A slash word this session does not know. Kept as the user typed it, minus
    /// the slash, so the message can quote it back.
    Unknown(&'a str),
    /// A prompt for the model, with surrounding whitespace removed.
    Prompt(&'a str),
}

/// Classify one line of input.
///
/// A line whose first non-blank character is a slash is a command, whatever else it
/// looks like. That is the usual REPL convention, and it is also the safe direction
/// to be wrong in: an unknown command is reported back and never reaches the model,
/// whereas guessing wrong the other way would send `/help` to the model as a prompt
/// and leave the user talking to something that cannot talk back. The word is
/// matched case-insensitively and tolerates a space after the slash, because a
/// mistyped command that silently went to the model would be worse than one that did
/// not.
fn classify(line: &str) -> Line<'_> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Line::Blank;
    }
    let Some(rest) = trimmed.strip_prefix('/') else {
        return Line::Prompt(trimmed);
    };
    // Only the first word matters, so `/exit now` is still `/exit`.
    let word = rest.split_whitespace().next().unwrap_or_default();
    if word.eq_ignore_ascii_case("exit") || word.eq_ignore_ascii_case("quit") {
        return Line::Command(Command::Exit);
    }
    if word.eq_ignore_ascii_case("help") {
        return Line::Command(Command::Help);
    }
    Line::Unknown(word)
}

/// The per-turn footer, so the cost of a turn is visible without a debug log.
fn footer(iterations: usize, tool_calls: usize) -> String {
    format!("--- {iterations} iteration(s), {tool_calls} tool call(s) ---")
}

/// Print `text` to `out` followed by a blank line.
///
/// Model answers are separated from the prompt that asked for them, so a transcript
/// piped through a file still reads as a dialogue.
fn write_answer(out: &mut dyn Write, text: &str) -> Result<()> {
    writeln!(out, "\n{text}")?;
    Ok(())
}

/// The read loop.
///
/// `run` is the whole reason this is not written inline in [`chat`]: it takes one
/// prompt and reports what the model did, so the loop can be exercised against a
/// stub instead of a live Ollama. The agent that backs it is built once by the
/// caller and borrowed by every call, which is what makes the turns share a
/// history.
async fn repl<F, Fut>(
    input: &mut (impl AsyncBufRead + Unpin),
    out: &mut dyn Write,
    mut run: F,
) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<AgentOutcome>>,
{
    let mut line = String::new();
    loop {
        write!(out, "\n{PROMPT}> ")?;
        out.flush()?;

        line.clear();
        let read = input
            .read_line(&mut line)
            .await
            .context("cannot read from stdin")?;
        if read == 0 {
            // End of stream, not an empty line. Without this check the loop would
            // re-read the exhausted stream forever.
            writeln!(out)?;
            return Ok(());
        }

        match classify(&line) {
            Line::Blank => continue,
            Line::Command(Command::Exit) => return Ok(()),
            Line::Command(Command::Help) => write_answer(out, HELP)?,
            Line::Unknown(word) => {
                write_answer(out, &format!("unknown command: /{word}"))?;
                writeln!(out, "try /help")?;
            }
            Line::Prompt(prompt) => match run(prompt.to_string()).await {
                Ok(outcome) => {
                    write_answer(out, &outcome.final_text)?;
                    writeln!(
                        out,
                        "{}",
                        footer(outcome.iterations, outcome.tool_calls.len())
                    )?;
                }
                Err(err) => {
                    // Reported, not fatal: the next prompt must still come up, with
                    // the history intact, so the user can retry or change course.
                    write_answer(out, &format!("error: {err}"))?;
                }
            },
        }
        out.flush()?;
    }
}

/// Chat with the model until the user leaves.
///
/// Reads one line at a time from stdin, answers it with the same agent every time so
/// the conversation accumulates, and closes every MCP connection before returning —
/// including when a turn failed or stdin died, because the loop's result is kept and
/// the shutdown runs on the way out instead of being skipped by `?`.
pub async fn chat(
    model: String,
    cfg: &LumeConfig,
    tools: ToolRegistry,
    connections: Vec<McpServerConnection>,
) -> Result<()> {
    let agent = ReActAgent::new(
        Arc::new(OllamaBackend::new(&cfg.ollama_url, &model)),
        Conversation::new(cfg.context_window),
        AgentConfig {
            max_iterations: cfg.max_iterations,
            // The last link in the sampling chain: `LUME_TEMPERATURE` and
            // `LUME_SEED` land here and, through `AgentConfig::params`, on every
            // request the agent makes. Nothing else reads these two config fields.
            params: SamplingParams {
                temperature: cfg.temperature,
                seed: cfg.seed,
                ..SamplingParams::default()
            },
            ..AgentConfig::default()
        },
    );

    println!(
        "lume chat · model {model} · {} tool(s) · temperature {} · seed {}",
        tools.specs().len(),
        temperature_label(cfg.temperature),
        seed_label(cfg.seed),
    );
    println!("type /help for commands, /exit or Ctrl-D to leave");

    let tools = &tools;
    let agent = &agent;
    let mut input = BufReader::new(tokio::io::stdin());
    let session = repl(
        &mut input,
        &mut std::io::stdout(),
        move |prompt| async move { agent.run(&prompt, tools).await.map_err(anyhow::Error::from) },
    )
    .await;

    close_all(&connections).await;
    session
}

/// How the configured temperature is shown in the banner.
///
/// `default` is the honest label: an unset temperature leaves Ollama's own default
/// in place, and printing `0` for it would claim a setting nobody made.
fn temperature_label(temperature: Option<f32>) -> String {
    match temperature {
        Some(value) => value.to_string(),
        None => "default".to_string(),
    }
}

/// How the configured seed is shown in the banner.
fn seed_label(seed: Option<u64>) -> String {
    match seed {
        Some(value) => value.to_string(),
        None => "unset".to_string(),
    }
}

/// Close every connection and kill its server process.
///
/// Closing is not a formality: the registry hands the same clients out as `Arc`s,
/// so a dropped connection does not stop the child process. Errors are dropped too,
/// because a server that refuses to shut down cleanly must not turn a good session
/// into a failed exit code.
async fn close_all(connections: &[McpServerConnection]) {
    for connection in connections {
        let _ = connection.close().await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Command, Line, classify, footer, repl, seed_label, temperature_label};
    use lume_agent::AgentOutcome;

    /// A turn that took `iterations` model calls and made `tool_calls` tool calls.
    fn outcome(final_text: &str, iterations: usize, tool_calls: usize) -> AgentOutcome {
        AgentOutcome {
            final_text: final_text.to_string(),
            iterations,
            tool_calls: (0..tool_calls).map(|n| format!("tool_{n}()")).collect(),
        }
    }

    /// Run a whole session over `input` and report what the model was asked, plus
    /// everything the loop printed.
    ///
    /// `turns` supplies one answer per prompt; `None` fails that turn. The timeout
    /// is the important part: a loop that forgot to break on EOF would hang, and a
    /// hang would stall the whole test run instead of failing one test.
    async fn session(input: &str, turns: &[Option<&str>]) -> (Vec<String>, String) {
        let mut asked: Vec<String> = Vec::new();
        let mut replies = turns.iter();
        let mut transcript: Vec<u8> = Vec::new();
        {
            let out = &mut transcript;
            tokio::time::timeout(Duration::from_secs(10), async {
                let mut stream = input.as_bytes();
                repl(&mut stream, out, |prompt| {
                    asked.push(prompt.clone());
                    let reply = replies.next().copied().flatten();
                    async move {
                        match reply {
                            Some(text) => Ok(outcome(text, 1, 1)),
                            None => Err(anyhow::anyhow!("the model is unreachable")),
                        }
                    }
                })
                .await
            })
            .await
            .expect("the session must end rather than spin on an exhausted stdin")
            .expect("the session itself must not fail");
        }
        (
            asked,
            String::from_utf8(transcript).expect("a transcript of our own text"),
        )
    }

    #[test]
    fn a_line_of_only_whitespace_is_nothing_to_send() {
        assert_eq!(classify(""), Line::Blank);
        assert_eq!(classify("   "), Line::Blank);
        assert_eq!(classify("\t \n  "), Line::Blank);
    }

    #[test]
    fn a_prompt_is_a_line_without_a_leading_slash() {
        assert_eq!(classify("list the files"), Line::Prompt("list the files"));
        assert_eq!(
            classify("  read notes.md and summarise it  "),
            Line::Prompt("read notes.md and summarise it"),
            "surrounding whitespace must not reach the model"
        );
        assert_eq!(classify("42"), Line::Prompt("42"));
    }

    #[test]
    fn a_slash_word_is_a_command_however_it_is_spelled() {
        for line in [
            "/exit",
            "/EXIT",
            "  /exit  ",
            "/ exit",
            "/exit now",
            "/Exit",
        ] {
            assert_eq!(
                classify(line),
                Line::Command(Command::Exit),
                "line: {line:?}"
            );
        }
        assert_eq!(classify("/quit"), Line::Command(Command::Exit));
        assert_eq!(classify("/QUIT"), Line::Command(Command::Exit));
        for line in ["/help", "/Help", "  /HELP  ", "/ help me"] {
            assert_eq!(
                classify(line),
                Line::Command(Command::Help),
                "line: {line:?}"
            );
        }
    }

    #[test]
    fn an_unknown_slash_word_is_reported_rather_than_guessed_at() {
        assert_eq!(classify("/frobnicate"), Line::Unknown("frobnicate"));
        assert_eq!(classify("/Frobnicate now"), Line::Unknown("Frobnicate"));
        assert_eq!(
            classify("/"),
            Line::Unknown(""),
            "a bare slash is unknown, not an empty prompt"
        );
        assert_eq!(
            classify("/etc/hosts is broken"),
            Line::Unknown("etc/hosts"),
            "a path typed at the start of a line is an unknown command, not a prompt"
        );
    }

    #[test]
    fn the_footer_counts_this_turn_only() {
        assert_eq!(footer(1, 0), "--- 1 iteration(s), 0 tool call(s) ---");
        assert_eq!(footer(3, 2), "--- 3 iteration(s), 2 tool call(s) ---");
    }

    #[test]
    fn the_banner_does_not_invent_a_sampling_setting() {
        assert_eq!(temperature_label(None), "default");
        assert_eq!(temperature_label(Some(0.0)), "0");
        assert_eq!(temperature_label(Some(0.25)), "0.25");
        assert_eq!(seed_label(None), "unset");
        assert_eq!(seed_label(Some(0)), "0");
        assert_eq!(seed_label(Some(987_654_321)), "987654321");
    }

    #[tokio::test]
    async fn a_blank_line_never_reaches_the_model() {
        let (asked, _) = session("\n   \n\t\n\nwrite a haiku\n/exit\n", &[Some("ok")]).await;

        assert_eq!(
            asked,
            vec!["write a haiku".to_string()],
            "only the one real prompt may be sent"
        );
    }

    #[tokio::test]
    async fn an_immediately_exhausted_stdin_ends_the_session() {
        // What `printf '' | lume chat` does. The empty result is the whole point:
        // this is the case a naive `read_line` loop spins on forever.
        let (asked, transcript) = session("", &[]).await;

        assert!(asked.is_empty(), "no prompt, no model call");
        assert_eq!(
            transcript.matches("you> ").count(),
            1,
            "exactly one prompt may be drawn: {transcript:?}"
        );
    }

    #[tokio::test]
    async fn a_last_line_without_a_newline_runs_and_then_eof_ends_the_session() {
        // No trailing newline: `read_line` still returns the text, and the next read
        // returns 0, so the session must answer the line and then leave.
        let (asked, transcript) = session("hello", &[Some("hi there")]).await;

        assert_eq!(asked, vec!["hello".to_string()]);
        assert!(transcript.contains("hi there"), "{transcript:?}");
    }

    #[tokio::test]
    async fn a_failing_turn_is_reported_and_the_session_continues() {
        let (asked, transcript) = session(
            "first\nsecond\nthird\n/exit\n",
            &[Some("answer-one"), None, Some("answer-three")],
        )
        .await;

        assert_eq!(
            asked,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ],
            "the turn after a failure must still be asked"
        );
        assert!(
            transcript.contains("error: the model is unreachable"),
            "the failure must be reported: {transcript:?}"
        );
        assert!(transcript.contains("answer-one"), "{transcript:?}");
        assert!(transcript.contains("answer-three"), "{transcript:?}");
    }

    #[tokio::test]
    async fn exit_and_quit_both_end_the_session_without_asking_the_model() {
        for command in ["/exit", "/quit", "  /EXIT  "] {
            let (asked, transcript) = session(&format!("{command}\nnever asked\n"), &[]).await;

            assert!(asked.is_empty(), "{command} must not reach the model");
            assert!(
                !transcript.contains("never asked"),
                "{command} must stop the loop: {transcript:?}"
            );
        }
    }

    #[tokio::test]
    async fn help_and_an_unknown_command_are_answered_here_not_by_the_model() {
        let (asked, transcript) = session("/help\n/frobnicate\n/exit\n", &[]).await;

        assert!(asked.is_empty(), "neither line is a prompt");
        assert!(transcript.contains("/exit"), "help must list the commands");
        assert!(transcript.contains("unknown command: /frobnicate"));
    }

    #[tokio::test]
    async fn turns_are_answered_in_the_order_they_were_typed() {
        let (asked, transcript) = session(
            "one\ntwo\nthree\n/exit\n",
            &[Some("answer-one"), Some("answer-two"), Some("answer-three")],
        )
        .await;

        assert_eq!(
            asked,
            vec!["one".to_string(), "two".to_string(), "three".to_string()],
            "one runner, asked once per prompt, in order"
        );
        let first = transcript.find("answer-one").expect("first answer printed");
        let second = transcript
            .find("answer-two")
            .expect("second answer printed");
        let third = transcript
            .find("answer-three")
            .expect("third answer printed");
        assert!(first < second && second < third, "{transcript:?}");
    }

    #[tokio::test]
    async fn every_turn_is_answered_and_costed() {
        let (asked, transcript) = session("go\n/exit\n", &[Some("done")]).await;

        assert_eq!(asked.len(), 1);
        assert!(transcript.contains("done"), "{transcript:?}");
        assert!(
            transcript.contains(&footer(1, 1)),
            "the footer must reach the transcript: {transcript:?}"
        );
    }
}
