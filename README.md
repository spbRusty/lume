# lume

A small agent harness that runs entirely on local Qwen2.5 weights. No hosted API, no
cloud inference — a `git clone`, a running `ollama`, and a binary.

Written in Rust, with a two-tier orchestrator and Model Context Protocol (MCP) tool
integration.

**Status: early.** The crate graph, the ReAct loop, the orchestrator, and MCP protocol
handling are implemented and covered by 161 unit tests. `lume run` has been exercised
against live `qwen2.5-coder:1.5b` and `qwen2.5-coder:7b`: the model requests `write_file`
and the file lands on disk. Sampling is non-deterministic (`temperature` 0.7 by default),
so tool-call quality varies between runs and a failed call is reported rather than hidden.
One 7B turn costs roughly 9s once the weights are resident, and ~45s on the first run while
4.7 GB loads into RAM. `lume chat` and MCP server connection are implemented and verified
against live models. See [Roadmap](#roadmap) for what actually works today.

## Why this exists

Existing harnesses assume a fast model behind an API. This one assumes the opposite:
a 6-core CPU, 15 GiB of RAM, no GPU, and weights already sitting on disk.

That constraint changes the design in three concrete ways:

1. **Two model tiers, not one.** On this hardware `qwen2.5:7b` generates roughly three
   tokens per second and `qwen2.5-coder:1.5b` is about three times faster. Spending 7B
   tokens on every decision is untenable. Lume classifies each task, runs *planning and
   decisions* on the 7B tier, and pushes *well-specified execution* down to 1.5B.
2. **Cost is a first-class budget, not an afterthought.** Token ceilings and iteration
   caps are enforced in code (`TokenBudget`, `RetryPolicy`), so a runaway loop fails
   loudly instead of melting the box.
3. **Tools are confined to a workspace root.** File tools resolve every path against a
   single root and refuse anything that escapes it, and the shell tool carries a denylist
   for the handful of commands that would ruin a machine. This is defence in depth, not a
   sandbox — see [Safety model](#safety-model).

## Architecture

```
                       ┌──────────────────────────────┐
                       │        lume (CLI)            │
                       └───────────────┬──────────────┘
                                       │
                       ┌───────────────▼──────────────┐
                       │       lume-orchestrator      │
                       │  classify → tier             │
                       │  decompose → subtasks        │
                       │  retry / token budget        │
                       └───┬──────────────────────┬───┘
                           │ Large tier           │ Small tier
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │ qwen2.5:7b  │        │  1.5b-coder │
                    └──────┬──────┘        └──────┬──────┘
                           └─────────┬───────────┘
                       ┌─────────────▼─────────────┐
                       │         lume-llm          │
                       │  ChatML renderer           │
                       │  ollama backend (HTTP)     │
                       │  llama.cpp backend (later) │
                       └─────────────┬─────────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │        lume-agent         │
                       │  ReAct loop, compaction   │
                       │  built-in fs / shell tools│
                       └─────────────┬─────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
   ┌─────────▼──────────┐                        ┌───────────▼────────┐
   │   built-in tools   │                        │     lume-mcp       │
   │ read/write/edit    │                        │ JSON-RPC 2.0       │
   │ list/grep/shell    │                        │ stdio + HTTP       │
   └────────────────────┘                        │ tools/list, call   │
                                                 └────────────────────┘
```

### Crates

| Crate | Responsibility |
|---|---|
| `lume-core` | Shared vocabulary: `Message`, `ToolCall`, `ToolSpec`, the `Model` and `Tool` traits, `LumeError`, `LumeConfig` |
| `lume-llm` | ChatML rendering for Qwen2.5, sampling parameters, the ollama HTTP backend |
| `lume-mcp` | MCP client: JSON-RPC 2.0, stdio and streamable-HTTP transports, tool registry |
| `lume-agent` | The ReAct loop, context compaction, built-in filesystem and shell tools |
| `lume-orchestrator` | Task classification, model-tier routing, decomposition, retry and token budget |
| `lume-cli` | The `lume` binary: `run`, `chat`, `models`, `doctor`, `mcp` |

Dependencies form an acyclic graph with no cycles and no back-references: `core` sits at
the bottom, `llm` and `mcp` build on it, `agent` on those, and `orchestrator` and `cli` on
top. It is not a strict chain — `orchestrator` also reaches `llm` for a backend, and `cli`
depends on every crate it needs to wire them together. The load-bearing property is that
`agent` talks to the `Model` trait rather than to any backend, so swapping ollama for
something else does not touch the loop.

## The Qwen2.5 ChatML renderer

Qwen2.5 is not a generic chat model — it expects a specific wire format, and a harness
that gets this wrong produces plausible-looking nonsense rather than an error. The
renderer in `lume-llm/src/chat_template.rs` is the piece this project exists to get
right, and it is unit-tested against exact expected strings:

```
<|im_start|>system
Available tools:
[{"name":"read_file","description":"…","input_schema":{…}}]<|im_end|>
<|im_start|>user
refactor this function<|im_end|>
<|im_start|>assistant
{"id":"call_1","name":"read_file","arguments":{"path":"src/lib.rs"}}<|im_end|>
<|im_start|>user
<tool_response>
{"ok":true}
</tool_response><|im_end|>
```

## Requirements

- Rust 1.85+ (edition 2024)
- `ollama` with the models pulled:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull qwen2.5-coder:1.5b
  ```

## Quickstart

```bash
git clone https://github.com/spbRusty/lume
cd lume
cargo build --release

# Check the runtime wiring before spending any tokens.
cargo run --release -p lume-cli -- doctor

# What is actually served by your local ollama.
cargo run --release -p lume-cli -- models

# One-shot task through the orchestrator (verified against a live qwen2.5-coder:1.5b).
cargo run --release -p lume-cli -- run "summarise the public API of src/"

# Pin both tiers to one model instead of letting the router choose.
cargo run --release -p lume-cli -- run --model qwen2.5:7b "review this diff"

# MCP servers from your config file.
cargo run --release -p lume-cli -- mcp list --config lume.toml
```

`lume chat` is an interactive REPL that reuses one `ReActAgent` across turns, so
conversation history accumulates. Type `/help` for commands, `/exit` or Ctrl-D to leave.

Configuration comes from environment variables (see `.env.example`) or a TOML file
(see `mcp.example.toml`). Environment variables win over file values, so a `LUME_*`
variable always overrides the same key in the file.

## Reproducibility

Set `LUME_SEED` and optionally `LUME_TEMPERATURE` to make runs deterministic:

```bash
LUME_SEED=42 LUME_TEMPERATURE=0 cargo run --release -p lume-cli -- run "fix the typo in README"
```

`LUME_SEED` is passed to Ollama on every model turn. `LUME_TEMPERATURE=0` disables
sampling. Both can also be set in `lume.toml`:

```toml
seed = 42
temperature = 0
```

## Safety model

- **Path confinement.** Every filesystem tool canonicalises its argument against
  `workspace_root`. Anything resolving outside it returns an error instead of a path.
- **Shell denylist, normalised before matching.** The command is lowercased and its
  whitespace collapsed before matching, and the no-space variants (`rm -fr`,
  `mkfs.ext4`) are covered alongside `rm -rf /`, `shutdown`, `reboot`, and fork bombs.
  Path-like tokens in the command are also resolved against `workspace_root` and
  rejected when they escape it.
- **Grep scope.** Recursive search skips `.git`, `target`, and `node_modules`, and caps
  output at 200 matches.
- **Bounded everything.** `max_iterations` stops the ReAct loop; `TokenBudget` stops
  generation; tool output is truncated on a character boundary so UTF-8 sequences are
  never cut in half.

**The shell tool is not a sandbox, and neither is anything else here.** Token extraction
from a shell command is a heuristic — a determined escape will get through. What you get
is defence in depth: the process runs with its working directory pinned to
`workspace_root`, the denylist catches the common footguns, and path-like arguments are
checked. That raises the cost of an accident; it does not prevent one. If you point this
at a machine you care about, keep the workspace root narrow and do not run it as root.

## Roadmap

Roughly in order:

- [ ] Ollama backend verified against `qwen2.5:7b` (it is verified against
      `qwen2.5-coder:1.5b` and `qwen2.5-coder:7b`)
- [x] End-to-end ReAct run: prompt → tool call → tool result → effect on disk
- [ ] A natural-language closing answer — neither local model reliably ends with prose;
      1.5b emits an empty code fence and 7b an empty tool call, so `final_text` falls back
      to a notice
- [ ] Tighter tool discipline — both models repeat a tool call that already succeeded,
      costing an extra round trip
- [x] `lume chat` — the interactive loop with history accumulation
- [x] Connect MCP servers for real — stdio and HTTP transports, negotiation, tool registry,
      and live `mcp probe` verified against `@modelcontextprotocol/server-filesystem`
- [ ] Real token accounting — `TokenBudget` currently estimates from the final answer text
      rather than reading usage from the backend, so it ignores prompts and tool output
- [ ] `openai_compatible` backend — llama.cpp, vLLM, LM Studio, and llamafile all speak
      this protocol, so one backend covers every non-ollama local runtime and lets the
      harness talk to a GGUF directly without a daemon in the middle
- [ ] llama.cpp FFI backend for in-process inference, bypassing HTTP entirely
- [ ] Streaming wired through the CLI for incremental output
- [ ] Smarter routing — replace the deterministic heuristic with a cheap classifier
- [ ] Conversation persistence and session resume
- [ ] Integration tests against recorded transcripts

## Contributing

Issues and pull requests are welcome. Please keep the dependency surface small — the
point of the project is to run cheaply on modest hardware, and every transitive
dependency is paid for in build time and binary size.

## License

MIT — see [LICENSE](LICENSE).