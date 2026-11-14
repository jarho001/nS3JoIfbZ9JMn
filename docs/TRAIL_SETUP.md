# Trail-code setup

Trail is required before project work on local machines, cloud workers and
worktrees. AGENTS.md owns this policy. Each checkout installs its own pinned
CLI/server from `tools/trail-code/package-lock.json`; a global installation and
copied caches are unnecessary. Node.js 22.3+ (or 20.16+) and npm are required.

## One setup command

Run from the repository root, including in a cloud environment's setup script:

```sh
node scripts/setup-trail.cjs
```

Setup installs the locked dependencies, skips ONNX CUDA downloads, and builds
the deterministic source graph without semantic models or LLM calls. The graph
is local and ignored by Git. Query commands refresh it when source changes.
No simulator tests, GPU validation or profiling run during setup.

The setup command is safe to repeat after changing the pin or recreating an
environment. It needs package-registry access for dependency installation.
Configure the same command in each saved cloud environment's setup/startup
settings for automatic provisioning. If that setting is absent, agents must
run setup themselves before proceeding, as required by AGENTS.md. This
repository cannot change a host's saved environment settings by itself.

## Verify and use

```sh
node scripts/check-trail.cjs
node scripts/trail.cjs ask "<current task>"
node scripts/trail.cjs callers absorbs
node scripts/trail.cjs status
```

The preflight checks the pinned version, runs a real query against current
source, and starts the MCP server to verify initialization, tool discovery and
a real code query. A successful `--version` or an existing cache alone is not
sufficient. Do not silently continue project work if verification fails.
The Node entrypoints work on Linux, macOS and Windows; Windows execution is
not validated in this cloud environment. Bash wrappers remain available for
existing shell commands. The launcher resolves the repository from its own
location, so an absolute path to `scripts/trail.cjs` also works when called outside the repository.

## Agent hosts

Project MCP configuration is committed in `.codex/config.toml` (Codex) and
`.mcp.json` (hosts supporting that format, including Claude Code). Launch the
server with the repository as the working directory:

```sh
node scripts/trail.cjs mcp
```

Other hosts can register that same stdio command with the repository as cwd.
Restart/reload the host after changing MCP configuration and grant its normal
project trust/tool access if requested. This does not inject tools into an
already-running chat. Until the host exposes MCP, use the same pinned CLI;
Trail remains required and usable through the terminal.

CLAUDE.md points to AGENTS.md for the shared policy. No automatic git hooks,
branches, pushes, deep summaries or optional semantic builds are installed.
The launcher opts out of telemetry/update checks with `DO_NOT_TRACK=1`.

Historical `.trail/learnings` files are retained as project evidence; they do
not replace the live source graph. Trail's current package does not create
those older learning/hook files. On a new machine, build a fresh local cache.

In restricted cloud agent sandboxes, setup/preflight may need the host
permission mechanism to spawn npm or the test MCP process. Retry the exact
setup/preflight with that permission if denied; do not bypass Trail. Normal
queries use the Node launcher in-process.
