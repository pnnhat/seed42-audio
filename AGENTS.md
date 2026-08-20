# AGENTS.md

Instructions for AI coding assistants (and a useful reference for people) working in
this repository. Any assistant that supports this file should read it before making
changes. Merge with your own judgement on trivial tasks.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgement.

---

## Project: seed42 audio analysis module

**What this is.** A standalone Python module that analyses live or recorded audio in real
time and outputs a classification (later, a text prompt plus generation parameters) that
seed42's visual engine can read. We build the analysis and mapping layer only, not the
renderer. seed42 already owns the rendering pipeline.

**Hard constraints:**
- Causal only. Code may use only the past X seconds of audio, never the whole track and
  never future samples. All audio reaches the rest of the code through `io/stream.py` in
  segments. Do not load a whole file and analyse it as one piece.
- Must run efficiently enough to repeat every X seconds on a live feed. Prefer simple,
  fast operations over heavy ones.
- Phase 1 uses trained and off-the-shelf models (librosa, Essentia). Do NOT add the
  Phase 2 statistical methods yet (point-process periodogram, modified least squares,
  change-point detection, Kalman fusion, energy forecasting). Those come only after
  Phase 1 works.
- The shared output is the `music_state` object. Keep that interface stable so later work
  is a drop-in, not a rewrite.

**Environment:**
- Windows members: plain `essentia` will not install (no Windows wheel). Use
  `essentia-tensorflow`, or defer Essentia and do the librosa work first. librosa,
  numpy, scipy, soundfile and pyyaml install everywhere.
- Use the project virtual environment (`.venv`). Never commit `.venv`.
- Never commit audio files, API keys, credentials, or anything seed42 shares in
  confidence. See `.gitignore`.

**Workflow:**
- Work on a feature branch, open a pull request into `main`, one review (Hatim, Quality
  Manager) before merge. No direct pushes to `main`.
- Keep pull requests small and single-purpose. Commit messages say what changed and why,
  not just the file name.

---

## Style

- No em dashes anywhere: not in code, comments, docstrings, commit messages, or generated
  docs. Use commas, colons, parentheses, or separate sentences instead. This applies to en
  dashes used as em dashes too. Hyphens for ranges (for example 7-16) and in identifiers
  are fine.
- Quick check before a merge: `grep -rn "—" src/ scripts/ tests/` should return nothing.
- Match the existing code style in the file you are editing, even if you would do it
  differently.

---

## 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

## 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it. Do not delete it.

When your changes create orphans:
- Remove imports, variables, and functions that YOUR changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the request.

## 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" becomes "Write tests for invalid inputs, then make them pass."
- "Fix the bug" becomes "Write a test that reproduces it, then make it pass."
- "Refactor X" becomes "Ensure tests pass before and after."

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require
constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due
to overcomplication, no em dashes slipping into the code, and clarifying questions come
before implementation rather than after mistakes.
