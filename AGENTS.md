# Working in this repository

## Branches, commits and review

- Make changes on a dedicated branch and open a PR. Do not push directly to
  `main` or merge without the user's explicit approval.
- Keep each PR focused on one clear purpose. Reuse existing code and directory
  structure; avoid unrelated refactoring and unnecessary documentation.
- Before handing work back or launching an evaluation, commit all code and
  configs used, including required new files, and push the branch. Do not commit
  credentials, model weights, datasets or bulky generated outputs.
- Committing and pushing preserves work; it does not authorize merging or GPU
  execution. A user-authorized smoke test may use a committed development branch.
- Run checks appropriate to the change. In the PR, explain what changed, why,
  what was tested and any remaining limitations.

## Evaluation execution

- Use this repository's existing orchestration, runners and launchers for future
  localization evaluations. Bring required fixes into version control rather
  than maintaining patched evaluator copies in experiment directories.
- Keep FindIt's native prompts, parser and scorer in its own versioned pipeline.
  Record its code revision and integration settings; do not pass its responses
  through the localization JSON decoder.
- Before submitting, inspect existing jobs, run records and outputs to avoid
  duplicate jobs. Resume existing work when appropriate.
- GPU submissions require user authorization. An already authorized campaign
  may proceed within its agreed scope without repeated permission requests.
- Validate the relevant path with appropriate checks and a small smoke test
  before a substantial evaluation. Inspect raw responses, decoded coordinates
  and scoring, not merely whether the process exited successfully.
- Do not silently change prompts, tokenizer/processor assets, output decoding,
  coordinate conventions, manifests or scoring. Explain such changes in the PR
  and identify their effect on result comparability.

## Make every evaluation findable

This is an agent workflow requirement, not a requirement to build an automatic
registry. After submitting an evaluation, add or update a small Markdown record
at `runs/<NNNN>-<descriptive-name>.md`, commit it and push the branch. Allocate
the next unused four-digit number from the run index before submission; check
the remote branches and open PRs for a number already reserved there. Never
renumber existing records. Keep retries under the same number with separate
attempts; a different checkpoint, config or manifest gets a new number.
Commit and push the prepared command before launching. Link it from `runs/README.md`
(create the index with the first record). Record failed or cancelled runs too.

Each record must contain:

- Model/checkpoint identity and location, plus tokenizer/processor identity.
- Suite, manifest identity (including hash when available), config path and
  resolved generation settings, with prompt/decoder and coordinate convention.
- Evaluator commit and revisions of external runtime dependencies used.
- Exact launch command, relevant non-secret environment settings and job IDs.
- Output directory and paths or links to predictions, scores and logs; link
  existing contracts and submission receipts rather than duplicating them.
- Current status, expected and completed coverage, errors and available scores.
  Mark anything not yet known as pending or unknown.

Update the same record after completion, failure, cancellation, repair or resume.
Preserve previous job IDs and describe changes between attempts. Keep heavy
artifacts in external storage. Historical records must distinguish verified
evidence from missing information; do not reconstruct unsupported facts.

## Protect work and report accurately

- Do not delete or overwrite checkpoints, datasets or results, or cancel
  unrelated jobs, without explicit user authorization.
- Preserve other people's uncommitted work. Do not discard changes to obtain a
  clean checkout; use an isolated checkout when necessary.
- Report actual evidence, failures, incomplete coverage and uncertainty. Never
  invent results or present a partial run as complete.
