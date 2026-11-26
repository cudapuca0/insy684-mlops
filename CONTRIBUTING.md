# Team workflow (graded!)

The commit and PR history is part of the grade. Every member must **author** PRs and **review** PRs.

## Rules

1. **Never commit to `main` directly.** Protect it: Settings → Branches → Add rule for `main`:
   - Require a pull request before merging (1 approval)
   - Require status checks to pass: `lint`, `unit-tests`, `integration-test`
   - Allow merge commits and/or rebase merging; **disable squash merging** (Settings → General)
2. **One branch per change**, named by type:
   `feat/training-flow`, `feat/grafana-alerts`, `fix/api-input-validation`, `test/drift-detection`,
   `docs/readme-screenshots`, `ci/ghcr-publish`.
3. **Open a PR** using the template: what changed and how it was tested.
4. **A teammate who is not the author reviews**, leaving real comments (a question, a suggestion, a
   line comment), not just a click on Approve. The author replies or pushes fixes.
5. **Merge** with a merge commit or rebase. Delete the branch afterwards.
6. **No secrets.** `.env` is git-ignored; CI uses GitHub Actions secrets.

## Suggested way to commit this starter code as PRs

Don't push the whole starter in one commit. Split it so the history shows real collaboration
(author → reviewer):

| # | Branch | Contents | Author → Reviewer |
|---|---|---|---|
| 1 | `chore/repo-setup` | `.gitignore`, `.env.example`, `pyproject.toml`, `config.yaml`, `CONTRIBUTING.md`, PR template | A → B |
| 2 | `feat/core-data-and-features` | `src/mlops_core/{config,data,features,modeling}.py` + data/preprocessing tests | A → C |
| 3 | `feat/training-flow` | `services/training`, `services/mlflow`, `postgres/`, compose: postgres + mlflow + prefect + training | A → D |
| 4 | `feat/model-api` | `services/api` + `test_api.py`, compose: api | B → A |
| 5 | `feat/frontend` | `services/frontend` + `test_frontend.py`, compose: frontend | B → C |
| 6 | `ci/lint-and-tests` | `.github/workflows/ci.yml`, `scripts/export_model.py`, integration test | D → B |
| 7 | `ci/publish-images` | `.github/workflows/cd.yml` | D → A |
| 8 | `feat/simulator-and-drift` | `services/simulator`, `src/mlops_core/drift.py`, `test_drift.py` | C → D |
| 9 | `feat/evidently-monitor` | `services/monitor`, `src/mlops_core/db.py`, compose: monitor | C → A |
| 10 | `feat/grafana-dashboards` | `monitoring/` (Prometheus, Grafana dashboards + alerts), compose: prometheus + grafana | C → B |
| 11 | `docs/architecture-and-readme` | `docs/architecture.md`, `README.md` | D → C |
| 12 | `feat/real-dataset` | the team's dataset + `config.yaml` changes | everyone |
| 13 | `docs/monitoring-screenshots` | screenshots of normal vs drift, alert, retrain | any → any |

Note: CI (PR 6) only starts running after it lands on `main`, so you may want to merge PR 6 early.
