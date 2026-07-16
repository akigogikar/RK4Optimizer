# arXiv submission — RK-Adam paper

**Status: ready.** Source compiles clean in a fresh clean-room (10 pages), all
11 arXiv LaTeX gotchas pass, tarball is 0.08 MB, and every metadata field is
plain ASCII with no arXiv-rejected characters. This file is the single source of
truth for the submission — the metadata below is pulled verbatim from
`submission.yaml`.

## Files
| File | Purpose |
|---|---|
| `arxiv_upload.tar.gz` | The source tarball to upload (main.tex, main.bbl, 4 figs, style files) |
| `submission.yaml` | Metadata (used by the co-pilot script; also copy-paste source below) |
| `paper/main.pdf` | Local render — what readers will see |

## Copy-paste metadata

> **IMPORTANT:** copy from THIS file (plain ASCII), never from the PDF. The PDF
> renders "Runge-Kutta"/"RK-Adam" with Unicode en-dashes and the abstract with
> math symbols (%, subscripts, ~) that arXiv's metadata form rejects as "bad
> characters."

**Title**
```
Adaptive Runge-Kutta Step Control Buys Training Loss, Not Generalization: An Honest Compute-Matched Study of RK-Adam Optimizers
```

**Authors**
```
Akhilesh Gogikar
```

**Abstract** (1906 chars — under arXiv's 1920 limit)
```
Interpreting optimizers as gradient-flow discretizations has motivated applying higher-order Runge-Kutta (RK) integrators to neural networks. We build a representative Adam variant (Bogacki-Shampine 3(2) RK pair, FSAL reuse, local-error step control) and evaluate it under a strict compute-matched protocol giving every method the same gradient-evaluation budget - an accounting this literature rarely enforces. Under it the RK variant loses to plain Adam on training loss in both minibatch and full-batch (RK's best-case) training. Instrumenting it shows the "adaptivity" is illusory: normalized error stays far below tolerance, the step size pins at its growth cap from step one (98-100 percent of steps), and no rtol x hmax x h0 setting makes it act; tolerances spanning 100x give bit-identical trajectories. The method is exactly fixed-step Adam with an averaged gradient at 3-4x cost. Repairing it (true reject branch; error on the applied map) reverses the full-batch result - about 40x lower training loss than tuned Adam - and a fixed-step control isolates adaptivity (an emergent warmup-and-growth schedule) as the mechanism. But the gain is fragile to the initial step size and does not reach test accuracy. A pre-registered follow-up rules out the obvious explanations: deeper minimization does not overfit, and an explicit temperature knob only hurts - leaving a trajectory effect, the controller selecting a minimum generalizing 1.3-3.4 points below first-order descent at equal depth. An n=10 study confirms one secondary effect: gradient averaging is a genuine implicit regularizer, beating lr-matched Adam and AdamW on 10/10 seeds - yet RMSprop and NAdam match or beat it at a third the per-step cost. Higher-order adaptive integration buys deeper deterministic minimization and a small regularization effect, but nothing a cheaper, well-tuned first-order baseline does not already provide.
```

**Comments**
```
10 pages, 4 figures. Code, logs, and result JSONs: https://github.com/akigogikar/RK4Optimizer
```

**Categories** — primary `cs.LG`; cross-list `stat.ML, math.OC` (set by hand in arXiv's picker)

**ACM class** `G.1.7; I.2.6`   |   **License** `CC BY 4.0` (chosen in the browser)

## How to submit

Two ways — both need YOU to log in (credentials never leave your browser).

### A. Co-pilot script (recommended)
```
cd /Users/akhileshgogikar/RK4Optimizer
python3 ~/.claude/skills/publish-to-arxiv/scripts/arxiv_submit.py \
  --metadata submission.yaml --tarball arxiv_upload.tar.gz --dry-run
```
A visible browser opens; you log in, and it walks each stage, auto-filling what
it can and pausing for you. `--dry-run` stops at the preview and submits nothing.
Swap `--dry-run` for `--submit` when the preview looks right.

### B. By hand at https://arxiv.org/submit
1. **Start a New Submission** -> accept the submission policy.
2. **License:** pick `CC BY 4.0` (permanent; CC BY 4.0 is the common open-access choice).
3. **Upload** `arxiv_upload.tar.gz`. Wait for a SUCCESSFUL compile; check the
   generated PDF's figures/refs. Do not proceed on a failed compile.
4. **Metadata:** paste Title / Authors / Abstract / Comments from above.
   (arXiv's form labels didn't match the script's auto-fill for Authors/Abstract/
   ACM last run — just paste those by hand.)
5. **Categories:** primary `cs.LG`, cross-list `stat.ML, math.OC`. ACM-class `G.1.7; I.2.6`.
6. **Preview:** read it as a reader would. Submit.

## Endorsement (line this up EARLY — it can block for days)

As a first-time submitter to `cs.LG` you may need an endorsement from an
established author in that category. If arXiv asks, it gives you a 6-char code at
https://arxiv.org/auth/need-endorsement . Send a colleague who has published in
`cs.LG` this note (fill the [brackets] and the [CODE]):

> **Subject:** arXiv endorsement request — cs.LG (Akhilesh Gogikar)
>
> Dear [Dr./Prof. Endorser],
>
> [One line on who you are / how you know them.]
>
> I'm about to submit my first paper to the **cs.LG** section of arXiv and, as a
> first-time submitter there, need an endorsement from an established author.
> Would you be willing to endorse me?
>
> The paper is:
> > **"Adaptive Runge-Kutta Step Control Buys Training Loss, Not Generalization: An Honest Compute-Matched Study of RK-Adam Optimizers"**
> > A compute-matched study showing that adaptive Runge-Kutta step control for
> > Adam-style optimizers buys training-loss but not generalization, and that a
> > representative method's "adaptive" controller is inert until repaired.
>
> Endorsing only confirms I'm a legitimate researcher submitting appropriate
> work — it is not a review and takes about a minute:
> > https://arxiv.org/auth/endorse?x=[CODE]
>
> Happy to send the PDF first. No pressure at all.
>
> Best regards,
> Akhilesh Gogikar — Independent Researcher — https://github.com/akigogikar/RK4Optimizer

## After you submit
- Paper enters a **moderation hold**, gets an identifier, and announces on the
  next mailing (submit before 14:00 US Eastern Mon-Fri to announce that evening).
- Revisions and adding a journal-ref/DOI later are done as a **replacement**
  (v2, v3) under the same identifier — never a fresh submission.
