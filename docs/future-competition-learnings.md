# March Machine Learning Mania 2026: Retrospective and Future Competition Learnings

This note is meant to do two jobs at once: capture what we actually shipped for March Machine Learning Mania 2026, and turn the top-3 Kaggle writeups into a practical playbook for future competitions. We finished roughly in the middle of the pack, so the goal here is not to defend our approach. It is to be honest about where it was solid, where it was limited, and what the winners understood more clearly than we did.

## Context And Baseline

Our shipped 2026 pipeline was a clean, reproducible, leakage-aware system, but it was also fairly conservative in the kind of information it used.

- We trained separate men's and women's models with the same overall framework.
- We used Kaggle-provided data only: regular-season results, tournament seeds, and derived Elo. We did not use outside rankings, injury feeds, betting lines, or prediction markets.
- The final shipped feature sets were 27 men's differential features and 23 women's differential features.
- The feature space was mostly regular-season efficiency, score-margin, rolling form, Elo, and seed-based matchup deltas.
- The final production model for each gender was an isotonic-calibrated weighted blend of logistic regression, histogram gradient boosting, XGBoost, CatBoost, and Elo.

The saved train reports show how heavily the final blends still leaned on simpler models. For men, the tuned blend weights were roughly logistic `47.2%`, CatBoost `27.7%`, Elo `21.2%`, XGBoost `3.5%`, and HGB `0.3%`. For women, they were roughly logistic `55.9%`, CatBoost `31.3%`, XGBoost `9.5%`, Elo `2.6%`, and HGB `0.8%`. The calibrated OOF Brier scores were `0.1902` for men and `0.1338` for women.

One artifact nuance matters for any retrospective: our shipped Stage 2 submission used the latest saved men/women bundles, even though the men's variant did not clear our internal promotion gate. `artifacts/reports/runs_index.csv` marks the March 16, 2026 run with `w_promoted=1.0` and `m_promoted=0.0`, while still pointing to the shipped Stage 2 submission artifact. So when this document says "our approach," it means the actual shipped submission path, not only the internally promoted configuration.

The repo evidence also makes clear what the model really trusted. In both genders, seed features dominated the logistic coefficients and tree-model SHAP rankings, with score margin and net-efficiency style signals coming next. That is important because it means our model was not truly broad-based, even though the ensemble looked broad on paper.

## What The Top 3 Did

The winners did not all build the same system, but they converged on a few common ideas.

### Shared Themes

- They started from the assumption that seeds already encode a lot of committee knowledge, so the job was to add only the information seeds miss.
- They framed features as matchup differentials, which keeps the model focused on relative strength instead of team identity.
- They were more disciplined than we were about model complexity. Even when they used tree models, calibration, clipping, pruning, and careful feature selection mattered more than adding one more model family.
- They treated the men's and women's tournaments as meaningfully different problems. Two teams modeled them separately, and the runner-up still handled gender-specific signal gaps explicitly rather than pretending the data situation was the same on both sides.
- They optimized for Brier score directly, including how to handle the tails of the probability distribution.

Another useful pattern: the top teams were better at adding targeted signal than at adding lots of signal. They were not winning by building the biggest feature warehouse. They were winning by finding a few high-value sources of information that regular-season box-score deltas do not fully capture.

### Distinctive Edges

**1st place**

- Used a seed-first philosophy and then layered on a hand-tuned custom rating to capture what seeding misses.
- Added men's injury adjustments manually, which is exactly the kind of late-breaking information our pipeline ignored.
- Used isotonic calibration and then deliberately sharpened the extreme probabilities for Brier-score upside.
- Kept the feature set narrow instead of trying to make every plausible basketball stat matter.

**2nd place**

- Stayed fully within Kaggle data, but extracted more from it than we did by building a broader yet still disciplined ranking stack.
- Combined efficiency, Elo, seeds, recent form, strength of schedule, and men's Massey consensus rankings into a relatively compact LR + XGBoost blend.
- Used clipping as an explicit Brier-risk control at the tails.
- Treated consensus ranking information as a major gap-filler beyond seeds, especially on the men's side.

**3rd place**

- Leaned hardest into the idea that outside markets know things a pre-tournament stat model does not.
- Used aggressive feature pruning and a logistic-regression-first philosophy instead of trusting boosted trees to generalize.
- Added game-specific Round 1 market information from ESPN BPI, Vegas, and Kalshi, then tapered that influence later in the bracket.
- Built custom rating systems and interaction terms, but only after a long ablation process that removed more than it added.

## Winners Vs. Ours

| Dimension | Our shipped approach | What the top teams did better |
|---|---|---|
| Data sources | Kaggle regular-season results, seeds, and derived Elo only | Added higher-value signal where available: injuries, Massey consensus, Barttorvik-style ratings, women's NET, or market probabilities |
| Model philosophy | Five-way blended ensemble with calibration; stacking available but not ultimately selected | Simpler center of gravity: LR-first or lean blends, with less faith that extra model families would rescue small-data generalization |
| Feature philosophy | Generic rolling efficiency and score-margin deltas with seed/Elo features | More targeted features: consensus rankings, custom ratings, market priors, and selective interactions that added information seeds did not already encode |
| Calibration and tails | Isotonic calibration only | Calibration plus explicit Brier-aware tail management through clipping, sharpening, or market blending |
| Men vs. women treatment | Separate pipelines, but broadly similar inputs and philosophy | More aggressive gender-specific treatment, especially where men's and women's ranking ecosystems differ |
| Matchup-specific information | Very little beyond relative team stats, seeds, and Elo | Stronger handling of late, matchup-specific context such as injuries, market priors, and game-level projections |
| Validation posture | Rolling season holdouts plus internal promotion gates and observability | Stronger emphasis on ablation, pruning, and validation setups designed around future-tournament generalization rather than model breadth |

The short version is that our system was stronger on engineering discipline than on competition-specific edge. The winners were stronger on deciding what not to model, what not to include, and where one extra source of information was worth more than another model or another rolling window.

## What We Got Right

We should keep the parts of the repo that were genuinely good.

- **Leakage-safe feature generation:** the pipeline was careful about shifted rolling features and regular-season-only snapshots.
- **Separate men's and women's pipelines:** this gave us room to diverge by gender, even if we did not exploit that enough.
- **Calibration awareness:** we at least treated probability quality as a first-class concern instead of only chasing classifiers.
- **Reproducibility and observability:** the saved reports, explainability outputs, and run index make this retrospective possible. That is real infrastructure value.

Those are not small wins. A lot of competition codebases never reach this level of cleanliness. The issue is that strong engineering hygiene does not automatically create predictive edge.

## Where We Left Value On The Table

### 1. We were probably too ensemble-heavy for the size of the problem

The final blends already tell the story. Logistic regression carried the largest weight in both genders, CatBoost carried most of the remaining learned weight, HGB was almost irrelevant, and the stacked meta-model did not beat the tuned blend. That is a sign that we likely spent too much complexity budget on model variety and not enough on getting the strongest simple baseline truly right.

### 2. Our final feature set was still dominated by seed and margin proxies

The explainability artifacts show that `diff_seed_num` and `diff_seed_is_low_better` were among the most important signals for both genders, with score margin and efficiency features behind them. That means our pipeline mostly learned a polished version of "trust the seeds, plus some regular-season quality." The winners also trusted seeds, but they added more meaningful non-seed information on top.

### 3. We skipped the outside signals the winners leaned on most

This is probably the clearest gap. We had no injuries, no market odds, no public ranking consensus, no Barttorvik/KenPom-style external strength signals, and no women's ranking replacement for men's-only public data. In a tournament competition, once seeds are known, that missing information is often exactly where the edge lives.

### 4. We did not prune hard enough toward a great linear baseline

The winners, especially 2nd and 3rd, were much more willing to cut features that sounded smart but made the model worse. Our pipeline had good infrastructure for feature families, but the shipped model still looks more like a general-purpose framework than a ruthlessly pruned tournament predictor. Future-us should assume that many plausible basketball features are noise until proven otherwise.

### 5. We treated calibration seriously, but not submission strategy seriously enough

Isotonic calibration helped us, but the winners went further. They clipped probabilities, sharpened selected edges, or blended in market views where Brier risk and upside were asymmetric. We mostly stopped at "calibrate the model," while the winners also asked, "what should the final submitted probability distribution look like for this scoring rule?"

## Future Competition Playbook

If we do another competition like this, the priority order should be tighter than it was here.

### Do First

Build a brutally simple logistic baseline before anything else. Start with:

- seed features
- one or two consensus strength ratings
- a small number of matchup differentials
- post-hoc calibration

If that baseline is not strong, adding tree ensembles is probably a distraction.

### Do Next

Set up external data plumbing early, not late.

- rankings and consensus ratings
- injury/availability data
- odds and market probabilities
- robust team-ID mapping tables
- a women's ranking substitute whenever men's-only public signals exist

The best time to solve these integrations is before the last week of the competition.

### Do Selectively

Tune men and women more independently. Separate pipelines are not enough if they still consume nearly the same information with nearly the same philosophy. Future versions should assume from day one that the optimal feature mix, priors, and clipping policy may differ by gender.

### Do With Discipline

Make ablation and pruning a required step, not a cleanup step.

- Start narrow.
- Add one feature family at a time.
- Remove anything that does not improve holdout Brier.
- Prefer interpretable wins over "maybe the ensemble will figure it out."

The winners were much better at killing their darlings.

### Do For Submission Strategy

Treat final probability shaping as part of the model.

- evaluate isotonic vs Platt vs none
- test clipping ranges explicitly
- test limited Brier-aware edge sharpening
- test whether market blending should be round-specific

In this competition, probability management was part of the edge, not just packaging.

### Explore Later

After the baseline, data plumbing, and pruning workflow are solid, then it is worth exploring:

- coaching effects
- bracket-path priors
- market blending beyond Round 1
- custom rating systems such as Colley, SRS, or GLM-quality variants
- carefully chosen interaction terms

These are good second-wave ideas, not substitutes for the basics.

## Bottom Line

Our 2026 repo was better at building a stable modeling system than at identifying the smallest set of high-leverage signals. The top-3 teams were better at focusing on exactly what the tournament setup rewards: seeds plus a few missing pieces, strong probability discipline, and a willingness to stay simple until complexity earned its place.

For the next competition, the main lesson is not "use more models." It is "use fewer ideas, but make each one count more."

## Repo Evidence Used For This Retrospective

This synthesis of our own approach was grounded mainly in:

- `docs/methodology.md`
- `docs/results.md`
- `artifacts/reports/M_train_report.json`
- `artifacts/reports/W_train_report.json`
- `artifacts/reports/explainability/M_explainability.json`
- `artifacts/reports/explainability/W_explainability.json`
- `artifacts/reports/runs_index.csv`

The top-3 comparison was based on the copied 1st, 2nd, and 3rd place writeups reviewed after the competition ended.
