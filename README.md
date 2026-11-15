# Comparing-Playoff-Formats

Simulated comparisons of NHL playoff formats: how much do seeding rules,
bracket structure, and reseeding change the likelihood that good teams meet
late — and how do the formats trade off competitive quality against travel
burden?

Built on a Bayesian season simulation engine (100,000 simulated
league seasons per format, common random numbers so every format sees the
*same* regular seasons), developed as part of another hockey analytics
project on team talent estimation and playoff simulation.

## The question

Since 2013 the NHL seeds its playoffs within divisions, with two wild-card
teams per conference. Critics argue this produces weaker late-round matchups
than a format where teams are seeded at the conference level. This project simulates eight formats on identical regular
seasons and measures:

- P(the league's best team wins the Cup)
- P(the two best teams meet in the Cup final)
- P(the top-4 / top-8 teams all survive to the final four / round 2)
- Expected matchup quality per round (mean talent gap in each series)
- League-total postseason travel miles

## The eight formats

| # | Format | Round 1 seeding | Reseeding after R1/R2 |
|---|--------|-----------------|----------------------|
| 0 | Current (baseline) | Division bracket + wild cards | No |
| 7 | Current + reseed | Division bracket + wild cards | Yes, within conference |
| 1 | 2-6 bracket | 2 division winners + next 6 by points | No |
| 2 | 1-8 bracket | Strict conference top-8 by points | No |
| 3 | 1-16 bracket | League-wide top-16 by points | No |
| 4 | 2-6 + reseed | 2-6 eligibility | Yes, within conference |
| 5 | 1-8 + reseed | 1-8 eligibility | Yes, within conference |
| 6 | 1-16 + reseed | 1-16 eligibility | Yes, league-wide |

Reseeding follows the 1998-99 NHL convention: after each of rounds 1 and 2,
the top remaining seed meets the lowest remaining seed. Round 1 home ice
goes to the higher seed in all new formats; the current format keeps its
real bracket rules. Cup-final home ice uses the regular-season tiebreaker
chain where seed numbers collide (as in real NHL practice).

## Caveats (important)

- All results are model-based. Team talent is drawn once per season from a
  fitted distribution; game goals are Poisson. Absolute probabilities are
  model outputs; comparisons across formats share the model.

## Repository layout

The repo covers the code only; data, parquet outputs, figures, and reports
live outside it. All simulation code is Python 3, vectorized with
NumPy/pandas.

| File | Role |
|------|------|
| `pf_brackets.py` | Format specs: seed orders, round-1 pairings, reseeding rule |
| `pf_postseason.py` | Simulates one full postseason under any of the 8 formats |
| `pf_mass_sim.py` | 100k-season harness for scenarios 1-7 (common random numbers) |
| `pf_scenario0.py` | Baseline (current format) 100k run + cross-checks vs the talent project's run |
| `pf_travel.py` | Great-circle travel accounting per series/season/scenario |
| `pf_report.py` | Results report, tables, figures |
| `pf_conditional.py` / `pf_conditional_standings.py` | Possibility-adjusted metrics: P(outcome is structurally possible) vs P(outcome \| possible), via full 2^8 round-1 outcome enumeration |
| `pf_table_headline.py` / `pf_table_headline_standings.py` | 12-column headline tables (4 metrics x 3 probability types) |
| `pf_rank_matrix.py` / `pf_fig_rank_matrix.py` | Talent-rank x standings-rank joint distributions and the 2-panel rank-matrix figure |
| `pf_fig_tornado.py` | Tornado/sensitivity figure |

## Reproduction notes

- The engine depends on a companion repository (team-talent estimation and
  the season/game simulation core — multiplicative-Poisson games, Poisson-race
  OT, 2-2-1-1-1 series). Scripts reference it via a sibling `likens/code`
  directory; the shared virtualenv lives there too.
- Locked model constants (talent SD 0.0415, league GPG 5.4933, home-ice
  advantage 0.2563) are estimated in the companion project and reused here.
- Runtime is ~19 s per 100k-season scenario (~2.5 min for all seven new
  scenarios). All runs use a fixed seed (20261028); regular seasons are
  bit-identical across formats by construction, so format differences are
  attributable to the format alone.
- Scripts fail loudly on structure violations (bracket sizes, round counts,
  pairing topology, team-index order) rather than silently producing numbers.


