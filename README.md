# Hotel Recommendation Systems

End-to-end design, evaluation, and business interpretation of recommender systems on real TripAdvisor hotel-review data. Group project for the Chatbots & Recommender Systems course (IE University, 2026).

Research, design, and implementation by Matteo Colantoni. Elias Nmeir, Nikolas Lafrentz, and Kenny Tohme reviewed the work and contributed to the final polish.

Everything lives in one thoroughly documented notebook, [`hotel_recommendation_systems.ipynb`](hotel_recommendation_systems.ipynb) (~570 cells), which builds the full pipeline from raw reviews to business recommendations:

1. **Data** — Kaggle download, preprocessing, dataset reduction, train/test split, EDA and sparsity diagnostics (matrix density, long-tail concentration, temporal coverage).
2. **Recommender families** — non-personalized baselines (random, popular, Bayesian/weighted popular), memory-based collaborative filtering (user- and item-based), model-based CF (biased SVD matrix factorization with latent-factor interpretation), and content-based models comparing **Bag-of-Words, TF-IDF, and BERT** (sentence-transformers) hotel profiles built from review text.
3. **Context-awareness** — context-aware variants of the memory- and content-based models, with a dedicated with-vs-without-context impact analysis.
4. **Hybrids** — weighted/ensemble combinations of the best models, with grid-searched mixing weights.
5. **Evaluation best practices** — offline evaluation of rating and ranking quality, cross-validation and cross-validation-through-time (CVTT), early stopping, hyperparameter tuning and sensitivity analysis.
6. **Beyond accuracy** — cold-start, popularity bias and fairness considerations, exploration–exploitation with **multi-armed bandits**, and a closing business-value interpretation for the company.

The [`results/`](results) folder contains the exported metric tables (evaluation summaries, context-impact deltas per text representation, hybrid grid searches, CV/CVTT fold results) referenced throughout the notebook.

## Data

The dataset is [joebeachcapital/hotel-reviews](https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews) on Kaggle (TripAdvisor hotel reviews). It is not committed to the repo; the notebook's first section downloads it via the Kaggle API into `data/hotel_reviews/`. You need your own Kaggle API token (`kaggle.json`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab hotel_recommendation_systems.ipynb
```

The notebook is self-contained and runs top to bottom; the BERT-based sections benefit from a GPU but run on CPU.

## License

Released under the [MIT License](LICENSE).
