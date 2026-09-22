# Extension Cell Universe

Decision token: `EXTENSION_DATASET_FOODCOM_SELECTED_FOR_EXECUTION`

This report materializes the current operational raw-experiment and result-cell
universes over Beauty, Sports, ML-1M, LastFM, and Food.com. No LLM inference,
recommender training, downloads, or raw-data reconstruction were performed.

## Complete Gemma parity

Gemma raw experiments classified: 46
Gemma result cells classified: 826
Gemma planned LLM requests from raw table: 20928

## Food.com final fifth dataset

Food.com primary raw experiments: 6
Food.com primary planned LLM requests: 6000
Food.com history raw experiments: 18
Food.com history planned LLM requests: 3600

## Offline and training cells

Offline-only result cells currently classified: 30
Training-required raw rows: 4

## Blocked cells

Blocked rows are limited to source-rank prerequisites and global policy/input
definitions that are not resolved in the frozen protocol. Blocked result cells:
147

## Active dataset set

beauty, sports, ml1m, lastfm, foodcom

## Validation

Passed: `True`
