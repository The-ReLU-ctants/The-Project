# Big Data Project: Wildlife Species Classification (WildlifeReID-10k)

This project builds a Vision Transformer (ViT) that classifies an animal photo into one of 5 species (dog, polar bear, sea star, nyala, lion), serves it through an inference API, and wraps it in an upload app, per the Big Data Analytics project brief.

Dataset source: https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k

## Setup

1. Clone the repo.
2. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Downloading the dataset

The full WildlifeReID-10k dataset is about 26GB, so instead we pull 5 species individually from their original smaller sources (about 4.1GB total). No Kaggle account is needed for this subset.

```
python scripts/download_subset.py
```

If it stops partway on the nyala or lion step with a `TypeError` about writing `None` to a license file, that is a known bug in the `wildlife-datasets` library, not a real failure. Run this once to finish those two:

```
python scripts/fix_nyala_lion.py
```

This downloads everything into `data/raw_subset/` (about 4.1GB, gitignored, not pushed to GitHub). Expect it to take a while and to need re-running once or twice if a connection drops partway, since the script skips anything already finished.

## Preprocessing (done, Temos's part)

These scripts turn the raw downloaded folders into a clean, model-ready pipeline. Run them in this order after the download finishes:

```
python scripts/build_manifest.py
python scripts/make_splits.py
python scripts/build_label_map.py
```

What each one does:

1. `build_manifest.py`: walks `data/raw_subset/<species>/...` and builds `data/processed/manifest.csv` with one row per image (path, species, identity). Each of the 5 sources has a different folder layout, so this normalises them into one flat table.
2. `make_splits.py`: polar bear has 138,363 images in the raw data because it is extracted video frames, versus 740 to 8,363 for the other species, so this caps every class at 2,500 images (spread across individuals first, not just the first ones found) and produces a 70/15/15 train/val/test split, grouped by identity, so the same animal never appears in more than one split. Outputs `data/processed/{train,val,test}.csv`.
3. `build_label_map.py`: freezes the species to integer label mapping to `data/processed/label_map.json`, so training, evaluation, and the inference server all agree on the same indices.

`src/dataset.py` is the transform and `Dataset` pipeline on top of those CSVs: `WildlifeDataset` loads an image, resizes it to 224x224, normalises it, and applies augmentation for the training split. `load_label_map()` reads the frozen label mapping. Import these two directly, there is no need to touch the CSVs by hand.

Result sizes after capping: train 6,997 images, val 1,436, test 1,436, roughly balanced across the 5 classes (polar bear has only 13 distinct individuals in the source data, worth a mention in the dataset properties writeup).

## What Sihle needs to do: model, training, evaluation, embeddings

Important hardware note: this is being developed on a CPU only machine (no dedicated GPU, 11.7GB RAM). We agreed to train from scratch with no pretrained weights, so keep the ViT small (fewer layers, smaller hidden size, maybe a lower input resolution than 224 if training is too slow) or training will take an unreasonable amount of time. Do a 1 epoch timing test before committing to a long run.

1. **Model architecture**: build a ViT (you can use the architecture classes from `transformers` or `torchvision`, but initialise with random weights, not a pretrained checkpoint, and write your own classification head for 5 classes). Put this in `src/model.py`.
2. **Training loop**: in `scripts/train.py`, load `WildlifeDataset` and `load_label_map` from `src/dataset.py` for the train and val splits, train with CrossEntropyLoss and an optimiser of your choice, checkpoint the best model on validation accuracy. Save the final weights to `models/vit_species.pt` (create the `models/` folder).
3. **Evaluation**: in `scripts/evaluate.py`, load the test split and the saved model, report accuracy, per-class precision/recall/F1, and a confusion matrix.
4. **Embeddings and dimensional reduction**: this is a required pipeline step in the brief. Extract the ViT's pooled embeddings for the test set and run PCA or t-SNE on them, then plot the result coloured by species. This also doubles as a results visualisation for the slides and paper.

Once the model is trained, evaluated, and saved to `models/vit_species.pt`, hand the code back to Temos to build the inference engine and the upload UI.

## Still to do after that (Temos): inference engine and UI

1. **Inference engine**: build a REST (or gRPC) server, Flask or FastAPI is simplest, that loads `models/vit_species.pt` and `data/processed/label_map.json`, accepts an uploaded image, and returns the predicted species and confidence. Put this under `src/inference_server.py` or similar, and document how to start it (e.g. `python src/inference_server.py`, what port it listens on, and the request/response format for the predict endpoint).
2. **Upload UI**: a web or mobile app that calls the inference server and displays the predicted species and confidence.
3. **Final packaging**: the submission folder structure required by the brief: `src/` (all source), `dist/` (an executable), `data/` (two sample files plus the dataset URL above), `ss/` (slideshow).
