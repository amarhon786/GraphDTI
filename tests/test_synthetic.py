import pandas as pd


def test_synth_generator_produces_csvs(synth_data_dir):
    train = pd.read_csv(synth_data_dir / "train.csv")
    val = pd.read_csv(synth_data_dir / "val.csv")
    assert {"smiles", "protein_sequence", "protein_id", "label"} <= set(train.columns)
    assert len(train) == 64 and len(val) == 32
    # the threshold is chosen so we get both classes; if not, the dataset is degenerate
    assert train["label"].nunique() == 2
