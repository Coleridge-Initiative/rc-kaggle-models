import os

from transformers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    model_names = [
        "allenai/scibert_scivocab_cased",
        "allenai/biomed_roberta_base",
    ]
    pretrained_folders = [
        "scibert-base-cased",
        "biomed_roberta_base",
    ]

    for model_name, pretrained_folder in zip(model_names, pretrained_folders):
        os.makedirs(f"./pretrained/{pretrained_folder}", exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        tokenizer.save_pretrained(f"./pretrained/{pretrained_folder}")
        model.save_pretrained(f"./pretrained/{pretrained_folder}")
