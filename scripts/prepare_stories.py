import json
import glob
import os
from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import numpy as np
from tqdm import tqdm

from lit_llama import Tokenizer
import lit_llama.packed_dataset as packed_dataset


filename_sets = {
    "stories": "**/*.jsonl"
}


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path)
    print("Tokenizer")

    # set_name is the name of dataset
    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue
        is_cc = set_name == "common_crawl"
        print(os.path.join(source_path, pattern))
        # filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)
        filenames = glob.iglob(os.path.join(source_path, pattern), recursive=True)
        try:
            next(filenames)
        except StopIteration as exc:
            raise RuntimeError(
                f"No files matching {pattern} found at {source_path}"
            ) from exc
        # TODO: We can not filenames is empty or not
        # if not filenames:
        #     raise RuntimeError(
        #         f"No files matching {pattern} found at {source_path}. \n"
        #         "Make sure you download the data, e.g. wget -i https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through \n"
        #         "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T \n"
        #         "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
        #     )
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )
        for name in filenames:
            print(f"Processing {name}")

            if is_cc:
                with zstd.open(open(name, "rb"), "rt", encoding="utf-8") as f:
                    for row in tqdm(f):
                        text = json.loads(row)["text"]
                        text_ids = tokenizer.encode(text)
                        builder.add_array(np.array(text_ids, dtype=builder.dtype))
            else:
                with open(name, encoding="utf-8") as f:
                    for row in tqdm(f):
                        text = json.loads(row)["content"]
                        text_ids = tokenizer.encode(text)
                        builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,  # 2048 block size + 1 for causal (from LLama), 1024 blocks
    match: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    prepare_full(
        source_path=source_path,
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=chunk_size,
        match=match,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
