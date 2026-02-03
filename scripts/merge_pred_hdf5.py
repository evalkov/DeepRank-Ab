#!/usr/bin/env python3
"""
Merge multiple DeepRank-Ab prediction HDF5 files (pred_batch*.hdf5) into one.

Input files must contain:
  /epoch_0000/pred/mol      (N,)
  /epoch_0000/pred/outputs  (N,)

Output file will contain the same datasets, concatenated in input order.
Streaming append (does not hold all mols in RAM).
"""

import argparse
from pathlib import Path
import h5py
import numpy as np


def iter_preds(path: Path, step: int = 200_000):
    with h5py.File(path, "r") as f:
        g = f["epoch_0000"]["pred"]
        mol = g["mol"]
        outs = g["outputs"]
        n = len(mol)
        for i in range(0, n, step):
            j = min(n, i + step)
            yield mol[i:j], outs[i:j]


def as_bytes(arr) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            out.append(bytes(x))
        else:
            out.append(str(x).encode("utf-8"))
    return np.array(out, dtype="S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output merged HDF5 path")
    ap.add_argument("inputs", nargs="+", help="Input pred_*.hdf5 files")
    args = ap.parse_args()

    out_path = Path(args.out)
    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"Missing input: {p}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as out:
        grp = out.create_group("epoch_0000/pred")

        d_mol = grp.create_dataset(
            "mol",
            shape=(0,),
            maxshape=(None,),
            dtype="S256",
            chunks=(100_000,),
            compression="gzip",
            compression_opts=1,
        )
        d_out = grp.create_dataset(
            "outputs",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(100_000,),
            compression="gzip",
            compression_opts=1,
        )

        n_total = 0
        for inp in inputs:
            for mol_chunk, out_chunk in iter_preds(inp):
                mol_chunk = as_bytes(mol_chunk)
                out_chunk = np.array(out_chunk, dtype=np.float32)
                n = len(out_chunk)
                if n == 0:
                    continue

                d_mol.resize((n_total + n,))
                d_out.resize((n_total + n,))
                d_mol[n_total:n_total + n] = mol_chunk
                d_out[n_total:n_total + n] = out_chunk
                n_total += n

        if n_total < 1:
            raise SystemExit("Merged 0 predictions; check inputs.")
        out.flush()

    print(f"✓ Merged {n_total} predictions -> {out_path}")


if __name__ == "__main__":
    main()

