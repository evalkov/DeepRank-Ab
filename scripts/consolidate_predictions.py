#!/usr/bin/env python3
import argparse, gzip, json
from pathlib import Path
from time import perf_counter
import h5py

# P² quantile estimator (O(1) memory)
class P2Quantile:
    def __init__(self, p: float):
        self.p = p
        self.n = 0
        self.initial = []
        self.q = [0.0]*5
        self.np = [0.0]*5
        self.ni = [0]*5
        self.dn = [0.0, p/2.0, p, (1.0+p)/2.0, 1.0]

    def add(self, x: float):
        self.n += 1
        if self.n <= 5:
            self.initial.append(x)
            if self.n == 5:
                self.initial.sort()
                self.q = self.initial[:]
                self.ni = [1,2,3,4,5]
                self.np = [1.0, 1.0 + 2.0*self.p, 1.0 + 4.0*self.p, 3.0 + 2.0*self.p, 5.0]
            return

        if x < self.q[0]:
            self.q[0] = x; k = 0
        elif x >= self.q[4]:
            self.q[4] = x; k = 3
        else:
            k = 0
            while k < 4 and not (self.q[k] <= x < self.q[k+1]):
                k += 1
            if k == 4:
                k = 3

        for i in range(5):
            if i >= k+1:
                self.ni[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        for i in (1,2,3):
            d = self.np[i] - self.ni[i]
            if (d >= 1 and self.ni[i+1] - self.ni[i] > 1) or (d <= -1 and self.ni[i-1] - self.ni[i] < -1):
                di = 1 if d >= 1 else -1
                qip = self.q[i] + di * (
                    (self.ni[i] - self.ni[i-1] + di) * (self.q[i+1] - self.q[i]) / (self.ni[i+1] - self.ni[i]) +
                    (self.ni[i+1] - self.ni[i] - di) * (self.q[i] - self.q[i-1]) / (self.ni[i] - self.ni[i-1])
                ) / (self.ni[i+1] - self.ni[i-1])

                if self.q[i-1] < qip < self.q[i+1]:
                    self.q[i] = qip
                else:
                    self.q[i] = self.q[i] + di * (self.q[i] - self.q[i-di]) / (self.ni[i] - self.ni[i-di])

                self.ni[i] += di

    def value(self):
        if self.n == 0:
            return float("nan")
        if self.n <= 5:
            s = sorted(self.initial)
            idx = int(round((len(s)-1)*self.p))
            return float(s[idx])
        return float(self.q[2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    preds_dir = run_root / "exchange" / "preds"
    files = sorted(preds_dir.glob("pred_shard_*.hdf5"))
    if not files:
        raise SystemExit(f"No pred_shard_*.hdf5 in {preds_dir}")

    out_dir = run_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / f"all_predictions_{args.run_id}.tsv.gz"
    out_stats = out_dir / f"stats_{args.run_id}.json"

    q10,q25,q50,q75,q90 = P2Quantile(0.10),P2Quantile(0.25),P2Quantile(0.50),P2Quantile(0.75),P2Quantile(0.90)

    count=0
    sum_x=0.0
    min_x=float("inf")
    max_x=float("-inf")
    high=med=low=0

    tmp = out_tsv.with_suffix(out_tsv.suffix + ".tmp")
    t0 = perf_counter()
    with gzip.open(tmp, "wt") as out:
        # FIX: header is correct and stable
        out.write("shard\tpdb_id\tpredicted_dockq\n")
        for pf in files:
            shard = pf.stem.replace("pred_shard_", "")
            with h5py.File(pf, "r") as h:
                g = h["epoch_0000"]["pred"]
                mol = g["mol"][()]
                outs = g["outputs"][()]
                for m, o in zip(mol, outs):
                    m = m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
                    x = float(o)
                    out.write(f"{shard}\t{m}\t{x}\n")
                    count += 1
                    sum_x += x
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    if x >= 0.49: high += 1
                    elif x >= 0.23: med += 1
                    else: low += 1
                    q10.add(x); q25.add(x); q50.add(x); q75.add(x); q90.add(x)

    tmp.replace(out_tsv)

    if count < 1:
        raise SystemExit("ERROR: consolidated 0 predictions")

    stats = dict(
        run_id=args.run_id,
        shards=len(files),
        rows=count,
        mean=(sum_x / count),
        median=q50.value(),
        min=min_x,
        max=max_x,
        p10=q10.value(),
        p25=q25.value(),
        p75=q75.value(),
        p90=q90.value(),
        high_ge_0_49=high,
        med_0_23_0_49=med,
        low_lt_0_23=low,
        wall_s=perf_counter()-t0,
    )
    out_stats.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"✓ Wrote: {out_tsv}")
    print(f"✓ Stats: {out_stats}")

if __name__ == "__main__":
    main()

