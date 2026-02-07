import json
import warnings
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import os

from Bio.PDB import PDBParser
from Bio import SeqIO, pairwise2
from anarci import anarci

BASE = Path(__file__).resolve().parent
hmmscan_path = BASE / "ANARCI"

# residue code map (includes MSE -> M)
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

# IMGT CDR ranges
CDR_L = {"L1": range(27, 39), "L2": range(56, 66), "L3": range(105, 118)}
CDR_H = {"H1": range(27, 39), "H2": range(56, 66), "H3": range(105, 118)}


def extract_chain_sequence(chain) -> str:
    """
    Return one-letter amino acid sequence for a chain.

    Important: return 'X' for unknown/modified residues so sequence length
    stays consistent and alignment/locate indices remain meaningful.
    """
    seq = []
    for res in chain:
        aa = THREE_TO_ONE.get(res.get_resname(), "X")
        # Some residues in PDB may be non-amino (e.g. waters) but they usually
        # aren’t in ATOM records of protein chains; keep 'X' to be safe.
        seq.append(aa)
    return "".join(seq)


def region_l(position: int) -> str:
    """Label light-chain IMGT position as FR/CDR; fallback CONST."""
    for label, rng in CDR_L.items():
        if position in rng:
            return label
    return "CONST"


def region_h(position: int) -> str:
    """Label heavy-chain IMGT position as FR/CDR; fallback CONST."""
    for label, rng in CDR_H.items():
        if position in rng:
            return label
    return "CONST"


def locate(subseq: str, fullseq: str, tag: str):
    """
    Local alignment of subseq in fullseq; return (start, end) indices in fullseq.
    """
    alignment = pairwise2.align.localms(
        fullseq, subseq, 2, -1, -0.5, -0.1, one_alignment_only=True
    )
    if not alignment:
        raise RuntimeError(f"{tag}: alignment failed")
    _, _, _, start, end = alignment[0]
    return start, end


def _norm_role_from_id(rec_id: str) -> str:
    """
    Expect FASTA record ids like:
      foo.H or foo.L (your pipeline writes this)
    We classify by last character.
    """
    rid = rec_id.strip().upper()
    return rid[-1] if rid else ""


def annotate_single_pdb(fasta_file: Path, pdb_file: Path, antigen_chainid: str = "A"):
    """
    Annotate CDR/FR/CONST for a single PDB using a reference fasta.

    Supports:
      - 2-seq FASTA: H and L (classic antibody)
      - 1-seq FASTA: H only (VHH / nanobody)

    The annotation output is a dict:
      { "<model>.pdb": [ (aa, region_label), ... ] }
    where the list indexes correspond to the *merged antibody chain sequence*
    (the non-antigen chain in the merged PDB).
    """
    fasta_file = Path(fasta_file)
    pdb_file = Path(pdb_file)

    parser = PDBParser(QUIET=True)

    seqs = [
        record
        for record in SeqIO.parse(str(fasta_file), "fasta")
        if record.seq and str(record.seq).strip()
    ]
    if len(seqs) not in (1, 2):
        raise ValueError(
            f"{fasta_file}: expected 1 (H only) or 2 (H and L) sequences, found {len(seqs)}"
        )

    ref = {}
    for rec in seqs:
        role = _norm_role_from_id(rec.id)
        if role not in ("H", "L"):
            raise ValueError(f"{fasta_file}: unexpected record id '{rec.id}' (expected id ending with H or L)")
        ref[role] = str(rec.seq).replace("\n", "").strip()

    if "H" not in ref or not ref["H"]:
        raise ValueError(f"{fasta_file}: missing heavy chain sequence (H)")

    seqH = ref["H"]
    seqL = ref.get("L", "")  # may be absent for VHH

    model_name = pdb_file.stem
    struct = parser.get_structure(model_name, str(pdb_file))
    model0 = struct[0]

    all_chainids = [chain.id for chain in model0]
    if antigen_chainid not in all_chainids:
        raise ValueError(f"{model_name}: antigen chain '{antigen_chainid}' not found (have {all_chainids})")

    # antibody chain(s) = any chain not antigen
    chainid_antibody = [cid for cid in all_chainids if cid != antigen_chainid]
    if not chainid_antibody:
        raise ValueError(f"{model_name}: no antibody chain found (only antigen '{antigen_chainid}' present?)")

    # In merged PDBs this should be a single chain (e.g. A), but keep first non-antigen chain.
    ab_cid = chainid_antibody[0]
    chainAb_seq = extract_chain_sequence(model0[ab_cid])

    # Locate heavy in merged antibody chain
    try:
        h0, _ = locate(seqH, chainAb_seq, f"{model_name}_H")
        orig_h0 = h0
        while h0 > 0 and chainAb_seq[h0] != seqH[0]:
            h0 -= 1
        if h0 != orig_h0:
            warnings.warn(f"{model_name}: heavy missing {orig_h0 - h0} N-term residues, shifting start")
    except RuntimeError as e:
        warnings.warn(str(e))
        return None

    # Locate light only if present (classic Ab)
    l0 = None
    if seqL:
        try:
            l0, _ = locate(seqL, chainAb_seq, f"{model_name}_L")
            orig_l0 = l0
            while l0 > 0 and chainAb_seq[l0] != seqL[0]:
                l0 -= 1
            if l0 != orig_l0:
                warnings.warn(f"{model_name}: light missing {orig_l0 - l0} N-term residues, shifting start")
        except RuntimeError as e:
            warnings.warn(str(e))
            return None

    # variable domains (~first 130 aa)
    varH = seqH[:130] if seqH else ""
    varL = seqL[:130] if seqL else ""

    records = []
    mapping = []

    if varH:
        records.append((f"{model_name}_H", varH))
        mapping.append({"model": model_name, "chain": "H", "start": h0, "seq": chainAb_seq})

    if varL and l0 is not None:
        records.append((f"{model_name}_L", varL))
        mapping.append({"model": model_name, "chain": "L", "start": l0, "seq": chainAb_seq})

    if not records:
        warnings.warn(f"{model_name}: no variable domains found")
        return None

    # Run ANARCI with IMGT numbering
    numbering, _, _ = anarci(
        records,
        scheme="imgt",
        assign_germline=True,
        output=False,
        hmmerpath=hmmscan_path,
    )

    annotations = {}
    for info, num in zip(mapping, numbering):
        model_key = f"{info['model']}.pdb"
        chain_role = info["chain"]   # 'H' or 'L'
        start = info["start"]
        seq = info["seq"]

        # default: everything CONST
        annotations.setdefault(model_key, [(aa, "CONST") for aa in seq])
        ann = annotations[model_key]

        if not num:
            continue

        aligned = num[0][0]  # list of ((position, ins), aa)
        region_fn = region_h if chain_role == "H" else region_l

        idx = start
        for ((pos, _ins), aa) in aligned:
            if aa == "-":
                continue

            # Advance until aa matches or we run out
            while idx < len(ann) and ann[idx][0] != aa:
                idx += 1
            if idx >= len(ann):
                break

            ann[idx] = (aa, region_fn(pos))
            idx += 1

    return annotations


def annotate_folder(folder: Path, output_dir: Path, fasta_file: Path = None, antigen_chainid: str = "A"):
    """
    Annotate all PDB files in a folder using a single fasta file.
    (Kept for backwards compatibility.)
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    fasta_file = Path(fasta_file) if fasta_file is not None else None

    pdb_files = list(folder.glob("*.pdb"))

    all_annotations = {}
    for pdb_file in pdb_files:
        try:
            annotations = annotate_single_pdb(fasta_file, pdb_file, antigen_chainid)
            if not annotations:
                continue
            all_annotations.update(annotations)
        except Exception as e:
            warnings.warn(f"{pdb_file}: annotation failed: {e}")
            continue

    out_dir = folder / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "annotations_cdrs.json"
    with open(out_file, "w") as f:
        json.dump(all_annotations, f, indent=2)



def _annotate_single_wrapper(args):
    """Helper to call annotate_single_pdb with try/except in multiprocessing."""
    fasta_file, pdb_file, antigen_chainid = args
    try:
        return annotate_single_pdb(fasta_file, pdb_file, antigen_chainid)
    except Exception as e:
        warnings.warn(f"{pdb_file}: annotation failed: {e}")
        return None


def _extract_sequences_for_anarci(args):
    """
    Phase 1: Extract sequences from a single PDB without running ANARCI.
    Returns records for ANARCI and metadata for later annotation mapping.
    """
    fasta_file, pdb_file, antigen_chainid = args
    fasta_file = Path(fasta_file)
    pdb_file = Path(pdb_file)

    try:
        parser = PDBParser(QUIET=True)

        seqs = [
            record
            for record in SeqIO.parse(str(fasta_file), "fasta")
            if record.seq and str(record.seq).strip()
        ]
        if len(seqs) not in (1, 2):
            return None

        ref = {}
        for rec in seqs:
            role = _norm_role_from_id(rec.id)
            if role not in ("H", "L"):
                return None
            ref[role] = str(rec.seq).replace("\n", "").strip()

        if "H" not in ref or not ref["H"]:
            return None

        seqH = ref["H"]
        seqL = ref.get("L", "")

        model_name = pdb_file.stem
        struct = parser.get_structure(model_name, str(pdb_file))
        model0 = struct[0]

        all_chainids = [chain.id for chain in model0]
        if antigen_chainid not in all_chainids:
            return None

        chainid_antibody = [cid for cid in all_chainids if cid != antigen_chainid]
        if not chainid_antibody:
            return None

        ab_cid = chainid_antibody[0]
        chainAb_seq = extract_chain_sequence(model0[ab_cid])

        # Locate heavy
        try:
            h0, _ = locate(seqH, chainAb_seq, f"{model_name}_H")
            orig_h0 = h0
            while h0 > 0 and chainAb_seq[h0] != seqH[0]:
                h0 -= 1
        except RuntimeError:
            return None

        # Locate light
        l0 = None
        if seqL:
            try:
                l0, _ = locate(seqL, chainAb_seq, f"{model_name}_L")
                orig_l0 = l0
                while l0 > 0 and chainAb_seq[l0] != seqL[0]:
                    l0 -= 1
            except RuntimeError:
                return None

        # Variable domains
        varH = seqH[:130] if seqH else ""
        varL = seqL[:130] if seqL else ""

        records = []
        mapping = []

        if varH:
            records.append((f"{model_name}_H", varH))
            mapping.append({"model": model_name, "chain": "H", "start": h0, "seq": chainAb_seq})

        if varL and l0 is not None:
            records.append((f"{model_name}_L", varL))
            mapping.append({"model": model_name, "chain": "L", "start": l0, "seq": chainAb_seq})

        if not records:
            return None

        return {"records": records, "mapping": mapping}

    except Exception:
        return None


def _apply_numbering_to_mapping(mapping_entry, numbering_entry):
    """
    Phase 3: Apply ANARCI numbering to create annotation for a single chain.
    """
    model_key = f"{mapping_entry['model']}.pdb"
    chain_role = mapping_entry["chain"]
    start = mapping_entry["start"]
    seq = mapping_entry["seq"]

    # Default: everything CONST
    ann = [(aa, "CONST") for aa in seq]

    if not numbering_entry:
        return model_key, ann

    aligned = numbering_entry[0][0]  # list of ((position, ins), aa)
    region_fn = region_h if chain_role == "H" else region_l

    idx = start
    for ((pos, _ins), aa) in aligned:
        if aa == "-":
            continue
        while idx < len(ann) and ann[idx][0] != aa:
            idx += 1
        if idx >= len(ann):
            break
        ann[idx] = (aa, region_fn(pos))
        idx += 1

    return model_key, ann

def _run_anarci_batch(records):
    """
    Run ANARCI for one batch of records.
    Top-level helper so it can be used by ProcessPoolExecutor.
    """
    numbering, _, _ = anarci(
        records,
        scheme="imgt",
        assign_germline=True,
        output=False,
        hmmerpath=hmmscan_path,
    )
    return numbering


def annotate_folder_one_by_one_mp(
    folder: Path,
    fasta_folder: Path,
    output_dir: Path,
    n_cores: int = None,
    antigen_chainid: str = "A",
):
    """
    Annotate all PDB files in a folder using corresponding fasta files.
    Uses batched ANARCI call to reduce HMMER startup overhead.

    For each PDB: expects FASTA named: <pdb_stem>_HL.fasta

    Supports:
      - H+L FASTA (2 records)
      - H-only FASTA (1 record) for VHH
    """
    folder = Path(folder)
    fasta_folder = Path(fasta_folder)
    output_dir = Path(output_dir)

    pdb_files = list(folder.glob("*.pdb"))
    tasks = []

    for pdb_file in pdb_files:
        fasta_file = fasta_folder / f"{pdb_file.stem}_HL.fasta"
        if not fasta_file.exists():
            warnings.warn(f"{fasta_file} not found, skipping {pdb_file}")
            continue
        tasks.append((fasta_file, pdb_file, antigen_chainid))

    if not tasks:
        out_dir = folder / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "annotations_cdrs.json"
        with open(out_file, "w") as f:
            json.dump({}, f, indent=2)
        return

    n_processes = int(n_cores) if n_cores else (os.cpu_count() or 1)

    # Phase 1: Extract sequences in parallel
    with Pool(processes=n_processes) as pool:
        extraction_results = pool.map(_extract_sequences_for_anarci, tasks)

    # Collect all records for batched ANARCI call
    all_records = []
    all_mappings = []
    record_indices = []  # Track which extraction result each record belongs to

    for i, result in enumerate(extraction_results):
        if result is None:
            continue
        for j, rec in enumerate(result["records"]):
            all_records.append(rec)
            all_mappings.append(result["mapping"][j])
            record_indices.append(i)

    if not all_records:
        out_dir = folder / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "annotations_cdrs.json"
        with open(out_file, "w") as f:
            json.dump({}, f, indent=2)
        return

    # Phase 2: Batched ANARCI call(s)
    # Set ANARCI_PARALLEL_BATCHES>1 to split very large record sets across processes.
    anarci_parallel_batches = max(1, int(os.environ.get("ANARCI_PARALLEL_BATCHES", "1")))
    if anarci_parallel_batches <= 1 or len(all_records) < 32:
        numbering, _, _ = anarci(
            all_records,
            scheme="imgt",
            assign_germline=True,
            output=False,
            hmmerpath=hmmscan_path,
        )
    else:
        n_batches = min(anarci_parallel_batches, len(all_records))
        chunk_size = (len(all_records) + n_batches - 1) // n_batches
        chunks = [all_records[i:i + chunk_size] for i in range(0, len(all_records), chunk_size)]
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            numbering_chunks = list(executor.map(_run_anarci_batch, chunks))
        numbering = [item for chunk in numbering_chunks for item in chunk]
        if len(numbering) != len(all_mappings):
            raise RuntimeError(
                f"ANARCI numbering length mismatch: got {len(numbering)} entries for {len(all_mappings)} mappings"
            )

    # Phase 3: Map results back to annotations
    all_annotations = {}
    for mapping_entry, num_entry in zip(all_mappings, numbering):
        model_key, ann = _apply_numbering_to_mapping(mapping_entry, num_entry)
        # Merge annotations (H and L chains for same model)
        if model_key in all_annotations:
            existing = all_annotations[model_key]
            for idx, (aa, region) in enumerate(ann):
                if region != "CONST" and idx < len(existing):
                    existing[idx] = (aa, region)
        else:
            all_annotations[model_key] = ann

    out_dir = folder / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "annotations_cdrs.json"
    with open(out_file, "w") as f:
        json.dump(all_annotations, f, indent=2)



def annotate_folder_one_by_one_mp_single_fasta(
    folder: Path,
    fasta_file_path: Path,
    output_dir: Path,
    n_cores: int = None,
    antigen_chainid: str = "A",
):
    """
    Annotate all PDB files in a folder using a single fasta file in parallel.
    """
    folder = Path(folder)
    fasta_file_path = Path(fasta_file_path)
    output_dir = Path(output_dir)

    pdb_files = list(folder.glob("*.pdb"))
    tasks = [(fasta_file_path, pdb_file, antigen_chainid) for pdb_file in pdb_files]

    n_processes = int(n_cores) if n_cores else (os.cpu_count() or 1)

    all_annotations = {}
    if tasks:
        with Pool(processes=n_processes) as pool:
            results = pool.map(_annotate_single_wrapper, tasks)

        for res in results:
            if not res:
                continue
            all_annotations.update(res)

    out_dir = folder / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "annotations_cdrs.json"
    with open(out_file, "w") as f:
        json.dump(all_annotations, f, indent=2)

