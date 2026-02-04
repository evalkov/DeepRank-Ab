import os
import sys
import glob
import h5py
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Queue, Process
import pickle
import warnings
import json
from Bio.PDB.PDBParser import PDBParser
from Bio import BiopythonWarning

warnings.filterwarnings("ignore", category=BiopythonWarning)

sys.path.append(os.path.dirname(__file__))


# Global worker configuration (set once per worker process)
_WORKER_CONFIG = None


def _worker_init(config):
    """
    Initialize worker process with configuration.
    Called once per worker at startup.
    """
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _worker_process_graph(pdb_path):
    """
    Worker function: build a single graph and return it.
    Uses global config set by _worker_init.
    """
    global _WORKER_CONFIG
    cfg = _WORKER_CONFIG
    
    try:
        if cfg['graph_type'] == 'residue':
            from ResidueGraph import ResidueGraph
            g = ResidueGraph(
                pdb=pdb_path,
                biopython=cfg['biopython'],
                region_map=cfg['region_map'],
                add_orientation=cfg['add_orientation'],
                contact_features=cfg['contact_features'],
                antigen_chainid=cfg['antigen_chainid']
            )
        elif cfg['graph_type'] == 'atom':
            from AtomGraph import AtomGraph
            g = AtomGraph(
                pdb=pdb_path,
                antigen_chainid=cfg['antigen_chainid'],
                region_map=cfg['region_map'],
                use_voro=cfg['use_voro'],
                contact_features=cfg['contact_features'],
            )
        else:
            raise ValueError(f"Unknown graph_type: {cfg['graph_type']}")
        
        # Optionally compute score against reference
        if cfg['ref'] is not None:
            g.get_score(cfg['ref'])
        
        return g
    
    except Exception as e:
        # Return error tuple instead of raising
        return ('ERROR', pdb_path, str(e))


def _writer_process(result_queue, outfile, total_count, use_tqdm=True):
    """
    Writer process: drain result queue and write graphs to HDF5.
    Runs continuously until receiving sentinel value.
    """
    graphs_written = 0
    errors = []
    
    # Open HDF5 file once
    with h5py.File(outfile, 'w') as f5:
        if use_tqdm:
            pbar = tqdm(total=total_count, desc="Writing graphs", file=sys.stdout)
        
        while True:
            result = result_queue.get()
            
            # Sentinel: None means all work is done
            if result is None:
                break
            
            # Handle errors
            if isinstance(result, tuple) and result[0] == 'ERROR':
                _, pdb_path, error_msg = result
                errors.append((pdb_path, error_msg))
                if use_tqdm:
                    pbar.update(1)
                continue
            
            # Write graph to HDF5
            try:
                result.nx2h5(f5)
                graphs_written += 1
            except Exception as e:
                errors.append((getattr(result, 'pdb', 'UNKNOWN'), str(e)))
            
            if use_tqdm:
                pbar.update(1)
        
        if use_tqdm:
            pbar.close()
    
    # Report errors if any
    if errors:
        print(f"\n{len(errors)} graphs failed:")
        for pdb, err in errors[:10]:  # Show first 10
            print(f"  {pdb}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"Successfully wrote {graphs_written}/{total_count} graphs to {outfile}")


class GraphHDF5(object):
    def __init__(
        self,
        pdb_path,
        ref_path=None,
        graph_type="residue",
        embedding_path=None,
        select=None,
        outfile="graph.hdf5",
        nproc=1,
        use_tqdm=True,
        tmpdir="./",
        limit=None,
        biopython=False,
        use_regions=True,
        region_json=None,
        add_orientation=False,
        contact_features: bool = True,
        antigen_chainid: str = 'A',
        use_voro: bool = False,
    ):
        """Entry point that orchestrates graph construction and output."""
        # --- gather PDB files ---
        pdbs = [f for f in os.listdir(pdb_path) if f.endswith(".pdb")]
        pdbs = [f for f in pdbs if '_xray_tidy' not in f]
        
        if select:
            pdbs = [f for f in pdbs if f.startswith(select)]
        pdbs = [os.path.join(pdb_path, f) for f in pdbs]
        if limit is not None:
            pdbs = pdbs[limit[0]:limit[1]] if isinstance(limit, list) else pdbs[:limit]
        
        if len(pdbs) == 0:
            raise FileNotFoundError(f"no .pdb files found under {pdb_path}")
        
        # --- Sort PDBs by file size (descending) for better load balancing ---
        try:
            pdbs = sorted(pdbs, key=lambda x: os.path.getsize(x), reverse=True)
        except OSError:
            pass  # If sizing fails, proceed with unsorted list
        
        # --- reference structure ---
        base = os.path.splitext(os.path.basename(pdbs[0]))[0] if pdbs else "model"
        ref = os.path.join(ref_path, base + ".pdb") if ref_path else None
        
        # --- embeddings are optional ---
        self.embedding_path = embedding_path
        if self.embedding_path is None:
            pass
        
        self.antigen_chainid = antigen_chainid
        self.use_regions = use_regions
        self.graph_type = graph_type
        self.use_voro = use_voro
        
        # --- region mapping (optional) ---
        self.region_map = {}
        if self.use_regions:
            if not region_json or not os.path.isfile(region_json):
                raise FileNotFoundError(f"region_json not found: {region_json}")
            with open(region_json) as f:
                raw = json.load(f)
            
            parser = PDBParser(QUIET=True)
            for pdbfile in pdbs:
                model_name = os.path.basename(pdbfile).replace('.pdb', '')
                if model_name not in raw:
                    continue
                
                annot = raw[model_name]
                struct = parser.get_structure(model_name, pdbfile)
                chainA = struct[0][next(c.get_id() for c in struct[0] if c.get_id() != antigen_chainid)]
                pdb_nums = [r.get_id()[1] for r in chainA.get_residues()]
                num_annot = len(annot)
                
                for i, resnum in enumerate(pdb_nums):
                    if i < num_annot:
                        _, tag = annot[i]
                    else:
                        tag = "UNK"
                    self.region_map[(model_name, resnum)] = tag
        
        # feature flags
        self.add_orientation = add_orientation
        self.contact_features = contact_features
        
        # --- build graphs ---
        if nproc == 1:
            # Sequential mode (unchanged for simplicity and debugging)
            self.get_all_graphs(
                pdbs, ref, outfile,
                use_tqdm=use_tqdm,
                biopython=biopython,
                region_map=self.region_map,
                antigen_chainid=self.antigen_chainid
            )
        else:
            # Producer-consumer parallel pipeline
            self._parallel_pipeline(
                pdbs, ref, outfile, nproc, use_tqdm, biopython
            )
        
        # --- clean up temporary files ---
        for ext in ("*.izone", "*.lzone", "*.refpairs"):
            for fn in glob.glob(ext):
                try:
                    os.remove(fn)
                except OSError:
                    pass
        
        # --- append embeddings only when provided ---
        if self.embedding_path:
            try:
                self._add_embedding(outfile=outfile, pdbs=pdbs, embedding_path=self.embedding_path)
            except Exception as e:
                print(f"warning: embedding step failed; graphs are ready: {e}", flush=True)
    
    
    def _parallel_pipeline(self, pdbs, ref, outfile, nproc, use_tqdm, biopython):
        """
        Producer-consumer pipeline:
        - Workers pull PDB paths from work queue, compute graphs, put on result queue
        - Writer process drains result queue and writes to HDF5
        - No pickle intermediate files
        """
        # Configuration passed to each worker once at initialization
        worker_config = {
            'graph_type': self.graph_type,
            'biopython': biopython,
            'region_map': self.region_map if self.use_regions else None,
            'add_orientation': self.add_orientation,
            'contact_features': self.contact_features,
            'antigen_chainid': self.antigen_chainid,
            'use_voro': self.use_voro,
            'ref': ref,
        }
        
        # Result queue with bounded size to prevent memory overflow
        # Adjust maxsize based on available memory (50-100 is reasonable)
        result_queue = Queue(maxsize=100)
        
        # Start writer process
        writer = Process(
            target=_writer_process,
            args=(result_queue, outfile, len(pdbs), use_tqdm)
        )
        writer.start()
        
        # Create worker pool with initialization function
        pool = mp.Pool(
            processes=nproc,
            initializer=_worker_init,
            initargs=(worker_config,)
        )
        
        # Process graphs and feed results to queue
        # imap_unordered for better load balancing (results arrive as completed)
        try:
            for result in pool.imap_unordered(_worker_process_graph, pdbs, chunksize=1):
                result_queue.put(result)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            pool.terminate()
            pool.join()
            result_queue.put(None)  # Signal writer to stop
            writer.join()
            raise
        
        # Clean shutdown
        pool.close()
        pool.join()
        
        # Signal writer that all work is done
        result_queue.put(None)
        writer.join()
    
    
    def get_all_graphs(
            self,
            pdbs,
            ref,
            outfile,
            use_tqdm=True,
            biopython=False,
            region_map=None,
            antigen_chainid='A'
        ):
        """Generate all graphs sequentially (nproc=1 mode)."""
        graphs = []
        
        if use_tqdm:
            desc = "{:25s}".format("   create HDF5")
            lst = tqdm(pdbs, desc=desc, file=sys.stdout)
        else:
            lst = pdbs
        
        for name in lst:
            try:
                g = self._get_one_graph(
                    name,
                    ref,
                    biopython,
                    region_map=self.region_map if self.use_regions else None,
                    antigen_chainid=self.antigen_chainid,
                    embedding_path=self.embedding_path,
                )
                if g is None:
                    print(f"[WARN] Skipped graph for {name}")
                    continue
                graphs.append(g)
            except Exception as e:
                print(f"error computing graph {name}: {e}")
        
        with h5py.File(outfile, "w") as f5:
            for g in graphs:
                if g is None:
                    continue
                try:
                    g.nx2h5(f5)
                except Exception as e:
                    print(f"error writing graph {getattr(g,'pdb','UNKNOWN')}: {e}")
    
    
    def _get_one_graph(self, name, ref, biopython=False, region_map=None, antigen_chainid='A', embedding_path=None):
        """Build one graph and return it."""
        try:
            if self.graph_type == 'residue':
                from ResidueGraph import ResidueGraph
                g = ResidueGraph(
                    pdb=name,
                    biopython=biopython,
                    region_map=self.region_map,
                    add_orientation=self.add_orientation,
                    contact_features=self.contact_features,
                    antigen_chainid=antigen_chainid
                )
                if ref is not None:
                    g.get_score(ref)
            
            elif self.graph_type == 'atom':
                from AtomGraph import AtomGraph
                g = AtomGraph(
                    pdb=name,
                    antigen_chainid=antigen_chainid,
                    region_map=region_map,
                    use_voro=self.use_voro,
                )
            return g
        
        except Exception as e:
            print(f"[WARN] Failed graph {name}: {e}")
            return None
    
    
    @staticmethod
    def _add_embedding(outfile, pdbs, embedding_path):
        """Append per-residue embeddings to the HDF5 if available."""
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"PyTorch is required to add embeddings: {e}")
        
        with h5py.File(outfile, 'r+') as f:
            mols = list(f.keys())
            for mol in mols:
                residues = f[mol]['nodes'][()]
                emb_tensor = torch.zeros(len(residues), 1)

                try:
                    pdb_file = next(i for i in pdbs if mol in i)
                except StopIteration:
                    pdb_file = None

                # Cache loaded .pt files to avoid repeated loading for same chain
                pt_cache = {}

                for i in range(len(residues)):
                    chainID = residues[i][0].decode()
                    resID   = residues[i][1].decode()
                    pt_name = mol + '.' + chainID + '.pt'
                    pt_path = os.path.join(embedding_path, pt_name)
                    res_number = int(resID)
                    try:
                        # Load from cache or file
                        if pt_path not in pt_cache:
                            data = torch.load(pt_path)
                            pt_cache[pt_path] = data["representations"][33]
                        vecs = pt_cache[pt_path]
                        embedding = vecs[res_number - 1].mean()
                        emb_tensor[i] = embedding
                    except Exception:
                        emb_tensor[i] = 0

                grp = f[mol].require_group('node_data')
                if 'embedding' in grp:
                    del grp['embedding']
                grp.create_dataset('embedding', data=emb_tensor.numpy())
            
            print(f"embeddings written to {outfile}")
