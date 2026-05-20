import os
import json
import mygene

# Path to cache file in a persistent location or alongside the module
CACHE_FILE = os.path.join(os.path.dirname(__file__), "gene_symbol_cache.json")

class GeneMapper:
    """
    Utility class to map Ensembl Gene IDs (ENSG...) to standard Gene Symbols (e.g., EGFR).
    Uses mygene API and local caching to avoid repeated slow network calls.
    """
    def __init__(self):
        self.mg = mygene.MyGeneInfo()
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[GeneMapper] Error loading cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[GeneMapper] Error saving cache: {e}")

    def map_ensembl_to_symbols(self, ensembl_ids: list[str]) -> dict[str, str]:
        """
        Maps a list of Ensembl IDs to Gene Symbols.
        Returns a dictionary { "ENSG00000...": "EGFR", ... }
        """
        results = {}
        missing_ids = []

        # Check cache first
        for eid in ensembl_ids:
            # Ensembl IDs often have versions (e.g., ENSG00000146648.14). Remove the version for querying.
            base_id = eid.split(".")[0] if "." in eid else eid
            if base_id in self.cache:
                results[eid] = self.cache[base_id]
            else:
                missing_ids.append(base_id)

        if not missing_ids:
            return results

        # Query mygene for missing IDs
        try:
            print(f"[GeneMapper] Querying mygene for {len(missing_ids)} missing IDs...")
            # Query in batches implicitly handled by mygene or explicit
            query_results = self.mg.querymany(
                missing_ids, 
                scopes='ensembl.gene', 
                fields='symbol', 
                species='human',
                as_dataframe=False,
                verbose=False
            )

            # Process results
            for hit in query_results:
                query_id = hit.get('query')
                symbol = hit.get('symbol', query_id) # fallback to query_id if no symbol
                
                # Cache the base_id
                self.cache[query_id] = symbol

            self._save_cache()

            # Re-map the original full IDs
            for eid in ensembl_ids:
                base_id = eid.split(".")[0] if "." in eid else eid
                results[eid] = self.cache.get(base_id, eid)

        except Exception as e:
            print(f"[GeneMapper] Query failed: {e}")
            # Fallback: just return the IDs themselves
            for eid in ensembl_ids:
                if eid not in results:
                    results[eid] = eid

        return results

# Singleton instance for easy import
gene_mapper = GeneMapper()
