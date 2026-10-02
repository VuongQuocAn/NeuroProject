# Local demo patients

`local-demo.zip` is included in normal Git (about 26 MB). It contains a
`manifest.json`, checksums and the actual storage files for two demo cases
exported from `neuroproject-tunnel`:

| Patient | Bundled inputs |
|---|---|
| `UCSF-003` | 3 MRI images, 3 WSI series with 100 tiles each, RNA expression CSV, clinical record |
| `UCSF-001` | 3 MRI images; no current WSI, RNA or clinical record in the source database |

Both cases also include their available diagnosis/prognosis results, XAI
images, explanations and patient history records. Historical results can
reflect inputs available at the time of that run. Failed/pending jobs are not
seeded. WSI/RNA attached to UCSF-003 were uploaded for testing from the
TCGA-12-1093 sample; these are demonstration inputs, not a claim of a matched
biological cohort. The WSI files are processed tiles, and the RNA CSV is an
expression matrix, not raw SVS/FASTQ/BAM data.

The local backend verifies every archived file, copies storage objects into
local MinIO and inserts the records with newly assigned database IDs. The two
patients belong to the local `admin` account. Existing patients with the same
external code and existing data are kept as they are. An empty profile owned by
local admin can receive the bundled data without changing its ID or name.
Restarting does not duplicate or overwrite populated cases. No source accounts,
passwords, API keys, chat sessions or board posts
are included. `local-demo.manifest.json` is the readable inventory of the ZIP.

`docker-compose.local.yml` enables this automatically. Other stacks leave the
patient seed disabled unless `LOCAL_DEMO_SEED=true` is explicitly set. Set
`LOCAL_DEMO_SEED=false` in the root `.env` before the first local startup to
start without these two cases. This flag does not delete existing patients.

To export a new snapshot from the source backend, after reviewing which demo
data will be shared:

```powershell
docker exec neuroproject-tunnel-backend-1 python scripts/export_local_demo.py --output /tmp/local-demo.zip
docker cp neuroproject-tunnel-backend-1:/tmp/local-demo.zip backend/demo_data/local-demo.zip
docker cp neuroproject-tunnel-backend-1:/tmp/local-demo.manifest.json backend/demo_data/local-demo.manifest.json
```

This export only reads the source data. Fresh clones need no R2 credentials
or connection to the author's database to load the bundled cases.
