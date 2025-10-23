# Music Beats Insurance MLOps — End-to-End Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.10-blue) ![MongoDB](https://img.shields.io/badge/database-MongoDB-green) ![AWS](https://img.shields.io/badge/cloud-AWS-orange) ![Docker](https://img.shields.io/badge/docker-ready-lightblue)

**One-line:** Production-style MLOps template built for **Music Beats Insurance** data — from raw data (MongoDB) → validated pipelines → trained models → model registry (S3) → Dockerized deployment with CI/CD (GitHub Actions → ECR → EC2).

---

## Why this project matters (recruiter elevator pitch)

Shows end-to-end MLOps skills — data engineering (MongoDB), validation & feature engineering, model lifecycle management, infra automation (AWS S3/ECR/EC2), containerization, and CI/CD. Ideal to demonstrate practical production-readiness and infra-aware ML best practices.

---

## Key highlights

* Modular components: **Data Ingestion → Validation → Transformation → Trainer → Evaluator → Pusher**
* MongoDB Atlas for raw/key-value storage + notebooks for EDA & data push
* Model registry on **AWS S3** and CI/CD pipeline with **Docker + GitHub Actions → ECR → EC2**
* Structured logging and custom exception handling with `demo.py`
* Clear code layout, config-driven validation (`config/schema.yaml`) and unit-testable components

---

## Tech stack

Python 3.10 · Conda · Pandas · scikit-learn · MongoDB Atlas · AWS (S3, ECR, EC2, IAM) · Docker · GitHub Actions · Jupyter Notebooks

---

## Repository layout (essential)

```
music-mlops/
├─ src/
│  ├─ configuration/         # mongo_db_connections.py, aws_connection.py
│  ├─ data_access/           # proj1_data.py (fetch → df)
│  ├─ components/            # data_ingestion/validation/transformation/trainer/evaluator
│  ├─ entity/                # config_entity.py, artifact_entity.py, estimator.py, s3_estimator.py
│  ├─ aws_storage/           # s3 client / helpers
│  └─ utils/                 # main_utils.py (validation helpers)
├─ notebook/                 # mongoDB_demo.ipynb, EDA_feature_engg.ipynb
├─ templates/template.py
├─ demo.py
├─ app.py
├─ requirements.txt
├─ setup.py, pyproject.toml
├─ Dockerfile
└─ .github/workflows/aws.yaml
```

---

## Quickstart — copy, paste, run

> Replace placeholders before running (see checklist below).

```bash
# 1. Generate template (if not present)
python templates/template.py

# 2. Create & activate Conda env
conda create -n music python=3.10 -y
conda activate music

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install package locally so imports work
pip install -e .

# 5. Set MongoDB URL (example)
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/mydb"
# Windows PowerShell:
# $env:MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/mydb"

# 6. Run demo (data ingestion + logging demo)
python demo.py

# 7. Start web app (prediction / training endpoints)
python app.py
# If deployed on EC2, visit: http://<EC2_PUBLIC_IP>:5080
```

---

## Core implementation notes (what to open first)

* `src/configuration/mongo_db_connections.py` — Mongo connection helpers
* `data_access/proj1_data.py` — fetch raw KV records and convert to DataFrame
* `components/data_ingestion.py` — ingestion orchestration & artifacts
* `utils/main_utils.py` + `config/schema.yaml` — validation rules & helpers
* `components/data_transformation.py` + `entity/estimator.py` — feature pipeline & estimator contract
* `components/model_trainer.py` & `components/model_evaluation.py` — train and compare model (threshold in constants)
* `src/aws_storage/s3_client.py` & `entity/s3_estimator.py` — push/pull models from S3

---

## AWS infra (summary)

1. Create IAM user with appropriate policy (for demo you can use `AdministratorAccess` — **do not use long-term keys in production**).
2. Export credentials:

```bash
export AWS_ACCESS_KEY_ID="<ID>"
export AWS_SECRET_ACCESS_KEY="<SECRET>"
export AWS_DEFAULT_REGION="us-east-1"
```

3. Create S3 bucket: `my-model-mlopsproj` (region: `us-east-1`) — set constants in `src/constants/__init__.py`:

```py
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

4. Implement S3 push/pull logic in `src/aws_storage` and use `s3_estimator.py` for model registry operations.

---

## CI/CD & Deployment (high level)

* Dockerfile builds the app image; `.github/workflows/aws.yaml` builds + pushes to ECR.
* Create ECR repo `musicproj` and add GitHub secrets:

  * `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `ECR_REPO`
* Optional: create a self-hosted runner on EC2 for heavier deploy steps (connect via GitHub Actions → Settings → Runners).

---

## Demo endpoints (example)

* `GET /` — app home
* `POST /predict` — model prediction (JSON)
* `GET /training` — trigger training pipeline (or `POST` depending on implementation)

---

## Small model snapshot (paste real values)

| Metric         |                                    Value |
| -------------- | ---------------------------------------: |
| Model type     |         RandomForest / CatBoost / Custom |
| Validation AUC |                           0.92 (example) |
| Baseline AUC   |                                     0.88 |
| Deployed       | S3: `my-model-mlopsproj/model-registry/` |

*(Replace with your actual numbers — short, strong metrics help recruiters instantly.)*

---

## Quick safety & pre-publish checklist (MUST DO)

* Replace all placeholders: `<username>`, `<password>`, `<cluster>`, `<EC2_PUBLIC_IP>`, `<ID>`, `<SECRET>`, `email@example.com`, `[LinkedIn URL]`.
* **Never** commit secrets or `.env` with keys — use GitHub Secrets for CI.
* Add to `.gitignore`: `artifact/`, `*.ckpt`, `*.pkl`, `.env`, `__pycache__/`.
* Validate `setup.py` & `pyproject.toml` work by running `pip install -e .` then `python -c "import music_mlops"`.
* Ensure `config/schema.yaml` accurately reflects your dataset (used by validation).
* Add a short demo GIF or link (hosted) if possible — recruiters love clickable proof.

---

## Want this README tailored to your repo?

I can:

* Replace placeholders with your real links & metrics.
* Produce a **one-page recruiter version** (shorter) or a **detailed developer version** (longer with setup scripts).
* Create a short demo script `run_demo.sh` you can include.

**Contact:** Anil Kumar — [nanil6304@gmail.com](mailto:nanil6304@gmail.com)

---

