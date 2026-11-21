# Deploying FORENSIKADA Web App to Hugging Face Spaces (Free 24/7 Public URL)

## Step 1: Build / Update the Pre-Indexed Web Video Library
Whenever you add new videos (such as your groupmates' new staged non-handgun videos into `expert_evaluation_videos/NO_WEAPON/`), run:
```powershell
.\.venv\Scripts\python.exe web_app/build_web_library.py
```
This automatically runs **Model 12**, transcodes every video to browser-compatible H.264 MP4, computes SHA-256 hashes, and updates `web_app/library_data/library_index.json`.

## Step 2: Test Locally in Your Browser
```powershell
.\.venv\Scripts\streamlit.exe run web_app/app.py
```
Open `http://localhost:8501` in Chrome or Edge and verify Single-Video, Multi-Camera, Observation Review, Custom Upload, and Forensic PDF Report downloads.

## Step 3: Package & Deploy 24/7 to Hugging Face Spaces (Free Tier: 16 GB RAM, Open Access)
1. Run the one-command packager to build a clean, self-contained `hf_space_bundle/` folder (with root `Dockerfile`, HF Spaces `README.md`, `.gitattributes` for Git LFS, Model 12 weights, and the pre-indexed H.264 video library):
   ```powershell
   .\.venv\Scripts\python.exe web_app/prepare_hf_space_bundle.py
   ```
2. Go to **https://huggingface.co/spaces** and click **Create new Space**.
3. Name it **`forensikada-cctv`**, select **Docker** as the Space SDK, choose **CPU basic (2 vCPU, 16 GB RAM — Free)**, and set Visibility to **Public** (open access, no password).
4. Clone your new Space repo locally, copy everything inside `hf_space_bundle/` into your cloned Space folder, and push:
   ```powershell
   git lfs install
   git add .
   git commit -m "Deploy FORENSIKADA Web App with Model 12"
   git push
   ```
5. Within ~3–5 minutes, Hugging Face builds the Docker container and your permanent 24/7 public link is live at:  
   `https://huggingface.co/spaces/<your-username>/forensikada-cctv`

