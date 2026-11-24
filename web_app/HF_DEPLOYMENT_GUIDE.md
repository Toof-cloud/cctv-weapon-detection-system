# Deploying FORENSIKADA Web App (Free 24/7 Public URL on Streamlit Community Cloud)

## 1. Deploy 24/7 on Streamlit Community Cloud (100% Free, Auto-Updates from GitHub)
1. Open **[https://share.streamlit.io](https://share.streamlit.io)** and sign in with your **GitHub** account.
2. Click **Create app** (top right) -> **Deploy a public app from GitHub**.
3. Fill in the deployment form:
   - **Repository:** `Toof-cloud/cctv-weapon-detection-system` (or your forked repository)
   - **Branch:** `integrate-haawaard-final-ui`
   - **Main file path:** `web_app/app.py`
   - **App URL (optional):** `forensikada-cctv` -> gives you `https://forensikada-cctv.streamlit.app`
4. Click **Deploy!** — within ~2 minutes your permanent 24/7 public URL is live.

---

## 2. How to Change or Swap Any of the 17 Videos Anytime
All 87 Model 12 annotated candidate videos live in:
`outputs/00_GOOGLE_DRIVE_MODEL12_ANNOTATED_VIDEOS_FOR_SELECTION/`

If your groupmates decide to swap any video later:
1. Open [`web_app/update_selected_17_library.py`](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/web_app/update_selected_17_library.py) and change the `candidate_name` in `SELECTED_SINGLE_VIDEOS` or `SELECTED_MULTICAM_SCENES` to any candidate from `ALL_CANDIDATES_MODEL12_LEADERBOARD.csv`.
2. Run:
   ```powershell
   .\.venv\Scripts\python.exe web_app/update_selected_17_library.py
   ```
3. Commit and push to GitHub:
   ```powershell
   git add web_app/library_data web_app/update_selected_17_library.py
   git commit -m "Update selected evaluation videos"
   git push origin integrate-haawaard-final-ui
   ```
   **Streamlit Community Cloud automatically detects the GitHub push and updates the live 24/7 web app within seconds!**

---

## 3. Test Locally in Your Browser
```powershell
.\.venv\Scripts\streamlit.exe run web_app/app.py
```
Open `http://localhost:8501` in Chrome or Edge to test Single-Video, Multi-Camera, Observation Review, and Forensic PDF Report downloads locally.


