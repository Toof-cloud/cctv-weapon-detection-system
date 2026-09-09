# Forensikada video detection mockup

The interface visualizes the existing weapon-detection system:
**Import video → confirm threshold and start detection → scan with the third trained model → play the annotated
video and show its detection summary.** The service loads the trained model,
detects its **handgun** and **knife** classes, and produces the boxes and report.

**Original project files modified: NONE.** Changes are confined to `mockup_ui/`.
No model was retrained, replaced, copied or modified. The original whole-video
detection function is imported and called directly.

## Run

From the repository root:

```powershell
.\mockup_ui\.venv\Scripts\python.exe -B .\mockup_ui\app.py
```

Or use the existing working detector environment:

```powershell
python -B mockup_ui/app.py
```

Optional video on startup:

```powershell
python -B mockup_ui/app.py --video "D:\your-videos\cctv.mp4"
```

Replace the example video path with your real file. The locally created `.venv/`
contains PySide6, OpenCV, Pillow, NumPy, PyTorch 2.13.0 CPU and torchvision 0.28.0
CPU. Your original `requirements.txt` and other environments are unchanged. Use
an existing working CUDA environment for GPU inference if available.

## Demonstration

1. **Import video** or drop a local video onto the viewer. The video loads without starting detection.
2. Review the video details and confidence threshold in the configuration dialog.
   Select **Start detection** to continue, or cancel and use **Analyze video** later.
3. The detection service loads the trained model and
   scans every video frame. The UI shows loading/scanning progress. Playback and
   result controls remain disabled; summary values stay pending while scanning.
4. Once processing and output validation finish, the annotated video automatically
   plays from the beginning at its source frame rate. It contains the boxes,
   labels and confidence scores drawn by the existing detection service.
5. The summary and observation table show results for the **entire processed video**.
   Play, pause, seek, or click a detection row to review a particular frame. Only
   **Current frame** details change during playback; full-video totals stay fixed.
6. **Forensic report** opens the completed analysis inside the application.
7. **Save video + report** asks for a destination and exports an annotated MP4, the original CSV/JSON results,
   and a readable forensic report with structured records.

After analysis, **Observation review** opens the analyst decision panel. Saved
reviews can be resumed with **Open saved review…**, including after restarting
the application. Opening a review never reruns detection.

Importing never starts analysis. The toolbar contains **Import video**,
**Forensic report**, and **Save video + report**. An **Analyze video** control opens
the threshold confirmation whenever an imported video remains unprocessed.
The app does not open a model picker. If the required model is missing after the
user explicitly starts detection, the imported video waits while the required path
is checked every two seconds. Adding `best_weapon_detector_third_model.pth` to
`mockup_ui/models/` resumes that requested scan without reimporting the video.

Importing displays a still preview and metadata. The player is a result viewer:
its playback controls become available after detection completes. You can adjust
the threshold before importing a video and compare Original and Detected views afterward.
Playing or seeking the completed video does not rerun inference or change its
summary. Importing another video clears the previous result. Reimport a video
to scan it again with a changed threshold or retry after a processing error.

Short clips are best for the defense, especially on CPU. Processing every frame
can take several minutes. The UI remains responsive, but only one analysis runs
at a time. The unchanged detection helper does not offer cancellation; allow the
scan to finish before closing the window. Processing speed affects how long you
wait for the result, while completed playback uses the original frame rate.

The player and OpenCV-generated annotated video are video-only; audio is not
played or included in the export. Timestamps use frame number divided by source
FPS, so fixed-frame-rate footage is preferable. Saved video retains the source
FPS regardless of how long analysis took.

## Trained checkpoint

The UI uses **only `mockup_ui/models/best_weapon_detector_third_model.pth`** for
every video. The Trained model panel displays its filename and full-path tooltip.
The baseline and retrained files may remain in the directory but are never selected.
There is no fallback: if the third checkpoint is missing, scanning waits for it.

Model selection, its keyboard shortcut, and the `--model` option have been removed.
Legacy `settings.json` paths and `FORENSIKADA_MODEL_PATH` overrides are ignored.
The application reads the required checkpoint in place without changing its weights
or filename. `mockup_ui/models/` is created automatically and is excluded from Git.
The application has no simulated-detection mode.

Class mapping remains the original service's mapping:
`1 = handgun`, `2 = knife`, with background ignored.

The original `get_model()` requests torchvision `weights="DEFAULT"` before loading
your trained state dict. It therefore still needs its initialization-weight cache.
If that cache is unavailable, the original loader may download it on first use;
the mockup directs that cache into `mockup_ui/temp/torch/hub/`. An existing cache
is read in place. This initialization requirement does not replace the trained
weapon detector. Run a real clip with your checkpoint successfully before the
presentation, then confirm a second run works offline.

## Forensic report

After processing, **Forensic report** displays video information, detection
configuration, full-video totals, processing details, timeline rows, bounding boxes,
confidence scores, and review status in a large in-app dialog.

**Save video + report** opens a destination-folder chooser. Cancelling it writes
nothing. Confirming creates a uniquely named result folder there containing:

- `forensic_report.pdf`: paginated report with separate automated and analyst results.
- `forensic_report.html`: readable browser version with the same saved decisions.
- `forensic_report.json`: report metadata and every structured detection record.
- `forensic_records.csv`: one row per observation, including source and model references.
- `observation_reviews.json`: reopenable review snapshot with source references,
  detection details, decisions, notes, UTC timestamps, and review revision history.

Each record includes a report/record ID, source-video path/name, zero-based frame
number, video-relative time, object label, bounding box, confidence, and validation
status. Camera identifier and source recording timestamp fields are included but
marked **not recorded by the current pipeline**. File modification dates and
report-generation times are never substituted for a recording timestamp.

Every detection begins with **no analyst decision**. Completed processing and high
confidence do not select a decision. Repeated detections across frames do not count
unique weapons. Reports retain all observations, including Reject and Uncertain.
Detection totals and annotated boxes continue to represent the original model output.

## Observation Review

The panel lists every observation and shows its annotated frame, source path,
frame number, video offset, object class, confidence, box, and stable observation ID.
Select one of exactly three decision options:

| Decision | Notes | Save behavior |
| --- | --- | --- |
| Accept | Optional | Empty notes are valid. |
| Reject | Required | Empty or whitespace-only notes are blocked with a reason-required message. |
| Uncertain | Optional | Empty notes are valid. |

The options are mutually exclusive. No option is preselected for a new observation.
Changing the decision immediately updates the notes label, placeholder, and validation
rules. Save without selecting a decision is blocked. **Save review** persists a
trimmed note and UTC review timestamp; previously saved decisions can be revised.
Unsaved drafts survive navigation within the panel and are explicitly discarded
only when closing with confirmation. Saved decisions survive reopening and app restart.

Completed analyses are stored under `mockup_ui/reviews/<run-id>.json`. Each new
analysis receives a distinct run ID so reviews cannot silently carry over to a
different model run. Use **Open saved review…** and choose that file, or choose
`observation_reviews.json` from an export. The saved source/video paths are retained;
keep the original annotated-video files for frame previews. Detection records and
reviews remain readable if those files are unavailable.

**Automated Validation Result** is read-only and separate from **Analyst Review
Decision**. The current pipeline supplies detections and confidence filtering but
does not perform independent observation validation, so its value is **Not performed**.
If a detection explicitly supplies `automated_validation_status`, that exact value
is preserved. An analyst choice never overwrites the automated status.

Review files are written atomically and checked against the saved run's detection
details and source references. Export includes the latest saved decisions and is a
snapshot; export again after changing a decision to generate an updated PDF. Viewing
the in-app report does not create a PDF. The PDF is generated only on explicit export.

PDF generation uses ReportLab in the isolated app environment. To set up another
working detector environment with the UI/PDF dependencies:

```powershell
python -m pip install -r mockup_ui/requirements-ui.txt
```

The report records the model actually used by its saved run. Historical results
retain their original model reference, even though new processing uses only the
third checkpoint. The CSV and summary must agree before a report can be generated.
Available source and input-record SHA-256 hashes are measured at report generation;
they are not presented as an earlier ingestion record or chain-of-custody history.
Newly saved runs also record resolution, source frame count, and analysis completion
time. Older saved results leave unavailable fields blank with explicit statuses.

Create a report from an existing export without rerunning inference:

```powershell
.\mockup_ui\.venv\Scripts\python.exe -B -m mockup_ui.forensic_report "mockup_ui\outputs\YOUR_SAVED_RUN\summary.json"
```

This creates a new report subfolder and preserves the previous export.

## How the UI calls the existing system

```text
Video upload and preview
  -> app.services.video_service.VideoService (validation and metadata)
  -> mockup_ui.model_bridge.ModelBridge.analyze_video()
  -> run_full_pipeline.detect_video_with_model()
     -> app.services.detection_service.DetectionService
     -> dataset_analysis.build_model.get_model()
     -> existing trained checkpoint
     -> DetectionService.detect_frame() for every frame
     -> draw_detections()
     -> complete annotated MP4 and detection CSV
  -> validate completed output and read the service's report
  -> automatically play annotated video and populate full-video summary
```

The background worker calls the existing whole-video helper without altering it.
The helper owns model loading, inference, filtering, box drawing, video writing
and CSV generation. The adapter chooses isolated output paths, requests every
frame, translates console progress into UI updates, checks output completeness,
and reads the resulting CSV. Summary counts come from that completed report.
The existing `VideoService` supplies validation and metadata. Result playback
and seeking use OpenCV with a Qt timer after the worker produces a valid result.

The original helper initializes the detector once per video run.
It handles all frames within that run with the same loaded third model. Choosing a new
video or rerunning detection clears old results so they
cannot be mistaken for the current run. Failed/incomplete processing is not
presented as a completed result. Failed runs keep totals pending and disable
playback/export; any partial files remain isolated under `temp/`.

The application does not run video enhancement, training, multi-camera comparison,
object tracking, or case management. Analyst decisions are entered manually.

## What the results mean

- **Detections:** the total number of handgun/knife observations across analyzed frames.
- **Handguns / Knives:** those same observations grouped by class.
- **Frames with detections:** how many distinct frames contain at least one observation.
- **Confidence:** the detector's score, rounded by the existing CSV exporter.
- **Frame:** zero-based frame number; the corresponding time is frame number divided by FPS.
- **Bounding box:** `[x1, y1, x2, y2]` in that frame's pixel coordinates.

These counts do not represent unique physical weapons. A knife detected in ten
frames contributes ten observations. The model is not an identity tracker. A
no-detection result means no supported object met the selected confidence
threshold. Changing the threshold affects the next run, not previous results.

## Files and storage

```text
mockup_ui/
├── app.py                       Video interface, results table and background worker
├── model_bridge.py              Adapter calling the original whole-video detection function
├── forensic_report.py           Readable report and structured detection-record exports
├── observation_review.py        Stable observation IDs and atomic review persistence
├── review_panel.py              Analyst decision controls, notes and frame preview
├── pdf_report.py                Paginated PDF with automated and analyst results
├── requirements-ui.txt          Desktop UI and PDF dependencies
├── video_player.py              OpenCV playback, pause, timeline and exact-frame seeking
├── styles.qss                   Figma-derived Qt styles
├── test_mockup.py               Video integration and UI regression checks
├── test_forensic_report.py      Report traceability, missing metadata and consistency checks
├── test_observation_review.py   Decision rules, reopening, persistence, and PDF checks
├── README.md                    These instructions
├── .gitignore                   Excludes runtime files
├── models/                      Required location of best_weapon_detector_third_model.pth
├── reviews/                     Persistent completed runs and analyst review records
├── assets/
│   ├── forensikada-logo.png      Original Figma logo
│   ├── import-image.png         Existing Figma folder icon, reused for video import
│   ├── run-analysis.svg         Original Figma analysis icon
│   ├── save-result.svg          Original Figma report icon
│   ├── InterVariable.ttf        Bundled Inter font
│   └── Inter-LICENSE.txt        Original SIL Open Font License
├── .venv/                       Isolated local runtime; unchanged by this video update
├── temp/                        Logs, test renders and per-run playback files
│   └── video-run-*/
│       ├── annotated.mp4         Completed/partial annotated video for that run
│       └── detections.csv        Per-frame observations from the original detector
└── outputs/
    └── <video-name>_<time>_<id>/
        ├── annotated.mp4
        ├── detections.csv
        ├── summary.json
        ├── forensic_report.html
        ├── forensic_report.pdf
        ├── forensic_report.json
        ├── observation_reviews.json
        └── forensic_records.csv
```

Uploaded videos are read from their original locations. They are not modified,
moved or duplicated. Each analysis gets a fresh directory under `temp/`, and
explicit exports get unique directories under `outputs/`. Earlier outputs are
not overwritten. Temporary run files remain available for playback; they can be
removed when the application is closed. Logs are in `temp/mockup.log`.

The visual reference remains the [Forensikada Figma dashboard](https://www.figma.com/design/woIsbmBEZTJ4RYb5b7tgcd/Forensikada-UI?node-id=2-6):
charcoal toolbar, white side panels, central video, observation table, Inter font
and indigo selection accents. Existing exported logo/icons and the Inter license
are reused locally, with no new visual downloads or dependencies in this update.

## Verification

```powershell
.\mockup_ui\.venv\Scripts\python.exe -B -m unittest mockup_ui.test_mockup mockup_ui.test_forensic_report mockup_ui.test_observation_review -v
```

All **19 tests passed**, including six observation-review checks and three forensic-report checks. They cover video validation, Unicode paths, metadata,
missing checkpoints, changed inputs, incomplete processing, original per-frame
inference/preprocessing, actual MP4 encoding/decoding, bounding-box rendering,
CSV/JSON exports, playback/seeking, progress, repeated-click prevention, no
detections, exact-frame navigation, error recovery and clearing stale results.
Workflow checks verify that playback and totals wait until processing finishes,
the processed file starts playing automatically, and full-video totals remain
constant when reviewing other frames. They also cover manual threshold confirmation,
cancelled configuration retaining the imported video, finding a newly added
third checkpoint while a video waits, exclusive selection despite other available
weights or legacy overrides, and refusal to fall back if the required file is removed.

For these tests, only checkpoint loading/predictions are controlled fixtures;
the original whole-video loop, `detect_frame()` and renderer execute on a
generated clip. Separate UI fixtures check scan progress and result review.
No alternative trained weights are created or used, and the production
application has no test-prediction fallback. These tests verify software integration.
The uploaded third checkpoint passed a separate real-model loading and one-frame
processing check on CPU, producing an annotated MP4 and CSV. The log is
`temp/third-model-validation.log`, including the exact checkpoint and output paths.
That blank-frame check verifies compatibility and output generation. Detection
accuracy on representative CCTV footage has not yet been evaluated here.

The updated UI was rendered at 1450 × 880 and 1060 × 760. Previews are in
`temp/auto-import-preview.png` and `temp/auto-import-preview-small.png`; test output is in
`temp/third-model-tests.txt` for the current exclusive-model checks. Renders of the scan and completed-result states use
clearly marked test predictions in `temp/scanning-test.png` and
`temp/processed-result-test.png`.
Worker updates use explicitly queued GUI slots and
thread completion before disposal, following [Qt's thread-affinity guidance](https://doc.qt.io/qtforpython-6/tutorials/basictutorial/signals_and_slots.html#thread-affinity).

The current combined run is recorded in `temp/observation-review-tests.txt`.
Review checks cover all three decision rules, whitespace rejection, changing
decisions, no default selection, navigation, reopening from disk, new-run isolation,
failed-write recovery, source/status preservation, and matching PDF/report values.
The panel was rendered at 1120 × 800 and 920 × 700. All three pages of the synthetic
PDF fixture, including long notes, were rendered and inspected. QA artifacts are
under `temp/review-visual-check/` and contain explicitly synthetic observations and
test analyst decisions only.
