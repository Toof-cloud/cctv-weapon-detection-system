from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.cctv_intelligence import TemporalConsistencyFilter, CCTVIntelligenceFilter


def test_temporal_filter():
    print("=" * 60)
    print("TESTING TEMPORAL CONSISTENCY FILTER")
    print("=" * 60)

    temporal = TemporalConsistencyFilter(min_hits=2, max_missing_frames=3, max_match_distance=50.0, enable_label_smoothing=True)

    # Test 1: Single isolated false alarm on Frame 10
    cand_flicker = [{"class_name": "knife", "box": [100, 100, 150, 150], "confidence": 0.70}]
    res_flicker = temporal.update(cand_flicker, frame_idx=10)
    assert res_flicker[0]["temporal_hits"] == 1
    assert res_flicker[0]["is_temporally_consistent"] is False
    print("Test 1 Passed: Frame 10 isolated hit has hits=1 and is_temporally_consistent=False")

    # Test 2: Multi-frame genuine threat starting at Frame 20
    cand_f20 = [{"class_name": "handgun", "box": [200, 200, 240, 240], "confidence": 0.88}]
    res_f20 = temporal.update(cand_f20, frame_idx=20)
    assert res_f20[0]["temporal_hits"] == 1
    assert res_f20[0]["is_temporally_consistent"] is False

    # Frame 21: same weapon moves slightly (drift = 5px)
    cand_f21 = [{"class_name": "handgun", "box": [203, 204, 243, 244], "confidence": 0.91}]
    res_f21 = temporal.update(cand_f21, frame_idx=21)
    assert res_f21[0]["temporal_hits"] == 2
    assert res_f21[0]["is_temporally_consistent"] is True
    print("Test 2 Passed: Frame 21 consecutive detection reaches hits=2 and is_temporally_consistent=True!")

    # Test 3: Label smoothing (Handgun -> momentary knife shadow -> Handgun)
    cand_f22 = [{"class_name": "handgun", "box": [205, 206, 245, 246], "confidence": 0.93}]
    temporal.update(cand_f22, frame_idx=22)

    cand_f23 = [{"class_name": "knife", "box": [206, 207, 246, 247], "confidence": 0.75}]  # Shadow knife
    res_f23 = temporal.update(cand_f23, frame_idx=23)
    assert res_f23[0]["class_name"] == "handgun", f"Expected smoothed label handgun, got {res_f23[0]['class_name']}"
    print("Test 3 Passed: Momentary knife shadow on Frame 23 was smoothed back to handgun!")

    # Test 4: Video reconciliation pass
    dummy_video_records = [
        # Track 1: Isolated 1-frame flicker
        {"frame_number": 5, "temporal_track_id": 101, "class_name": "knife", "status": "CONFIRMED_ALERT", "rejection_reason": None},
        # Track 2: Genuine weapon with 3 frames
        {"frame_number": 20, "temporal_track_id": 102, "class_name": "handgun", "status": "PENDING_CONFIRMATION", "rejection_reason": None},
        {"frame_number": 21, "temporal_track_id": 102, "class_name": "handgun", "status": "CONFIRMED_ALERT", "rejection_reason": None},
        {"frame_number": 22, "temporal_track_id": 102, "class_name": "handgun", "status": "CONFIRMED_ALERT", "rejection_reason": None},
    ]

    cctv_intel = CCTVIntelligenceFilter(enable_temporal_consistency=True, min_temporal_hits=2)
    reconciled = cctv_intel.reconcile_video_records(dummy_video_records, min_hits=2)

    # Track 1 should now be SUPPRESSED_TEMPORAL_FLICKER
    assert reconciled[0]["status"] == "SUPPRESSED"
    assert "SUPPRESSED_TEMPORAL_FLICKER" in reconciled[0]["rejection_reason"]
    # Track 2 initial frame should now be CONFIRMED_ALERT
    assert reconciled[1]["status"] == "CONFIRMED_ALERT"
    assert reconciled[1]["validation_status"] == "VALIDATED_TEMPORAL"
    print("Test 4 Passed: Video reconciliation suppressed 1-frame flicker and confirmed all genuine weapon frames!")

    print("\nALL TEMPORAL CONSISTENCY TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    test_temporal_filter()
