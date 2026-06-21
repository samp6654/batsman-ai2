"""
model.py
========
Pipeline:
  Video → MediaPipe (keypoints per frame) → Feature Engineering
  → XGBoost model (.pkl) → risk_level + risk_score → Report

The feature vector matches exactly what final.ipynb trained on:
  52 features = 3 angles + 10 engineered + 39 coordinate columns
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib

# ── Load XGBoost model artifacts ─────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_model    = None
_scaler   = None
_le       = None
_feat_cols = None

def _load_artifacts():
    global _model, _scaler, _le, _feat_cols
    if _model is not None:
        return True
    try:
        _model     = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
        _scaler    = joblib.load(os.path.join(MODEL_DIR, "risk_scaler.pkl"))
        _le        = joblib.load(os.path.join(MODEL_DIR, "risk_label_encoder.pkl"))
        _feat_cols = joblib.load(os.path.join(MODEL_DIR, "risk_feature_cols.pkl"))
        print("[model] XGBoost artifacts loaded OK")
        return True
    except FileNotFoundError as e:
        print(f"[model] WARNING: {e} — falling back to rule-based scoring")
        return False


# ── MediaPipe setup ───────────────────────────────────────────────────
mp_pose = mp.solutions.pose

# Landmark indices (matching pose_data.csv column names)
JOINTS = {
    "nose":       0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}


# ── Geometry helpers ──────────────────────────────────────────────────
def _angle(a, b, c):
    """Interior angle at b (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# ── Build feature row (matching final.ipynb FEATURE_COLS, 52 features) ──
def _build_feature_row(lm):
    """
    lm: list of MediaPipe NormalizedLandmark (33 items)
    Returns: dict with all 52 feature columns
    """
    def pt(idx):
        return lm[idx].x, lm[idx].y, lm[idx].z

    def vis(idx):
        return lm[idx].visibility

    # ── Coordinate columns (39 = 13 joints × 3 axes) ──────────────────
    row = {}
    for name, idx in JOINTS.items():
        x, y, z = pt(idx)
        row[f"{name}_x"]   = x
        row[f"{name}_y"]   = y
        row[f"{name}_z"]   = z
        row[f"{name}_vis"] = vis(idx)

    # ── Core angles ────────────────────────────────────────────────────
    # shoulder_angle: angle at left shoulder (elbow–shoulder–hip)
    shoulder_angle = _angle(pt(13), pt(11), pt(23))

    # hip_angle: angle at left hip (shoulder–hip–knee)
    hip_angle = _angle(pt(11), pt(23), pt(25))

    # back_rotation: angle between shoulder-vector and hip-vector
    sv = np.array([lm[12].x - lm[11].x, lm[12].y - lm[11].y])
    hv = np.array([lm[24].x - lm[23].x, lm[24].y - lm[23].y])
    cos_r = np.dot(sv, hv) / (np.linalg.norm(sv) * np.linalg.norm(hv) + 1e-9)
    back_rotation = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))

    row["shoulder_angle"] = shoulder_angle
    row["hip_angle"]      = hip_angle
    row["back_rotation"]  = back_rotation

    # ── Engineered features (matching final.ipynb) ─────────────────────
    row["shoulder_sym"]  = abs(lm[11].y - lm[12].y)
    row["hip_sym"]       = abs(lm[23].y - lm[24].y)
    row["trunk_lean_x"]  = (lm[11].x + lm[12].x) / 2 - (lm[23].x + lm[24].x) / 2
    row["trunk_lean_y"]  = (lm[11].y + lm[12].y) / 2 - (lm[23].y + lm[24].y) / 2
    row["l_arm_raise"]   = lm[11].y - lm[13].y
    row["r_arm_raise"]   = lm[12].y - lm[14].y

    vis_vals = [vis(i) for i in JOINTS.values()]
    row["avg_vis"] = float(np.mean(vis_vals))
    row["min_vis"] = float(np.min(vis_vals))

    row["combo1"] = int(abs(hip_angle) < 120 and abs(back_rotation) > 30)
    row["combo2"] = int(abs(shoulder_angle) > 60 and abs(hip_angle) < 140)

    return row, shoulder_angle, hip_angle, back_rotation


# ── Rule-based fallback (when pkl not available) ──────────────────────
def _rule_risk_score(shoulder_angle, hip_angle, back_rotation, avg_vis=1.0):
    def ss(a):
        a = abs(a)
        return 0 if a < 20 else 1 if a < 60 else 2 if a < 90 else 3

    def hs(a):
        a = abs(a)
        return 0 if a > 160 else 1 if a > 140 else 2 if a > 100 else 3

    def bs(a):
        a = abs(a)
        return 0 if a < 10 else 1 if a < 30 else 2 if a < 60 else 3

    posture = ss(shoulder_angle) + hs(hip_angle) + bs(back_rotation)
    combo = 0
    if abs(hip_angle) < 120 and abs(back_rotation) > 30: combo += 2
    if abs(shoulder_angle) > 60 and abs(hip_angle) < 140: combo += 2
    return float(np.clip((posture + combo) * avg_vis, 0, 10))


def _risk_label(score):
    if score <= 2:   return "Very Low"
    if score <= 4:   return "Low"
    if score <= 6:   return "Medium"
    if score <= 8:   return "High"
    return "Very High"


# ── Shot biomechanical profiles (expected angle ranges per shot type) ──
# Each shot has a characteristic range for [shoulder_angle, hip_angle, back_rotation]
# If measured angles fall outside these ranges, a penalty is applied.
SHOT_PROFILES = {
    # shot_name: (shoulder_min, shoulder_max, hip_min, hip_max, back_min, back_max)
    "Cover Drive":           (35,  80, 130, 175, 5,  30),
    "Pull Shot":             (70, 125, 95,  140, 30, 75),
    "Straight Drive":        (25,  70, 140, 180, 5,  25),
    "Cut Shot":              (50,  95, 115, 160, 20, 55),
    "Hook Shot":             (80, 135, 90,  140, 35, 75),
    "Sweep Shot":            (25,  60, 75,  120, 40, 85),
    "Slog Shot":             (70, 135, 85,  130, 40, 85),
    "Flick Shot":            (45,  90, 105, 150, 20, 55),
    "Scoop Shot":            (15,  55, 80,  130, 30, 75),
    "Pick Up Shot":          (55, 100, 105, 150, 25, 60),
    "Lofted Straight Drive": (50, 105, 125, 175, 15, 50),
}


def _shot_mismatch_penalty(shot, avg_shoulder, avg_hip, avg_back):
    """
    Returns a penalty (0.0 – 3.5) based on how far the measured angles
    deviate from the expected profile for the selected shot type.
    A large penalty means the user's body movement does not match
    the declared shot — indicating wrong technique or wrong shot selected.
    """
    profile = SHOT_PROFILES.get(shot)
    if profile is None:
        return 0.0

    s_lo, s_hi, h_lo, h_hi, b_lo, b_hi = profile
    sa, ha, br = abs(avg_shoulder), abs(avg_hip), abs(avg_back)

    def overshoot(val, lo, hi):
        """How far val is outside [lo, hi], normalised to range."""
        span = max(hi - lo, 1)
        if val < lo:
            return (lo - val) / span
        if val > hi:
            return (val - hi) / span
        return 0.0

    s_pen = overshoot(sa, s_lo, s_hi)
    h_pen = overshoot(ha, h_lo, h_hi)
    b_pen = overshoot(br, b_lo, b_hi)

    # Weighted: hip and back matter most for cricket shots
    total_pen = (s_pen * 0.30 + h_pen * 0.40 + b_pen * 0.30)
    # Scale to max 1.0 penalty — subtle correction, not score inflation
    return float(min(total_pen * 1.0, 1.0))


# ── Body area sub-scores (rule-based, for report cards) ───────────────
def _area_scores(shoulder_angle, hip_angle, back_rotation, lm):
    def norm(v, lo, hi):
        return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1) * 100)

    shoulder = (norm(abs(shoulder_angle), 0, 90) * 0.5 +
                norm(abs(back_rotation) * 0.6, 0, 60) * 0.3 +
                norm(abs(lm[11].y - lm[12].y) * 1000, 0, 50) * 0.2)

    lower_back = (norm(180 - abs(hip_angle), 0, 80) * 0.55 +
                  norm(abs(back_rotation), 0, 60) * 0.30 +
                  norm(abs(shoulder_angle) * 0.4, 0, 40) * 0.15)

    fk = _angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
    bk = _angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
    knee_hip = (norm(180 - fk, 0, 80) * 0.35 +
                norm(180 - bk, 0, 80) * 0.35 +
                norm(180 - abs(hip_angle), 0, 80) * 0.30)

    le = _angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
    re = _angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
    wrist_elbow = (norm(180 - (le + re) / 2, 0, 80) * 0.55 +
                   norm(abs(lm[15].z - lm[13].z) * 100, 0, 30) * 0.30 +
                   norm(abs(lm[11].y - lm[12].y) * 1000, 0, 50) * 0.15)

    def label(v):
        return "Low" if v < 30 else "Moderate" if v < 55 else "High"

    return {
        "shoulder":    {"avg": round(float(shoulder),    1), "status": label(shoulder)},
        "lower_back":  {"avg": round(float(lower_back),  1), "status": label(lower_back)},
        "knee_hip":    {"avg": round(float(knee_hip),    1), "status": label(knee_hip)},
        "wrist_elbow": {"avg": round(float(wrist_elbow), 1), "status": label(wrist_elbow)},
    }


# ── Main entry point ──────────────────────────────────────────────────
def analyze_video(video_path, shot="Unknown"):
    has_model = _load_artifacts()

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    frames_data = []      # list of feature dicts
    angles_data  = []     # (shoulder, hip, back) per frame
    lm_list_last = None   # last valid landmark list for area scores

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            continue

        lm = result.pose_landmarks.landmark
        lm_list_last = lm

        feat_row, sa, ha, br = _build_feature_row(lm)
        frames_data.append(feat_row)
        angles_data.append((sa, ha, br))

    cap.release()
    pose.close()

    # ── No pose detected ──────────────────────────────────────────────
    if not frames_data:
        return {
            "risk_score": 0,
            "risk_level": "Unknown",
            "areas": {k: {"avg": 0, "status": "Low"} for k in
                      ["shoulder", "lower_back", "knee_hip", "wrist_elbow"]},
            "frame_count": 0,
            "model_used": "none",
        }

    # ── XGBoost prediction path ───────────────────────────────────────
    if has_model and _feat_cols is not None:
        df = pd.DataFrame(frames_data)

        # Align columns to training order (fill missing with 0)
        for col in _feat_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[_feat_cols].fillna(0)

        # Predict per frame
        # XGBoost was trained on raw features (not scaled)
        preds  = _model.predict(df.values)              # encoded label indices
        probas = _model.predict_proba(df.values)        # shape (n_frames, n_classes)

        # Decode labels: le.classes_ = ['High','Low','Medium','Very High','Very Low']
        pred_labels = _le.inverse_transform(preds)

        # Map label → numeric risk score (same thresholds as notebook)
        LABEL_SCORE = {
            "Very Low": 1.0,
            "Low":      3.0,
            "Medium":   5.0,
            "High":     7.5,
            "Very High": 9.5,
        }
        frame_scores = np.array([LABEL_SCORE.get(l, 5.0) for l in pred_labels])

        # Weighted aggregate: top-30% risky frames weighted 60%, mean 40%
        top30     = sorted(frame_scores, reverse=True)[:max(1, len(frame_scores) // 3)]
        final_risk = float(np.clip(np.mean(top30) * 0.6 + np.mean(frame_scores) * 0.4, 0, 10))
        final_risk = round(final_risk, 2)
        risk_level = _risk_label(final_risk)

        # Most-common predicted label as secondary info
        from collections import Counter
        dominant_label = Counter(pred_labels).most_common(1)[0][0]

        # Average confidence for dominant class
        dom_idx     = list(_le.classes_).index(dominant_label)
        avg_conf    = float(np.mean(probas[:, dom_idx]) * 100)

        model_used = "xgboost"

    else:
        # ── Fallback: rule-based ──────────────────────────────────────
        frame_scores = np.array([
            _rule_risk_score(sa, ha, br,
                             avg_vis=frames_data[i].get("avg_vis", 1.0))
            for i, (sa, ha, br) in enumerate(angles_data)
        ])
        top30      = sorted(frame_scores, reverse=True)[:max(1, len(frame_scores) // 3)]
        final_risk = round(float(np.clip(np.mean(top30) * 0.6 + np.mean(frame_scores) * 0.4, 0, 10)), 2)
        risk_level    = _risk_label(final_risk)
        dominant_label = risk_level
        avg_conf       = 0.0
        model_used     = "rule-based"

    # ── Body area sub-scores (using last frame landmarks) ─────────────
    last_sa, last_ha, last_br = (
        float(np.mean([a[0] for a in angles_data])),
        float(np.mean([a[1] for a in angles_data])),
        float(np.mean([a[2] for a in angles_data])),
    )
    areas = _area_scores(last_sa, last_ha, last_br, lm_list_last)

    # ── Shot mismatch penalty ──────────────────────────────────────────
    penalty = _shot_mismatch_penalty(shot, last_sa, last_ha, last_br)
    if penalty > 0.1:
        print(f"[model] Shot mismatch penalty: +{penalty:.2f} for '{shot}' "
              f"(shoulder={last_sa:.1f}, hip={last_ha:.1f}, back={last_br:.1f})")
    penalised_risk  = float(np.clip(final_risk + penalty, 0, 10))
    penalised_risk  = round(penalised_risk, 2)
    penalised_level = _risk_label(penalised_risk)

    return {
        "risk_score":           penalised_risk,
        "risk_level":           penalised_level,
        "dominant_label":       dominant_label,
        "confidence":           round(avg_conf, 1),
        "areas":                areas,
        "frame_count":          len(frames_data),
        "avg_shoulder_angle":   round(last_sa, 1),
        "avg_hip_angle":        round(last_ha, 1),
        "avg_back_rotation":    round(last_br, 1),
        "model_used":           model_used,
        "shot_penalty":         round(penalty, 2),
    }