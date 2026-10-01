"""
Real-time AVSC liveness with:
(1) Lip-sync correlation,
(2) Eye-blink detection (EAR),
(3) Phrase fuzzy transcription,
(4) Haar face presence.
Best-of-5 rule (>=3/5 passes => liveness). CPU-only; Whisper (base) with Vosk fallback.
UI: smaller video display, blink-triggered points, and LIVE metric plots.
"""

import streamlit as st
import cv2
import numpy as np
import torch
import sounddevice as sd
import torchaudio
import mediapipe as mp
import time, random
import difflib
import matplotlib.pyplot as plt
from collections import deque

# ------------------- Transcription (CPU) -------------------
def transcribe_audio(wave, sr=16000):
    try:
        import whisper
        model = whisper.load_model("base", device="cpu")
        torchaudio.save("temp.wav", wave, sr)
        result = model.transcribe("temp.wav", fp16=False, language="en")
        return result["text"].lower()
    except Exception as e:
        print("⚠️ Whisper unavailable, using Vosk fallback:", e)
        from vosk import Model, KaldiRecognizer
        import wave, json
        torchaudio.save("temp.wav", wave, sr)
        wf = wave.open("temp.wav", "rb")
        model = Model(lang="en-us")
        rec = KaldiRecognizer(model, wf.getframerate())
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text += " " + res.get("text", "")
        return text.lower()

# ----------------------- Utilities ------------------------
def capture_audio(duration=6.0, sr=16000):
    st.sidebar.write("🎙️ Recording audio…")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return torch.tensor(audio.T)

def compute_lipsync(audio_wave, mouth_motion):
    a = np.abs(audio_wave.numpy()).flatten()
    v = np.array(mouth_motion).flatten()
    if len(a) < 5 or len(v) < 5:
        return 0.5
    L = min(len(a), len(v))
    a = a[:L] - a[:L].mean()
    v = v[:L] - v[:L].mean()
    denom = (np.std(a) * np.std(v) * L) + 1e-8
    corr = np.correlate(a, v, "valid")[0] / denom
    return float(np.clip(corr, 0, 1))

def ear(pts):
    vert = np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])
    horiz = np.linalg.norm(pts[0]-pts[3]) + 1e-6
    return vert / (2.0 * horiz)

def fuzzy_phrase_match(transcript, target):
    if not transcript.strip():
        return 0.0
    ratio = difflib.SequenceMatcher(None, transcript, target).ratio()
    return max(0.3, ratio) if ratio > 0.3 else ratio

# ----------------------- Streamlit UI ---------------------
st.set_page_config(page_title="AVSC Liveness (Live Plots)", layout="wide")
st.title("🎯 Real-Time AVSC Liveness (Blink-Triggered Points + Live Plots)")

st.markdown("""
Signals used: 1️⃣ Lip-Sync, 2️⃣ Phrase Match, 3️⃣ Blink (EAR), 4️⃣ Haar Face.  
Facial points (Haar corners + key landmarks) are drawn **only when a blink is detected**.  
Best-of-5 rule: **≥ 3/5** passes → **Person Liveness Detected**.
""")

phrases = [
    "security is my priority",
    "i am a real person",
    "open ai makes research easy",
    "artificial intelligence is the future",
    "the quick brown fox jumps over the lazy dog",
]
target = random.choice(phrases)
st.subheader(f"🗣️ Say this phrase clearly: **“{target}”**")

threshold = st.slider("Liveness threshold", 0.0, 1.0, 0.60, 0.05)
run = st.toggle("▶️ Start Real-Time Verification")

# Layout: video on left, live plots on right
col_video, col_plots = st.columns([1, 1])
video_placeholder = col_video.empty()
plots_placeholder = col_plots.empty()

status_box = st.empty()
st.sidebar.markdown("### 📊 Live Metrics")
attempts_box = st.sidebar.empty()
window_box = st.sidebar.empty()

# ------------------- Init on start ------------------------
if run:
    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # Buffers / histories
    mouth_hist, ear_hist = [], []
    lipsync_hist, phrase_hist, blink_hist = [], [], []
    blink_count, prev_eye_state = 0, True

    # Blink visual gating
    BLINK_FLASH_FRAMES = 10
    blink_flash_counter = 0

    # Best-of-5
    eval_results = deque(maxlen=5)
    eval_scores  = deque(maxlen=5)
    last_eval_ts = time.time()

    st.info("🎙️ Speak the phrase when prompted…")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible.")
            break

        frame = cv2.resize(frame, (600, 600))
        h, w, _ = frame.shape

        # ---- Haar face ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE
        )
        haar_face_present = len(faces) > 0

        # ---- FaceMesh ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.multi_face_landmarks:
            lm = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])
            try:
                left_eye = lm[[33, 160, 158, 133, 153, 144]]
                right_eye = lm[[362, 385, 387, 263, 373, 380]]
                ear_val = (ear(left_eye) + ear(right_eye)) / 2.0
                ear_hist.append(ear_val)

                eyes_open = ear_val > 0.21
                if not eyes_open and prev_eye_state:
                    blink_count += 1
                    blink_flash_counter = BLINK_FLASH_FRAMES
                prev_eye_state = eyes_open

                mouth_pts = lm[[13, 14, 87, 317, 82, 312]]
                mar_val = np.std(mouth_pts[:, 1])
                mouth_hist.append(mar_val)
            except Exception:
                pass

        # ---- Draw points ONLY on blink event ----
        if blink_flash_counter > 0:
            if haar_face_present:
                (x, y, fw, fh) = faces[0]
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 210, 255), 2)
                for cx, cy in [(x, y), (x+fw, y), (x, y+fh), (x+fw, y+fh)]:
                    cv2.circle(frame, (cx, cy), 4, (0, 210, 255), -1)
                cv2.putText(frame, "HAAR FACE", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,210,255), 2)
            if res.multi_face_landmarks:
                try:
                    for idx in [33,160,158,133,153,144, 362,385,387,263,373,380, 13,14,87,317,82,312]:
                        px, py = int(lm[idx][0]), int(lm[idx][1])
                        cv2.circle(frame, (px, py), 2, (255, 80, 80), -1)
                except Exception:
                    pass
            blink_flash_counter -= 1

        # ---- Periodic evaluation & LIVE plots update ----
        if time.time() - last_eval_ts > 5.0:
            wave = capture_audio(duration=6.0)
            lipsync_score = compute_lipsync(wave, mouth_hist[-60:])
            lipsync_hist.append(lipsync_score)

            transcript = transcribe_audio(wave)
            st.sidebar.write(f"🗣️ Transcript: `{transcript}`")
            phrase_score = fuzzy_phrase_match(transcript, target)
            phrase_hist.append(phrase_score)

            blink_score = min(blink_count / 3.0, 1.0)
            blink_hist.append(blink_score)

            haar_score = 0.15 if haar_face_present else 0.0
            validation = min((lipsync_score + phrase_score + blink_score) / 3.0 + haar_score, 1.0)
            passed = validation >= threshold
            eval_results.append(passed)
            eval_scores.append(validation)

            st.sidebar.metric("1️⃣ LipSync", f"{lipsync_score:.2f}")
            st.sidebar.metric("2️⃣ Phrase", f"{phrase_score:.2f}")
            st.sidebar.metric("3️⃣ Blink", f"{blink_score:.2f}")
            st.sidebar.metric("👤 Haar Face", "Yes" if haar_face_present else "No")
            st.sidebar.progress(validation)
            st.sidebar.write(f"Validation Score: `{validation:.2f}`")

            attempts_box.markdown(f"**Best-of-5 Window:** {list(eval_results)}")
            window_box.markdown(f"**Scores:** {[round(s, 2) for s in list(eval_scores)]}")

            # ---- LIVE plots (update every eval) ----
            fig, axes = plt.subplots(4, 1, figsize=(7, 8))
            axes[0].plot(ear_hist[-300:], color="blue"); axes[0].set_title("1️⃣ EAR — Blink Detection"); axes[0].grid(True)
            axes[1].plot(mouth_hist[-300:], color="orange"); axes[1].set_title("2️⃣ Mouth Motion (MAR proxy)"); axes[1].grid(True)
            axes[2].plot(lipsync_hist[-20:], color="purple", marker="o"); axes[2].set_title("3️⃣ LipSync Correlation"); axes[2].grid(True)
            axes[3].plot(phrase_hist[-20:], color="green", marker="x", label="Phrase")
            axes[3].plot(blink_hist[-20:], color="red", marker="o", label="Blink")
            axes[3].set_title("4️⃣ Phrase & Blink Scores"); axes[3].legend(); axes[3].grid(True)
            plt.tight_layout()
            plots_placeholder.pyplot(fig, clear_figure=True)

            # Best-of-5 decision
            if len(eval_results) == 5:
                passes = sum(eval_results)
                if passes >= 3:
                    status_box.success("🎉 Person Liveness Detected (≥3/5 passes)")
                    st.balloons()
                else:
                    status_box.error("❌ Failed Detection (<3/5 passes)")
                eval_results.clear()
                eval_scores.clear()
                blink_count = 0

            last_eval_ts = time.time()

        # ---- HUD (removed frame size text) ----
        cv2.putText(frame, f"Blinks: {blink_count}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        cv2.putText(frame, f"Threshold: {threshold:.2f}", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        if blink_flash_counter > 0:
            cv2.putText(frame, "Blink detected → points shown", (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)

        # ---- Smaller display preview (processing still at 600×600) ----
        display_frame = cv2.resize(frame, (600, 600))
        video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                channels="RGB", use_container_width=False)

        if not run:
            break

    cap.release()
    st.success("🔴 Session ended.")
