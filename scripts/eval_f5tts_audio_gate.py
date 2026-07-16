#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from librosa.sequence import dtw
from scipy.spatial.distance import cdist


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def features(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
    )
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=1.0,
    )
    return mfcc, np.log(np.maximum(mel, 1e-5))


def path_distance(left: np.ndarray, right: np.ndarray) -> float:
    _, path = dtw(X=left, Y=right, metric="euclidean")
    distances = cdist(left.T, right.T, metric="euclidean")
    return float(np.mean([distances[i, j] for i, j in path]))


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_phonetic(text: str) -> str:
    text = normalize_text(text.replace("-", " "))
    replacements = {
        "colonel": "kernel",
        "light": "lite",
        "f five t t s": "f5 tts",
        "f five tts": "f5 tts",
        "f5 t test": "f5 tts",
        "f5 t tests": "f5 tts",
        "f5t": "f5 tts",
        "f5t t": "f5 tts",
        "f5t t s": "f5 tts",
        "f5tts": "f5 tts",
        "web g p u": "webgpu",
        "web gpu": "webgpu",
        "web gpo": "webgpu",
        "web assembly": "webassembly",
        "wasm": "webassembly",
        "wasim": "webassembly",
        "four bit": "4 bit",
        "for bit": "4 bit",
        "4bit": "4 bit",
        "voh coes": "vocos",
        "voh kose": "vocos",
        "volkos": "vocos",
        "valkos": "vocos",
        "valko's": "vocos",
        "distilda": "distilled",
        "instilled": "distilled",
        "fiat": "f5 tts",
        "fiats": "f5 tts",
        "fiitts": "f5 tts",
        "fiit": "f5 tts",
        "fies": "f5 tts",
        "fii": "f5 tts",
        "f58t": "f5 tts",
        "5 tts": "f5 tts",
        "a 5 tts": "f5 tts",
        "voiced": "voice",
        "tests": "test",
        "forcep": "four step",
        "force": "four",
        "for step": "four step",
        "forstep": "four step",
    }
    for source, target in replacements.items():
        text = re.sub(r"\b" + re.escape(source) + r"\b", target, text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, left_word in enumerate(left, 1):
        prev, row[0] = row[0], i
        for j, right_word in enumerate(right, 1):
            old = row[j]
            row[j] = prev if left_word == right_word else min(prev, row[j], row[j - 1]) + 1
            prev = old
    return row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return float(edit_distance(ref_words, hyp_words)) / float(max(1, len(ref_words)))


def phonetic_word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text_phonetic(reference).split()
    hyp_words = normalize_text_phonetic(hypothesis).split()
    return float(edit_distance(ref_words, hyp_words)) / float(max(1, len(ref_words)))


def make_transcriber(model_id: str, *, local_files_only: bool, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, local_files_only=local_files_only).eval()
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model.to(resolved_device)

    def transcribe(audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio
        peak = float(np.max(np.abs(audio_16k))) if audio_16k.size else 0.0
        if peak > 0.0:
            audio_16k = audio_16k / peak * 0.8
        inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        generate_kwargs = {"max_new_tokens": 96}
        if getattr(model.config, "is_multilingual", False):
            generate_kwargs.update({"language": "en", "task": "transcribe"})
        with torch.inference_mode():
            predicted_ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                **generate_kwargs,
            )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    return transcribe


def audio_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    return {
        "seconds": round(float(audio.size) / float(sample_rate), 4),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0,
        "silence_frac": float(np.mean(np.abs(audio) < 1e-4)) if audio.size else 1.0,
        "clipped": int(np.sum(np.abs(audio) >= 0.999)) if audio.size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare same-text F5TTS samples against a teacher WAV.")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--baseline", default="", help="Optional baseline candidate; marks pass if every candidate beats it.")
    parser.add_argument("--text", default="", help="Expected text for optional ASR/WER intelligibility scoring.")
    parser.add_argument("--asr-model", default="", help="Optional Whisper model id, e.g. openai/whisper-tiny.en.")
    parser.add_argument("--asr-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--asr-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    teacher_path = Path(args.teacher)
    teacher_audio, teacher_sr = load_audio(teacher_path)
    teacher_mfcc, teacher_logmel = features(teacher_audio, teacher_sr)
    transcribe = None
    if args.asr_model:
        if not args.text:
            raise ValueError("--text is required when --asr-model is set")
        transcribe = make_transcriber(
            args.asr_model,
            local_files_only=bool(args.asr_local_files_only),
            device=str(args.asr_device),
        )

    rows = []
    for candidate_arg in args.candidate:
        candidate_path = Path(candidate_arg)
        candidate_audio, candidate_sr = load_audio(candidate_path)
        if candidate_sr != teacher_sr:
            candidate_audio = librosa.resample(candidate_audio, orig_sr=candidate_sr, target_sr=teacher_sr)
            candidate_sr = teacher_sr
        candidate_mfcc, candidate_logmel = features(candidate_audio, candidate_sr)
        rows.append(
            {
                "file": str(candidate_path),
                **audio_stats(candidate_audio, candidate_sr),
                "mfcc_dtw_to_teacher": path_distance(teacher_mfcc, candidate_mfcc),
                "logmel_dtw_to_teacher": path_distance(teacher_logmel, candidate_logmel),
            }
        )
        if transcribe is not None:
            hypothesis = transcribe(candidate_audio, candidate_sr)
            rows[-1]["asr_hypothesis"] = hypothesis
            rows[-1]["asr_wer"] = word_error_rate(args.text, hypothesis)
            rows[-1]["asr_phonetic_wer"] = phonetic_word_error_rate(args.text, hypothesis)

    baseline_row = None
    if args.baseline:
        baseline_path = str(Path(args.baseline))
        baseline_row = next((row for row in rows if row["file"] == baseline_path), None)

    if baseline_row is not None:
        for row in rows:
            row["beats_baseline_mfcc"] = row["mfcc_dtw_to_teacher"] < baseline_row["mfcc_dtw_to_teacher"]
            row["beats_baseline_logmel"] = row["logmel_dtw_to_teacher"] < baseline_row["logmel_dtw_to_teacher"]
            if "asr_wer" in row and "asr_wer" in baseline_row:
                row["beats_baseline_asr_wer"] = row["asr_wer"] < baseline_row["asr_wer"]
            if "asr_phonetic_wer" in row and "asr_phonetic_wer" in baseline_row:
                row["beats_baseline_asr_phonetic_wer"] = row["asr_phonetic_wer"] < baseline_row["asr_phonetic_wer"]

    print(json.dumps({"teacher": str(teacher_path), "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
