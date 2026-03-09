#!/usr/bin/env python3
import sys
import json
import argparse
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio
import onnxruntime as ort
from pathlib import Path

def extract_features(audio, sample_rate=16000):
    win_length = 320
    hop_length = 160
    n_fft = 320
    n_mels = 64

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=False,
        power=2.0,
        norm='slaney',
        mel_scale='htk',
    )

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    mel = mel_spec(audio)
    log_mel = torch.log(mel + 1e-6)
    return log_mel

def load_tokens(tokens_path):
    id2token = {}
    with open(tokens_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                token, idx = parts
                id2token[int(idx)] = token
    return id2token

def decode(encoder_out, decoder, joiner, id2token, blank_id=1024):
    pred_hidden = 320
    pred_layers = 1

    h = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
    c = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)

    x = np.array([[blank_id]], dtype=np.int64)
    unused_len = np.zeros(1, dtype=np.int64)

    decoder_out, _, h, c = decoder.run(
        None,
        {
            decoder.get_inputs()[0].name: x,
            decoder.get_inputs()[1].name: unused_len,
            decoder.get_inputs()[2].name: h,
            decoder.get_inputs()[3].name: c,
        }
    )

    result_tokens = []
    total_steps = encoder_out.shape[2]
    step_logs = []

    for t in range(total_steps):
        encoder_step = encoder_out[:, :, t:t+1]

        logits = joiner.run(
            None,
            {
                joiner.get_inputs()[0].name: encoder_step,
                joiner.get_inputs()[1].name: decoder_out
            }
        )[0]

        pred_token = np.argmax(logits, axis=-1).item()
        max_logit = np.max(logits)

        step_logs.append({
            "step": t,
            "best_token": int(pred_token),
            "logit": float(max_logit),
            "token_text": id2token.get(int(pred_token), "") if pred_token != blank_id else ""
        })

        if pred_token != blank_id:
            result_tokens.append(pred_token)

            x = np.array([[pred_token]], dtype=np.int64)
            decoder_out, _, h, c = decoder.run(
                None,
                {
                    decoder.get_inputs()[0].name: x,
                    decoder.get_inputs()[1].name: unused_len,
                    decoder.get_inputs()[2].name: h,
                    decoder.get_inputs()[3].name: c,
                }
            )

    # Сохраняем логи в файл
    import json
    log_file = f"/tmp/gigaam_py_steps_{time.time_ns()}.json"
    with open(log_file, 'w') as f:
        json.dump(step_logs, f, indent=2)
    print(f"  [PY] Логи шагов сохранены в {log_file}", file=sys.stderr)

    text = ''.join([id2token.get(t, '') for t in result_tokens])
    text = text.replace('▁', ' ').strip()
    return text, total_steps, log_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # Загрузка аудио
    audio, sr = sf.read(args.wav, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    print(f"  [PY] Аудио: {len(audio)} сэмплов, {len(audio)/sr:.2f} сек", file=sys.stderr)

    # Извлечение фичей
    features = extract_features(audio)
    audio_signal = features.numpy().astype(np.float32)
    num_frames = audio_signal.shape[2]
    print(f"  [PY] Фичи: {audio_signal.shape}, фреймов: {num_frames}", file=sys.stderr)

    length = np.array([num_frames], dtype=np.int64)

    # Загрузка моделей ONNX
    providers = ['CPUExecutionProvider']
    encoder = ort.InferenceSession(f"{args.model}/encoder.int8.onnx", providers=providers)
    decoder = ort.InferenceSession(f"{args.model}/decoder.onnx", providers=providers)
    joiner = ort.InferenceSession(f"{args.model}/joiner.onnx", providers=providers)

    # Загрузка токенов
    id2token = load_tokens(f"{args.model}/tokens.txt")

    # Энкодер
    print(f"  [PY] Запуск энкодера...", file=sys.stderr)
    encoder_out, encoder_len = encoder.run(
        None,
        {
            encoder.get_inputs()[0].name: audio_signal,
            encoder.get_inputs()[1].name: length
        }
    )
    print(f"  [PY] Выход энкодера: {encoder_out.shape}", file=sys.stderr)

    # Декодирование
    print(f"  [PY] Запуск декодирования...", file=sys.stderr)
    text, steps, log_file = decode(encoder_out, decoder, joiner, id2token)
    print(f"  [PY] Распознано {steps} шагов, лог: {log_file}", file=sys.stderr)

    result = {
      "text": text,
      "time": time.time() - start_time,
      "steps": steps,
      "frames": num_frames,
      "log_file": log_file,
    }

    # Только JSON в stdout
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
