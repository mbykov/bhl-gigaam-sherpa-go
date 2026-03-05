# GigaAM v3 RNNT модуль для Go

Модуль для распознавания речи с использованием модели GigaAM v3 RNNT через sherpa-onnx.

## Структура

.
├── cmd/
│ ├── run.go # тестовая программа на Go
│ └── test.py # тестовый инструмент на Python
├── config.json # конфигурация
├── gigaam.go # основной модуль
├── go.mod # Go модуль
└── README.md

## Требования

- Go 1.25+
- sherpa-onnx-go (github.com/k2-fsa/sherpa-onnx-go)
- Модель GigaAM v3 RNNT (обычная или INT8)

## Установка

```bash
go get github.com/k2-fsa/sherpa-onnx-go

{
    "model_path": "/path/to/gigaam_v3_e2e_rnnt_onnx_int8",
    "sample_rate": 16000,
    "feature_dim": 64,
    "num_threads": 2,
    "provider": "cpu",
    "decoding_method": "greedy_search",
    "quantized": true
}

## Исплользование

import "gv3rnnt/gigaam"

cfg := gigaam.Config{
    ModelPath:  "/path/to/model",
    SampleRate: 16000,
    FeatureDim: 64,
}

module, _ := gigaam.New(cfg)
defer module.Close()

text, _ := module.ProcessAudio(pcmData)
fmt.Println(text)
