package gigaam

import (
    "encoding/json"
    "fmt"
    "math"
    "os"
    "strings"
    "sync"

    // "github.com/go-audio/wav" // Убираем неиспользуемый импорт
    "github.com/madelynnblue/go-dsp/fft"
    ort "github.com/yalue/onnxruntime_go"
)

const (
    WinLength = 320
    HopLength = 160
    NFFT      = 320
    NMels     = 64
)

// Config структура для загрузки из JSON
type Config struct {
    ModelPath  string `json:"model_path"`
    SampleRate int    `json:"sample_rate"`
    FeatureDim int    `json:"feature_dim"`
    NumThreads int    `json:"num_threads"`
    Provider   string `json:"provider"`
}

// GigaAMModule представляет модуль для распознавания речи
type GigaAMModule struct {
    config      Config
    encoder     *ort.AdvancedSession
    decoder     *ort.AdvancedSession
    joiner      *ort.AdvancedSession
    tokens      map[int]string
    blankID     int64
    predHidden  int
    predLayers  int
    encoderDim  int
    initialized bool
    mu          sync.Mutex
}

// Result представляет результат распознавания
type Result struct {
    Text        string
    IsProcessed bool
}

// New создает новый экземпляр GigaAM модуля
func New(cfg Config) (*GigaAMModule, error) {
    module := &GigaAMModule{
        config:     cfg,
        blankID:    1024,
        predHidden: 320,
        predLayers: 1,
        encoderDim: 768,
    }

    if err := module.initONNX(); err != nil {
        fmt.Printf("⚠️ GigaAM модель не загружена (%v), использую заглушку\n", err)
        return module, nil
    }

    module.initialized = true
    fmt.Println("✅ GigaAM модель инициализирована (ONNX Runtime)")
    return module, nil
}


// initONNX инициализирует ONNX Runtime и загружает модели
func (m *GigaAMModule) initONNX() error {
    // Пути к файлам модели
    encoderPath := m.config.ModelPath + "/encoder.int8.onnx"
    decoderPath := m.config.ModelPath + "/decoder.onnx"
    joinerPath := m.config.ModelPath + "/joiner.onnx"
    tokensPath := m.config.ModelPath + "/tokens.txt"

    // Проверяем существование файлов
    for _, path := range []string{encoderPath, decoderPath, joinerPath, tokensPath} {
        if _, err := os.Stat(path); err != nil {
            return fmt.Errorf("file not found: %s", path)
        }
    }

    // Загружаем токены
    m.tokens = loadTokens(tokensPath)
    if len(m.tokens) == 0 {
        return fmt.Errorf("failed to load tokens")
    }

    // Инициализируем ONNX Runtime
    ort.SetSharedLibraryPath("/home/michael/go/ort/lib/libonnxruntime.so")

    // Пытаемся инициализировать
    err := ort.InitializeEnvironment()
    if err != nil {
        // Если ошибка "already initialized" - это нормально, продолжаем
        if strings.Contains(err.Error(), "already been initialized") {
            fmt.Println("  ℹ️ ONNX Runtime уже инициализирован (другим модулем)")
        } else {
            return fmt.Errorf("failed to initialize ONNX Runtime: %v", err)
        }
    } else {
        fmt.Println("  ✅ ONNX Runtime инициализирован")
    }

    // options, err := ort.NewSessionOptions()
    // if err != nil {
    //     return fmt.Errorf("failed to create session options: %v", err)
    // }

    // Сессии будут создаваться динамически при каждом запросе
    // потому что в AdvancedSession нужно передавать тензоры при создании

    return nil
}

// ProcessAudio обрабатывает аудио и возвращает распознанный текст
func (m *GigaAMModule) ProcessAudio(pcm []byte) (Result, error) {
    m.mu.Lock()
    defer m.mu.Unlock()

    if !m.initialized {
        return Result{
            Text:        "привет мир [stub]",
            IsProcessed: false,
        }, nil
    }

    // Конвертируем PCM в float32 сэмплы
    samples := make([]float32, len(pcm)/2)
    for i := 0; i < len(pcm); i += 2 {
        sample := int16(pcm[i]) | int16(pcm[i+1])<<8
        samples[i/2] = float32(sample) / 32768.0
    }

    // Извлекаем фичи
    features := extractFeatures(samples)
    numFrames := len(features[0])

    // Подготавливаем вход для энкодера
    flatFeatures := make([]float32, NMels*numFrames)
    for c := 0; c < NMels; c++ {
        for t := 0; t < numFrames; t++ {
            flatFeatures[c*numFrames+t] = features[c][t]
        }
    }

    // Создаем тензоры
    audioTensor, _ := ort.NewTensor(ort.NewShape(1, NMels, int64(numFrames)), flatFeatures)
    defer audioTensor.Destroy()

    lengthTensor, _ := ort.NewTensor(ort.NewShape(1), []int64{int64(numFrames)})
    defer lengthTensor.Destroy()

    // Выход энкодера
    outSteps := int64(numFrames / 4)
    encoderOutData := make([]float32, 1*m.encoderDim*int(outSteps))
    encoderOutTensor, _ := ort.NewTensor(ort.NewShape(1, int64(m.encoderDim), outSteps), encoderOutData)
    defer encoderOutTensor.Destroy()

    outLenTensor, _ := ort.NewTensor(ort.NewShape(1), []int64{0})
    defer outLenTensor.Destroy()

    // Создаем и запускаем сессию энкодера
    options, _ := ort.NewSessionOptions()
    defer options.Destroy()

    encoder, err := ort.NewAdvancedSession(
        m.config.ModelPath+"/encoder.int8.onnx",
        []string{"audio_signal", "length"},
        []string{"encoded", "encoded_len"},
        []ort.Value{audioTensor, lengthTensor},
        []ort.Value{encoderOutTensor, outLenTensor},
        options,
    )
    if err != nil {
        return Result{}, fmt.Errorf("failed to create encoder session: %v", err)
    }
    defer encoder.Destroy()

    if err := encoder.Run(); err != nil {
        return Result{}, fmt.Errorf("encoder failed: %v", err)
    }

    // Инициализация декодера
    hData := make([]float32, m.predLayers*1*m.predHidden)
    cData := make([]float32, m.predLayers*1*m.predHidden)

    hTensor, _ := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), hData)
    defer hTensor.Destroy()
    cTensor, _ := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), cData)
    defer cTensor.Destroy()

    tokenInTensor, _ := ort.NewTensor(ort.NewShape(1, 1), []int64{m.blankID})
    defer tokenInTensor.Destroy()
    unusedLenIn, _ := ort.NewTensor(ort.NewShape(1), []int64{0})
    defer unusedLenIn.Destroy()

    decOutData := make([]float32, 1*m.predHidden*1)
    decOutTensor, _ := ort.NewTensor(ort.NewShape(1, int64(m.predHidden), 1), decOutData)
    defer decOutTensor.Destroy()
    decUnusedOut, _ := ort.NewTensor(ort.NewShape(1), []int64{0})
    defer decUnusedOut.Destroy()
    hOutTensor, _ := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), hData)
    defer hOutTensor.Destroy()
    cOutTensor, _ := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), cData)
    defer cOutTensor.Destroy()

    // Создаем сессию декодера
    decoder, err := ort.NewAdvancedSession(
        m.config.ModelPath+"/decoder.onnx",
        []string{"x", "unused_x_len.1", "h.1", "c.1"},
        []string{"dec", "unused_x_len", "h", "c"},
        []ort.Value{tokenInTensor, unusedLenIn, hTensor, cTensor},
        []ort.Value{decOutTensor, decUnusedOut, hOutTensor, cOutTensor},
        options,
    )
    if err != nil {
        return Result{}, fmt.Errorf("failed to create decoder session: %v", err)
    }
    defer decoder.Destroy()

    // Первый шаг декодера
    if err := decoder.Run(); err != nil {
        return Result{}, fmt.Errorf("first decoder step failed: %v", err)
    }

    // Подготовка джойнера
    encStepData := make([]float32, 1*m.encoderDim*1)
    encStepTensor, _ := ort.NewTensor(ort.NewShape(1, int64(m.encoderDim), 1), encStepData)
    defer encStepTensor.Destroy()

    jointOutData := make([]float32, 1*1*1*(int(m.blankID)+1))
    jointOutTensor, _ := ort.NewTensor(ort.NewShape(1, 1, 1, int64(m.blankID)+1), jointOutData)
    defer jointOutTensor.Destroy()

    // Создаем сессию джойнера
    joiner, err := ort.NewAdvancedSession(
        m.config.ModelPath+"/joiner.onnx",
        []string{"enc", "dec"},
        []string{"joint"},
        []ort.Value{encStepTensor, decOutTensor},
        []ort.Value{jointOutTensor},
        options,
    )
    if err != nil {
        return Result{}, fmt.Errorf("failed to create joiner session: %v", err)
    }
    defer joiner.Destroy()

    // Цикл декодирования
    var resultTokens []string
    blankCount := 0

    for t := 0; t < int(outSteps); t++ {
        // Копируем шаг энкодера
        encStep := encStepTensor.GetData()
        for c := 0; c < m.encoderDim; c++ {
            encStep[c] = encoderOutData[c*int(outSteps)+t]
        }

        // Запускаем джойнер
        if err := joiner.Run(); err != nil {
            return Result{}, fmt.Errorf("joiner failed at step %d: %v", t, err)
        }

        // Находим лучший токен
        logits := jointOutTensor.GetData()
        bestToken := 0
        maxVal := logits[0]
        for i := 1; i <= int(m.blankID); i++ {
            if logits[i] > maxVal {
                maxVal = logits[i]
                bestToken = i
            }
        }

        if int64(bestToken) != m.blankID {
            blankCount = 0
            if token, ok := m.tokens[bestToken]; ok {
                resultTokens = append(resultTokens, token)
            }

            // Обновляем декодер с новым токеном
            tokenInTensor.GetData()[0] = int64(bestToken)
            if err := decoder.Run(); err != nil {
                return Result{}, fmt.Errorf("decoder update failed: %v", err)
            }
        } else {
            blankCount++
            if blankCount > 50 {
                break
            }
        }
    }

    // Формируем текст
    text := strings.Join(resultTokens, "")
    text = strings.ReplaceAll(text, "▁", " ")

    return Result{
        Text:        strings.TrimSpace(text),
        IsProcessed: true,
    }, nil
}

// ProcessText для совместимости
func (m *GigaAMModule) ProcessText(text string) (Result, error) {
    return Result{
        Text:        text,
        IsProcessed: true,
    }, nil
}

// Close освобождает ресурсы
func (m *GigaAMModule) Close() {
    ort.DestroyEnvironment()
    fmt.Println("🔚 [GigaAM] Модуль закрыт")
}

// LoadConfig загружает конфигурацию из JSON файла
func LoadConfig(path string) (Config, error) {
    var cfg Config
    data, err := os.ReadFile(path)
    if err != nil {
        return cfg, fmt.Errorf("error reading config file: %v", err)
    }
    err = json.Unmarshal(data, &cfg)
    if err != nil {
        return cfg, fmt.Errorf("error parsing config JSON: %v", err)
    }
    return cfg, nil
}

// --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

func loadTokens(path string) map[int]string {
    tokens := make(map[int]string)
    data, err := os.ReadFile(path)
    if err != nil {
        return tokens
    }
    lines := strings.Split(string(data), "\n")
    for i, line := range lines {
        fields := strings.Fields(line)
        if len(fields) > 0 {
            tokens[i] = fields[0]
        }
    }
    return tokens
}

func extractFeatures(samples []float32) [][]float32 {
    fb := getMelFilterbank()
    numFrames := (len(samples)-WinLength)/HopLength + 1
    features := make([][]float32, NMels)
    for i := range features {
        features[i] = make([]float32, numFrames)
    }

    window := make([]float64, WinLength)
    for i := range window {
        window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(WinLength-1)))
    }

    for f := 0; f < numFrames; f++ {
        start := f * HopLength
        frame := make([]complex128, NFFT)
        for i := 0; i < WinLength; i++ {
            frame[i] = complex(float64(samples[start+i])*window[i], 0)
        }

        spectrum := fft.FFT(frame)

        for m := 0; m < NMels; m++ {
            var melEnergy float64
            for k := 0; k <= NFFT/2; k++ {
                magSq := real(spectrum[k])*real(spectrum[k]) + imag(spectrum[k])*imag(spectrum[k])
                melEnergy += magSq * fb[m][k]
            }
            features[m][f] = float32(math.Log(melEnergy + 1e-6))
        }
    }
    return features
}

func getMelFilterbank() [][]float64 {
    hzToMel := func(hz float64) float64 { return 2595.0 * math.Log10(1.0+hz/700.0) }
    melToHz := func(mel float64) float64 { return 700.0 * (math.Pow(10, mel/2595.0) - 1.0) }

    sampleRate := 16000.0
    minMel := hzToMel(0)
    maxMel := hzToMel(sampleRate / 2.0)

    melPts := make([]float64, NMels+2)
    for i := 0; i < NMels+2; i++ {
        melPts[i] = melToHz(minMel + float64(i)*(maxMel-minMel)/float64(NMels+1))
    }

    fb := make([][]float64, NMels)
    for i := 0; i < NMels; i++ {
        fb[i] = make([]float64, NFFT/2+1)
        for k := 0; k <= NFFT/2; k++ {
            hz := float64(k) * sampleRate / float64(NFFT)
            if hz >= melPts[i] && hz <= melPts[i+1] {
                fb[i][k] = (hz - melPts[i]) / (melPts[i+1] - melPts[i])
            } else if hz >= melPts[i+1] && hz <= melPts[i+2] {
                fb[i][k] = (melPts[i+2] - hz) / (melPts[i+2] - melPts[i+1])
            }
        }
        enorm := 2.0 / (melPts[i+2] - melPts[i])
        for k := 0; k < len(fb[i]); k++ {
            fb[i][k] *= enorm
        }
    }
    return fb
}
