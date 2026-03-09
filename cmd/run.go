package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "log"
    "os"

    "github.com/mbykov/bhl-gigaam-go"
)

func main() {
    configPath := flag.String("config", "config.json", "путь к файлу конфигурации")
    audioFile := flag.String("audio", "", "путь к WAV файлу для тестирования")
    debug := flag.Bool("debug", false, "включить отладочный вывод")
    flag.Parse()

    cfg, err := gigaam.LoadConfig(*configPath)
    if err != nil {
        log.Fatalf("❌ Ошибка загрузки конфигурации: %v", err)
    }

    if *audioFile == "" {
        log.Fatal("❌ Укажите аудио файл через -audio")
    }

    fmt.Printf("🔧 Конфигурация GigaAM:\n")
    fmt.Printf("  Модель: %s\n", cfg.ModelPath)
    fmt.Printf("  Частота: %d Hz\n", cfg.SampleRate)

    module, err := gigaam.New(cfg, *debug)
    if err != nil {
        log.Fatalf("❌ Ошибка создания GigaAM модуля: %v", err)
    }
    defer module.Close()

    fmt.Printf("\n🎧 Загрузка аудио файла: %s\n", *audioFile)

    wavData, err := os.ReadFile(*audioFile)
    if err != nil {
        log.Fatalf("❌ Ошибка чтения WAV файла: %v", err)
    }

    if len(wavData) < 44 {
        log.Fatalf("❌ Файл слишком маленький")
    }
    audioData := wavData[44:]

    fmt.Printf("📊 Аудио данных: %d байт\n", len(audioData))

    result, err := module.ProcessAudio(audioData)
    if err != nil {
        log.Fatalf("❌ Ошибка обработки аудио: %v", err)
    }

    fmt.Printf("\n✨ РЕЗУЛЬТАТ:\n%s\n", result.Text)

    if *debug && result.Debug != nil {
        debugJSON, _ := json.MarshalIndent(result.Debug, "", "  ")
        fmt.Printf("\n🔍 Отладка:\n%s\n", debugJSON)
    }
}
