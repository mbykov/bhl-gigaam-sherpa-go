package main

import (
    "flag"
    "fmt"
    "log"
    "os"

    "github.com/mbykov/bhl-gigaam-go"
)

func main() {
    configPath := flag.String("config", "config.json", "путь к файлу конфигурации")
    audioFile := flag.String("audio", "", "путь к WAV файлу для тестирования")
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

    module, err := gigaam.New(cfg)
    if err != nil {
        log.Fatalf("❌ Ошибка создания GigaAM модуля: %v", err)
    }
    defer module.Close()

    fmt.Printf("\n🎧 Загрузка аудио файла: %s\n", *audioFile)

    // Читаем WAV файл
    wavData, err := os.ReadFile(*audioFile)
    if err != nil {
        log.Fatalf("❌ Ошибка чтения WAV файла: %v", err)
    }

    // Пропускаем WAV заголовок (44 байта)
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
}
