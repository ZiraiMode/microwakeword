// Command mic listens to a microphone and prints when a wake word is detected.
//
// Requires miniaudio via github.com/gen2brain/malgo.
//
// Usage:
//
//	mic -model hey_jarvis.tflite [-cutoff 0.85] [-window 5] [-step 10]
package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/gen2brain/malgo"
	mww "github.com/zserge/microwakeword"
)

func main() {
	modelPath := flag.String("model", "", "Path to .tflite v2 model")
	cutoff := flag.Float64("cutoff", 0.85, "Probability cutoff for detection")
	window := flag.Int("window", 5, "Sliding window size")
	stepMs := flag.Int("step", 10, "Feature step size in ms")
	verbose := flag.Bool("v", false, "Print each inference frame")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: mic -model <model.tflite>\n\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	cfg := mww.Config{
		ModelPath:         *modelPath,
		ProbabilityCutoff: *cutoff,
		SlidingWindowSize: *window,
		FeaturesStepMs:    *stepMs,
	}

	det, err := mww.NewDetector(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating detector: %v\n", err)
		os.Exit(1)
	}

	stride := 3
	if *verbose {
		det.OnInference = func(frameIdx int, rawProb uint8, slidingAvg float64) {
			prob := float64(rawProb) / 255.0
			timeS := float64(frameIdx) * float64(stride) * float64(*stepMs) / 1000.0
			bar := makeBar(prob, 30)
			fmt.Printf("\rframe %4d  t=%5.1fs  prob=%.3f  avg=%.3f  %s",
				frameIdx, timeS, prob, slidingAvg, bar)
		}
	}

	// Initialize miniaudio context
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing audio context: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = ctx.Uninit() }()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Duplex)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = 16000
	deviceConfig.Playback.Format = malgo.FormatS16
	deviceConfig.Playback.Channels = 1
	deviceConfig.Alsa.NoMMap = 1

	var mu sync.Mutex
	detections := 0

	onRecvFrames := func(outputSamples, inputSamples []byte, framecount uint32) {
		if len(inputSamples) == 0 {
			return
		}

		// Convert bytes to int16
		n := len(inputSamples) / 2
		samples := make([]int16, n)
		for i := 0; i < n; i++ {
			samples[i] = *(*int16)(unsafe.Pointer(&inputSamples[i*2]))
		}

		mu.Lock()
		detected := det.ProcessAudio(samples)
		if detected {
			detections++
			count := detections
			avg := det.SlidingAverage()
			mu.Unlock()
			fmt.Printf("\n\n*** WAKE WORD DETECTED (#%d) at %s ***\n",
				count, time.Now().Format("15:04:05"))
			fmt.Printf("    sliding average: %.4f (cutoff: %.4f)\n\n", avg, *cutoff)
			// Reset to allow re-detection
			mu.Lock()
			det.Reset()
			mu.Unlock()
		} else {
			mu.Unlock()
		}
	}

	deviceCallbacks := malgo.DeviceCallbacks{
		Data: onRecvFrames,
	}
	device, err := malgo.InitDevice(ctx.Context, deviceConfig, deviceCallbacks)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing capture device: %v\n", err)
		os.Exit(1)
	}

	err = device.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting capture: %v\n", err)
		os.Exit(1)
	}
	defer device.Uninit()

	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Config: cutoff=%.2f window=%d step=%dms\n", *cutoff, *window, *stepMs)
	fmt.Printf("Listening on microphone... (Ctrl+C to quit)\n\n")

	// Wait for Ctrl+C
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	<-sig

	fmt.Printf("\n\nStopped. Total detections: %d\n", detections)
}

func makeBar(prob float64, width int) string {
	filled := int(prob * float64(width))
	if filled > width {
		filled = width
	}
	return "|" + strings.Repeat("#", filled) + strings.Repeat(".", width-filled) + "|"
}
