package audiofrontend

import "math"

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// makeHannWindow creates a Hann window of the given size.
// Uses the "periodic" variant: 0.5 - 0.5*cos(2*pi*i/N)
func makeHannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	return w
}

// realFFT computes the FFT of a real signal of length n (must be power of 2).
// Returns interleaved [re0, im0, re1, im1, ..., re_{n/2}, im_{n/2}].
func realFFT(x []float64, n int) []float64 {
	// Implement Cooley-Tukey radix-2 DIT FFT on complex input,
	// then extract the real FFT result.
	re := make([]float64, n)
	im := make([]float64, n)
	copy(re, x[:n])

	// Bit-reversal permutation
	j := 0
	for i := 0; i < n; i++ {
		if i < j {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
		m := n >> 1
		for m >= 1 && j >= m {
			j -= m
			m >>= 1
		}
		j += m
	}

	// FFT butterfly
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		angle := -2 * math.Pi / float64(size)
		wRe := math.Cos(angle)
		wIm := math.Sin(angle)
		for start := 0; start < n; start += size {
			uRe, uIm := 1.0, 0.0
			for k := 0; k < half; k++ {
				tRe := uRe*re[start+k+half] - uIm*im[start+k+half]
				tIm := uRe*im[start+k+half] + uIm*re[start+k+half]
				re[start+k+half] = re[start+k] - tRe
				im[start+k+half] = im[start+k] - tIm
				re[start+k] += tRe
				im[start+k] += tIm
				newURe := uRe*wRe - uIm*wIm
				uIm = uRe*wIm + uIm*wRe
				uRe = newURe
			}
		}
	}

	// Pack into interleaved output: n/2+1 complex bins
	numBins := n/2 + 1
	out := make([]float64, numBins*2)
	for i := 0; i < numBins; i++ {
		out[2*i] = re[i]
		out[2*i+1] = im[i]
	}
	return out
}

// hzToMel converts frequency in Hz to mel scale.
func hzToMel(hz float64) float64 {
	return 1127.0 * math.Log(1.0+hz/700.0)
}

// melToHz converts mel scale to Hz.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Exp(mel/1127.0) - 1.0)
}

// makeMelFilterbank creates triangular mel-spaced filterbank weights.
func makeMelFilterbank(numChannels, fftSize, sampleRate int, lowerHz, upperHz float64) [][]float64 {
	numBins := fftSize/2 + 1
	lowerMel := hzToMel(lowerHz)
	upperMel := hzToMel(upperHz)

	// Create numChannels+2 equally spaced mel points
	melPoints := make([]float64, numChannels+2)
	for i := range melPoints {
		melPoints[i] = lowerMel + float64(i)*(upperMel-lowerMel)/float64(numChannels+1)
	}

	// Convert to FFT bin indices
	binPoints := make([]float64, numChannels+2)
	for i := range binPoints {
		hz := melToHz(melPoints[i])
		binPoints[i] = hz * float64(fftSize) / float64(sampleRate)
	}

	// Create triangular filters
	filters := make([][]float64, numChannels)
	for ch := 0; ch < numChannels; ch++ {
		filters[ch] = make([]float64, numBins)
		left := binPoints[ch]
		center := binPoints[ch+1]
		right := binPoints[ch+2]
		for bin := 0; bin < numBins; bin++ {
			fb := float64(bin)
			if fb >= left && fb <= center && center > left {
				filters[ch][bin] = (fb - left) / (center - left)
			} else if fb > center && fb <= right && right > center {
				filters[ch][bin] = (right - fb) / (right - center)
			}
		}
	}
	return filters
}
