import serial
import time

# Setup serial port (adjust COM port for your system)
ser = serial.Serial('COM5', 9600)  # Use correct port on Windows, /dev/ttyUSB0 on Linux
time.sleep(2)  # Wait for ESP32 to reset

def analyze_ecg_windows_and_send(signal, fs=360, window_sec=10, step_sec=2, model=model, hr_threshold=HR_THRESHOLD):
    samples_per_window = fs * window_sec
    samples_per_step = fs * step_sec
    num_windows = (len(signal) - samples_per_window) // samples_per_step + 1

    for w in range(num_windows):
        start = w * samples_per_step
        end = start + samples_per_window
        window_signal = bandpass_filter(signal[start:end])

        peaks, _ = find_peaks(window_signal, height=np.mean(window_signal) + np.std(window_signal), distance=fs//3)

        if len(peaks) < 2:
            ser.write(b'0')  # No pacing
            continue

        beat_segments = []
        for pos in peaks:
            if pos - 75 >= 0 and pos + 75 < len(window_signal):
                seg = window_signal[pos - 75:pos + 75]
                seg = (seg - np.mean(seg)) / np.std(seg)
                beat_segments.append(seg)

        if len(beat_segments) == 0:
            ser.write(b'0')
            continue

        X_beats = np.array(beat_segments)[..., np.newaxis]
        preds = model.predict(X_beats, verbose=0)
        pred_classes = np.argmax(preds, axis=1)

        rr_intervals = np.diff(peaks) / fs
        avg_rr = np.mean(rr_intervals)
        avg_hr = 60 / avg_rr if avg_rr > 0 else resting_hr
        arrhythmia_ratio = np.sum(pred_classes != 0) / len(pred_classes)

        if arrhythmia_ratio > 0.2:
            dynamic_threshold = hr_threshold + 5
        elif arrhythmia_ratio < 0.05:
            dynamic_threshold = hr_threshold - 3
        else:
            dynamic_threshold = hr_threshold

        pace = avg_hr < dynamic_threshold or arrhythmia_ratio > 0.5

        print(f"Window {w+1}: HR={avg_hr:.1f}, Arrhythmia={arrhythmia_ratio:.2f}, PACE={'YES' if pace else 'NO'}")

        # Send signal to ESP32
        ser.write(b'1' if pace else b'0')
        time.sleep(2)  # Wait before next window
