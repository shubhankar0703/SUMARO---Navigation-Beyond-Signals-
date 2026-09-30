import asyncio
import os
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
import edge_tts
import imageio_ffmpeg

# Scene specifications: (tag, start_time, duration, target_rate, text)
# Total video duration: exactly 90.0 seconds
SCENES = [
    ("scene_01", 0.0, 8.0, "+15%",
     "We trust navigation to get us where we need to go. But what happens when GNSS disappears?"),
    ("scene_02", 8.0, 10.0, "+50%",
     "In tunnels, underground parking, dense urban areas or during signal outages, GNSS can become unavailable. Navigation can freeze, jump, or lose the vehicle's true position."),
    ("scene_03", 18.0, 4.0, "+15%",
     "So how do we keep navigating without it?"),
    ("scene_04", 22.0, 8.0, "+40%",
     "Meet SUMARO—an AI-ML-assisted smartphone-based dead reckoning system designed to keep estimating vehicle position when GNSS is unavailable."),
    ("scene_05", 30.0, 9.0, "+40%",
     "When GNSS is available, SUMARO combines GNSS and smartphone sensor data through sensor fusion, while calibrating the phone's orientation to the vehicle."),
    ("scene_06", 39.0, 9.0, "+50%",
     "When GNSS is lost, SUMARO switches to dead reckoning. Using the vehicle's motion captured by the smartphone, it continues estimating where the vehicle is moving."),
    ("scene_07", 48.0, 11.0, "+70%",
     "But dead reckoning has a major challenge: small sensor errors accumulate over time, causing position drift. SUMARO uses AI and machine learning to identify motion patterns, sensor noise and anomalies, helping reduce these errors."),
    ("scene_08", 59.0, 11.0, "+25%",
     "SUMARO combines these estimates through adaptive sensor fusion and uses offline map matching to determine which road the vehicle is most likely following."),
    ("scene_09", 70.0, 10.0, "+30%",
     "When GNSS returns, SUMARO compares the estimated position with the recovered GNSS position, corrects the state, and smoothly returns to normal navigation."),
    ("scene_10", 80.0, 4.7, "+45%",
     "For the driver, it stays simple: just navigation that keeps working."),
    ("scene_11", 84.7, 5.3, "+70%",
     "When GNSS disappears, SUMARO doesn't stop navigating—it estimates, corrects, and keeps going.")
]

async def build_audio_track():
    os.makedirs("video_assets/audio", exist_ok=True)
    voice = "en-IN-PrabhatNeural"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    sr = 44100
    total_samples = int(90.0 * sr)
    master_voice = np.zeros(total_samples, dtype=np.float32)
    voice_mask = np.zeros(total_samples, dtype=np.float32)

    print("--- Generating Individual Scene Narration ---")
    for tag, start_sec, dur_sec, target_rate, text in SCENES:
        mp3_path = f"video_assets/audio/{tag}.mp3"
        wav_path = f"video_assets/audio/{tag}.wav"

        comm = edge_tts.Communicate(text, voice, rate=target_rate)
        await comm.save(mp3_path)
        
        # Convert to 44.1k wav
        subprocess.run([ffmpeg, '-y', '-i', mp3_path, '-ar', str(sr), '-ac', '1', wav_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        _, data = wavfile.read(wav_path)
        clip_dur = len(data) / sr

        print(f"[{tag}] Window: {start_sec:4.1f}s - {start_sec+dur_sec:4.1f}s ({dur_sec:4.1f}s) | Audio: {clip_dur:4.2f}s | Rate: {target_rate}")
        
        # Place into master track (start 0.25s after scene start)
        offset = int((start_sec + 0.25) * sr)
        audio_norm = (data.astype(np.float32) / 32768.0)
        # Apply gentle 20ms fade in/out
        fade_len = int(0.02 * sr)
        audio_norm[:fade_len] *= np.linspace(0, 1, fade_len)
        audio_norm[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        end_idx = min(offset + len(audio_norm), total_samples)
        valid_len = end_idx - offset
        master_voice[offset:end_idx] += audio_norm[:valid_len]
        voice_mask[offset:end_idx] = 1.0

    # Expand voice_mask slightly for smooth music ducking (200ms pre/post)
    kernel_len = int(0.4 * sr)
    smoothed_mask = np.convolve(voice_mask, np.ones(kernel_len)/kernel_len, mode='same')
    smoothed_mask = np.clip(smoothed_mask * 1.5, 0.0, 1.0)

    print("--- Synthesizing Cinematic Background Music & Audio Cues ---")
    t = np.linspace(0, 90.0, total_samples, endpoint=False)
    
    # 1. Warm Ambient Synth Pad (A minor chords: Am - F - C - G progression)
    # A2 (110Hz), C3 (130.8Hz), E3 (164.8Hz), A3 (220Hz), E4 (329.6Hz)
    chords = [
        (0.0, 18.0, [110.0, 164.81, 220.0, 329.63]),      # Am (Problem setup)
        (18.0, 30.0, [87.31, 130.81, 174.61, 261.63]),    # Fmaj7 (Introduction)
        (30.0, 48.0, [130.81, 164.81, 196.00, 261.63]),   # Cmaj (Calibration / Outage action)
        (48.0, 70.0, [98.00, 146.83, 196.00, 293.66]),    # Gsus4 -> G (Dead reckoning & ML fusion)
        (70.0, 85.0, [110.0, 130.81, 164.81, 261.63]),   # Am7 (Recovery)
        (85.0, 90.0, [110.0, 164.81, 220.0, 329.63, 440.0]) # Am (Resolution & End Card)
    ]
    
    pad = np.zeros(total_samples, dtype=np.float32)
    for c_start, c_end, freqs in chords:
        c_mask = (t >= c_start) & (t < c_end)
        t_c = t[c_mask]
        chord_signal = np.zeros(len(t_c), dtype=np.float32)
        for f in freqs:
            # Add fundamental and soft warm harmonics with gentle LFO vibrato
            lfo = 1.0 + 0.003 * np.sin(2 * np.pi * 3.5 * t_c)
            chord_signal += 0.3 * np.sin(2 * np.pi * f * lfo * t_c)
            chord_signal += 0.15 * np.sin(2 * np.pi * (2 * f) * t_c)
            chord_signal += 0.05 * np.sin(2 * np.pi * (3 * f) * t_c)
        # Apply smooth chord envelope
        env = np.ones(len(t_c), dtype=np.float32)
        fade_chord = min(int(1.0 * sr), len(t_c)//4)
        if fade_chord > 0:
            env[:fade_chord] = np.linspace(0, 1, fade_chord)
            env[-fade_chord:] = np.linspace(1, 0, fade_chord)
        pad[c_mask] += chord_signal * env

    # Sub-bass pulse (warm 55Hz sine with 120 bpm rhythmic swell)
    sub_pulse = 0.15 * np.sin(2 * np.pi * 55.0 * t) * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 1.0 * t)))
    music = (pad * 0.12 + sub_pulse * 0.08)

    # Master music ducking: duck by -15 dB when voiceover is active
    duck_gain = 1.0 - (0.75 * smoothed_mask)
    music *= duck_gain

    # 2. Discrete Audio Cues
    sfx = np.zeros(total_samples, dtype=np.float32)
    
    # SFX 1: GNSS Lost Alert Ping (t = 7.6s, two descending warning pulses: 880Hz -> 660Hz)
    def add_tone(start_t, dur, freq, amp):
        idx0 = int(start_t * sr)
        idx1 = int((start_t + dur) * sr)
        t_seg = np.linspace(0, dur, idx1 - idx0, endpoint=False)
        envelope = np.exp(-t_seg * 12.0)
        sfx[idx0:idx1] += amp * np.sin(2 * np.pi * freq * t_seg) * envelope

    # GNSS warning tone
    add_tone(7.6, 0.25, 880.0, 0.35)
    add_tone(7.9, 0.35, 660.0, 0.40)

    # SFX 2: SUMARO Activation Pulse (t = 39.0s, high-tech resonant sweep 150Hz -> 450Hz)
    idx0 = int(39.0 * sr)
    idx1 = int(39.8 * sr)
    t_pulse = np.linspace(0, 0.8, idx1 - idx0, endpoint=False)
    freq_sweep = np.linspace(140.0, 440.0, len(t_pulse))
    sweep_phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
    pulse_env = np.sin(np.pi * np.linspace(0, 1, len(t_pulse)))
    sfx[idx0:idx1] += 0.35 * np.sin(sweep_phase) * pulse_env

    # SFX 3: Map Matching Snap Chime (t = 66.0s, elegant double chime: 1046Hz, 1318Hz - C6, E6)
    add_tone(66.0, 0.3, 1046.5, 0.28)
    add_tone(66.15, 0.4, 1318.5, 0.25)

    # SFX 4: GNSS Restored Success Chime (t = 70.8s, major triad arpeggio: 523, 659, 784, 1046 Hz)
    add_tone(70.8, 0.2, 523.25, 0.25)
    add_tone(71.0, 0.2, 659.25, 0.25)
    add_tone(71.2, 0.2, 783.99, 0.25)
    add_tone(71.4, 0.5, 1046.5, 0.30)

    # SFX 5: Final Resolution Whoosh (t = 85.0s, subtle deep cinematic boom)
    idx0 = int(85.0 * sr)
    idx1 = int(86.5 * sr)
    t_boom = np.linspace(0, 1.5, idx1 - idx0, endpoint=False)
    sfx[idx0:idx1] += 0.3 * np.sin(2 * np.pi * 65.0 * t_boom) * np.exp(-t_boom * 2.5)

    # Final Audio Mix: Voice + Music + SFX
    final_mix = master_voice * 0.95 + music + sfx
    
    # Final master fade out at 89.5 - 90.0s
    fade_end = int(0.5 * sr)
    final_mix[-fade_end:] *= np.linspace(1, 0, fade_end)
    
    # Peak limiter / normalization
    peak = np.max(np.abs(final_mix))
    if peak > 0.95:
        final_mix = (final_mix / peak) * 0.95

    # Export master voiceover and final mix
    master_voice_int16 = (np.clip(master_voice, -0.98, 0.98) * 32767).astype(np.int16)
    final_mix_int16 = (np.clip(final_mix, -0.98, 0.98) * 32767).astype(np.int16)

    wavfile.write("video_assets/audio/master_voiceover.wav", sr, master_voice_int16)
    wavfile.write("video_assets/audio/final_audio_mix.wav", sr, final_mix_int16)
    print(f"Master voiceover written: video_assets/audio/master_voiceover.wav (duration: {len(master_voice)/sr:.1f}s)")
    print(f"Final audio mix written: video_assets/audio/final_audio_mix.wav (duration: {len(final_mix)/sr:.1f}s)")

if __name__ == "__main__":
    asyncio.run(build_audio_track())
