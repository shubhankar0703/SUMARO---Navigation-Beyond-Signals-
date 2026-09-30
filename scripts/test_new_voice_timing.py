import asyncio
import os
import subprocess
import edge_tts
import imageio_ffmpeg

SCENES = [
    ("scene_01", 0.0, 8.0, "+12%",
     "We trust navigation to get us where we need to go. But what happens when GNSS disappears?"),
    ("scene_02", 8.0, 10.0, "+30%",
     "In tunnels, underground parking, dense urban areas or during signal outages, GNSS can become unavailable. Navigation can freeze, jump, or lose the vehicle's true position."),
    ("scene_03", 18.0, 4.0, "+8%",
     "So how do we keep navigating without it?"),
    ("scene_04", 22.0, 8.0, "+20%",
     "Meet SUMARO—an AI-ML-assisted smartphone-based dead reckoning system designed to keep estimating vehicle position when GNSS is unavailable."),
    ("scene_05", 30.0, 9.0, "+20%",
     "When GNSS is available, SUMARO combines GNSS and smartphone sensor data through sensor fusion, while calibrating the phone's orientation to the vehicle."),
    ("scene_06", 39.0, 9.0, "+25%",
     "When GNSS is lost, SUMARO switches to dead reckoning. Using the vehicle's motion captured by the smartphone, it continues estimating where the vehicle is moving."),
    ("scene_07", 48.0, 11.0, "+35%",
     "But dead reckoning has a major challenge: small sensor errors accumulate over time, causing position drift. SUMARO uses AI and machine learning to identify motion patterns, sensor noise and anomalies, helping reduce these errors."),
    ("scene_08", 59.0, 11.0, "+15%",
     "SUMARO combines these estimates through adaptive sensor fusion and uses offline map matching to determine which road the vehicle is most likely following."),
    ("scene_09", 70.0, 10.0, "+18%",
     "When GNSS returns, SUMARO compares the estimated position with the recovered GNSS position, corrects the state, and smoothly returns to normal navigation."),
    ("scene_10", 80.0, 4.7, "+15%",
     "For the driver, it stays simple: just navigation that keeps working."),
    ("scene_11", 84.7, 5.3, "+22%",
     "When GNSS disappears, SUMARO doesn't stop navigating—it estimates, corrects, and keeps going.")
]

async def test_timing():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    voice = "en-US-AndrewMultilingualNeural"
    print(f"Testing voice: {voice}")
    
    for tag, start, dur, rate, text in SCENES:
        out = f"video_assets/audio/check_{tag}.mp3"
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(out)
        
        res = subprocess.run([ffmpeg, '-i', out], stderr=subprocess.PIPE, text=True)
        clip_dur = 0.0
        for line in res.stderr.split('\n'):
            if 'Duration:' in line:
                dur_str = line.split('Duration:')[1].split(',')[0].strip()
                p = dur_str.split(':')
                clip_dur = float(p[0])*3600 + float(p[1])*60 + float(p[2])
                
        margin = dur - clip_dur
        print(f"[{tag}] Window: {dur:4.1f}s | Audio: {clip_dur:4.2f}s | Margin: {margin:+4.2f}s | Rate: {rate}")

if __name__ == "__main__":
    asyncio.run(test_timing())
