import asyncio
import edge_tts
import os
import subprocess
import imageio_ffmpeg

# Scene windows: (start_time, end_time, duration)
SCENE_TIMINGS = [
    ("scene_01", 0.0, 8.0, 8.0, "+12%", "We trust navigation to get us where we need to go. But what happens when GNSS disappears?"),
    ("scene_02", 8.0, 18.0, 10.0, "+36%", "In tunnels, underground parking, dense urban areas or during signal outages, GNSS can become unavailable. Navigation can freeze, jump, or lose the vehicle's true position."),
    ("scene_03", 18.0, 22.0, 4.0, "+8%", "So how do we keep navigating without it?"),
    ("scene_04", 22.0, 30.0, 8.0, "+25%", "Meet SUMARO—an AI-ML-assisted smartphone-based dead reckoning system designed to keep estimating vehicle position when GNSS is unavailable."),
    ("scene_05", 30.0, 39.0, 9.0, "+25%", "When GNSS is available, SUMARO combines GNSS and smartphone sensor data through sensor fusion, while calibrating the phone's orientation to the vehicle."),
    ("scene_06", 39.0, 48.0, 9.0, "+28%", "When GNSS is lost, SUMARO switches to dead reckoning. Using the vehicle's motion captured by the smartphone, it continues estimating where the vehicle is moving."),
    ("scene_07", 48.0, 58.0, 10.0, "+30%", "Dead reckoning faces a major challenge: small sensor errors accumulate over time, causing position drift. SUMARO uses AI and machine learning to identify motion patterns and reduce these errors."),
    ("scene_08", 58.0, 70.0, 12.0, "+15%", "SUMARO combines these estimates through adaptive sensor fusion and uses offline map matching to determine which road the vehicle is most likely following."),
    ("scene_09", 70.0, 80.0, 10.0, "+20%", "When GNSS returns, SUMARO compares the estimated position with the recovered GNSS position, corrects the state, and smoothly returns to normal navigation."),
    ("scene_10", 80.0, 87.0, 7.0, "+10%", "For the driver, it stays simple: just navigation that keeps working."),
    ("scene_11", 87.0, 90.0, 3.0, "+28%", "When GNSS disappears, SUMARO doesn't stop navigating—it estimates, corrects, and keeps going.")
]

async def generate_speech():
    os.makedirs("video_assets/audio", exist_ok=True)
    voice = "en-IN-PrabhatNeural"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    for tag, start, end, max_dur, default_rate, text in SCENE_TIMINGS:
        out_file = f"video_assets/audio/{tag}.mp3"
        rate_int = int(default_rate.replace('%', ''))
        
        # Iteratively find rate that fits comfortably (with >= 0.4s margin)
        while True:
            rate_str = f"{rate_int:+d}%"
            communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch="+0Hz")
            await communicate.save(out_file)
            
            res = subprocess.run([ffmpeg_exe, '-i', out_file], stderr=subprocess.PIPE, text=True)
            dur = 0.0
            for line in res.stderr.split('\n'):
                if 'Duration:' in line:
                    dur_str = line.split('Duration:')[1].split(',')[0].strip()
                    p = dur_str.split(':')
                    dur = float(p[0])*3600 + float(p[1])*60 + float(p[2])
            if dur <= max_dur - 0.3 or rate_int >= 60:
                break
            rate_int += 5
            
        print(f"[{tag}] Allocated: {max_dur:.1f}s | Audio: {dur:.2f}s | Margin: {max_dur - dur:.2f}s | Rate: {rate_str}")

if __name__ == "__main__":
    asyncio.run(generate_speech())
