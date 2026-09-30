import asyncio
import edge_tts

text = "Meet SUMARO—an AI-ML-assisted smartphone-based dead reckoning system designed to keep estimating vehicle position when GNSS is unavailable."

candidates = [
    ("andrew_multi", "en-US-AndrewMultilingualNeural"),
    ("brian_multi", "en-US-BrianMultilingualNeural"),
    ("christopher", "en-US-ChristopherNeural"),
    ("guy", "en-US-GuyNeural"),
    ("eric", "en-US-EricNeural"),
]

async def main():
    for tag, voice in candidates:
        out = f"video_assets/audio/test_{tag}.mp3"
        comm = edge_tts.Communicate(text, voice, rate="+15%")
        await comm.save(out)
        print(f"Generated {out} with voice {voice}")

if __name__ == "__main__":
    asyncio.run(main())
