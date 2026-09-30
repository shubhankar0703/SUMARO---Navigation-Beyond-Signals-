import asyncio
import edge_tts

async def main():
    voices = await edge_tts.list_voices()
    print("=== English Voices ===")
    for v in voices:
        if v["Locale"].startswith("en-US") or v["Locale"].startswith("en-GB") or v["Locale"].startswith("en-CA"):
            name = v["ShortName"]
            gender = v["Gender"]
            friendly = v["FriendlyName"]
            print(f"{name:<32} {gender:<8} {friendly}")

if __name__ == "__main__":
    asyncio.run(main())
