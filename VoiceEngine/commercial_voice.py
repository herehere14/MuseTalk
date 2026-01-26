from melo.api import TTS
import sys
import os

# --- CONFIGURATION ---
TEXT = "您好，欢迎来到本行。我是您的智能银行助理，很高兴为您服务。无论您需要查询账户余额、转账汇款、了解理财产品，还是咨询贷款与信用卡信息，我都可以随时协助您。我们的目标是让您的每一次金融操作都更加安全、高效、便捷。如果您有任何疑问或特殊需求，请直接告诉我，我会一步一步为您解答，并为您提供最合适的解决方案。感谢您的信任，祝您一天愉快."
OUTPUT_FILE = "output_raw.wav"

def generate_safe_speech(text):
    # 1. Initialize Model (EN = English, ZH = Chinese/English Mix)
    # Using 'cuda' for GPU acceleration
    model = TTS(language='EN', device='cuda') 
    
    # 2. Get Speaker ID (Standard American English)
    speaker_ids = model.hps.data.spk2id
    
    print(f"Generating audio for: '{text}'...")
    
    # 3. Generate Audio
    # speed=1.0 is normal. 0.9 is more formal/slow.
    model.tts_to_file(text, speaker_ids['EN-US'], OUTPUT_FILE, speed=1.0)
    
    print(f"DONE! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Allow command line text input
    if len(sys.argv) > 1:
        TEXT = sys.argv[1]
    generate_safe_speech(TEXT)
