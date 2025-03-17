import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import torch
import numpy as np
import soundfile as sf
import logging
from datetime import datetime
from cli.SparkTTS import SparkTTS
print("_________________________!!!!!!!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

def generate_tts_audio(
    text,
    model_dir=r"C:\Users\Prasanna\Documents\GitHub\Spark-TTS\pretrained_models\Spark-TTS-0.5B",
    device="cuda:0",
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
    seed=None,
    save_dir=r"C:\Users\Prasanna\Documents\GitHub\Spark-TTS\example\results",
    segmentation_threshold=250,  # Do not go above this if you want to crash or you have better GPU
    prompt_audio=None,
    filename=None
):
    """
    Generates TTS audio from input text, splitting into segments if necessary.

    Args:
        text (str): Input text for speech synthesis.
        model_dir (str): Path to the model directory.
        device (str): Device identifier (e.g., "cuda:0" or "cpu").
        prompt_speech_path (str, optional): Path to prompt audio for cloning.
        prompt_text (str, optional): Transcript of prompt audio.
        gender (str, optional): Gender parameter ("male"/"female").
        pitch (str, optional): Pitch parameter (e.g., "moderate").
        speed (str, optional): Speed parameter (e.g., "moderate").
        save_dir (str): Directory where generated audio will be saved.
        segmentation_threshold (int): Maximum number of words per segment.

    Returns:
        str: The unique file path where the generated audio is saved.
    """
    import pathlib
    model_dir = pathlib.WindowsPath(model_dir)
    save_dir = pathlib.WindowsPath(save_dir)
    model_dir = str(model_dir)
    save_dir = str(save_dir)
    # print("prompt_speech_path", prompt_speech_path)
    logging.info("Initializing TTS model...")
    device = torch.device(device)
    model = SparkTTS(model_dir, device)

    # Ensure the save directory exists.
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")
    print(f"{save_dir}_________________________!!!!!!!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Check if the text is too long.
    words = text.split()
    if len(words) > segmentation_threshold:
        logging.info("Input text exceeds threshold; splitting into segments...")
        segments = [
            " ".join(words[i : i + segmentation_threshold])
            for i in range(0, len(words), segmentation_threshold)
        ]
        wavs = []
        for seg in segments:
            with torch.no_grad():
                wav = model.inference(
                    seg,
                    prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    seed=seed,
                )
            wavs.append(wav)
        final_wav = np.concatenate(wavs, axis=0)
    else:
        with torch.no_grad():
            final_wav = model.inference(
                text,
                prompt_speech_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
                seed=seed,
            )

    # Save the generated audio.
    if filename:
        save_path = os.path.join(save_dir, f"{filename}.wav")
    sf.write(save_path, final_wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")
    return save_path


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="tts_generation.log",  # Log file name
    filemode="a"  # Append mode (use "w" to overwrite on each run)
)

    log = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)





    # Sample input (feel free to adjust)
    sample_text = (
        "The mind that opens to a new idea never returns to its original size. "
        "Hellstrom’s Hive: Chapter 1 – The Awakening. Mara Vance stirred from a deep, dreamless sleep, "
        "her consciousness surfacing like a diver breaking through the ocean's surface. "
        "A dim, amber light filtered through her closed eyelids, warm and pulsing softly. "
        "She hesitated to open her eyes, savoring the fleeting peace before reality set in. "
        "A cool, earthy scent filled her nostrils—damp soil mingled with something sweet and metallic. "
        "The air was thick, almost humid, carrying with it a faint vibration that resonated in her bones. "
        "It wasn't just a sound; it was a presence. "
        "Her eyelids fluttered open. Above her stretched a ceiling unlike any she'd seen—organic and alive, "
        "composed of interwoven tendrils that glowed with the same amber light. They pulsated gently, "
        "like the breathing of some colossal creature. Shadows danced between the strands, creating shifting patterns."
    )

    f_text = "Hi there everyone! I am a cloned voice. I belong to a new era of voice cloning technology."

    # Call the function (adjust parameters as needed)
    output_file = generate_tts_audio(
        sample_text, gender="female",seed=42, pitch="moderate", speed="moderate"
    )
    # output_file = generate_tts_audio(
    #     f_text, 
    #     # prompt_speech_path="./output.wav",
    # )
    print("Generated audio file:", output_file)