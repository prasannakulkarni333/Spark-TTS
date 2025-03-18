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
    segmentation_threshold=450,  # Do not go above this if you want to crash or you have better GPU
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





 
    # Call the function (adjust parameters as needed)
    import pathlib, os
    path_vid = pathlib.Path(r"C:\Users\Prasanna\Documents\GitHub\pop_crawler\pop_crawler\pop_crawler\vids")

    for root, dirs, files in os.walk(path_vid):
        print("root",root)
        print("dirs",dirs)
        print("files",files)
        text = ""
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    sample_text = f.read()
                    text = text + ". " +sample_text
        print("tststgsetswt",text)
        if text:
            no_of_words = len(text.split())
            print("no_of_words",no_of_words)
            import subprocess
            # python tts_cli.py --text "Let's test some voice features." --gender female --pitch high --emotion HAPPY --seed 42
            # subprocess.run([f"python tts_cli.py --text {text} --gender female --pitch high --emotion HAPPY --seed 42 --save_dir {root}"], shell=True)
            output_file = generate_tts_audio(
                str(no_of_words) + text,
                gender="female",
                seed=42,
                pitch="moderate",
                speed="moderate",
                save_dir=root
            )

    # output_file = generate_tts_audio(
    #     sample_text, gender="female",seed=42, pitch="moderate", speed="moderate"
    # )
    # output_file = generate_tts_audio(
    #     f_text, 
    #     # prompt_speech_path="./output.wav",
    # )
    # print("Generated audio file:", output_file)