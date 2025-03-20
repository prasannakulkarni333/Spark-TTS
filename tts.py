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
from pathlib import Path

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
    segmentation_threshold=130,  # Do not go above this if you want to crash or you have better GPU
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
    save_path = os.path.join(save_dir, f"output.wav")

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
    vidarray = []
    log = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
    # Call the function (adjust parameters as needed)
    import pathlib, os
    path_vid = pathlib.Path(r"C:\Users\Prasanna\Documents\GitHub\pop_crawler\pop_crawler\pop_crawler\vids")

    for root, dirs, files in os.walk(path_vid):
        print("root",root)
        print("dirs",dirs)
        print("files",files)
        text = ""
        vid_arr = []
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    sample_text = f.read()
                    text = text + ". " +sample_text
                    generate_tts_audio(
                        sample_text,
                        gender="female",
                        seed=42,
                        pitch="moderate",
                        speed="moderate",
                        save_dir=root,
                        filename=file.split(".")[0]
                    )
        if len(text) > 10:            
            generate_tts_audio(
                text,
                gender="female",
                seed=42,
                pitch="moderate",
                speed="moderate",
                save_dir=root,
                filename="output"
            )
    for root, dirs, files in os.walk(path_vid):
        for file in files:
            if file.endswith(".wav") and "output" not in file:
                # get length of audio
                audio = AudioFileClip(os.path.join(root, file))
                audio_duration = audio.duration
                from moviepy import *
                image = file.split(".")[0] + ".png"
                image_clip = ImageClip(str(Path(root).joinpath(image)))
                video_clip = image_clip
                video_clip.duration = audio_duration
                video_clip.fps = 30
                vidarray.append(video_clip)

                # get length of                        
    import subprocess

    # Convert WAV to AAC using FFmpeg
    wav_file = os.path.join(root, "output.wav")
    aac_file = os.path.join(root, "output.aac")

    # FFmpeg command to convert WAV to AAC
    subprocess.run([
        "ffmpeg", "-i", wav_file, "-c:a", "aac", "-b:a", "192k", aac_file
    ])

    # Use the converted AAC audio in MoviePy
    concatenate_videoclips(vidarray, method="compose").write_videofile(
        os.path.join(root, "output.mp4"),
        audio=aac_file,
        threads=12,
        codec="mpeg4",
        audio_codec="aac"
)
    # concatenate_videoclips(vidarray, method="compose").write_videofile(
    #     os.path.join(root, "output.mp4"), audio = os.path.join(root, "output.wav"), threads = 12,codec="mpeg4",
    # # audio_codec="aac"
    # )

