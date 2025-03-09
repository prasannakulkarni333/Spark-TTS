import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import torch
import numpy as np
import soundfile as sf
import logging
from datetime import datetime
from cli.SparkTTS import SparkTTS


def generate_tts_audio(
    text,
    model_dir="pretrained_models/Spark-TTS-0.5B",
    device="cuda:0",
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
    segmentation_threshold=150,  # Do not go above this if you want to crash or you have better GPU
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
    logging.info("Initializing TTS model...")
    device = torch.device(device)
    model = SparkTTS(model_dir, device)

    # Ensure the save directory exists.
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

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
            )

    # Save the generated audio.
    sf.write(save_path, final_wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")
    return save_path


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
    text ="""
    Welcome to another week full of AI releases and there seems to be common theme here. We have a lot of Chinese releases coming at us following the DeepSeek announcement. All of them are open source, but also chat GPT is shipping various updates that most people didn't even talk about. And even MetaAI is updating its chat bots all in the wake of the DeepSeek hype. So all that and so much more in this week's episode of AI News. You can use the show that breaks down all of this week's AI releases and filters for the ones that you can actually put to work today. So let's just start by talking about chat GPT and there's multiple changes here. Some of them regional, some of them global. So when I start by talking about the general updates, which refer to GPT 4.0, the first one is that they updated the knowledge cutoff. Now it has data up to June 2024, meaning if you ask it about any significant events from June 2024, it will know about it. Secondly, they improved the image understanding capabilities, and this seems to also be a reaction to some of the Chinese releases that happened earlier this week. We'll talk about that later in this video. There's a new Quen model that is state-of-the-art that image recognition. So chat, you be these forced to ship some updates that they probably had in their pocket. Beyond that, the model is better at stem subjects and I guess it uses more emojis if you prompt with emojis. So if you don't want those don't use emojis in your prompt. And then there's two regional updates for anybody in the use Switzerland Liechtenstein or Iceland. Now everybody has these updated custom instructions which make it even easier for people who are not that deep into this to customize their chat GPT experience. The same thing goes for the video features. If you're using the desktop app and you're in the EU, Norway, Iceland, Liechtenstein or Switzerland, you can now update your desktop app. And if you update it, you should be able to use the mobile app with the video features. We covered this when it came out. Now it's just available to everybody. And also there has been one more change in the interface, which is the following. If you open up a brand new 4.0 chat, they're just rolling out this new thing, but interesting, I was preparing the video on my laptop and here on my desktop, it's not even available. So this is just rolling out, but it's essentially a light bulb button that you can select, which will activate O1. And I'm bringing this up not because there's a new button, but because the main rumor right now is that O3 mini should be shipping today, but it hasn't happened and I have to record this video at some point. So just as a reminder, O3 mini will have the intelligence of O1 or DeepSeek. I guess they're relatively on the same level, but it will be way faster. And then O3 full will surpass anything that we have out of the US or China. 03 there's no release date on 03mini they said end of January, which is now and I suppose this is going to ship very, very soon. And that's why they're also changing up the interface. Okay, but hold up, there's even more chat GPT news and the changes they made to the canvas here are actually some of my favorites in recent history. Okay, so this one really went under the radar. I mean, even operator went under the radar last week with all the deep seek news. But I personally am a really big fan of upgrades to tools that I'm actually using and canvas is something that I use all the time and they made it way better. So let's just have a look at this in the chat jibbit interface. And the first upgrade we see here is that now O1 is not grayed out anymore. So the thinking model that is exceptional at larger projects that require more foresight or planning. And of course, as you might know, these thinking models also Excel in coding. So now we can use this instead of GBC 4.0 with Canvas. So I just have to go down here, enable Canvas. And the second update here is that it can now display HTML code, meaning you can build front ends of websites with this and actually see what it is doing. This was one of my main points of criticism of the Canvas feature when it came out. Just quickly before I show you this, I want to point out that this was already available inside of Anthropics Cloud. Their artifact feature is state of the art at building front ends and showing them to you right away. It's really good. But guess what? With the use of O1 that is really good at coding and this amazing Canvas feature that, matter of fact, is the best thing there is for casual users to quickly produce code is a very potent combination. So let's just give you something simple like create a dashboard to track my personal finances as a video freelancer and when I run this, one will think through the different steps and alternatives here, Canvas will build and display this. So this is what we were used to. It writes all the code but when it wrote HTML code, web code, it never showed us and you needed to bring this outside of the app which is a pain because iterating quickly is what this is good at. So let's click preview. Boom, there it is. It just works and it built a personal finance dashboard for you in seconds. I think this is really impressive. I do want to point out that it still lacks some of the amazing features of Anthropics artifacts. Like for example, this website being actually usable, none of these buttons actually do anything right here. Or the website being shareable with others with artifacts. You can build a little tool and it just works and they host it too. But for non-coders, this combo of a thinking model and its ability to create websites in one shot. And if you want to just change it, then you can talk to right here, you could even use advanced voice, or you could use one of these presets here. Like, hey, I don't understan
    """

    # Call the function (adjust parameters as needed)
    output_file = generate_tts_audio(
        text, gender="male", pitch="moderate", speed="moderate"
    )
    print("Generated audio file:", output_file)