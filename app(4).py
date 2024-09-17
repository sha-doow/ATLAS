 import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import whisper, uuid
from huggingface_hub import login
import os
from dotenv import load_dotenv  # Import the dotenv library

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("ATLAS: Multilingual Translation & Speech AI Assistant")

# Retrieve Hugging Face token from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Log in with your Hugging Face API key
login(huggingface_token)

# Aya model for translation
checkpoint = "CohereForAI/aya-101"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=huggingface_token)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_auth_token=huggingface_token)

# Whisper model for speech-to-text
whisper_model = whisper.load_model("base")  # Whisper model used for STT, not TTS

# Function to convert text to speech using gTTS
def text_to_speech(text, output_filename, lang='en'):
    tts = gTTS(text, lang=lang)  # Specify language ('en' for English, 'fr' for French)
    tts.save(output_filename)

# Translation Section
st.header("ATLAS: Text Translation and Speech Synthesis")

# Select the source language
source_lang = st.selectbox("Select the source language", options=["Yoruba", "Hausa", "Igbo", "French", "English"])

# Select the target language (English or French)
target_lang = st.selectbox("Select the target language", options=["English", "French"])

# Input for user text
input_text = st.text_area("Enter text for translation")
st.subheader("OR")
# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file for transcription", type=["wav", "mp3"])

if st.button("Translate"):
    if input_text:
        # Prepare inputs for the Aya model based on the selected language
        lang_prefix = f"Translate {source_lang} to {target_lang}: "

        # Tokenize and generate translation
        inputs = tokenizer.encode(f"{lang_prefix}{input_text}", return_tensors="pt")
        outputs = aya_model.generate(inputs, max_new_tokens=128)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display translated text
        st.subheader("Translated Text")
        st.write(translated_text) # TODO CHANGE TO CHAT

        # Generate TTS and provide audio file download
        output_audio_file = f"translation_{''.join(str(uuid.uuid4()).split('-'))}.wav"
        lang_code = 'en' if target_lang == "English" else 'fr'
        text_to_speech(translated_text, output_audio_file, lang=lang_code)

        # Provide audio file for listening
        st.audio(output_audio_file)
        with open(output_audio_file, "rb") as file:
            st.download_button("Download Audio", file, file_name=output_audio_file)

        # Clean up the temporary audio file
        os.remove(output_audio_file)


    if uploaded_file:
        # Speech-to-Text Section
        st.header("Speech-to-Text (ATLAS Transcription)")
        # Save the uploaded file
        audio_file_path = os.path.join("uploaded_audio", uploaded_file.name)
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcribe the audio file using Whisper
        st.write("Transcribing audio...")
        result = whisper_model.transcribe(audio_file_path)
        transcription = result["text"]

        # Display transcription
        st.subheader("Transcribed Text")
        st.write(transcription)

        # Save the transcription to a text file and provide download
        transcription_file = f"transcription_{''.join(str(uuid.uuid4()).split('-'))}.txt"
        with open(transcription_file, "w") as f:
            f.write(transcription)

        with open(transcription_file, "rb") as file:
            st.download_button("Download Transcription", file, file_name=transcription_file)

        # Clean up the temporary audio file
        os.remove(transcription_file)

    # # Model saving
    # if st.button("Save Aya Model"):
    #     model_save_path = "./aya_translation_model"
    #     aya_model.save_pretrained(model_save_path)
    #     tokenizer.save_pretrained(model_save_path)
    #     st.success(f"Model saved to {model_save_path}")
