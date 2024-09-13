import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import whisper
from huggingface_hub import login
import os

# Streamlit app title
st.title("Multilingual Translation & Speech Assistant")

# Log in with Hugging Face API key
huggingface_token = "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"  # Your Hugging Face API Key
login(huggingface_token)

# Aya model for translation
checkpoint = "CohereForAI/aya-101"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=huggingface_token)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_auth_token=huggingface_token)

# Whisper model for speech-to-text
whisper_model = whisper.load_model("base")  # Whisper model used for STT

# Function to convert text to speech using gTTS
def text_to_speech(text, output_filename, lang='en'):
    tts = gTTS(text, lang=lang)  # Specify language ('en' for English, 'tr' for Turkish, etc.)
    tts.save(output_filename)

# Translation Section
st.header("Text Translation and Speech Synthesis")

# Select the source language
source_lang = st.selectbox("Select the source language", options=["Turkish", "Hindi"])

# Input for user text
input_text = st.text_area("Enter text for translation")

if st.button("Translate"):
    if input_text:
        # Prepare inputs for the Aya model based on the selected language
        if source_lang == "Turkish":
            lang_prefix = "Translate to English: "
        elif source_lang == "Hindi":
            lang_prefix = ""

        # Tokenize and generate translation
        inputs = tokenizer.encode(f"{lang_prefix}{input_text}", return_tensors="pt")
        outputs = aya_model.generate(inputs, max_new_tokens=128)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display translated text
        st.subheader("Translated Text")
        st.write(translated_text)

        # Generate TTS and download link
        output_audio_file = f"translation_{source_lang.lower()}.wav"
        text_to_speech(translated_text, output_audio_file, lang='en' if source_lang == "Turkish" else 'hi')
        
        # Provide audio file download
        st.audio(output_audio_file)
        with open(output_audio_file, "rb") as file:
            st.download_button("Download Audio", file, file_name=output_audio_file)

# Speech-to-Text Section
st.header("Speech-to-Text (Whisper Transcription)")

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file for transcription", type=["wav", "mp3"])

if uploaded_file:
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
    transcription_file = "transcription.txt"
    with open(transcription_file, "w") as f:
        f.write(transcription)
    
    with open(transcription_file, "rb") as file:
        st.download_button("Download Transcription", file, file_name=transcription_file)

