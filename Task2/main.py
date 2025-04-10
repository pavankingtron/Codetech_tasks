
import speech_recognition as sr

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        print("Listening to the audio...")
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        print("\n--- Transcription ---\n")
        print(text)
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

if __name__ == "__main__":
    audio_path = "audio_sample.wav"  
    transcribe_audio(audio_path)
