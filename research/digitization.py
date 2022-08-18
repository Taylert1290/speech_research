import speech_recognition as sr


def recognizer():
    r = sr.Recognizer()
    sample = sr.AudioFile("../audio_files/Speaker26_001.wav")
    with sample as source:
        audio = r.record(source, duration=4)

    # Recognize the speech
    text = r.recognize_google(audio)
    print(text)


if __name__ == "__main__":
    recognizer()
