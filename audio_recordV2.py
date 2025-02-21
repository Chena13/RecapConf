from gpiozero import LED, Button
from time import sleep
import threading
import sounddevice as sd
import numpy as np
import wave

from diarization import prediction, transform_dataframe,summarize 
print ("end of import")
# GPIO setup
led = LED(27)
button = Button(18)

# State to track recording
is_recording = False
recording_thread = None  # Placeholder for the recording thread
audio_data = []  # To store the audio data while recording

# Sampling rate and duration settings
sampling_rate = 44100  # Samples per second (standard for audio)
channels = 1  # Mono audio
dtype = np.int16  # Data type for audio

# File path for saving the recorded audio
output_file = "recorded_audio.wav"

def audio_callback(indata, frames, time, status):
    """Callback function for real-time audio recording."""
    if status:
        print(status)
    if is_recording:
        audio_data.append(indata.copy())  # Store the audio data while recording

def start_recording():
    """Start recording audio."""
    global audio_data
    print("Recording launched.")
    audio_data = []  # Clear any previous audio data
    with sd.InputStream(callback=audio_callback, channels=channels, samplerate=sampling_rate, dtype=dtype):
        while is_recording:
            sleep(1)  # Continue recording while is_recording is True
    print("Recording stopped.")

def save_audio():
    """Save the recorded audio to a file."""
    global audio_data
    print("Saving recorded audio...")
    try :
        audio_data_array = np.concatenate(audio_data, axis=0)  # Concatenate all recorded chunks
    except ValueError:
        pass
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(np.dtype(dtype).itemsize)
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_data_array.tobytes())  # Write audio data to a WAV file
    print("Audio saved.")

def toggle_recording():
    global is_recording, recording_thread

    if is_recording:
        # Stop recording
        is_recording = False
        led.off()  # Turn off the LED
        print("Stopping recording...")
        if recording_thread and recording_thread.is_alive():
            recording_thread.join()  # Wait for the recording thread to finish
        # Save the audio after recording
        save_audio()
        print("end of audio")
        sleep(10)
        pred_df = prediction("recorded_audio.wav","training_df.json")
        print("end of prediction")
        transform_dataframe(pred_df)
        print("end of transform") ## debug
        sleep(10)
        summarize("transcript.txt")
        print ("end of summarize. You can find the summary in the summary file")
        
        
    else:
        # Start recording
        is_recording = True
        led.blink(on_time=0.2, off_time=0.2)  # Blink during recording
        print("Starting recording...")
        recording_thread = threading.Thread(target=start_recording)
        recording_thread.start()

# Attach the button press event to the function
button.when_pressed = toggle_recording

# Keep the program running
try:
    while True:
        sleep(0.1)  # Reduce CPU usage
except KeyboardInterrupt:
    print("\nExiting gracefully")
