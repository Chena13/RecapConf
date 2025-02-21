import pandas as pd
import torch
import torchaudio
import tempfile
import os
from pyannote.audio import Model, Inference
import whisper
from scipy.spatial.distance import cosine
from transformers import pipeline

def prediction(recording_path,reference_path):
    new_file=recording_path
    training_df = reference_path
    # Load the test file
    #new_file = "audio_files/testfile3.wav"
   
    # Load models
    token = "hugginface token"
    speaker_model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
    inference = Inference(speaker_model, window="whole")
    whisper_model = whisper.load_model("medium.en")
    #transcribe recording
    result = whisper_model.transcribe(new_file, word_timestamps=True)

    # Load JSON and convert embeddings back to tensors
    training_df = pd.read_json('training_df.json', orient='records', lines=True)
    training_df['embeddings'] = training_df['embeddings'].apply(lambda x: torch.tensor(x))

    # Initialize lists for the predicted labels
    predicted_labels = []
    predicted_phrases = []
    speaker_files = []

    # Process words in groups of 3
    for segment in result['segments']:
        words = segment['words']
        for i in range(0, len(words), 3):
            group = words[i:i + 3]
            
            # Concatenate text and calculate start/end times
            group_text = " ".join([word_info['word'] for word_info in group])
            group_start_time = group[0]['start']
            group_end_time = group[-1]['end']
            
            # Extract the audio for the word group using its timestamps
            waveform, fs = torchaudio.load(new_file)  # Load the entire file
            group_audio = waveform[:, int(group_start_time * fs): int(group_end_time * fs)]  # Extract group audio
            
            # Ensure group_audio has the shape [1, num_samples] (2D tensor)
            group_audio = group_audio.unsqueeze(0) if group_audio.ndimension() == 1 else group_audio
            
            # Optionally pad the group audio to ensure it has a sufficient length for processing
            target_length = 16000  # Define a target length, adjust as needed
            if group_audio.size(1) < target_length:
                padding = target_length - group_audio.size(1)
                group_audio = torch.nn.functional.pad(group_audio, (0, padding))  # Pad with zeros
            
            # Save the group audio to a temporary file for Pyannote to process
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
                torchaudio.save(temp_filename, group_audio, fs)  # Save group audio as .wav file
                
                # Extract embedding for the group in the new file using Pyannote
                group_embedding = inference(temp_filename)
            
            # Safely delete the temporary file after inference is done
            try:
                os.remove(temp_filename)
            except PermissionError as e:
                print(f"Error removing temporary file {temp_filename}: {e}")
            
            # Classify the group by calculating cosine similarity with the training embeddings
            similarities = [1 - cosine(group_embedding, train_embedding) for train_embedding in training_df['embeddings']]
            best_match_idx = similarities.index(max(similarities))  # Get the index of the best match
            
            predicted_label = training_df.iloc[best_match_idx]['label']
            
            # Store the results
            predicted_labels.append(predicted_label)
            predicted_phrases.append(group_text)

    # Create the dataframe for predicted labels
    predicted_df = pd.DataFrame({
        'phrase': predicted_phrases,
        'predicted_label': predicted_labels
    })

    speaker_words = {}
    for _, row in predicted_df.iterrows():
        label = row['predicted_label']
        phrase = row['phrase']

        if label not in speaker_words:
            speaker_words[label] = []
        speaker_words[label].append(phrase)

    return predicted_df

def transform_dataframe(pred_df):
    #create transcript in more natural format
    #function input is the prediction df variable returned by prediction ()
    predicted_df = pred_df

    # Prepare the conversation text with speaker changes
    conversation = ""
    last_speaker = None
    current_text = ""

    for _, row in predicted_df.iterrows():
        current_speaker = row['predicted_label']
        if current_speaker != last_speaker:
            # If there's a change in speaker, append the previous text and start a new one
            if current_text:
                conversation += f"Speaker {last_speaker}: {current_text.strip()}\n"
            last_speaker = current_speaker
            current_text = ""
        # Add the current row's text to the ongoing text for the current speaker
        current_text += f" {row['phrase']}"

    # Append the final speaker's text
    if current_text:
        conversation += f"Speaker {last_speaker}: {current_text.strip()}\n"

    #print(conversation)
    # save the transformed version
    with open("transcript.txt", 'w') as file:
        file.write(f"{conversation}\n")
    file.close()
    return

def summarize(transcript_path):
    #file = open("transcript.txt", "r")
    file = open(transcript_path, "r")
    summary = file.read()
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sum = summarizer(summary, max_length=150, min_length=20, do_sample=False)
    
    # Extract the summary text
    summary_text = sum[0]['summary_text']

    # Print the summary
    #print(summary_text)
    file.close()
    #store as file
    with open("summary.txt", 'w') as file:
        file.write(f"{summary_text}\n")
    file.close()
    return

## get the file : scp kafando@raspi.local:/home/kafando/AIoT_env/recorded_audio.wav D:\travail\NTU_courses\AIoT
if __name__ == "__main__":
    pred_df = prediction("recorded_audio.wav","training_df.json")
    print("end of prediction")
    transform_dataframe(pred_df)
    print("end of transform") ## debug
    time.sleep(10)
    summarize("transcript.txt")
    print ("end of summarize. You can find the summary in the summary file")
