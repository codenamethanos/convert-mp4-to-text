from moviepy.editor import VideoFileClip
import os
import whisper
import time
import numpy as np
from tqdm import tqdm

def convert_mp4_to_wav(input_file, output_file):
    """
    Convert an MP4 video file directly to a WAV audio file using MoviePy,
    ensuring the output is mono and at the correct sample rate.
    """
    print(f"Converting {input_file} to WAV format...")
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    
    # Set parameters for audio conversion
    ffmpeg_params = [
        "-ac", "1",  # Set audio channels to mono
        "-ar", "16000"  # Set audio sample rate to 16000 Hz
    ]
    
    audio_clip.write_audiofile(output_file, codec='pcm_s16le', ffmpeg_params=ffmpeg_params)
    audio_clip.close()
    print(f"Conversion complete. WAV file saved as {output_file}")

def transcribe_audio_streaming(audio_path, model, output_path):
    """
    Transcribe the given audio file using the Whisper model with streaming,
    displaying results in real-time and writing to file incrementally.
    """
    # Load audio
    audio = whisper.load_audio(audio_path)
    
    # Determine chunk size (30 seconds)
    chunk_length = 30 * 16000  # 30 seconds * 16000 sample rate
    
    # Calculate number of chunks
    num_chunks = int(np.ceil(len(audio) / chunk_length))
    
    # Open the output file in append mode
    with open(output_path, 'a', encoding='utf-8') as f:
        # Process audio in chunks
        with tqdm(total=num_chunks, desc="Transcribing", unit="chunk") as pbar:
            for i in range(num_chunks):
                # Extract chunk
                chunk_start = i * chunk_length
                chunk_end = min(len(audio), (i + 1) * chunk_length)
                audio_chunk = audio[chunk_start:chunk_end]

                # Pad audio chunk if it's shorter than 30 seconds
                if len(audio_chunk) < chunk_length:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_length - len(audio_chunk)))

                # Transcribe chunk
                result = model.transcribe(audio_chunk, fp16=False)
                chunk_text = result["text"].strip()

                # Display chunk result
                print(f"\nChunk {i+1}/{num_chunks}: {chunk_text}")

                # Write chunk result to file
                f.write(chunk_text + ' ')
                f.flush()  # Ensure the text is written immediately

                # Update progress bar
                pbar.update(1)

    print("\nTranscription complete. Full transcript saved to", output_path)

def process_videos(video_files, media_dir, audio_dir, output_dir):
    """
    Process a list of video files to convert to audio, transcribe, and save transcripts.
    """
    # Load Whisper model once
    print("Loading Whisper model (small)...")
    model = whisper.load_model("small")

    for video_file in video_files:
        print(f"\nProcessing video: {video_file}")
        start_time = time.time()  # Start timing

        input_video = os.path.join(media_dir, video_file)
        output_audio = os.path.join(audio_dir, video_file.replace(".mp4", ".wav"))
        output_transcript = os.path.join(output_dir, video_file.replace(".mp4", ".txt"))

        # Convert MP4 to WAV
        convert_mp4_to_wav(input_video, output_audio)

        # Transcribe the audio file
        print("Starting audio transcription. This may take a while...")
        transcribe_audio_streaming(output_audio, model, output_transcript)

        end_time = time.time()  # End timing
        duration = end_time - start_time  # Calculate duration

        print(f"Transcription complete for {video_file}. Transcript saved to {output_transcript}.")
        print(f"Time taken: {duration:.2f} seconds.")

def main():
    media_dir = "media"
    audio_dir = "audio"
    output_dir = "output"
    
    # Ensure output directories exist
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # List of video files to process
    video_files = [
        # "How to Build a Startup Without Funding by Pieter Levels Dojo Bali.mp4",
        "tobi-square.mp4"
    ]

    process_videos(video_files, media_dir, audio_dir, output_dir)

if __name__ == "__main__":
    main()