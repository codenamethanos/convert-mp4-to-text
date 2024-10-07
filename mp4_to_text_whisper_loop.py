from moviepy.editor import VideoFileClip
import os
import whisper
import time  # Import the time module

def convert_mp4_to_wav(input_file, output_file):
    """
    Convert an MP4 video file directly to a WAV audio file using MoviePy,
    ensuring the output is mono and at the correct sample rate.
    """
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    
    # Set parameters for audio conversion
    ffmpeg_params = [
        "-ac", "1",  # Set audio channels to mono
        "-ar", "16000"  # Set audio sample rate to 16000 Hz
    ]
    
    audio_clip.write_audiofile(output_file, codec='pcm_s16le', ffmpeg_params=ffmpeg_params)
    audio_clip.close()

def transcribe_audio(audio_path):
    """
    Transcribe the given audio file using the Whisper model.
    """
    model = whisper.load_model("small")  # Load the Whisper small model
    result = model.transcribe(audio_path)
    return result["text"]

def write_transcript(transcript_text, output_path):
    """
    Write the transcription results to a text file.
    """
    with open(output_path, 'w') as f:
        f.write(transcript_text + '\n')

def process_videos(video_files, media_dir, audio_dir, output_dir):
    """
    Process a list of video files to convert to audio, transcribe, and save transcripts.
    """
    for video_file in video_files:
        start_time = time.time()  # Start timing

        input_video = os.path.join(media_dir, video_file)
        output_audio = os.path.join(audio_dir, video_file.replace(".mp4", ".wav"))
        output_transcript = os.path.join(output_dir, video_file.replace(".mp4", ".txt"))

        # Convert MP4 to WAV
        convert_mp4_to_wav(input_video, output_audio)

        # Transcribe the audio file
        transcript_text = transcribe_audio(output_audio)

        # Write the transcription to a file
        write_transcript(transcript_text, output_transcript)

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
        "How to Build a Startup Without Funding by Pieter Levels Dojo Bali.mp4",
        "5.2-+Getting+lead+for+date.mp4",
        "5.3+-+Messaging+Sequence+For+Dates.mp4",
        "5.4+venus+section.mp4",
        "5.5+-+setting+up+date.mp4",
        "5.6-+texting+post+setup.mp4",
        "5.7-+Date+Mindset.mp4",
        "5.8-+what+to+do+on+date.mp4",
        "5.9-+What+To+Do+Back+at+the+House.mp4",
        "5.10-+Setting+Up+the+House.mp4",
    ]

    process_videos(video_files, media_dir, audio_dir, output_dir)

if __name__ == "__main__":
    main()
