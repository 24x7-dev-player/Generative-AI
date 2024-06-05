# pip install opencv-python --quiet
# pip install moviepy --quiet
import cv2
from moviepy.editor import VideoFileClip
import time
import base64
import os
VIDEO_PATH = "/content/videoplayback.mp4"

def process_video(video_path, seconds_per_frame=2):   
  base64Frames = []
  base_video_path, _ = os.path.splitext(video_path)
  video = cv2.VideoCapture(video_path)
  total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))   # Retrieves the total number of frames in the video.
  fps = video.get(cv2.CAP_PROP_FPS)                       
  # This line gets the frames per second (fps) of the video.
  frames_to_skip = int(fps * seconds_per_frame)           
  # This calculates how many frames to skip based on the desired sampling rate (frames per second multiplied by seconds per frame).
  curr_frame=0
  # Loop through the video and extract frames at specified sampling rate
  while curr_frame < total_frames - 1:
      video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
      success, frame = video.read()
      if not success:
          break
      _, buffer = cv2.imencode(".jpg", frame)
      base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
      curr_frame += frames_to_skip
  video.release()

  # Extract audio from video
  audio_path = f"{base_video_path}.mp3"
  clip = VideoFileClip(video_path)
  clip.audio.write_audiofile(audio_path, bitrate="32k")
  clip.audio.close()
  clip.close()

  print(f"Extracted {len(base64Frames)} frames")
  print(f"Extracted audio to {audio_path}")
  return base64Frames, audio_path

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)




response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content": "You are generating a video summary. Please provide a summary of the video. Respond in Markdown."},
    {"role": "user", "content": [
        "These are the frames from the video.",
        *map(lambda x: {"type": "image_url",
                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
        ],
    }
    ],
    temperature=0,
)
print(response.choices[0].message.content)




# Transcribe the audio
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb"),
)
## OPTIONAL: Uncomment the line below to print the transcription
#print("Transcript: ", transcription.text + "\n\n")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content":"""You are generating a transcript summary. Create a summary of the provided transcription. Respond in Markdown."""},
    {"role": "user", "content": [
        {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
        ],
    }
    ],
    temperature=0,
)
print(response.choices[0].message.content)


## Generate a summary with visual and audio
response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content":"""You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown"""},
    {"role": "user", "content": [
        "These are the frames from the video.",
        *map(lambda x: {"type": "image_url",
                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
        {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
        ],
    }
],
    temperature=0,
)
print(response.choices[0].message.content)