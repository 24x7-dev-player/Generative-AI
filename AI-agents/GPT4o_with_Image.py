#GPT4o 
from openai import OpenAI
import os
# Set the API key and model name
MODEL="gpt-4o"
client = OpenAI(api_key=api_key)
completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, 
    # This is the system message that provides context to the model
    {"role": "user", "content": "Hello! Could you solve 2+2?"}  
    # This is the user message for which the model will generate a response
  ]
)
print("Assistant: " + completion.choices[0].message.content)


#GPT4o for Image
from IPython.display import Image, display, Audio, Markdown
import base64
IMAGE_PATH = "/content/download (4).png"
# Preview image for context
display(Image(IMAGE_PATH))
# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
base64_image = encode_image(IMAGE_PATH)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
        {"role": "user", "content": [
            {"type": "text", "text": "What's the area of the triangle?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.0,
)
print(response.choices[0].message.content)

#from URL image
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
        {"role": "user", "content": [
            {"type": "text", "text": "What's the area of the triangle?"},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"}
            }
        ]}
    ],
    temperature=0.0,
)
