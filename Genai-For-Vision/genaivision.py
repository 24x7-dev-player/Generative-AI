# ! pip install openai requests pillow
from openai import OpenAI 
import requests 
from PIL import Image
from io import BytesIO


# Initialize OpenAI client
client = OpenAI(api_key="nanana")

# Set the prompt
prompt = "Generate a monkey working on a computer and writing program in Python."

# call the OpenAI API
response = client.images.generate(
    model = "dall-e-3",
    prompt=prompt,
    n=1,
    size="1024x1024",
    response_format="url",
)

generated_image_url = response.data[0].url
print(generated_image_url)

generated_image = requests.get(generated_image_url).content

# Open the image
display(Image.open(BytesIO(generated_image)))