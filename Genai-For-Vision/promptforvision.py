# !pip install -qU diffusers==0.11.1
# !pip install -qU transformers scipy ftfy accelerate
# !pip install -qU invisible_watermark safetensors
# !pip install --upgrade torch
# Downgrading jax and jaxlib to avoid errors
# !pip install jax==0.4.23 jaxlib==0.4.23
# Import the Stable Diffusion pipeline method and set up the pipeline
# Note: You can change the model here if you want
import torch
from diffusers import StableDiffusionPipeline

model_name = 'stabilityai/stable-diffusion-2-1'
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype = torch.float16)

# Convert the pipe to the CUDA language that is used by NVIDIA GPUs
pipe = pipe.to('cuda')


import random
import math

def gen_sd_img(prompt, negative_prompt = None, seed = 0):
  rnum = math.floor(random.random() * 1000000000)
  fseed = rnum
  if(seed != 'random'): fseed = seed

  image = pipe(prompt = prompt,
               negative_prompt = negative_prompt,
               num_inference_steps = 45,
               guidance_scale = 7,
               height = 768,
               width = 768,
               num_images_per_prompt = 1,
               generator = torch.Generator('cuda').manual_seed(fseed)).images[0]

  return image


gen_sd_img(prompt = 'An flying eagle', seed = 0)
gen_sd_img(prompt = 'A dog running in the river.',seed = 358870059)
gen_sd_img(prompt = 'A majestic golden eagle soaring gracefully over a rugged mountain landscape, with its wings spread wide and sunlight gleaming off its feathers',
           seed = 358870059)
gen_sd_img(prompt = 'A cat sitting on a window sill',
           seed = 358870059)
gen_sd_img(prompt = 'A digital illustration of a cat sitting on a window still',
           seed = 358870059)
gen_sd_img(prompt = 'A woman, oil painting',
           seed = 358870059)
gen_sd_img(prompt = 'A digital portrait of a woman',
           seed = 358870059)
gen_sd_img(prompt = 'A woman standing in a garden holding a flower',
           seed = 358870059)
gen_sd_img(prompt = 'A woman standing in a garden holding a flower, in a surrealist style',
           seed = 358870059)
gen_sd_img(prompt = 'Paint a vibrant market scene with fruits, vegetables, and people',
           seed = 358870059)
gen_sd_img(prompt = 'Paint a vibrant market scene with fruits, vegetables, and people in the style of Vincent van Gogh',
           seed = 358870059)
gen_sd_img(prompt = 'A serene landscape with mountains, a river, and a sunset',
           seed = 358870059)
gen_sd_img(prompt = 'A serene landscape reminiscent of a Bob Ross painting, featuring mountains, a river, and a sunset',
           seed = 358870059)
gen_sd_img(prompt = 'Create a digital artwork of a city skyline at night with neon lights',
           seed = 358870059)
gen_sd_img(prompt = 'Create a high-resolution digital artwork (at least 6000x4000 pixels) of a city skyline at night with neon lights',
           seed = 358870059)
gen_sd_img(prompt = 'A beach with a lighthouse',
           seed = 358870059)
gen_sd_img(prompt = 'A serene beach at sunrise with a weathered, red-and-white striped lighthouse, golden morning light, seagulls gracefully circling above the calm, azure waters',
           seed = 358870059)
gen_sd_img(prompt = 'A man waking in neighbourhood, with good outfit, perfect face',
           seed = 358870059)
gen_sd_img(prompt = 'A man waking in neighbourhood, with good outfit, perfect face, detailed facial features',
           negative_prompt = "ugly, poorly drawn hands, poorly drawn face",
           seed = 358870059)
gen_sd_img(prompt = 'Madara in battleground, holding a sword, detailed face, cinematic',
           seed = 358870059)
gen_sd_img(prompt = 'Madara in battleground, holding a sword, detailed face, cinematic',
           negative_prompt = 'ugly, weird faces, poorly drawn hands, poorly drawn legs, out of frame',
           seed = 358870059)
gen_sd_img(prompt = 'A man, perfect face with good outfit, detailed facial features',
           negative_prompt = 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face',
           seed = 358870059)
gen_sd_img(prompt = 'realistic photo of night sky with (clouds:1 and stars:1)',
           negative_prompt = 'birds, aircraft, tree branches',
           seed = 358870059)
gen_sd_img(prompt = 'realistic photo of night sky with (clouds:0.5 and stars:10)',
           negative_prompt = 'birds, aircraft, tree branches',
           seed = 358870059)