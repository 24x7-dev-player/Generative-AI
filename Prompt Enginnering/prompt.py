# Zero-shot Prompting
prompt_zero_shot = "Translate the following English sentence to French: 'Hello, how are you?'"

# One-shot Prompting
prompt_one_shot = "Translate the following English sentence to French: 'Good morning' -> 'Bonjour'. Now, translate: 'Hello, how are you?'"

# Few-shot Prompting
prompt_few_shot = "Translate the following English sentences to French: 'Good morning' -> 'Bonjour', 'Good night' -> 'Bonne nuit'. Now, translate: 'Hello, how are you?'"

# Chain-of-Thought Prompting
prompt_chain_of_thought = "To solve the math problem 'What is 12 + 15?', first add 10 and 10 to get 20, then add 2 and 5 to get 7, and finally add 20 and 7 to get 27. What is 12 + 15? Answer: 27"

# Instruction-based Prompting
prompt_instruction_based = "Summarize the following paragraph in one sentence: [paragraph text]"

# Role-based Prompting
prompt_role_based = "You are a helpful assistant. Provide a summary of the latest news article on climate change."

# Contextual Prompting
context = "The article discusses the impacts of climate change on coastal cities."
prompt_contextual = "Summarize the key points of the article."

# Meta-Prompting
prompt_meta = "Write a prompt that would instruct a language model to generate a short story about a space adventure."

# Interactive Prompting
user_input = input("Tell me a story about a dragon.")
response_interactive = gpt3.request(user_input)
# Model response

# Dynamic Prompting
prompt_dynamic = "Generate a poem based on the following lines: 'The sky is blue, and the sun is bright...'"

# Template-based Prompting
template = "Describe a {} {} in a detailed manner."
prompt_template = template.format("beautiful", "sunset")

# Contrastive Prompting
prompt_contrastive = "Explain why the following sentence is grammatically incorrect: 'She go to the store.' Compare it to the correct sentence: 'She goes to the store.'"

# Task-specific Prompting
prompt_task_specific = "Translate the sentence 'I love programming' into Spanish."

# Negative Prompting
prompt_negative = "Do not include any technical jargon in your response. Explain quantum mechanics in simple terms."

# Tree-based Prompting
prompt_tree_based = "Write a story about a character who finds a mysterious object in the forest. Describe the object and its origins."

# Creative Prompting
prompt_creative = "Write a poem about the beauty of nature."

# Critical Prompting
prompt_critical = "Discuss the ethical implications of genetic engineering."

# Reflective Prompting
prompt_reflective = "Reflect on a time when you faced a difficult decision. How did you approach it and what did you learn?"

# Collaborative Prompting
prompt_collaborative = "Collaborate with another participant to create a short story. Each of you will contribute alternating sentences."

# Socratic Prompting
prompt_socratic = "Explain the concept of democracy as if you were teaching it to a child."


import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

os.environ["OPENAI_API_KEY"]=api_key

import openai



def get_completion(prompt,model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,

    )


    return response.choices[0].message["content"]

text = f"""
You should express what you want a model to do by \
providing instructions that are as clear and \
specific as you can possibly make them. \
This will guide the model towards the desired output, \
and reduce the chances of receiving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short prompt. \
In many cases, longer prompts provide more clarity \
and context for the moyel, which can lead to
more detailed and releant outputs.

"""

prompt = f"""
give it in point wise manner  and also give the number to point the text delimited by triple dash \

---{text}---

"""


response = get_completion(prompt)
print(response)

prompt =f"""
Generate a list of three made-up book titles along \
with their authors and genres form you side \
Provide them in JSON format with the following keys:

Book Title
Author
Genre


"""



response = get_completion(prompt)
print(response)

text_1 = f"""

In the heart of a bustling metropolis, there lived a young man named Alex. Unbeknownst to the world, Alex possessed extraordinary abilities. He could move objects with his mind, run at incredible speeds, and even soar through the skies with nothing but his willpower.

Despite his incredible powers, Alex lived a humble life, working as a librarian by day and using his abilities to help those in need under the cover of darkness. He believed that with great power came great responsibility, a motto instilled in him by his late parents who had also possessed similar abilities.

One fateful night, as Alex patrolled the city streets, he heard the desperate cries of a woman in distress. Racing to the scene, he found a group of thugs harassing her. With lightning speed, Alex intervened, effortlessly disarming the assailants and ensuring the woman's safety.


"""




prompt = f"""
You will be provided with text delimited by triple quotes.

If it contains a sequence of instructions, \
re-write those instructions in the following format:
Step 1 - ...
Step 2 - ...
Step N - ..
If the text does not contain a sequence of instructions,\
then simply write \"No steps provided.
 ```{text_1}```

"""


response = get_completion(prompt)
print(response)

prompt = f"""
your task is to answer in consistent style.
<Student> How do I find the least common denominator for fractions with different denominators?

<Teacher> You can start by listing the multiples of each denominator and finding the smallest number they have in common. That's your least common denominator. Keep practicing, it'll get easier.

<student> how do i fine the Highest common factor for fractions with different denominators?
"""

response = get_completion(prompt)
print(response)

text = f"""


In a cozy village nestled between rolling hills, Lily and James, two adventurous friends, stumbled upon a hidden cottage tucked away in the woods. Intrigued by its mystery, they embarked on a quest, braving forests and scaling mountains. At last, atop a hill, they unearthed a treasure chest. Inside, amidst gleaming riches, they found the true treasure—their unbreakable bond, forged through shared adventures and endless laughter.
"""


prompt_1 = f"""
Perform the following actions:

1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into Hinglish.
3 - List each nime in the hindi summary.
4 - Output a json object that contains the following \

keys: hindi_summary, num_names.

Separate your answers with line breaks.
Text:```{text}```.
"""

response = get_completion(prompt_1)
print(response)

prompt = f"""

 give me detail secription about the Weiiqaeilnsoewseneo smart phone in english
"""
response = get_completion(prompt)
print (response)

prompt = f"""

 give me detail secription about the vivo smart phone in english ?

 give only if you find out infomartion about it otherwise give me nothing like this exist

"""
response = get_completion(prompt)
print (response)

fact_sheet_table = """
OVERVIEW

Part of a stylish collection of contemporary office furniture,
featuring desks, storage units, chairs, and accessories.
Available in various sizes and configurations to suit different needs.
Options include different finishes and materials for tabletop and legs.
Suitable for both home offices and commercial workspaces.
Designed for durability and functionality.
CONSTRUCTION

Sturdy steel frame with powder-coated finish for stability and longevity.
Tabletop available in high-quality laminate or solid wood options.
DIMENSIONS

WIDTH 120 CM | 47.24”
DEPTH 60 CM | 23.62”
HEIGHT 75 CM | 29.53”
OPTIONS

Choice of tabletop finishes: walnut laminate, oak laminate, or solid wood.
Legs available in matte black, brushed steel, or chrome finishes.
Optional cable management solutions for a clutter-free workspace.
MATERIALS
TABLETOP

High-pressure laminate or solid wood options.
Scratch-resistant and easy to clean for long-lasting use.

FRAME
Steel frame with powder-coated finish for durability and stability.

COUNTRY OF ORIGIN
Designed and manufactured in Denmark.

"""

prompt = f"""
Your task is to help a marketing team create a
description for a retail website of a product based
on a technical fact sheet.in a point wise manner

Write a product description based on the information
provided in the technical specifications delimited by
triple backticks.

use atmost 50 words only

At the end of the description, include every 7-character
Product ID in the technical specification.



Technical specifications: ```{fact_sheet_table}```
"""
response = get_completion(prompt)
print(response)

prompt = f"""
Your task is to help a marketing team create a
description for a retail website of a product based
on a technical fact sheet.in a point wise manner

Write a product description based on the information
provided in the technical specifications delimited by
triple backticks.

use atmost 50 words only

At the end of the description, include every 7-character
Product ID in the technical specification.




After the description, include a table that gives the
product's dimensions. The table should have two columns.
In the first column include the name of the dimension.
In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website.
Place the description in a <div> element.



Technical specifications: ```{fact_sheet_table}```
"""
response = get_completion(prompt)
print(response)

"""# summarizartion"""

prod_review = """
Bought this new smartphone for my wife's birthday, and she's thrilled with it! The sleek design and vibrant display make it a standout choice. The camera quality is impressive, capturing stunning photos and videos. However, the battery life could be better for the price point. Despite that, it arrived earlier than expected, allowing me to set it up with her favorite apps before surprising her. Overall, a great purchase that brings joy to my wife's everyday life.
"""

prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site.

Summarize the review below, delimited by triple
backticks, in at most 20 words.

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)

prompt = f"""
Your task is to extract relevant information from \
a product review from an ecommerce site to give \
feedback to the Shipping department.

From the review below, delimited by triple quotes \
extract the information relevant to camera quealuty . Limit to 30 words .

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)

review_5 = """
Purchased this new smartphone cover for my device, and I'm delighted with its quality! The design is sleek, providing excellent protection without adding bulk. The shipping was swift, arriving within two days of ordering. Overall, highly recommended!
"""

review_6 = """
Bought this stylish back cover for my smartphone, and it exceeded my expectations! The matte finish feels premium, and the snug fit offers excellent protection. Shipping was fast, arriving within two days. Couldn't be happier with my purchase!
"""

review_7 = """
Recently acquired these wireless earphones, and they're fantastic! The sound quality is impressive, delivering crisp highs and deep bass. The ergonomic design ensures a comfortable fit for extended wear. Shipping was quick, arriving within two days. Highly satisfied!
"""

reviews = [ review_5, review_6, review_7]

for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary  \
    review from an ecommerce site.

    Summarize the review below, delimited by triple \
    backticks in at most 20 words.

    Review: ```{reviews[i]}```
    """
response = get_completion(prompt)
print(i,response ,"\n")

backcover_review = """
Looking for a sleek back cover for my smartphone, and this one exceeded my expectations! and the cover made by the company and its name is The minimalist design and matte finish give it a premium look and feel. Shipping was swift, arriving within two days. Highly recommend!
"""

prompt = f"""
What is the sentiment of the following product review,
which is delimited with triple backticks?
Give your answer as either psotive or negative.

Review text: '''{backcover_review}'''
"""
response = get_completion(prompt)
print(response)

backcover_review = """
Looking for a sleek back cover for my smartphone, and this one exceeded my expectations! and the cover made by the company and its name is The minimalist design and matte finish give it a premium look and feel. Shipping was swift, arriving within two days. Highly recommend!
"""

prompt = f"""
Identify a list of emotions that the writer of the \
following review is expressing. Include no more than \
five items in the list. Format your answer as a list of \
lower-case words separated by commas.

Review text: '''{backcover_review}'''
"""
response = get_completion(prompt)
print(response)

prompt = f"""
Identify the following items from the review text:
- Item purchased by reviewer
- Company that made the item
What is the sentiment of the following product review


Give your answer as either psotive or negative

The review is delimited with triple backticks. \
Format your response as a JSON with columns as object with \

"Item" and "Brand" ,and sentiment as the keys.

If the information isn't present, use "sorry i m not able to get right now " \
as the value.

Make your response as short as possible.



Review text: '''{backcover_review}'''
"""
response = get_completion(prompt)
print(response)


"""
input
output
task
context
example
tone

"""

story = """
The Indian Space Research Organisation (ISRO) stands as a beacon of scientific excellence and innovation, spearheading India's remarkable advancements in space exploration and technology. Established in 1969, ISRO has consistently achieved remarkable milestones, including the successful launch of satellites, lunar missions, and interplanetary explorations. With a mission to harness space technology for national development and societal benefits, ISRO has significantly contributed to telecommunication, weather forecasting, disaster management, and navigation systems. Its endeavors, such as the Mars Orbiter Mission (Mangalyaan) and the Chandrayaan missions, have garnered international acclaim, showcasing India's prowess in space exploration on a global stage.
"""


prompt = f"""
Determine five topics that are being discussed in the \
following text, which is delimited by triple backticks.

Make each item one or two words long.

Format your response as a list of items separated by commas.

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)

prompt = f"""
Translate the following English text to HInglish \
```Hi, I would like to order a allo paratha```
"""
response = get_completion(prompt)
print(response)

data_json = { "resturant_employees" :[
{"name":"Shyam", "email":"shyamjaiswal@gmail.com", "उम्र": "25"},
{"name":"Bob", "email":"bob32@gmail.com", "उम्र": "28"},
{"name":"Jai", "email":"jai87@gmail.com", "उम्र": "30"}
]}

prompt = f"""
Translate the following python dictionary from JSON  in tablualr format
table with column headers and title: {data_json}
"""
response = get_completion(prompt)
print(response)

# expanding :
sentiment = "negative"

review = f"""
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

prompt = f"""
You are a prince chatbot assistant.

Your task is to send an email reply to a valued customer.
Given the customer email delimited by ```, \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service.

Make sure to use specific details from the review.
Write in a concise and chidlish tone.
Sign the email as `AI customer agent`.

Customer review: ```{review}```
Review sentiment: {sentiment}
"""
response = get_completion(prompt)
print(response)