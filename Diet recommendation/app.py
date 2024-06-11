from flask import Flask, render_template, request
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import re

import os
import re
os.environ['OPENAI_API_KEY'] = 'nanannan' # your openai key





app = Flask(__name__)

llm_resto = OpenAI( temperature=0.6)
prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 6 restaurants names, 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}."
             "Include only the names, not descriptions."
)




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == "POST":
        age = request.form['age']
        gender = request.form['gender']
        weight = request.form['weight']
        height = request.form['height']
        veg_or_noveg = request.form['veg_or_nonveg']
        disease = request.form['disease']
        region = request.form['region']
        allergics = request.form['allergics']
        foodtype = request.form['foodtype']

        chain_resto = LLMChain(llm=llm_resto, prompt=prompt_template_resto)
        input_data = {'age': age,
                              'gender': gender,
                              'weight': weight,
                              'height': height,
                              'veg_or_nonveg': veg_or_noveg,
                              'disease': disease,
                              'region': region,
                              'allergics': allergics,
                              'foodtype': foodtype}
        results = chain_resto.run(input_data)

        # Ensure results is a string
        if isinstance(results, dict) and 'text' in results:
            results = results['text']
        elif not isinstance(results, str):
            raise TypeError("Expected the result to be a string or dictionary containing 'text' key.")
          
        # Extracting the different recommendations using regular expressions
        restaurant_names = re.findall(r'Restaurants:(.*?)(?:Breakfast:|Dinner:|Workouts:|$)', results, re.DOTALL)
        breakfast_names = re.findall(r'Breakfast:(.*?)(?:Dinner:|Workouts:|$)', results, re.DOTALL)
        dinner_names = re.findall(r'Dinner:(.*?)(?:Workouts:|$)', results, re.DOTALL)
        workout_names = re.findall(r'Workouts:(.*?)$', results, re.DOTALL)

        # Cleaning up the extracted lists
        restaurant_names = [name.strip() for name in restaurant_names[0].strip().split('\n') if name.strip()] if restaurant_names else []
        breakfast_names = [name.strip() for name in breakfast_names[0].strip().split('\n') if name.strip()] if breakfast_names else []
        dinner_names = [name.strip() for name in dinner_names[0].strip().split('\n') if name.strip()] if dinner_names else []
        workout_names = [name.strip() for name in workout_names[0].strip().split('\n') if name.strip()] if workout_names else []

        # Customize the names format
        restaurant_names = [f"{i+1}. {name}" for i, name in enumerate(restaurant_names)]
        breakfast_names = [f"{i+1}. {name}" for i, name in enumerate(breakfast_names)]
        dinner_names = [f"{i+1}. {name}" for i, name in enumerate(dinner_names)]
        workout_names = [f"{i+1}. {name}" for i, name in enumerate(workout_names)]


        return render_template('result.html', restaurant_names=restaurant_names,breakfast_names=breakfast_names,dinner_names=dinner_names,workout_names=workout_names)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
