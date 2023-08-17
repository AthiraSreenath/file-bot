from langchain.llms import CTransformers
import os
import openai

file_path= '/Users/chochu/Downloads/Models'
model_path = os.path.join(file_path, 'llama-2-7b-chat.ggmlv3.q8_0.bin')

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model=model_path,
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})


# # Initialize OpenAI API with your API key
# openai.api_key = 'YOUR_OPENAI_API_KEY'
#
# # Define function to interact with OpenAI's GPT model
# def generate_text(prompt):
#     response = openai.Completion.create(
#         engine='text-davinci-003',
#         prompt=prompt,
#         temperature=0.7,  # You can adjust the temperature
#         max_tokens=2000,
#     )
#     generated_text = response.choices[0].text.strip()
#     return generated_text


