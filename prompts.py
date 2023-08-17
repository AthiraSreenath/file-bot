qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


# For the question {question}
# Create an answer from the following relevant text passages
# [text]
# Question: query text
# relevant text: [text1, text2....]