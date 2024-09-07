from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# Initialize Groq Langchain chat object
llm = ChatGroq(model_name="llama-3.1-70b-versatile")

# Prompt template
template =""" 
      You work in the field of writing and correct spelling and grammar errors. You correct the sentence if it contains errors and write the correct sentence.
      Write only the correct sentence and nothing else.
    
      For example:
      Sentence: She enoys raeding books in the evining.
      Correction: She enjoys reading books in the evening.
      Sentence: The qick broown fox jumps ovr the lazi dog.
      Correction: The quick brown fox jumps over the lazy dog.
    
      Your turn:
      Sentence: {sentence}
      Correction:
      """
    
prompt = ChatPromptTemplate.from_template(template)


# User input 
sentence = "lok at th ski and its stars i wold lik to have this mael"

# Format the prompt with the user input
formatted_prompt = prompt.format(sentence=sentence)
response = llm.invoke(formatted_prompt).content

# Print the response
print(response)

