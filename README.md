# Hospital LLM RAG

This project is a hospital management system developed as part of [Real Python](https://realpython.com/build-llm-rag-chatbot-with-langchain/) tutorial about LLM and RAG. It aims to offer a better visibility of data about hospitals, for patients and employees of the hospitals.
The main difference with the tutorial is that I adapted the code and worked with Gemini-1.5-pro as the LLM, and deployed this app with google cloud run (for FastAPI) and Streamlit Cloud 

## Features

- Ask about reviews of the hospital (example : What did the patients say about food in the hospitals?)
- Ask about information about doctors, patients, hospitals (number of room, number of visits in hospital...)
- Get the waiting time at a defined hospital
- Get the most available hospital (get the minimum waiting time at the hospital)

## Test the chatbot

You can go to [this Streamlit website](https://hospital-chatbot.streamlit.app/) to chat with the chatbot.
