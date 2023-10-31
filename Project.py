#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
import requests
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# Title of your Streamlit app
st.title("News Summarization")

# Sidebar title
st.sidebar.title("News Article URL")

# Create a text input field in the sidebar for the user to enter the URL
url = st.sidebar.text_input("Enter the URL of the news article")

# Create a button to process the URL
process_url_clicked = st.sidebar.button("Process URL")

# Main content placeholder
main_placeholder = st.empty()

if process_url_clicked:
    # Check if a URL has been provided
    if url:
            loader = UnstructuredURLLoader(urls=url)

            data = loader.load()

        
            main_placeholder.text("Data Loading...Started...✅✅✅")
            tokenizer = AutoTokenizer.from_pretrained('t5-base')
            model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

            inputs = tokenizer.encode(str(data),
            return_tensors='pt',
            max_length=1000,
            truncation=True)
            main_placeholder.text("Preparing Summary...✅✅✅")

            summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)

            summary = tokenizer.decode(summary_ids[0])

            st.header('Summary')
            st.write(summary)






