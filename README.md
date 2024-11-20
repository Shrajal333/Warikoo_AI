# Warikoo_AI
The repo contains the build for Warikoo AI, which is your ultimate chatbot if you have any questions related to Ankur Warikoo's content. The chat application provides with top 3 video results and the text summarization of these videos so that you can navigate the content with ease. The following steps were followed in building up the chatbot application.

Step 1: Data Collection
Ankur Warikoo's Youtube channel is scraped to collect the titles for over 1000 videos to build a solid title databse.

Step 2: Text to Vectors
FAISS and Langchain are leveraged to create vector embeddings, which transforms the video title for easy search and retrieval through cosine similarity.

Step 3: Storing the Data
The vector database is stored locally which can be called quickly for ease of comparing with the user input for top 3 video titles

Step 4: Building the Chatbot
The system retrives the transcript for the 3 videos and performs a text summarization using Gemma-7b model using stuff document chains.

![image](https://github.com/user-attachments/assets/2f7a6fbb-1967-4fc2-a5ba-2635eccf8306)
