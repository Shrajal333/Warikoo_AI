import faiss
import numpy as np
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain.schema import Document
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fetch_youtube_video_titles(channel_handle, youtube_api_key):

    youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    response = youtube.search().list(
        part = "snippet",
        q = channel_handle,
        type = "channel",
        maxResults = 1
    ).execute()

    channel_id = response["items"][0]["id"]["channelId"]

    response = youtube.channels().list(
        part = 'contentDetails',
        id = channel_id
    ).execute()

    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    all_titles = []
    next_page_token = None

    while True:
        playlist_response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        all_titles.extend(
            item['snippet']['title'] for item in playlist_response['items']
        )

        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break

    video_titles = [item for item in all_titles if not item.lower().endswith("shorts")]
    return video_titles

def search_vector_store(query, video_titles, model_name="all-MiniLM-L6-v2", top_k=3):

    model = SentenceTransformer(model_name)
    embeddings = model.encode(video_titles)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    metadata = {i: video_titles[i] for i in range(len(video_titles))}

    query_vector = model.encode([query])
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

    _, indices = index.search(query_vector, top_k)
    return [metadata[idx] for _, idx in enumerate(indices[0])]

def summarize_youtube_video(youtube_api_key, search_query, groq_api_key):

    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    request = youtube.search().list(
        part="snippet",
        q=search_query,
        type="video",
        maxResults=1
    )

    response = request.execute()
    video_id = response["items"][0]["id"]["videoId"]

    transcript_hindi = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['hi','en']).translate('en').fetch()
    transcript_english = transcript_hindi
    final_transcript = " ".join([transcript['text'] for transcript in transcript_english])

    documents = [Document(page_content=final_transcript)]
    final_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_documents(documents)

    template = """
                    Please summarize the following YouTube transcript segment:
                    Speech: {text}
                    Provide a detailed summary, highlighting key ideas and important points. 
                    Summary:
               """
    prompt = PromptTemplate(input_variables=['text'], template=template)

    model = ChatGroq(model="Gemma-7b-It", api_key=groq_api_key)
    summary_chain = load_summarize_chain(
        model, 
        chain_type="stuff", 
        prompt=prompt, 
        verbose=False)

    summary = summary_chain.invoke({'input_documents': final_documents})
    return video_id, summary['output_text']