# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SjIRe2PaszE_slhuWMTtQRi-QrmIADfn
"""





# !pip install youtube-transcript-api
# !pip install google-cloud-speech
# !pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"

# from youtube_transcript_api import YouTubeTranscriptApi

# data=YouTubeTranscriptApi.get_transcript("1RMhlSH27BY")

# import youtube_dl
# from google.cloud import speech_v1p1beta1 as speech

# def download_youtube_video(video_url):
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#             'preferredquality': '192',
#         }],
#         'outtmpl': 'audio.wav',
#         'verbose': True  # Add verbose flag
#     }

#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([video_url])


# def transcribe_youtube_video(video_url):
#     # Fetch the YouTube video using pytube
#     # youtube = YouTube(video_url)
#     # video = youtube.streams.get_audio_only()

#     # # Download the audio content
#     audio_path = 'audio.wav'  # Specify the path to save the audio file
#     # video.download(filename='audio')

#     # Transcribe the downloaded audio
#     client = speech.SpeechClient()

#     with open(audio_path, 'rb') as audio_file:
#         content = audio_file.read()

#     audio = speech.RecognitionAudio(content=content)
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=44100,  # Adjust based on your audio file
#         language_code='en-US'     # Adjust based on your audio language
#     )

#     response = client.recognize(config=config, audio=audio)

#     transcript = ''
#     for result in response.results:
#         transcript += result.alternatives[0].transcript + ' '

#     return transcript

# # Specify the YouTube video URL
# youtube_url = 'https://www.youtube.com/watch?v=8DvywoWv6fI'  # Replace with the actual YouTube URL

# # Download the YouTube video
# download_youtube_video(youtube_url)

# # Transcribe the downloaded YouTube video
# transcript = transcribe_youtube_video(youtube_url)
# print(transcript)

# text_combined = ""
# for item in data:
#     text_combined += item['text']

# print(text_combined)

"""The AI Starts from here"""



# def construct_index():
#     max_input_size = 4096
#     num_outputs = 3000
#     chunk_size_limit = 600
#     prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_size_limit=chunk_size_limit)
#     llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-003", max_tokens=num_outputs,openai_api_key='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'))
#     # BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
#     # loader = BeautifulSoupWebReader()
#     # documents=loader.load_data(url_scraped)
#     # YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
#     # loader = YoutubeTranscriptReader()
#     # documents = loader.load_data(ytlinks=['https://www.youtube.com/watch?v=8DvywoWv6fI'])
#     # DocxReader = download_loader("DocxReader")
#     # loader = DocxReader()
#     # documents = loader.load_data(file=Path('AITube.docx'))
#     PDFReader = download_loader("PDFReader")
#     loader = PDFReader()
#     documents = loader.load_data(file=Path('Transcript.pdf'))
#     parser=SimpleNodeParser()
#     nodes=parser.get_nodes_from_documents(documents)
#     service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#     index = GPTVectorStoreIndex.from_documents(nodes, service_context=service_context)
#     index.storage_context.persist(persist_dir='index.json')
#     return index


# construct_index()

# response=ask_ai('''Imagine yourself as the greatest course instructor, give me 2 programming questions based on the programming language in context, make sure you format the question in such a way that you give a beginner level questions based one the context, followed by the answer to the problem ''')

# print(response.response)

