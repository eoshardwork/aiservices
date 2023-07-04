from langchain.document_loaders import TextLoader, YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import streamlit as st

from sklearn.cluster import KMeans

import tiktoken

import numpy as np


import time

import urllib.parse

from concurrent.futures import ThreadPoolExecutor, as_completed

from pineconewapper import PineconeVectorstore


def doc_loader(file_path: str):
    """
    Load the contents of a text document from a file path into a loaded langchain Document object.

    :param file_path: The path to the text document to load.

    :return: A langchain Document object.
    """
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def token_counter(text: str):
    """
    Count the number of tokens in a string of text.

    :param text: The text to count the tokens of.

    :return: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding('cl100k_base')
    token_list = encoding.encode(text, disallowed_special=())
    tokens = len(token_list)
    return tokens


def doc_to_text(document):
    """
    Convert a langchain Document object into a string of text.

    :param document: The loaded langchain Document object to convert.

    :return: A string of text.
    """
    text = ''
    for i in document:
        text += i.page_content
    special_tokens = ['>|endoftext|', '<|fim_prefix|',
                      '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|']
    words = text.split()
    filtered_words = [word for word in words if word not in special_tokens]
    text = ' '.join(filtered_words)
    return text


def remove_special_tokens(docs):
    special_tokens = ['>|endoftext|', '<|fim_prefix|',
                      '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|>']
    for doc in docs:
        content = doc.page_content
        for special in special_tokens:
            content = content.replace(special, '')
            doc.page_content = content
    return docs


def init_embedding_db(docs, api_key):
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    doc_db = PineconeVectorstore().from_documents(
        docs=docs,
        embeddings=embeddings,
        index_name='aiservices', namespace="test"
    )
    return doc_db


def add_embedding_db(docs):
    # we use the openAI embedding model

    doc = PineconeVectorstore().add_documents(
        docs,
        index_name='aiservices', namespace="test"
    )
    return doc


def delete_db():
    PineconeVectorstore().delete_all_vectors('test')


def embed_docs_openai(docs, api_key):
    """
    Embed a list of loaded langchain Document objects into a list of vectors.

    :param docs: A list of loaded langchain Document objects to embed.

    :param api_key: The OpenAI API key to use for embedding.

    :return: A list of vectors.
    """
    docs = remove_special_tokens(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    return vectors


def create_summarize_chain(prompt_list):
    """
    Create a langchain summarize chain from a list of prompts.

    :param prompt_list: A list containing the template, input variables, and llm to use for the chain.

    :return: A langchain summarize chain.
    """
    template = PromptTemplate(
        template=prompt_list[0], input_variables=([prompt_list[1]]))
    chain = load_summarize_chain(
        llm=prompt_list[2], chain_type='stuff', prompt=template)
    return chain


def parallelize_summaries(summary_docs, initial_chain, progress_bar, max_workers=4):
    """
    Summarize a list of loaded langchain Document objects using multiple langchain summarize chains in parallel.

    :param summary_docs: A list of loaded langchain Document objects to summarize.

    :param initial_chain: A langchain summarize chain to use for summarization.

    :param progress_bar: A streamlit progress bar to display the progress of the summarization.

    :param max_workers: The maximum number of workers to use for parallelization.

    :return: A list of summaries.
    """
    doc_summaries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(
            initial_chain.run, [doc]): doc.page_content for doc in summary_docs}

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]

            try:
                summary = future.result()

            except Exception as exc:
                print(f'{doc} generated an exception: {exc}')

            else:
                doc_summaries.append(summary)
                num = (len(doc_summaries)) / (len(summary_docs) + 1)
                # Remove this line and all references to it if you are not using Streamlit.
                progress_bar.progress(num)
    return doc_summaries


def create_summary_from_docs(summary_docs, initial_chain, final_sum_list, api_key, use_gpt_4):
    """
    Summarize a list of loaded langchain Document objects using multiple langchain summarize chains.

    :param summary_docs: A list of loaded langchain Document objects to summarize.

    :param initial_chain: The initial langchain summarize chain to use.

    :param final_sum_list: A list containing the template, input variables, and llm to use for the final chain.

    :param api_key: The OpenAI API key to use for summarization.

    :param use_gpt_4: Whether to use GPT-4 or GPT-3.5-turbo for summarization.

    :return: A string containing the summary.
    """

    # Create a progress bar to show the progress of summarization.
    progress = st.progress(0)
    # Remove this line and all references to it if you are not using Streamlit.

    doc_summaries = parallelize_summaries(
        summary_docs, initial_chain, progress_bar=progress)

    summaries = '\n'.join(doc_summaries)
    count = token_counter(summaries)

    if use_gpt_4:
        max_tokens = 7500 - int(count)
        model = 'gpt-4'

    else:
        max_tokens = 3800 - int(count)
        model = 'gpt-3.5-turbo'

    final_sum_list[2] = ChatOpenAI(
        openai_api_key=api_key, temperature=0, max_tokens=max_tokens, model_name=model)
    final_sum_chain = create_summarize_chain(final_sum_list)

    summaries = Document(page_content=summaries)
    final_summary = final_sum_chain.run([summaries])

    # Remove this line and all references to it if you are not using Streamlit.
    progress.progress(1.0)
    # Remove this line and all references to it if you are not using Streamlit.
    time.sleep(0.4)
    # Remove this line and all references to it if you are not using Streamlit.
    progress.empty()

    return final_summary


def split_by_tokens(doc, chunk_size=2000):
    """
    Split a  langchain Document object into a list of smaller langchain Document objects.

    :param doc: The langchain Document object to split.

    :param minimum_tokens: The minimum number of tokens to use for splitting.

    :param maximum_tokens: The maximum number of tokens to use for splitting.

    :return: A list of langchain Document objects.
    """
    text_doc = doc_to_text(doc)
    overlap = int(chunk_size/10)

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_doc = splitter.create_documents([text_doc])
    return split_doc


def extract_summary_docs(langchain_document, api_key):
    """
    Automatically convert a single langchain Document object into a list of smaller langchain Document objects that represent each cluster.

    :param langchain_document: The langchain Document object to summarize.

    :return: A list of langchain Document objects.
    """
    split_document = split_by_tokens(langchain_document)

    summary_docs = init_embedding_db(
        split_document, api_key).similarity_search(" ")

    return summary_docs


def doc_to_final_summary(langchain_document, initial_prompt_list, final_prompt_list, api_key, use_gpt_4):
    """
    Automatically summarize a single langchain Document object using multiple langchain summarize chains.

    :param langchain_document: The langchain Document object to summarize.

    :param initial_prompt_list: The initial langchain summarize chain to use.

    :param final_prompt_list: A list containing the template, input variables, and llm to use for the final chain.

    :param api_key: The OpenAI API key to use for summarization.

    :param use_gpt_4: Whether to use GPT-4 or GPT-3.5-turbo for summarization.


    :return: A string containing the summary.
    """
    initial_prompt_list = create_summarize_chain(initial_prompt_list)
    summary_docs = extract_summary_docs(
        langchain_document, api_key)

    output = create_summary_from_docs(
        summary_docs, initial_prompt_list, final_prompt_list, api_key, use_gpt_4)
    return output


def summary_prompt_creator(prompt, input_var, llm):
    """
    Create a list containing the template, input variables, and llm to use for a langchain summarize chain.

    :param prompt: The template to use for the chain.

    :param input_var: The input variables to use for the chain.

    :param llm: The llm to use for the chain.

    :return: A list containing the template, input variables, and llm to use for the chain.
    """
    prompt_list = [prompt, input_var, llm]
    return prompt_list


def extract_video_id(video_url):
    """
    Extract the YouTube video ID from a YouTube video URL.

    :param video_url: The URL of the YouTube video.

    :return: The ID of the YouTube video.
    """
    parsed_url = urllib.parse.urlparse(video_url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]

    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):

        if parsed_url.path == '/watch':
            p = urllib.parse.parse_qs(parsed_url.query)
            return p.get('v', [None])[0]

        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[1]

        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/v/')[1]

    return None


def transcript_loader(video_url):
    """
    Load the transcript of a YouTube video into a loaded langchain Document object.

    :param video_url: The URL of the YouTube video to load the transcript of.

    :return: A loaded langchain Document object.
    """
    transcript = YoutubeLoader(video_id=extract_video_id(video_url))
    loaded = transcript.load()
    return loaded
