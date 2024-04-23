import os
from loguru import logger

# 遍历目录获取md和txt文件
def get_files(dir_path):
    file_list=[]
    for filepath,dirnames,filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_list.append(os.path.join(filepath,filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath,filename))
    logger.info("get file list done")
    return file_list

# 使用LangChain的加载工具 把文件列表转化为纯文本列表
from tqdm import tqdm
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader


def get_text(dir_path):
    file_list=get_files(dir_path)
    docs =[]
    for each_file in tqdm(file_list):
        file_type=each_file.split('.')[-1]
        if file_type == 'txt':
            loader = UnstructuredFileLoader(each_file)
        elif file_type == 'md':
            loader = UnstructuredMarkdownLoader(each_file)
        else: continue
        docs.extend(loader.load())
    return docs

# 使用LangChain的文本分块工具，这里是字符串递归分割器 分块大小为500 块重叠长度为150
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs=text_splitter.split_documents(docs)
    return split_docs

# sentence transformer embedding
from sentence_transformers import SentenceTransformer
def get_embedding_model(embedding_model_path):
    return SentenceTransformer(embedding_model_path,device='cuda')

from langchain.vectorstores.chroma import Chroma

def persist_vectordb(persist_dir, split_docs, embedding_model):
    vectordb=Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return True

def make_vectordb(tar_dir, embedding_model_path, persist_dir):
    docs=[]
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))
    logger.info("get docs done")
    split_docs=split_docs(docs)
    logger.info("split done")
    embedding_model=get_embedding_model(embedding_model_path)
    persist_vectordb(persist_dir, split_docs, embedding_model)
