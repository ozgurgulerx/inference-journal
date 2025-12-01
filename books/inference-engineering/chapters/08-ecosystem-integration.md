# Chapter 8: Ecosystem Integration

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. vLLM integrates seamlessly with the broader ML ecosystem.

---

## In This Chapter

- [Overview](#overview)
- [OpenAI SDK](#openai-sdk)
- [Hugging Face Hub](#hugging-face-hub)
- [LangChain](#langchain)
- [LlamaIndex](#llamaindex)
- [Other Integrations](#other-integrations)
- [API Compatibility Matrix](#api-compatibility-matrix)

---

## Overview

vLLM provides an OpenAI-compatible API, making it easy to integrate with existing tools and workflows. This chapter covers integration with:
- OpenAI SDK
- Hugging Face Hub
- LangChain and LlamaIndex
- Other ecosystem tools

---

## OpenAI SDK

vLLM's API is fully compatible with the OpenAI Python SDK.

### Basic Setup

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't require auth by default
)
```

### Chat Completions

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Explain AI briefly."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Completions (Legacy)

```python
response = client.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.8
)

print(response.choices[0].text)
```

### Embeddings

vLLM can also serve embedding models:

```bash
vllm serve BAAI/bge-base-en-v1.5 --port 8001
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")

response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=["What is AI?", "Machine learning explained"]
)

for embedding in response.data:
    print(f"Embedding dim: {len(embedding.embedding)}")
```

---

## Hugging Face Hub

### Direct Model Loading

vLLM loads models directly from Hugging Face:

```bash
# Public models
vllm serve meta-llama/Meta-Llama-3-8B-Instruct

# Gated models (requires token)
export HUGGING_FACE_HUB_TOKEN=hf_...
vllm serve meta-llama/Meta-Llama-3-70B-Instruct
```

### Using Local Models

```bash
# Serve from local path
vllm serve ./path/to/local/model

# Or download first
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./llama3
vllm serve ./llama3
```

### Custom Chat Templates

Hugging Face models include chat templates:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]

# Apply template
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
```

vLLM uses these templates automatically when serving chat models.

### Quantized Models from Hub

```bash
# GPTQ models
vllm serve TheBloke/Llama-2-7B-GPTQ --quantization gptq

# AWQ models
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq

# GGUF (experimental)
vllm serve ./model.gguf
```

---

## LangChain

LangChain provides abstractions for building LLM applications.

### Basic Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="EMPTY",
    temperature=0.7
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

### With Streaming

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="EMPTY",
    streaming=True
)

for chunk in llm.stream("Explain quantum computing."):
    print(chunk.content, end="", flush=True)
```

### Chains and Agents

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="EMPTY"
)

prompt = ChatPromptTemplate.from_template(
    "Summarize this topic in 3 bullet points: {topic}"
)

chain = {"topic": RunnablePassthrough()} | prompt | llm

response = chain.invoke("Machine Learning")
print(response.content)
```

### RAG with vLLM

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Use vLLM for chat
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="EMPTY"
)

# Use vLLM for embeddings (if serving embedding model)
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    openai_api_base="http://localhost:8001/v1",
    openai_api_key="EMPTY"
)

# Build vector store
texts = ["Document 1 content...", "Document 2 content..."]
vectorstore = FAISS.from_texts(texts, embeddings)

# RAG chain
retriever = vectorstore.as_retriever()
# ... build full RAG chain
```

---

## LlamaIndex

LlamaIndex (GPT Index) also works seamlessly with vLLM.

### Basic Setup

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    is_chat_model=True
)

response = llm.complete("What is deep learning?")
print(response.text)
```

### With Index

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure LLM
llm = OpenAILike(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is in these documents?")
print(response)
```

---

## Other Integrations

### FastAPI Wrapper

```python
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

class Query(BaseModel):
    message: str
    max_tokens: int = 256

@app.post("/chat")
async def chat(query: Query):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": query.message}],
        max_tokens=query.max_tokens
    )
    return {"response": response.choices[0].message.content}
```

### Gradio Interface

```python
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def chat(message, history):
    messages = [{"role": "user", "content": m[0]} for m in history]
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield full_response

demo = gr.ChatInterface(chat)
demo.launch()
```

### Streamlit App

```python
import streamlit as st
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

st.title("Chat with Llama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=st.session_state.messages,
            stream=True
        )
        response = st.write_stream(
            chunk.choices[0].delta.content or "" for chunk in stream
        )
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## API Compatibility Matrix

| Feature | OpenAI API | vLLM Support |
|---------|------------|--------------|
| Chat Completions | ✅ | ✅ |
| Completions | ✅ | ✅ |
| Embeddings | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Function Calling | ✅ | ✅ |
| Vision | ✅ | ✅ (with VLMs) |
| Audio | ✅ | ❌ |
| Fine-tuning API | ✅ | ❌ |

---

<p align="center">
  <a href="07-scaling.md">← Previous: Scaling</a> | <a href="../README.md">Table of Contents</a> | <a href="09-observability.md">Next: Observability →</a>
</p>
