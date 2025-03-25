## Usage

Open [Google Colab](https://colab.research.google.com) and click `GitHub` in the menu on the left side
Enter `https://github.com/rignitc/hAIck.py_2025/blob/main/hAIck.ipynb` in the link textbox and click Enter
> [!NOTE]  
> Be sure to connect to T4 GPU!

Click `Runtime` and click `Run all`
> [!TIP]
> If you want to include general search using Google API then:
>
> Add a Google Search API and Engine key in `rag-main.py`
> ```py
> def web_search(query, num_results=3):
>    search_api_key = ""
>    search_engine_id = ""
>    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={search_api_key}&cx={search_engine_id}"
>    response = requests.get(url)
> ..>```
> 
> This ensures that it can query the API with your own key and return answers as required
## Explanation

### hAIck.ipynb

The script begins by importing the `os` module, which is used for interacting with the operating system, such as changing directories or running shell commands. It then executes a shell command to clone a GitHub repository using `git clone https://github.com/rignitc/hAIck.py_2025.git`. This command fetches the repository from the remote location and stores it in the current working directory. Once the repository is successfully cloned, the script changes the working directory to `hAIck.py_2025` using `os.chdir('hAIck.py_2025')`, ensuring that subsequent commands operate within the cloned repository. To verify the contents of the directory, the script runs `!ls`, which lists all files and folders present in the working directory

After setting up the repository, the script proceeds with the installation of various dependencies required for running the project. It uses `pip install` to install several Python libraries that are essential for machine learning and large language models. The list of installed packages includes `transformers`, which is a library from Hugging Face for working with pre-trained AI models such as GPT and LLaMA. `datasets` is another Hugging Face library that provides easy access to a wide range of machine learning datasets. `faiss-gpu` and `faiss-cpu` are installed to enable efficient similarity search and clustering, particularly useful for applications like document retrieval or embeddings-based searches. The `langchain` package, along with related extensions such as `langchain_community`, `langchain-openai`, and `langchain-ollama`, is included to facilitate building LLM-powered applications by managing interactions with AI models, memory, and prompt engineering. Additionally, `scikit-learn` is installed, which provides various machine learning utilities such as vectorization and clustering, and `sentence-transformers` is used for embedding-based NLP tasks. The script also installs `pyngrok`, which is used to expose local servers to the internet, allowing remote access to locally hosted applications. To ensure a smooth front-end experience, `streamlit` is installed for creating interactive web-based applications, and the script suppresses unnecessary output by using the `-q` flag. Lastly, `apt install lshw` is executed to install `lshw`, a Linux system utility that provides hardware information, which may be useful for checking system capabilities before running computationally intensive AI models

Once all dependencies are installed, the script proceeds to set up `Ollama`, a runtime environment that allows serving AI models locally. This is done using a shell command that downloads and installs `Ollama` via `curl -fsSL https://ollama.com/install.sh | sh`. This command fetches and executes the installation script provided by the official Ollama source. After the installation is complete, the script starts the Ollama server by running `os.system("ollama serve &")`, which launches the service in the background, allowing other operations to proceed while the server initializes. Once the server is running, the script executes `!ollama pull llama3.1`, which downloads the `LLaMA 3.1` model. This model is a large language model that serves as the core of the AI system, enabling advanced text generation, retrieval-augmented generation (RAG), and other natural language processing tasks

With the AI model and backend services running, the script then configures `ngrok` to create a public-facing tunnel for the locally hosted application. First, it imports `pyngrok.ngrok`, which is a Python interface for the `ngrok` service. To ensure no conflicting processes are running, it executes `!pkill ngrok` and `!pkill streamlit`, which terminate any existing instances of `ngrok` and `streamlit`. Then, `ngrok.set_auth_token("2r8P6Ni1zQrVsBIBWXcxYgM86Vp_3v6MrwKNT2J52Y2m8cKmr")` is used to authenticate `ngrok` using a personal authentication token, enabling the user to create secure tunnels. The script then launches a Streamlit web application by running `!nohup streamlit run rag-main.py --server.port 5011 &`. The `nohup` command ensures that the process continues running even if the terminal session is closed, and the `&` symbol runs the command in the background, allowing the script to proceed with other tasks without waiting for the Streamlit app to fully initialize. The Streamlit application itself (`rag-main.py`) likely serves as an interactive interface for querying the AI model, possibly implementing a retrieval-augmented generation (RAG) pipeline that leverages embeddings and similarity search via `faiss`

Once the Streamlit server is running locally on port `5011`, the script establishes an `ngrok` tunnel to make it publicly accessible. This is done using `ngrok_tunnel = ngrok.connect(addr='5011', proto='http', bind_tls=True)`, which creates a secure HTTP tunnel for port `5011`. Finally, the script prints the public URL of the `ngrok` tunnel with `print(' * Tunnel URL:', ngrok_tunnel.public_url)`, allowing users to access the Streamlit application remotely through the generated link

There are also some commented-out sections in the script that serve additional functionality but are not executed by default. One section includes `# !pkill ngrok` and `# !pkill streamlit`, which, if uncommented, would manually terminate the running `ngrok` and `streamlit` processes. Another commented-out portion contains `# os.chdir('..')` and `# !ls`, which would change the working directory to its parent and list the files if needed. Further down, a section intended for Google Colab usage is present but commented out. This section starts with `# from google.colab import files`, which imports the `files` module for handling file downloads in Colab. It then includes `# !rm -rf hAIck.py_2025/nohup.out hAIck.py_2025/.git/*`, which removes log files and Git-related metadata to clean up the repository before archiving. The command `# !zip -r hAIck.py_2025.zip hAIck.py_2025/` compresses the entire project folder into a `.zip` file, and `# files.download('hAIck.py_2025.zip')` provides a direct link for downloading the zipped project. This functionality is useful for backing up the project or transferring it to another system

Overall, the script automates the process of setting up an AI-powered application by cloning the necessary repository, installing dependencies, configuring a local AI model with `Ollama`, launching a Streamlit web application, and exposing it to the internet using `ngrok`. It ensures that all required services are running smoothly and provides mechanisms for managing background processes, cleaning up unnecessary files, and even packaging the project for easy sharing. The integration of `faiss` for similarity search, `langchain` for AI application development, and `sentence-transformers` for embedding-based retrieval suggests that this setup is designed for an intelligent, interactive NLP system, potentially focused on retrieval-augmented generation or chatbot-like functionalities. If you have any additional files to analyze, feel free to provide them, and I can break down their functionality as well

---

### rag-main.py

#### Importing Required Libraries

The script begins by importing essential Python libraries:
- `os`: Handles file system operations- `numpy`: Used for numerical operations, particularly for handling cosine similarity calculations- `requests`: Fetches data from the web using HTTP requests- `streamlit`: Provides an interface for web-based interaction with the AI model- `langchain.document_loaders.TextLoader`: Loads text documents into LangChain’s pipeline- `langchain.embeddings.base.Embeddings`: Provides an abstraction for embedding models- `langchain.prompts.PromptTemplate`: Defines templates for generating AI responses- `langchain.text_splitter.RecursiveCharacterTextSplitter`: Splits documents into smaller chunks for better processing- `langchain.vectorstores.FAISS`: Manages a vector-based search index using FAISS (Facebook AI Similarity Search)- `langchain_core.output_parsers.StrOutputParser`: Extracts raw string output from LangChain's execution pipeline- `langchain_ollama.ChatOllama`: Provides an interface to the Ollama-based LLM- `sentence_transformers.SentenceTransformer`: Loads a pre-trained transformer model for embedding generation- `sklearn.metrics.pairwise.cosine_similarity`: Computes cosine similarity between two vectors


#### Web Search Function

The `web_search` function uses Google’s Custom Search API to fetch search results based on a user query. 

- It constructs a URL for the Google Search API using a provided API key and search engine ID- It sends an HTTP GET request to fetch search results- If the request is successful (HTTP 200), it extracts and returns the top `num_results` search results- If the request fails, it returns an empty list


#### Response Validation Function

The `validate_response` function checks whether the AI’s response aligns with the retrieved documents
- It converts the retrieved documents into embeddings using `embedding_model.embed_documents()`, which transforms textual data into dense numerical representations- It converts the AI-generated answer into an embedding using `embedding_model.embed_query()`- It calculates the cosine similarity between the answer embedding and document embeddings. The cosine similarity formula is:

  $$
  \text{similarity} = \frac{A \cdot B}{||A|| ||B||}
 $$

  where:
  - $$A$$ and $$B$$ are embedding vectors,
  - $$\cdot$$ denotes the dot product,
  - $$||A||$$ and $$||B||$$ are the magnitudes (norms) of the vectors
- The function averages these similarity scores and returns `True` if the similarity is above `0.5`, otherwise `False`

#### Hallucination Percentage Calculation

The `calculate_hallucination_percentage` function quantifies how much the AI-generated answer deviates from the source documents
- If no source documents are available, it returns 100% hallucination- It embeds the source documents and AI answer- It calculates cosine similarity between the AI-generated response and source embeddings- It converts the similarity score into a hallucination percentage:

$$
  \text{hallucination-percentage} = (1 - \text{similarity}) \times 100
 $$

  This means that if the similarity is high (close to 1), hallucination is low, and vice versa


#### Embedding Model

The `HuggingFaceEmbeddings` class wraps `SentenceTransformer` for text embeddings
- It loads a pre-trained transformer model from `sentence-transformers/all-MiniLM-L6-v2`, which converts text into 384-dimensional dense vector representations- The `embed_documents` function encodes multiple text inputs into embeddings- The `embed_query` function encodes a single text input and returns the first vector
Under the hood:
- The transformer model tokenizes the input using WordPiece tokenization- It passes tokens through multiple self-attention layers- The final embedding is computed using mean pooling over all token representations

#### Retrieval-Augmented Generation (RAG) Application

The `RAGApplication` class integrates retrieval-based and generative AI methods
##### Initialization
- The class takes a `retriever`, `llm` (large language model), and `embedding_model`

##### Execution Flow (`run` method)

1. **Retrieve Relevant Documents**: 
   - Queries the retriever for relevant documents based on vector similarity
   -    - If no documents are retrieved, it falls back to a web search
2. **Handle Web Search Fallback**:
   - If no documents exist, it queries Google Search
   - Extracts titles and passes them into a prompt
   - Uses the `PromptTemplate` to structure an LLM query
   - Generates an answer and calculates the hallucination percentage
3. **Generate Answer from Retrieved Documents**:
   - If documents are retrieved, they are formatted into a prompt
   - Uses `PromptTemplate` to structure the query
   - Invokes the LLM to generate a response
4. **Validate the Response**:
   - Uses `validate_response` to ensure answer accuracy
   - If the answer is invalid, it reattempts using web search
5. **Compute Hallucination Percentage**:
   - Uses `calculate_hallucination_percentage` to assess response credibility


#### Streamlit Web Application

The Streamlit application is responsible for user interaction
##### Page Configuration
- `st.set_page_config()` defines the page title, icon, and layout- `st.title()` sets the main title
##### Chat Session Management
- Uses `st.session_state` to maintain chat history across user interactions- Initializes a dictionary to store chat sessions
##### Sidebar for Chat Selection
- Lists all chat sessions- Allows users to start a new chat or clear the current chat
##### Display Chat History
- Iterates over previous interactions and displays them in Markdown format
##### User Input and Answer Generation
- `st.text_area()` collects user queries- If "Submit" is clicked:
  1. Loads document files (`combined_text.txt`)
  2. Splits them into smaller chunks using `RecursiveCharacterTextSplitter` (chunk size = 250)
  3. Creates embeddings using `HuggingFaceEmbeddings`
  4. Initializes FAISS vector store:
     - If the index file exists, it loads it
     - Otherwise, it creates a new FAISS index from document embeddings
  5. Initializes a retriever (`vectorstore.as_retriever(k=4)`) to fetch the top 4 similar documents
  6. Creates a `ChatOllama` instance (`llama3.1` model, temperature 0 for deterministic responses)
  7. Runs the `RAGApplication` with the user query
  8. Displays the generated answer
  9. Shows hallucination percentage
  10. Lists sources


#### Core Concepts: Tokenization, Vectorization, and Embeddings

##### Tokenization
- Tokenization is the process of breaking text into smaller units (tokens)- The transformer model uses **subword tokenization** (WordPiece/BPE)- Example:
  - Input: `"Hello world!"`
  - Tokenized: `["hello", "world", "!"]`

##### Vectorization
- Text tokens are mapped to numerical vectors (word embeddings)- Each token is assigned a high-dimensional vector (384 dimensions in MiniLM)- Example vector representation for "hello":
- `[0.21, -0.34, 0.55, ...]`

##### Embeddings
- Transformers generate contextual embeddings based on input context
- Example:
  - "bank" in **"river bank"** differs from **"money bank"**- Embeddings capture these nuances

#### FAISS: Efficient Similarity Search

FAISS (Facebook AI Similarity Search) is an optimized library for fast nearest-neighbor search
##### FAISS Under the Hood
1. **Indexing**
   - Each document embedding is stored in a vector index
   - Uses clustering to optimize search speed
2. **Retrieval**
   - Queries are embedded using the same model
   - FAISS computes cosine similarity with indexed vectors
   - Returns the top-k most similar documents

#### Final Execution Flow

1. User enters a query
2. Vectorstore retrieves relevant documents using FAISS
3. If no documents are found, Google Search provides fallback data
4. Retrieved content is passed into an LLM for answer generation5. Response is validated against the sources6. Hallucination percentage is computed7. Answer and sources are displayed

This system combines retrieval-based AI (searching documents) and generative AI (LLM responses), ensuring accurate, contextual, and explainable answers
