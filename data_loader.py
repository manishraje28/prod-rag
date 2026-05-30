from groq import Groq
from llama_index_readers import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client  = Groq()
