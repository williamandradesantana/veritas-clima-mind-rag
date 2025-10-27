from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFLoader:
    def __init__(self, path):
        self.path = path

    def extract_text_chunks(self, chunk_size=1000, chunk_overlap=100):
        reader = PdfReader(self.path)
        text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)
