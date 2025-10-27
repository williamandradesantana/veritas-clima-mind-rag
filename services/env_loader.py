import os
from decouple import config


class EnvLoader:
    def __init__(self):
        self.OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)
        self.PINECONE_API_KEY = config("PINECONE_API_KEY")
        self.GOOGLE_API_KEY = config("GOOGLE_API_KEY", default=None)
        self.PINECONE_CLOUD = config("PINECONE_CLOUD", default="aws")
        self.PINECONE_REGION = config("PINECONE_REGION", default="us-east-1")

        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY or ""
        os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
        if self.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
