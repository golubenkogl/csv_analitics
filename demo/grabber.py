import os
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


class web_graber:
    def __init__(self, config, text_limit = 3000):
        self.text_limit = text_limit
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['GOOGLE_APPLICATION_CREDENTIALS']
        self.__translator = None
        if config['use_translator']:            
            from google.cloud import translate_v2
            # Initialize the client
            self.__translator = translate_v2.Client()

    def extract_html_v2(self, url):
        size = 0
        content = ""
        error = ""
        try:
            # Load html
            loader = AsyncHtmlLoader([url])
            html = loader.load()
            # Transform
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(html)
            # Result
            text = "\n".join([doc.page_content for doc in docs_transformed])
            content = text
            # Translate text
            target_language = "en"
            if self.__translator:
                content = self.__translator.translate(text, target_language=target_language)
                content = content['translatedText']
            print(f"OK")
        except Exception as e:
            print("Error:", e)
            error = e.__cause__
        return {
            "url": url,
            "size": size,
            "content": content,
            "error": error
        }
