from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from googletrans import Translator
import time

class Graber:
    def __init__(self, urls):
        self.urls = urls
        self.translator = Translator()

    def extract_main_content(self):
        # Load HTML
        loader = AsyncChromiumLoader(self.urls)
        docs = loader.load()

        # Transform using BeautifulSoup
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["div", "p"])

        # Extracted content
        main_contents = [doc.page_content for doc in docs_transformed]

        # Transform HTML to text
        html2text = Html2TextTransformer()
        text_contents = html2text.transform_documents(docs)

        # Translate content
        translated_contents = []
        for content in text_contents:
            translated = self.translate_with_retry(content)
            translated_contents.append(translated)

        return main_contents, translated_contents

    def translate_with_retry(self, text, target_language='en', max_retries=3):
        for _ in range(max_retries):
            try:
                translation = self.translator.translate(text, dest=target_language)
                return translation.text
            except Exception as e:
                print(f"Translation failed. Retrying... Error: {e}")
                time.sleep(1)  # Adding a delay before retrying
        return "Translation failed"

# Test the class
if __name__ == "__main__":
    urls = ["http://www.abo-wind.de", "https://www.hannovermesse.de/exhibitor/abo-wind/N1545807"]
    graber = Graber(urls)
    main_contents, translated_contents = graber.extract_main_content()

    # Print extracted content
    print("Main Content (HTML):")
    for content in main_contents:
        print(content[:500])  # Print first 500 characters

    print("==================")

    print("Translated Content:")
    for content in translated_contents:
        print(content[:500])  # Print first 500 characters