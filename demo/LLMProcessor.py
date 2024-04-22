from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def extract_by_param(param:str, rec):
    if param.startswith("CSV:"):
        param_name_in_csv = param.split(":")[-1]
        value = rec[param_name_in_csv]
        return value
    elif param.startswith("GRABER:CSV:"):
        param_name_in_csv = param.split(":")[-1]
        # due to deps issue with httpx (open api and translate use different version) can't be used
        # graber = Graber(rec[param_name_in_csv])
        # main_contents, translated_contents = graber.extract_main_content()
        # as fall back used standard WebBaseLoader no traslation :(
        loader = WebBaseLoader(rec[param_name_in_csv])
        docs = loader.load()
        print(docs)
        return docs



def ask(prompt: PromptTemplate, data: hash, api_key: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    result = chain.invoke(data)    
    return result

def process(rec, prompts: dict, api_key: str):
    rec = rec.asdict()
    results = {}
    values = []
    for prompt_name in prompts:
        input_data = {}
        for param_name in prompts[prompt_name]:
            if param_name not in ("PROMPT", "OUT"):
                param_source = prompts[prompt_name][param_name]
                value = extract_by_param(param_source, rec)
                input_data[param_name] = value
                input_data.update(results)
        prompt = PromptTemplate(
            template=prompts[prompt_name]['PROMPT'],
            input_variables=list(input_data.keys())
        )
        answer = ask(prompt, input_data, api_key)
        results[prompt_name] = answer
        values.append(answer)
    return values