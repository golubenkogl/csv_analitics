from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from grabber import web_graber

def extract_by_param(param:str, rec, config):
    if param.startswith("CSV:"):
        param_name_in_csv = param.split(":")[-1]
        value = rec[param_name_in_csv]
        return value
    elif param.startswith("GRABER:CSV:"):
        param_name_in_csv = param.split(":")[-1]
        grabber = web_graber(config)
        web_content = grabber.extract_html_v2(rec[param_name_in_csv])
        return web_content



def ask(prompt: PromptTemplate, data: hash, config) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=config["apikey"])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    result = chain.invoke(data)    
    return result

def process(rec, config):
    prompts = config["prompts"]
    rec = rec.asdict()
    results = {}
    values = []
    for prompt_name in prompts:
        input_data = {}
        for param_name in prompts[prompt_name]:
            if param_name not in ("PROMPT", "OUT"):
                param_source = prompts[prompt_name][param_name]
                value = extract_by_param(param_source, rec, config)
                input_data[param_name] = value
                input_data.update(results)
        prompt = PromptTemplate(
            template=prompts[prompt_name]['PROMPT'],
            input_variables=list(input_data.keys())
        )
        answer = ask(prompt, input_data, config)
        results[prompt_name] = answer
        values.append(answer)
    return values