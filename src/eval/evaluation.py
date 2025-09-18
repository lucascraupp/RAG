from typing import Dict

import pandas as pd
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from supabase import create_client

from config import Config
from src.eval.retrievels import hybrid_search


def load_questions() -> Dict[str, str]:
    with open(f"src/eval/questions.yaml", "r") as file:
        params = yaml.safe_load(file)
    return params["questions"]


def answer_question(question: str, context: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
        Você é um especialista em análise de documentos financeiros.

        Baseado no contexto fornecido, responda a pergunta do usuário de forma clara e objetiva. Você deve apenas fornecer a resposta, sem explicações adicionais. Seja o mais simples possível, iformando apenas o valor e a unidade.

        Contexto:
        {context}
    """

    response = llm.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=question)]
    )
    return response.content


def evaluate_response(filename: str) -> None:
    supabase_client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    questions = load_questions()

    strategies = supabase_client.rpc(
        "get_strategies_by_filename", {"filename": filename}
    ).execute()

    results = pd.DataFrame(
        columns=["strategy", "question", "expected_answer", "llm_answer", "result"]
    )

    for strategy in strategies.data:
        filters = {"filename": filename, "strategy": strategy}

        for key, value in questions.items():
            question = value["question"]
            answer = value["answer"]

            context = hybrid_search(question, filters)

            llm_answer = answer_question(question, context)

            prompt = f"""
                Determine se a resposta do modelo está correta ou incorreta, com base na resposta esperada, considerando que valores muito próximos podem ser considerados corretos. Responda apenas com "correto" ou "incorreto".

                Resposta obtida: {llm_answer}
                Resposta esperada: {answer}
            """

            evaluation = llm.invoke([SystemMessage(content=prompt)])

            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "strategy": [strategy],
                            "question": [question],
                            "expected_answer": [answer],
                            "llm_answer": [llm_answer],
                            "result": [evaluation.content],
                        }
                    ),
                ],
                ignore_index=True,
            )

    results.to_csv(f"results_evaluation.csv", index=False)
