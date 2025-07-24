

import time
from rag.retriever import get_retriever
from rag.decomposer import get_decomposer
from rag.synthesizer import get_synthesizer
from rag.llm_provider import get_llm
from langchain.chains import RetrievalQA

retriever = get_retriever()
decomposer = get_decomposer()
synthesizer = get_synthesizer()
llm = get_llm()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type="stuff"
)

MAX_SUB_QUESTIONS = 2

def multi_hop_answer(question: str) -> str:
    print("\n🔍 Received question:", question)
    total_start = time.time()

    try:
        start = time.time()
        decomposed = decomposer.invoke({"question": question})
        lines = decomposed["text"].strip().split("\n")
        sub_questions = [
            line.split(". ", 1)[-1].strip()
            for line in lines if line.strip() and (line.startswith("1.") or line.startswith("2."))
        ][:MAX_SUB_QUESTIONS]
        print(f"🧠 Decomposition done in {time.time() - start:.2f}s")
    except Exception as e:
        return f"❌ Decomposition failed: {e}"

    if not sub_questions:
        return "⚠️ No sub-questions generated."

    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")

    answers = []
    for i, sub_q in enumerate(sub_questions):
        print(f"\n🔎 Answering sub-question {i+1}: {sub_q}")
        start = time.time()
        try:
            result = qa_chain.invoke({"query": sub_q})
            answers.append(result["result"])
        except Exception as e:
            answers.append(f"❌ Retrieval failed: {e}")
        print(f"✅ Answered in {time.time() - start:.2f}s")

    context = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in zip(sub_questions, answers))

    try:
        start = time.time()
        final = synthesizer.invoke({"sub_answers": context})
        final_answer = final.get("text", "⚠️ Could not synthesize a final answer.")
        print(f"📝 Synthesis done in {time.time() - start:.2f}s")
    except Exception as e:
        return f"❌ Synthesis failed: {e}"

    print(f"✅ Total time: {time.time() - total_start:.2f}s")
    return final_answer
