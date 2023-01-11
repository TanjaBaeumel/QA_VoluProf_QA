import json
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer, util


question_answerer = pipeline("question-answering", model="deepset/gelectra-large-germanquad",
                               tokenizer="deepset/gelectra-large-germanquad")


def retrieve_manual_pair(question: str, cut_off=0.8, manual_qa_file="manualQApairs.json") -> (bool, str, float):
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

    # Open the stored manual QA pairs
    with open(manual_qa_file) as json_file:
        manual_pairs = json.load(json_file)
    question_list = list(manual_pairs.keys())


    # Compute embedding for both the current question and the stored questions
    embeddings1 = model.encode([question], convert_to_tensor=True)
    embeddings2 = model.encode(question_list, convert_to_tensor=True)

    # Compare the given question to all stored questions by computing cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    #print(cosine_scores)

    # If we judge a stored question to be a paraphrase of the question, return the corresponding stored answer
    index_max = max(range(len(cosine_scores[0])), key=cosine_scores[0].__getitem__)
    #print(index_max)
    if cosine_scores[0][index_max] >= cut_off:
        #print(manual_pairs[question_list[index_max]])
        return True, manual_pairs[question_list[index_max]], cosine_scores[0][index_max].item()

    # If the question is not paraphrase of any of the stored questions, we did not find an answer
    return False, None, 0


# Return values:
# - requestID: int
# - answer: str
# - confidence: float
def answer_question(request_id: int, question: str, file: str, cut_off=0.8, manual_qa_file="manualQApairs.json") -> (int, str, float):
    manual_pair_exists, answer, confidence = retrieve_manual_pair(question)
    if manual_pair_exists:
        return (request_id, answer, confidence)

    with open(file, encoding='utf-8') as f:
        context = f.read()
        f.close()

        result = question_answerer(question, context)

    return request_id, result['answer'], round(result['score'], 4)
