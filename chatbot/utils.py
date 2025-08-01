import os
import time
import json
import torch
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, json_path='knowledge_base.json'):
        self.json_path = json_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = self.load_documents()
        self.document_embeddings = self.embed_documents()

    def load_documents(self):
        if not os.path.exists(self.json_path):
            return []
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def embed_documents(self):
        if not self.documents:
            return np.array([])
        texts = [doc['question'] + " " + doc['answer'] for doc in self.documents]
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def retrieve(self, query, top_k=3, threshold=0.6):  # lowered threshold
        if self.document_embeddings is None or len(self.document_embeddings) == 0:
            return []
        query_embedding = self.embedding_model.encode([query.lower()], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        for i, score in zip(top_indices, top_scores):
            logger.info(f"Doc {i} similarity: {score:.3f}")
        return [
            self.documents[i] for i, score in zip(top_indices, top_scores)
            if score > threshold
        ]

    def add_document(self, question, answer):
        self.documents.append({"question": question, "answer": answer})
        self.save_documents()
        self.document_embeddings = self.embed_documents()

    def save_documents(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

class AdvancedChatbot:
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        self.chatbot_name = "Nepsy"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.retriever = DocumentRetriever(json_path=knowledge_base_path)

        model_id = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device).eval()

    def generate_response(self, query, history=None):
        timings = {}
        start_total = time.time()

        try:
            t0 = time.time()
            retrieved_docs = self.retriever.retrieve(query)
            context = "\n".join([f"- {doc['answer']}" for doc in retrieved_docs]) if retrieved_docs else ""
            timings["retrieval"] = round(time.time() - t0, 4)

            t1 = time.time()
            prompt = f"""
You are an AI assistant. Use the following context to answer the user's question thoroughly.
Context:
{context or 'No relevant information available.'}

Question: {query}

Answer with complete details:
"""
            timings["prompt_construction"] = round(time.time() - t1, 4)

            t2 = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            timings["tokenization"] = round(time.time() - t2, 4)

            t3 = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            timings["generation"] = round(time.time() - t3, 4)

            t4 = time.time()
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            timings["post_processing"] = round(time.time() - t4, 4)

            # Fallback to context if response too short or generic
            if len(response) < 10 and context:
                response = context

            timings["total"] = round(time.time() - start_total, 4)
            return response, timings, retrieved_docs, False

        except Exception as e:
            logger.error("Response generation failed", exc_info=True)
            return "Sorry, I encountered an error while generating a response.", {}, [], False
