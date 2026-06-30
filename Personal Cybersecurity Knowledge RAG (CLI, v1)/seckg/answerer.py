import re
from typing import List
from seckg.config import settings
from seckg.models import Answer, RetrievalResult, Chunk

class Answerer:
    """Handles interaction with the Gemini API to generate source-grounded answers."""
    
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.client = None
        if self.api_key:
            try:
                # pyrefly: ignore [missing-import]
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Google GenAI Client: {e}")

    def answer(self, question: str, results: List[RetrievalResult]) -> Answer:
        """Queries the Gemini model with retrieved context and returns a grounded Answer object."""
        if not self.api_key or not self.client:
            return Answer(
                text="Error: GEMINI_API_KEY environment variable is not set. Please create a .env file and set GEMINI_API_KEY.",
                sources=[]
            )
            
        if not results:
            return Answer(
                text="I cannot answer this question based on the provided source material.",
                sources=[]
            )
            
        # Format the context text for the model prompt
        context_parts = []
        for i, res in enumerate(results):
            chunk = res.chunk
            header_info = f"Heading: {chunk.heading_path}" if chunk.heading_path else ""
            page_info = f"Page: {chunk.page_number}" if chunk.page_number is not None else ""
            source_detail = header_info or page_info
            
            part = (
                f"[{i + 1}] Source File: {chunk.source_file}\n"
                f"{source_detail}\n"
                f"Content:\n{chunk.content}\n"
            )
            context_parts.append(part)
            
        context_text = "\n---\n".join(context_parts)
        
        system_instruction = (
            "You are SecKnowledge, a precise, personal cybersecurity RAG system.\n"
            "Your task is to answer the user's question based strictly and ONLY on the provided context chunks.\n"
            "Follow these rules precisely:\n"
            "1. Answer the question using ONLY the facts explicitly mentioned in the context chunks.\n"
            "2. If the context chunks do not contain enough information to answer the question, state: "
            "'I cannot answer this question based on the provided source material.' and do NOT attempt to answer from general knowledge.\n"
            "3. In your response, cite the source(s) of your information. Every time you make a factual claim, cite it by placing the source number in square brackets, e.g., [1] or [2]. These numbers correspond to the context chunks listed below.\n"
            "4. At the very end of your response, write a single line starting with 'Citations: ' followed by a comma-separated list of the source numbers you actually used (e.g. 'Citations: 1, 3'). If no sources were used or you couldn't answer, write 'Citations: None'."
        )
        
        user_content = f"Context Chunks:\n{context_text}\n\nQuestion: {question}"
        
        try:
            # pyrefly: ignore [missing-import]
            from google.genai import types
            
            # Request completion from Gemini model
            response = self.client.models.generate_content(
                model=settings.llm_model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            
            raw_text = response.text or ""
            
            # Extract citations list from the raw text
            cited_chunks = []
            citations_match = re.search(r"Citations:\s*(.*)", raw_text, re.IGNORECASE)
            if citations_match:
                citation_str = citations_match.group(1).strip()
                # Find all integer indexes cited
                for num_str in re.findall(r"\d+", citation_str):
                    idx = int(num_str) - 1
                    if 0 <= idx < len(results):
                        chunk = results[idx].chunk
                        if chunk not in cited_chunks:
                            cited_chunks.append(chunk)
                            
            # Clean up the output text to hide the raw "Citations: ..." instruction line
            clean_text = re.sub(r"\n*Citations:\s*.*", "", raw_text, flags=re.IGNORECASE).strip()
            
            # If the model explicitly stated it cannot answer, return empty sources
            if "I cannot answer this question" in clean_text:
                cited_chunks = []
                
            return Answer(text=clean_text, sources=cited_chunks)
            
        except Exception as e:
            return Answer(
                text=f"Error generating answer: {e}",
                sources=[]
            )
