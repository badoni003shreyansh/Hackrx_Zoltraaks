import io
import json
import re
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
import requests

from PyPDF2 import PdfReader
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq  # Changed import
import cassio


load_dotenv()

# Configuration
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([API_BEARER_TOKEN, ASTRA_TOKEN, ASTRA_DB_ID]):
    raise ValueError("Missing required environment variables")

# Initialize FastAPI
app = FastAPI(title="Document QA System", version="3.1.0")
auth_scheme = HTTPBearer()

# Initialize database and models with optimizations
cassio.init(token=ASTRA_TOKEN, database_id=ASTRA_DB_ID)

# Optimized embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    # Batch processing settings for efficiency
    requests_per_minute=60,
    request_timeout=15
)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Pydantic models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to a PDF document")
    questions: List[str] = Field(..., min_items=1, description="List of questions")

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication
def clean_llm_json(llm_output: str):
    """
    Clean LLM output to extract valid JSON.
    Handles various formatting issues and extra characters.
    """
    if not llm_output:
        return "{}"
    
    # Remove markdown code fences (```json ... ```)
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", llm_output.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    
    # Find the first { and last } to extract just the JSON part
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    
    if start == -1 or end == 0 or start >= end:
        # No valid JSON structure found
        return "{}"
    
    # Extract the JSON portion
    json_part = cleaned[start:end]
    
    # Try to parse the extracted JSON
    try:
        # Test if it's valid JSON
        json.loads(json_part)
        return json_part
    except json.JSONDecodeError:
        # If still invalid, try to clean further
        # Remove any non-printable characters that might cause issues
        json_part = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_part)
        
        # Try parsing again
        try:
            json.loads(json_part)
            return json_part
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Fix common quote issues
                json_part = re.sub(r'([^\\])"([^"]*)"([^\\])', r'\1"\2"\3', json_part)
                # Remove any trailing commas before closing brackets/braces
                json_part = re.sub(r',(\s*[}\]])', r'\1', json_part)
                
                json.loads(json_part)
                return json_part
            except json.JSONDecodeError:
                # Last resort: return empty JSON
                return "{}"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )
    return credentials.credentials

# Optimized helper functions
def download_and_extract_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10, stream=True)  # Stream for large files
        response.raise_for_status()
        
        file_stream = io.BytesIO(response.content)
        pdf_reader = PdfReader(file_stream)
        
        # More efficient text extraction
        text_parts = []
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text_parts.append(content)
        
        return "".join(text_parts)
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process document: {str(e)}"
        )

def create_prompt_for_questions(text: str, questions: List[str]) -> str:
    """
    Creates a prompt for question answering based on text and questions.
    
    Args:
        text: The source text to analyze
        questions: List of questions to answer
        
    Returns:
        str: The formatted prompt ready for LLM
    """
    try:
        # Optimized text chunking - smaller chunks for faster search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,      # Reduced from 800
            chunk_overlap=100,
        )
        text_chunks = text_splitter.split_text(text)
        
        # Add texts in batch for efficiency
        astra_vector_store.add_texts(text_chunks)
        print(f"Inserted {len(text_chunks)} text chunks into the vector store.")

        # Optimized system message - more explicit about JSON formatting
        system_message = """You must return ONLY valid JSON with no additional text, formatting, or explanations.
        
        Format your response exactly like this:
        {"answers": ["answer1", "answer2", "answer3", ...]}
        
        Rules:
        - Each answer should be 600-800 characters
        - Be precise and based on the provided context
        - Do not include any text before or after the JSON
        - Do not use markdown formatting
        - Ensure all quotes are properly escaped
        - Return exactly the number of answers as there are questions"""

        # OPTIMIZED: Single combined similarity search
        # Combine all questions into one search query for better efficiency
        combined_query = " ".join(questions)
        
        # Get more documents in one search, then filter
        all_docs = astra_vector_store.similarity_search(
            combined_query, 
            k=min(len(questions) * 3, 15),  # Limit total docs
            search_type="similarity"  # Faster than MMR
        )
        
        # Create a combined context from all relevant docs
        combined_context = "\n".join(doc.page_content for doc in all_docs)
        
        # Build optimized prompt with all questions
        questions_text = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))
        
        full_prompt = f"""{system_message}

Context:
{combined_context}

Questions:
{questions_text}

Answers:"""

        return full_prompt
        
    except Exception as e:
        print(f"Prompt creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt creation failed: {str(e)}"
        )


def send_prompt_to_groq(prompt: str) -> str:
    """
    Sends a prompt to Groq API and returns the response.
    
    Args:
        prompt: The formatted prompt to send to Groq
        
    Returns:
        str: The response content from Groq API
    """
    try:
        # Single LLM call using native Groq client (non-streaming)
        groq_client = Groq(api_key=GROQ_API_KEY)
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",  # You can change this to your preferred model
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=800,
            top_p=0.8,
            stream=False,  # Set to False for non-streaming response
            stop=None
        )

        # Extract the response content
        reply = completion.choices[0].message.content
        
        # Log the raw response for debugging
        print(f"Raw Groq response: {reply[:200]}...")  # First 200 chars
        
        return reply
        
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq API call failed: {str(e)}"
        )


def process_questions(text: str, questions: List[str]) -> List[str]:
    """
    Main function to process questions using the separated prompt creation and API call functions.
    
    Args:
        text: The source text to analyze
        questions: List of questions to answer
        
    Returns:
        List[str]: List of answers corresponding to the questions
    """
    try:
        # Create the prompt
        full_prompt = create_prompt_for_questions(text, questions)
        
        # Send prompt to Groq and get response
        reply = send_prompt_to_groq(full_prompt)

        # Clean & parse JSON
        try:
            cleaned_output = clean_llm_json(reply)
            print(f"Cleaned output: {cleaned_output}")
            
            data = json.loads(cleaned_output)
            answers = data.get("answers", [])
            
            # Ensure we have the right number of answers
            if len(answers) != len(questions):
                print(f"Answer count mismatch: expected {len(questions)}, got {len(answers)}")
                # Fallback: create placeholder answers if mismatch
                while len(answers) < len(questions):
                    answers.append("Unable to find sufficient information in the document.")
                answers = answers[:len(questions)]
            
            return answers
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON: {e}")
            print(f"Raw LLM output: {reply}")
            print(f"Cleaned output: {cleaned_output}")
            
            # Try to extract answers using regex as a fallback
            try:
                # Look for patterns like "answer1", "answer2" etc.
                answer_pattern = r'"([^"]{50,800})"'  # Answers between 50-800 chars
                extracted_answers = re.findall(answer_pattern, reply)
                
                if extracted_answers and len(extracted_answers) >= len(questions):
                    return extracted_answers[:len(questions)]
                else:
                    # Final fallback
                    return ["Unable to process question due to parsing error."] * len(questions)
            except Exception:
                return ["Unable to process question due to parsing error."] * len(questions)

    except Exception as e:
        print(f"Question processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question processing failed: {str(e)}"
        )
# Optimized API Endpoint
@app.post("/api/v1/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def handle_query(request: QueryRequest):
    try:
        # Process document
        text = download_and_extract_text(request.documents)
        if not text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the document"
            )
        
        # Limit text size for processing efficiency
        if len(text) > 100000:  # 100KB limit
            text = text[:100000]  # Truncate if too large
        
        # Process questions
        answers = process_questions(text, request.questions)
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

def cleanup():
    try:
        if 'astra_vector_store' in globals():
            astra_vector_store.clear()
        if 'groq_client' in globals():
            del groq_client
    except Exception as e:
        print(f"Cleanup error: {e}")

if __name__ == "__main__":
    import uvicorn
    try:
        # Get port from environment variable for Render deployment
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        cleanup()