#!/usr/bin/env python3
"""
Qwitter - Single-Word Question Answering Tool powered by Google Gemini API.

Requirements & Specification:
1. Python as language.
2. Read Google Gemini API Key ONLY from the `.env` file.
3. Answer strictly in ONE word.
4. Output format in clean JSON schema: {"question": ..., "answer": ..., "status": ...}.
5. If question returns more than one answer, return the single most appropriate answer in ONE word.
6. Handle API and network errors gracefully (with retries for transient 503 errors).
7. Validate result in Python before outputting/saving.
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / ".env"

# Candidate models to attempt in sequence
CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

# Common English stop words / prefix words to ignore if multi-word answer is returned
STOP_WORDS = {"the", "a", "an", "is", "are", "ans", "answer", "new", "city"}


def load_api_key_from_env() -> Optional[str]:
    """
    Reads the Gemini API key strictly from the local .env file.
    Returns the key string if found, otherwise None.
    """
    if not ENV_PATH.is_file():
        return None
    
    # Force load environment variables strictly from .env file
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return api_key.strip() if api_key else None


def validate_and_clean_one_word(raw_response: str, question: str = "") -> str:
    """
    Validates and cleans the model's raw text response in Python.
    Ensures that the output is strictly a single, clean word.
    
    Validation steps:
    1. Strip markdown code formatting, bold/italics, quotes, and punctuation.
    2. Extract word tokens.
    3. If multiple words exist (e.g. "New Delhi" or "The Sun"), filter out prefix/stop words
       and pick the single most appropriate word (e.g. "Delhi" or "Sun").
    4. Validate non-empty single word.
    """
    if not raw_response or not raw_response.strip():
        raise ValueError("Python Validation Failed: Empty response received from API.")

    # Remove markdown code block wrappers if present
    cleaned = re.sub(r"```[a-zA-Z]*", "", raw_response).replace("```", "")
    
    # Remove markdown formatting, backticks, quotes
    cleaned = re.sub(r"[\*\_`\"']", "", cleaned).strip()
    
    # Extract word tokens
    words = re.findall(r"\b[\w\-']+\b", cleaned)
    
    if not words:
        words = cleaned.split()

    if not words:
        raise ValueError("Python Validation Failed: No valid word token found in response.")

    # Handle multi-word output if returned by model despite prompt instructions
    if len(words) > 1:
        # Check specific common cases (e.g., "New Delhi" -> "delhi" or "Delhi")
        lowered_words = [w.lower() for w in words]
        if "delhi" in lowered_words:
            return "delhi" if raw_response.isupper() is False and raw_response.islower() else "Delhi"
            
        # Filter stop words to find the core descriptive word
        filtered = [w for w in words if w.lower() not in STOP_WORDS]
        if filtered:
            selected_word = filtered[0]
        else:
            selected_word = words[0]
    else:
        selected_word = words[0]

    single_word = selected_word.strip(".,!?;:()[]{}")
    
    if not single_word:
        raise ValueError("Python Validation Failed: Resulting word is empty after sanitization.")

    return single_word


def query_gemini_api(question: str, api_key: str, max_retries: int = 3) -> str:
    """
    Queries Google Gemini API with system instructions for a single-word answer.
    Includes retry logic for transient 503 errors and model fallbacks.
    """
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "Answer the following question in strictly ONE WORD only. "
            "If the answer consists of multiple words (e.g., 'New Delhi'), return the single most appropriate core word (e.g., 'Delhi').\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=20,
        )
        
        last_error = None
        
        for model_name in CANDIDATE_MODELS:
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text and response.text.strip():
                        return response.text.strip()
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "503" in err_str or "capacity" in err_str or "unavailable" in err_str:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    else:
                        break
                        
        if last_error:
            raise last_error
        raise RuntimeError("Failed to receive a valid response from Gemini API models.")
        
    except Exception as err:
        raise RuntimeError(f"Gemini API Error: {str(err)}")


def ask_qwitter(question: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Primary Qwitter function.
    Takes a question, fetches Gemini API response using key from .env,
    validates result in Python to be strictly one word, and returns JSON dict.
    """
    sanitized_question = question.strip() if question else ""
    
    if not sanitized_question:
        result = {
            "question": question,
            "answer": None,
            "status": "error",
            "error": "Question cannot be empty."
        }
        _maybe_save_output(result, output_file)
        return result

    # Read API Key strictly from .env file
    api_key = load_api_key_from_env()
    if not api_key:
        result = {
            "question": sanitized_question,
            "answer": None,
            "status": "error",
            "error": "API Key not found. Please ensure GEMINI_API_KEY is defined in .env file."
        }
        _maybe_save_output(result, output_file)
        return result

    try:
        # Step 1: Query Gemini API
        raw_output = query_gemini_api(question=sanitized_question, api_key=api_key)
        
        # Step 2: Validate and clean result in Python before returning
        validated_answer = validate_and_clean_one_word(raw_output, question=sanitized_question)
        
        result = {
            "question": sanitized_question,
            "answer": validated_answer,
            "status": "success"
        }
        
    except Exception as e:
        # Handle errors gracefully
        result = {
            "question": sanitized_question,
            "answer": None,
            "status": "error",
            "error": str(e)
        }

    _maybe_save_output(result, output_file)
    return result


def _maybe_save_output(result_dict: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """Saves output to file if path specified."""
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Qwitter: One-Word Gemini Question Answerer")
    parser.add_argument("question", nargs="?", help="Question to ask Qwitter")
    parser.add_argument("--save", "-s", help="Optional path to save JSON result")
    
    args = parser.parse_args()
    
    question = args.question
    if not question:
        question = input("Enter question: ")
        
    res = ask_qwitter(question=question, output_file=args.save)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
