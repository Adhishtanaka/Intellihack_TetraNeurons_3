import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def generate_diverse_questions(chunk, num_questions=100):
    prompt = f"""
    Based on the following text from a document, generate {num_questions} diverse and specific questions that could be asked about this content.
    
    INSTRUCTIONS FOR QUESTION GENERATION:
    1. Make questions clear, direct, and unambiguous
    2. Ensure each question has a specific focus that can be answered from the text
    3. Vary complexity (factual, analytical, inferential)
    4. Use proper grammar and punctuation
    5. Phrase questions in a way that naturally elicits a 4-5 sentence response
    6. Avoid yes/no questions or questions that can be answered too briefly
    
    TEXT:
    {chunk}
    
    FORMAT YOUR RESPONSE AS A JSON ARRAY OF QUESTION STRINGS ONLY.
    """
    
    try:
        response = model.generate_content(prompt)
        questions_text = response.text
        json_start = questions_text.find('[')
        json_end = questions_text.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            questions_json = questions_text[json_start:json_end]
            questions = json.loads(questions_json)
            return questions
        else:
            return []
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def generate_answers(chunk, questions):
    prompt = f"""
    Based on the following text from a document, provide thorough and well-explained answers to each of the following questions.
    
    TEXT:
    {chunk}
    
    QUESTIONS:
    {json.dumps(questions)}
    
    INSTRUCTIONS FOR ANSWERING:
    1. EVERY answer MUST be exactly 4-5 sentences long - this is crucial
    2. Provide detailed explanations with relevant context from the text
    3. Include supporting evidence or reasoning in each answer
    4. Structure each answer to have a clear beginning, explanation, and conclusion
    5. Be concise but comprehensive within the 4-5 sentence constraint
    6. Start each answer with "Answer: " followed by your explanation
    
    FORMAT: 
    * Keep each answer self-contained
    * Separate answers with "---" on its own line
    * Don't repeat the questions in your response
    """
    
    try:
        response = model.generate_content(prompt)
        answers_text = response.text.strip()
        
        # Split answers by the separator
        answer_list = answers_text.split('---')
        
        # Clean up each answer
        processed_answers = []
        for answer in answer_list: 
            cleaned_answer = answer.strip()
            if cleaned_answer.startswith("Answer:"):
                cleaned_answer = cleaned_answer[7:].strip()
            if cleaned_answer: 
                processed_answers.append(cleaned_answer)
        
        return processed_answers
    except Exception as e:
        print(f"Error generating answers: {e}")
        return []

# Function to process a single Markdown file
def process_single_textfile(textfile_path, output_file, run_number):
    all_conversations = []
    
    try:
        with open(textfile_path, 'r', encoding='utf-8') as file:
            chunk = file.read()
        
        questions = generate_diverse_questions(chunk)
        answers = generate_answers(chunk, questions)
        
        for question, answer in zip(questions, answers):
            if question and answer:
                sentence_count = len([s for s in answer.split('.') if s.strip()])
                if 3 <= sentence_count <= 6:  
                    chatml_conversation = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                    all_conversations.append({
                        "messages": chatml_conversation,
                        "source_document": os.path.basename(textfile_path),
                        "run_number": run_number
                    })
        
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(all_conversations)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Run {run_number}: Generated {len(all_conversations)} QA pairs for {textfile_path}")
        return True

    except Exception as e:
        print(f"Run {run_number}: Error processing file {textfile_path}: {e}")
        return False

# Main function to loop through all .md files
def process_all_md_files(md_folder, output_file, run_number):
    md_files = [f for f in os.listdir(md_folder) if f.endswith('.md')]
    
    for md_file in md_files:
        md_path = os.path.join(md_folder, md_file)
        print(f"Run {run_number}: Processing: {md_path}")
        
        success = process_single_textfile(md_path, output_file, run_number)
        
        if success:
            print(f"Run {run_number}: Completed: {md_path}. Waiting 60 seconds before next file...")
            time.sleep(60)  # Wait 60 seconds before processing the next file
        else:
            print(f"Run {run_number}: Skipping: {md_path} due to an error.")

# Run the script 10 times
if __name__ == "__main__":
    md_folder = "md"
    output_file = "output.json"
    
    for run in range(1, 11):
        print(f"Starting run {run} of 10...")
        process_all_md_files(md_folder, output_file, run)
        
        if run < 10:
            wait_time = 1  
            print(f"Run {run} completed. Waiting {wait_time} seconds before starting next run...")
            time.sleep(wait_time)
    
    print("Done!. please run format.py for convert output.json to alpaca template format")