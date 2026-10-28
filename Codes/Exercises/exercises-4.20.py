from transformers import pipeline

# 1. Inisialisasi model
question_answerer = pipeline('question-answering')

# 2. Menentukan teks konteks baru (Different piece of text)
context_text = """
Mixed Reality (MR) merges real and virtual worlds to produce new environments 
where physical and digital objects co-exist and interact in real time. 
The first immersive mixed reality system was the Virtual Fixtures platform, 
developed at the U.S. Air Force's Armstrong Laboratories in the year 1992.
"""

# 3. Membuat daftar pertanyaan baru (Different questions)
questions = [
    "What does Mixed Reality merge?",
    "What was the name of the first immersive system?",
    "Where was the platform developed?",
    "In what year was it developed?"
]

# 4. Menjalankan model dan menampilkan hasil
print("="*50)
print("TEXT CONTEXT:")
print(context_text.strip())
print("="*50 + "\n")

for q in questions:
    result = question_answerer({'question': q, 'context': context_text})
    print(f"Question : {q}")
    print(f"Answer   : {result['answer']}")
    print(f"Score    : {result['score']:.4f}\n")