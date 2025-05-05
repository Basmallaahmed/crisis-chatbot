# import pandas as pd
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.docstore.document import Document
# import sqlite3
# import google.auth
# from langdetect import detect
# from fastapi import FastAPI
# from pydantic import BaseModel
# import time

# # Function to simulate typing effect
# def print_typing(text, delay=0.02):
#     for char in text:
#         print(char, end="", flush=True)
#         time.sleep(delay)
#     print()

# # Load credentials using the service account file
# credentials, project = google.auth.load_credentials_from_file(r"C:\Users\user\OneDrive - Thebes Academy\Desktop\chatbot_api\our-truth-457613-e8-782d73ebf3a1.json")

# # Initialize LLM and embeddings
# llm = GoogleGenerativeAI(model="gemini-2.0-flash", credentials=credentials)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
# qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# # Create the custom prompt templates
# prompt_en = """
# You are a highly experienced strategic business consultant specialized in saving projects and turning crises into opportunities.

# Your task is to provide **several complete, actionable, and realistic plans** (a main plan + one or more alternatives) to answer the following question, based only on the context provided.

# Structure your response like a professional case study report, focusing on:
# - A brief description of each plan.
# - Numbered action steps within each plan, with expected duration for each step (e.g., “Step 1: Internal team meeting – expected duration: 2 days”).
# - Clear success metrics (KPIs) for each plan, such as customer satisfaction, revenue growth, number of new clients, etc.
# - Tone enhancement: make it more human and encouraging. Instead of starting with phrases like “The company is facing a crisis…”, use something like “The company is facing a complex challenge that threatens its future, but there is hope with a well-designed strategic plan.”
# - Include a media reputation management strategy: this should address social media and traditional media coverage during the crisis.

# ✅ Use the following structure for each plan:
# 1. **Plan Title** (e.g., “Rapid Recovery Plan”, “Long-Term Growth Strategy”, “Alternative Backup Plan”).
# 2. **Brief Introduction** explaining the rationale and how it can help the company recover or improve.
# 3. **Numbered Execution Steps**, with the estimated time for each.
# 4. **Key Performance Indicators (KPIs)** to track progress and success (e.g., "Customer Satisfaction Score", "Increase in Sales", "Reduction in Outstanding Debt").
# 5. **When to Use / Advantages / Potential Risks**: clarify when this plan is best suited, what benefits it offers, and any possible risks or limitations.
# 6. **Media Reputation Strategy**: a concrete plan for managing the brand’s public image across social and traditional media channels during the crisis.

# ⛔ Do not add any information not found in the context.
# ⛔ If the context is not sufficient to build multiple plans, state that clearly.

# ✳️ Start directly with the plans. Do not include any introductions, signatures, or report-like headers.

# ---

# ### Context:
# {docs}

# ### Question:
# {query}

# ### Proposed Plans:
# """

# prompt_ar = """
# أنت مستشار أعمال استراتيجي ذو خبرة عالية في إنقاذ المشاريع وتحويل الأزمات إلى فرص.

# مهمتك هي تقديم **عدة خطط عملية متكاملة وقابلة للتنفيذ** (خطة رئيسية + خطة أو أكثر بديلة) للإجابة عن السؤال التالي، بناءً فقط على المعلومات في السياق المرفق.

# صِغ إجابتك بأسلوب دراسة حالة احترافية، وكأنك تكتب تقريرًا رسميًا لصاحب المشروع، مع التركيز على:
# - وصف مختصر لكل خطة.
# - خطوات تنفيذ مرقّمة داخل كل خطة، مع تحديد التوقيت الزمني المتوقع لكل خطوة.
# - تحديد مؤشرات قياس النجاح (KPIs) لكل خطة مثل رضا العملاء، العوائد المالية، وعدد العملاء الجدد.
# - تحسين لهجة الشرح: يجب أن تكون أكثر قربًا للقراء. بدلاً من البداية بجمل مثل "تواجه شركة منارة أزمة..."، يمكن أن تبدأ بجمل مشجعة مثل "تمر شركة منارة بأزمة معقدة تهدد بقاءها، ولكن هناك بصيص أمل مع خطة استراتيجية مدروسة."
# - تضمين خطة لإدارة السمعة الإعلامية: يجب تضمين استراتيجية للتعامل مع السمعة الإعلامية على وسائل التواصل الاجتماعي والإعلام التقليدي خلال الأزمة.

# ✅ استخدم الهيكل التالي لكل خطة:
# 1. **عنوان الخطة** (مثلاً: "خطة إنقاذ سريعة"، "خطة طويلة المدى"، "الخطة البديلة").
# 2. **مقدمة قصيرة** توضح مبرراتها، مع توضيح كيف يمكن أن تنقذ الشركة أو تحسن وضعها.
# 3. **خطوات مرقّمة للتنفيذ**، مع تحديد الوقت المتوقع لكل خطوة (مثل: "الخطوة 1: اجتماع مع الفريق الداخلي - الوقت المتوقع: يومان").
# 4. **مؤشرات قياس النجاح (KPIs)**: تحديد مؤشرات محددة لقياس تقدم الخطة، مثل "نسبة رضا العملاء" أو "زيادة في المبيعات" أو "تخفيض الديون المستحقة".
# 5. **متى تستخدمها / مزاياها / مخاطرها المحتملة**: تحديد الحالات التي تكون فيها الخطة الأمثل، وفوائدها، وأي تحديات قد تنشأ.
# 6. **إستراتيجية لإدارة السمعة الإعلامية**: وضع خطة للتعامل مع السمعة الإعلامية في الأزمة سواء على وسائل التواصل الاجتماعي أو في الإعلام التقليدي.

# ⛔ لا تضف معلومات غير موجودة في السياق.
# ⛔ إذا لم تكن المعلومات كافية لبناء خطط متعددة، اذكر ذلك بوضوح.

# ✳️ ابدأ مباشرة بكتابة الخطط بدون أي مقدمات أو توقيعات أو تقرير.

# ---

# ### السياق:
# {docs}

# ### السؤال:
# {query}

# ### الخطط المقترحة:
# """

# # SQLite memory functions
# def init_db():
#     conn = sqlite3.connect("memory.db")
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS memory (
#             question TEXT PRIMARY KEY,
#             answer TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# def save_to_memory(question, answer):
#     conn = sqlite3.connect("memory.db")
#     cursor = conn.cursor()
#     cursor.execute("INSERT OR REPLACE INTO memory (question, answer) VALUES (?, ?)", (question, answer))
#     conn.commit()
#     conn.close()

# def search_memory(question):
#     conn = sqlite3.connect("memory.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT answer FROM memory WHERE question = ?", (question,))
#     result = cursor.fetchone()
#     conn.close()
#     return result[0] if result else None

# # Load and process crisis cases
# def load_cases_from_file(file_path):
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()

#     documents = []
#     for _, row in df.iterrows():
#         content = f"""
# 📝 Case: {row['excerpt']}

# 📍 Type: {row['crisis type']} - 📍 Location: {row['location']}
# 🕒 Crisis Time: {row['crisis time']} | 🆘 Impact Level: {row['impact level']}

# 🧠 Before Resolution: {row['outcome before resolution']}
# ⚠️ Crisis Consequences: {row['consequences of the crisis']}

# ✅ Resolution Steps: {row['resolution steps']}
# 🎯 Outcome After Resolution: {row['outcome after resolution']}

# 📚 Source: {row['source']}
#         """
#         documents.append(Document(page_content=content))

#     vectorstore = FAISS.from_documents(documents, embeddings)
#     return vectorstore

# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0

# def detect_language(text):
#     if not text.strip():
#         return None
#     try:
#         return detect(text)
#     except Exception as e:
#         print(f"Language detection error: {e}")
#         return None


# # Function to get solution directly
# def get_solution(user_input, previous_context=""):
#     # Detect language (Arabic or English)
#     language = detect_language(user_input)
#     is_arabic = language == "ar"
#     prompt = prompt_ar if is_arabic else prompt_en

#     # Check if we have prior context and append user input to it
#     context = previous_context + "\n" + user_input if previous_context else user_input
#     question_prompt = f"{context}\n\nما هي الخطوات العملية لإنقاذ هذه الأزمة؟" if is_arabic else f"{context}\n\nWhat are the practical steps to resolve this crisis?"

#     # Search memory first to get answer if available
#     memory_response = search_memory(user_input)
#     if memory_response:
#         print("\n✅ (Retrieved from memory)\n")
#         print_typing(memory_response)
#         return memory_response, context  # Returning updated context without changes

#     # Search for similar documents in the vectorstore
#     similar_docs = vectorstore.similarity_search(user_input)

#     if similar_docs:
#         full_context = "\n\n".join([doc.page_content for doc in similar_docs])
#         full_prompt = prompt.format(docs=full_context, query=question_prompt)
#     else:
#         full_prompt = question_prompt

#     # Send query to LLM to get the answer
#     response = llm.invoke(full_prompt)
#     print("\n🤖 (chatbot)\n")
#     print_typing(response)

#     # Save the response to memory for future reference
#     save_to_memory(user_input, response)

#     updated_context = context + "\n" + response  # New context for the next question

#     return response, updated_context

# # FastAPI application setup
# app = FastAPI()
# # @app.on_event("startup")
# # def startup_event():
# #     global vectorstore
# #     init_db()
# #     vectorstore = load_cases_from_file("/content/Original-Final-.csv")

# # Load vectorstore globally once
# init_db()
# vectorstore = load_cases_from_file(r"C:\Users\user\OneDrive - Thebes Academy\Desktop\chatbot_api\Original-Final-.csv") 
# # Data model for user input
# class UserQuery(BaseModel):
#     user_input: str
#     previous_context: str = ""

# @app.post("/get_solution")
# async def get_solution_endpoint(query: UserQuery):
#     response, updated_context = get_solution(query.user_input, query.previous_context)
#     return {"response": response, "updated_context": updated_context}

# # Main interaction loop (for local testing)
# if __name__ == "__main__":
#     init_db()
#     vectorstore = load_cases_from_file(r"C:\Users\user\OneDrive - Thebes Academy\Desktop\chatbot_api\Original-Final-.csv")
#     print("🤖 Hello! Write a description of a crisis or scenario, and I will help you analyze it and suggest solutions.\n(Write 'exit' to end the conversation)\n")

#     previous_context = ""

#     while True:
#         user_input = input("👤 You: ").strip()
#         if user_input.lower() == "exit":
#             print("👋 Goodbye!")
#             break
#         response, previous_context = get_solution(user_input, previous_context)
import pandas as pd  # تأكد من استيراد pandas
import sqlite3
import google.auth
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import time
from langdetect import detect, DetectorFactory
from fastapi import FastAPI
from pydantic import BaseModel

# Function to simulate typing effect
def print_typing(text, delay=0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# Load credentials using the service account file
credentials, project = google.auth.load_credentials_from_file(r"C:\Users\user\OneDrive - Thebes Academy\Desktop\chatbot_api\our-truth-457613-e8-782d73ebf3a1.json")

# Initialize LLM and embeddings
llm = GoogleGenerativeAI(model="gemini-2.0-flash", credentials=credentials)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# Create the custom prompt templates
prompt_en = """
You are a highly experienced strategic business consultant specialized in saving projects and turning crises into opportunities.

Your task is to provide **several complete, actionable, and realistic plans** (a main plan + one or more alternatives) to answer the following question, based only on the context provided.

Structure your response like a professional case study report, focusing on:
- A brief description of each plan.
- Numbered action steps within each plan, with expected duration for each step (e.g., “Step 1: Internal team meeting – expected duration: 2 days”).
- Clear success metrics (KPIs) for each plan, such as customer satisfaction, revenue growth, number of new clients, etc.
- Tone enhancement: make it more human and encouraging. Instead of starting with phrases like “The company is facing a crisis…”, use something like “The company is facing a complex challenge that threatens its future, but there is hope with a well-designed strategic plan.”
- Include a media reputation management strategy: this should address social media and traditional media coverage during the crisis.

✅ Use the following structure for each plan:
1. **Plan Title** (e.g., “Rapid Recovery Plan”, “Long-Term Growth Strategy”, “Alternative Backup Plan”).
2. **Brief Introduction** explaining the rationale and how it can help the company recover or improve.
3. **Numbered Execution Steps**, with the estimated time for each.
4. **Key Performance Indicators (KPIs)** to track progress and success (e.g., "Customer Satisfaction Score", "Increase in Sales", "Reduction in Outstanding Debt").
5. **When to Use / Advantages / Potential Risks**: clarify when this plan is best suited, what benefits it offers, and any possible risks or limitations.
6. **Media Reputation Strategy**: a concrete plan for managing the brand’s public image across social and traditional media channels during the crisis.

⛔ Do not add any information not found in the context.
⛔ If the context is not sufficient to build multiple plans, state that clearly.

✳️ Start directly with the plans. Do not include any introductions, signatures, or report-like headers.

---

### Context:
{docs}

### Question:
{query}

### Proposed Plans:
"""

prompt_ar = """
أنت مستشار أعمال استراتيجي ذو خبرة عالية في إنقاذ المشاريع وتحويل الأزمات إلى فرص.

مهمتك هي تقديم **عدة خطط عملية متكاملة وقابلة للتنفيذ** (خطة رئيسية + خطة أو أكثر بديلة) للإجابة عن السؤال التالي، بناءً فقط على المعلومات في السياق المرفق.

صِغ إجابتك بأسلوب دراسة حالة احترافية، وكأنك تكتب تقريرًا رسميًا لصاحب المشروع، مع التركيز على:
- وصف مختصر لكل خطة.
- خطوات تنفيذ مرقّمة داخل كل خطة، مع تحديد التوقيت الزمني المتوقع لكل خطوة.
- تحديد مؤشرات قياس النجاح (KPIs) لكل خطة مثل رضا العملاء، العوائد المالية، وعدد العملاء الجدد.
- تحسين لهجة الشرح: يجب أن تكون أكثر قربًا للقراء. بدلاً من البداية بجمل مثل "تواجه شركة منارة أزمة..."، يمكن أن تبدأ بجمل مشجعة مثل "تمر شركة منارة بأزمة معقدة تهدد بقاءها، ولكن هناك بصيص أمل مع خطة استراتيجية مدروسة."
- تضمين خطة لإدارة السمعة الإعلامية: يجب تضمين استراتيجية للتعامل مع السمعة الإعلامية على وسائل التواصل الاجتماعي والإعلام التقليدي خلال الأزمة.

✅ استخدم الهيكل التالي لكل خطة:
1. **عنوان الخطة** (مثلاً: "خطة إنقاذ سريعة"، "خطة طويلة المدى"، "الخطة البديلة").
2. **مقدمة قصيرة** توضح مبرراتها، مع توضيح كيف يمكن أن تنقذ الشركة أو تحسن وضعها.
3. **خطوات مرقّمة للتنفيذ**، مع تحديد الوقت المتوقع لكل خطوة (مثل: "الخطوة 1: اجتماع مع الفريق الداخلي - الوقت المتوقع: يومان").
4. **مؤشرات قياس النجاح (KPIs)**: تحديد مؤشرات محددة لقياس تقدم الخطة، مثل "نسبة رضا العملاء" أو "زيادة في المبيعات" أو "تخفيض الديون المستحقة".
5. **متى تستخدمها / مزاياها / مخاطرها المحتملة**: تحديد الحالات التي تكون فيها الخطة الأمثل، وفوائدها، وأي تحديات قد تنشأ.
6. **إستراتيجية لإدارة السمعة الإعلامية**: وضع خطة للتعامل مع السمعة الإعلامية في الأزمة سواء على وسائل التواصل الاجتماعي أو في الإعلام التقليدي.

⛔ لا تضف معلومات غير موجودة في السياق.
⛔ إذا لم تكن المعلومات كافية لبناء خطط متعددة، اذكر ذلك بوضوح.

✳️ ابدأ مباشرة بكتابة الخطط بدون أي مقدمات أو توقيعات أو تقرير.

---

### السياق:
{docs}

### السؤال:
{query}

### الخطط المقترحة:
"""

# SQLite memory functions
def init_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            question TEXT PRIMARY KEY,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_memory(question, answer):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO memory (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def search_memory(question):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM memory WHERE question = ?", (question,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Load and process crisis cases
def load_cases_from_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    documents = []
    for _, row in df.iterrows():
        content = f"""
📝 Case: {row['excerpt']}

📍 Type: {row['crisis type']} - 📍 Location: {row['location']}
🕒 Crisis Time: {row['crisis time']} | 🆘 Impact Level: {row['impact level']}

🧠 Before Resolution: {row['outcome before resolution']}
⚠️ Crisis Consequences: {row['consequences of the crisis']}

✅ Resolution Steps: {row['resolution steps']}
🎯 Outcome After Resolution: {row['outcome after resolution']}

📚 Source: {row['source']}
        """
        documents.append(Document(page_content=content))

    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def detect_language(text):
    if not text.strip():
        return None
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection error: {e}")
        return None


# Function to get solution directly
def get_solution(user_input, previous_context=""):
    # Detect language (Arabic or English)
    language = detect_language(user_input)
    is_arabic = language == "ar"
    prompt = prompt_ar if is_arabic else prompt_en

    # Check if we have prior context and append user input to it
    context = previous_context + "\n" + user_input if previous_context else user_input
    question_prompt = f"{context}\n\nما هي الخطوات العملية لإنقاذ هذه الأزمة؟" if is_arabic else f"{context}\n\nWhat are the practical steps to resolve this crisis?"

    # Search memory first to get answer if available
    memory_response = search_memory(user_input)
    if memory_response:
        print("\n✅ (Retrieved from memory)\n")
        print_typing(memory_response)
        return {"response": memory_response, "updated_context": context}

    # Search for similar documents in the vectorstore
    similar_docs = vectorstore.similarity_search(user_input)

    if similar_docs:
        full_context = "\n\n".join([doc.page_content for doc in similar_docs])
        full_prompt = prompt.format(docs=full_context, query=question_prompt)
    else:
        full_prompt = question_prompt

    # Send query to LLM to get the answer
    response = llm.invoke(full_prompt)
    print("\n🤖 (chatbot)\n")
    print_typing(response)

    # Save the response to memory for future reference
    save_to_memory(user_input, response)

    # Parse and structure the response in a more organized manner
    structured_response = {
        "plans": []
    }

    # Assuming the response follows the structure defined in the prompt
    # Split the response into individual plans (this part can be adjusted based on your actual response structure)
    plans = response.split("1. **")  # Split by the first plan (assuming it's numbered starting from 1)
    for plan in plans[1:]:  # Skip the first split element which is an empty string
        plan_details = plan.split("\n")
        plan_title = plan_details[0].strip()
        plan_content = "\n".join(plan_details[1:]).strip()

        structured_response["plans"].append({
            "title": plan_title,
            "content": plan_content
        })

    updated_context = context + "\n" + response  # New context for the next question

    return {"response": structured_response, "updated_context": updated_context}

# FastAPI application setup
app = FastAPI()

# Load vectorstore globally once
init_db()
vectorstore = load_cases_from_file(r"C:\Users\user\OneDrive - Thebes Academy\Desktop\chatbot_api\Original-Final-.csv") 

# Data model for user input
class UserQuery(BaseModel):
    user_input: str
    previous_context: str = ""

@app.post("/get_solution")
async def get_solution_endpoint(query: UserQuery):
    result = get_solution(query.user_input, query.previous_context)
    response = result["response"]
    return {
        "response": response,  # Organized response with structured plans
        "updated_context": result["updated_context"]
    }

# Main interaction loop (for local testing)
if __name__ == "__main__":
    init_db()
    vectorstore = load_cases_from_file(r"C:\Users\user\OneDrive - Thebes Academy\Desktop\chatbot_api\Original-Final-.csv")
    print("🤖 Hello! Write a description of a crisis or scenario, and I will help you analyze it and suggest solutions.\n(Write 'exit' to end the conversation)\n")

    previous_context = ""

    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response, previous_context = get_solution(user_input, previous_context)

