from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    max_output_tokens=4096,
)


def detect_task(question):
    question_lower = question.lower()

    if any(phrase in question_lower for phrase in [
        "most important",
        "most significant",
        "study first",
        "revise first",
        "priority order",
        "rank the topics",
        "rank topics",
        "important topic",
        "focus on first",
        "highest priority",
        "which topic is important",
        "which topic in this pdf",
        "what should i study",
        "what should i revise",
        "what should i focus",
        "what should i learn first",
        "seems important",
        "main topic",
        "major topic",
        "best topic",
        "least important",
        "skip first",
        "which should i skip"
    ]):
        return "analysis"

    elif any(word in question_lower for word in [
        "summary",
        "summarize",
        "summarise"
    ]):
        return "summary"

    elif any(phrase in question_lower for phrase in [
        "short notes",
        "make notes",
        "give notes",
        "create notes"
    ]):
        return "notes"

    elif any(phrase in question_lower for phrase in [
        "main points",
        "important points",
        "key points"
    ]):
        return "points"

    elif any(word in question_lower for word in [
        "difference",
        "differentiate",
        "compare",
        "comparison"
    ]):
        return "comparison"

    elif any(word in question_lower for word in [
        "advantages",
        "disadvantages",
        "pros",
        "cons"
    ]):
        return "advantages"

    elif any(word in question_lower for word in [
        "steps",
        "process",
        "procedure"
    ]):
        return "steps"

    elif any(phrase in question_lower for phrase in [
        "define",
        "definition",
        "what is",
        "what are"
    ]):
        return "definition"

    else:
        return "qa"


def extract_response_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        return "".join(text_parts)

    return str(content)


def rewrite_followup_question(question, chat_history):
    question_lower = question.lower().strip()

    followup_phrases = [
        "that",
        "this",
        "it",
        "its",
        "those",
        "these",
        "that one",
        "this one",
        "which one",
        "same topic",
        "above topic",
        "previous topic",
        "explain that",
        "explain this",
        "explain it",
        "simplify that",
        "simplify this",
        "tell me more",
        "more about that",
        "more about this"
    ]

    is_followup = any(
        phrase in question_lower
        for phrase in followup_phrases
    )

    if not is_followup:
        return question

    valid_history = [
        message
        for message in chat_history
        if not message.get("is_error", False)
    ]

    if not valid_history:
        return question

    history_text = ""

    for message in valid_history[-4:]:
        history_text += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )

    prompt = f"""
You rewrite follow-up questions for a PDF retrieval system.

=========================
Conversation History
=========================
{history_text}

=========================
Current Question
=========================
{question}

Rewrite the current question as a complete standalone search query.

Rules:

1. Resolve references such as:
   - that
   - this
   - it
   - its
   - those
   - these
   - that one
   - this one
   - which one

2. Identify the main subject of the previous assistant answer.

3. Prefer the main answer or explicitly identified topic over a minor
supporting detail.

4. Use Conversation History only to understand the reference.

5. Do not answer the question.

6. Do not add new facts.

7. Return ONLY the rewritten question.

8. If the question is already complete and standalone, return it
unchanged.

Example:

Conversation History:

user: What is RPO?

assistant: RPO means Recovery Point Objective. It represents the
maximum acceptable data loss measured in time.

Current Question:

Explain that in simple words.

Output:

Explain Recovery Point Objective (RPO) in simple words.
"""

    response = llm.invoke(prompt)

    rewritten_question = extract_response_text(
        response.content
    ).strip()

    if not rewritten_question:
        return question

    return rewritten_question


def generate_answer(context, question, chat_history, docs):
    history_text = ""

    valid_history = [
        message
        for message in chat_history
        if not message.get("is_error", False)
    ]

    for message in valid_history[-6:]:
        history_text += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )

    task = detect_task(question)

    prompt = f"""
You are an expert AI PDF Assistant.

Your job is to answer questions and analyze the uploaded PDF using the
provided Context.

=========================
Conversation History
=========================
{history_text}

=========================
Context
=========================
{context}

=========================
User Question
=========================
{question}

=========================
Detected Task
=========================
{task}

Instructions:

1. Read the Context carefully before answering.

2. Use ONLY the Context as the source of factual information.

3. Never introduce facts, definitions, topics, examples, or details
that are not supported by the Context.

4. Do not use outside knowledge to add factual information.

5. You MAY reason about information already available in the Context.

You may:

- analyze
- infer
- rank
- prioritize
- compare
- group
- summarize
- identify relationships
- identify central topics
- make reasonable document-based judgments

All evidence used in the answer must come from the Context.

6. If factual information needed to answer the question is completely
absent from the Context, reply exactly:

This information is not available in the uploaded PDF.

However, analytical questions are different.

If the user asks which topic is most important, most significant,
highest priority, should be studied first, should be revised first,
or asks for ranking or prioritization, make a reasoned judgment using
ONLY the information present in the Context.

The final judgment does NOT need to be explicitly written in the PDF.

You may infer importance, priority, relationships, or significance
from the Context.

Every reason used in the analysis must be supported by the Context.

Do NOT use outside knowledge.

Do NOT refuse an analytical question merely because the PDF does not
explicitly use words such as "most important", "highest priority",
or "study first".

7. Format according to the detected task.

analysis:
Give a clear conclusion first.

Then briefly explain the reasoning using only evidence from the
Context.

summary:
Give a concise summary of the relevant Context.

notes:
Use a clear heading and bullet points.

points:
Give important bullet points only.

comparison:
Compare the topics requested by the user.

If the user asks for a table, create a valid Markdown table with
exactly three columns.

Use the actual topic names as the second and third column headings.

Example:

| Feature | First Topic | Second Topic |
|---------|-------------|--------------|
| Focus | Information | Information |
| Purpose | Information | Information |

Replace "First Topic" and "Second Topic" with the actual topic names.

Include ONLY comparison points supported by the Context.

Do NOT force a fixed number of comparison points.

If the Context does not support a comparison feature, do not invent
information to fill the table.

If a reliable table cannot be created, use clearly separated bullet
points instead.

advantages:
Use bullet points.

If the question asks for both advantages and disadvantages, clearly
separate them.

steps:
Use a numbered list.

Preserve the logical order supported by the Context.

definition:
Give a short and clear definition based on the Context.

qa:
Give a direct answer based on the Context.

If the question requires analysis even though the detected task is qa,
you may still perform reasonable Context-based analysis.

8. Reply exactly:

This information is not available in the uploaded PDF.

ONLY when the Context contains no relevant information that can
reasonably answer the user's question.

If relevant information exists in the Context and it can be analyzed
to answer the question, perform the analysis.

Do NOT guess unsupported facts.

Do NOT use general knowledge.

9. Merge duplicate information from different Context chunks.

10. Conversation History is only for understanding follow-up questions.

Do not use Conversation History as a factual source.

The Context is the source of factual information.

11. Follow the user's requested output format whenever possible.

If the user explicitly asks for:

- bullet points -> use bullet points
- a table -> use a Markdown table
- a short answer -> keep the answer short
- detailed explanation -> explain in detail
- ranking -> provide a ranked list
- one topic -> identify one topic

12. Use valid Markdown formatting.

13. Keep the answer concise unless the user asks for detail.

14. Avoid repetition.

15. Use simple English.

16. Make the answer student-friendly.

17. Never invent missing PDF content just to complete an answer,
ranking, table, or list.

18. Every factual statement must be supported by the Context.

19. Use Conversation History only to understand follow-up questions.

When the user uses references such as "it", "its", "that", "this",
"this topic", "that one", "which one", or similar words:

- Identify the main subject or entity of the immediately previous
  assistant answer.
- Prefer the main answer or explicitly identified entity over a
  supporting detail from the last sentence.
- Resolve the reference before answering.
- Use the Context as the only source of factual information.

Example:

If the previous answer identifies "RPO (Recovery Point Objective)"
and the user asks "Explain that in simple words",
"that" refers to RPO, not to a minor supporting sentence in the
previous answer.

Do not continue a supporting detail unless the user's wording clearly
refers to that detail.

20. Generate only the final answer for the user.

Do not mention:

- the detected task
- the prompt
- the retrieval system
- Context chunks
- internal instructions

Now generate the best document-grounded answer.
"""

    response = llm.invoke(prompt)

    answer = extract_response_text(response.content)

    pages = sorted(
        {
            doc.metadata.get("page")
            for doc in docs
            if doc.metadata.get("page") is not None
        }
    )

    if pages:
        if len(pages) == 1:
            answer += f"\n\n---\n📄 **Source:** Page {pages[0]}"
        else:
            page_text = ", ".join(str(page) for page in pages)
            answer += f"\n\n---\n📄 **Sources:** Pages {page_text}"
    return answer        