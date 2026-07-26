from src.llm.gemini import llm

def compress_context(context):

    prompt = f"""
You are an expert document analyzer.

Below are multiple retrieved chunks from a PDF.

Your job is to merge them into one concise context.

Rules:
- Remove duplicate information.
- Merge similar points.
- Preserve all important concepts.
- Keep headings.
- Use bullet points.
- Maximum 400 words.
- Do NOT answer any question.
- Only produce a clean summary of the retrieved content.

Retrieved Chunks:

{context}
"""

    response = llm.invoke(prompt)
    return response.content