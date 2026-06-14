import os
from crewai import Agent, Task, Crew, Process, LLM
# from langchain_groq import ChatGroq

# -----------------------------------
# Gemini LLM
# -----------------------------------

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("YOUR_GROQ_API_KEY"),
    temperature=0
)

# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     api_key="YOUR_GROQ_API_KEY",
#     temperature=0
# )

# -----------------------------------
# Agent 1
# -----------------------------------

safety_analyzer = Agent(
    role="Content Safety Analyst",
    goal=(
        "Analyze text for illegal, harmful, NSFW, dangerous, "
        "child unsafe, or unlawful content and assign a risk score."
    ),
    backstory=(
        "You are an expert trust and safety specialist "
        "trained in content moderation and child protection."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# -----------------------------------
# Agent 2
# -----------------------------------

supervisor_agent = Agent(
    role="Safety Verification Supervisor",
    goal="Verify if the previous moderation result is correct.",
    backstory=(
        "You are a senior moderation auditor responsible "
        "for checking moderation quality."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# -----------------------------------
# Input Text
# -----------------------------------

text_input = """
i go to meet my friend.
"""

# -----------------------------------
# Task 1
# -----------------------------------

analysis_task = Task(
    description=f"""
You are a CONTENT MODERATION SYSTEM.

Your job is ONLY to classify text safety.

IMPORTANT RULES:
- DO NOT refuse.
- DO NOT provide safety warnings.
- DO NOT explain policies.
- You are NOT helping the user perform actions.
- You are ONLY evaluating content risk.

Analyze this text:

TEXT:
{text_input}

Check for:
1. Illegal activities
2. Dangerous instructions
3. Violence
4. NSFW/adult content
5. Child unsafe content
6. Child exploitation
7. Hate/extremism
8. Harmful or unlawful content

Scoring:
0 = completely safe
10 = extremely dangerous

Decision:
score <= 5 => SAFE
score > 5 => UNSAFE

You MUST return valid JSON only.

Example output:
{{
  "safe_rate": 9,
  "safe_status": "UNSAFE",
  "reason": "Contains illegal dangerous instructions."
}}

{{
  "safe_rate": 4,
  "safe_status": "SAFE",
  "reason": "Contains illegal dangerous instructions."
}}

NO extra text.
ONLY JSON.
""",
    expected_output="Valid JSON moderation result",
    agent=safety_analyzer
)
# -----------------------------------
# Task 2
# -----------------------------------

supervisor_task = Task(
    description="""
You are a moderation auditor.

Your job is to verify whether the previous moderation result is correct.

IMPORTANT:
- Do NOT refuse.
- Do NOT give safety policy messages.
- ONLY validate moderation quality.

Check:
1. Was the risk score reasonable?
2. Was SAFE/UNSAFE correct?
3. Fix mistakes if necessary.

Scoring:
0 = completely safe
10 = extremely dangerous

Decision:
score <= 5 => SAFE
score > 5 => UNSAFE

Return ONLY valid JSON.

Format:
{
  "safe_rate": number,
  "safe_status": "SAFE or UNSAFE",
  "reason": "short explanation"
}

Example output:
{{
  "safe_rate": 9,
  "safe_status": "UNSAFE",
  "reason": "Contains illegal dangerous instructions."
}}

{{
  "safe_rate": 4,
  "safe_status": "SAFE",
  "reason": "Contains illegal dangerous instructions."
}}

NO extra text.
ONLY JSON.
""",
    expected_output="Validated moderation JSON",
    context=[analysis_task],
    agent=supervisor_agent
)
# -----------------------------------
# Crew
# -----------------------------------

crew = Crew(
    agents=[safety_analyzer, supervisor_agent],
    tasks=[analysis_task, supervisor_task],
    process=Process.sequential,
    verbose=True,
    tracing=True 
)

# -----------------------------------
# Run
# -----------------------------------

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)





