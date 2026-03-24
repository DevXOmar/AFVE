import requests
import uuid  # NEW

# ============================
# CONFIG
# ============================
AFVE_URL = "http://127.0.0.1:8000/verify"

# ============================
# FAKE GENERATION FUNCTION
# ============================
def generate_answer(prompt):
    fake_db = {
        "Who developed the theory of relativity?":
            "Albert Einstein developed the theory of relativity in 1905.",
        
        "Tell me about Einstein's Nobel Prize":
            "Einstein won the Nobel Prize for relativity in 1921.",
        
        "What are the benefits of drinking water?":
            "Drinking water helps maintain hydration, improves digestion, and regulates body temperature.",
        
        "Who discovered gravity and when?":
            "Isaac Newton discovered gravity in 1687 when an apple fell on his head.",
        
        "Explain quantum mechanics in simple terms":
            "Quantum mechanics explains how particles behave at very small scales, where they can exist in multiple states at once."
    }
    
    return fake_db.get(
        prompt,
        "Einstein discovered gravity and invented electricity in 1800."
    )

# ============================
# AFVE CALL
# ============================
def verify_with_afve(prompt, answer):
    payload = {
        "job_id": str(uuid.uuid4()),  # ✅ FIXED
        "prompt": prompt,
        "model_output": answer
    }
    
    response = requests.post(AFVE_URL, json=payload)
    
    print("DEBUG STATUS:", response.status_code)
    print("DEBUG RESPONSE:", response.text)
    
    try:
        return response.json()
    except:
        return {}

# ============================
# MAIN LOOP
# ============================
def run_test(prompt):
    print("\n" + "="*60)
    print(f"🧑 USER PROMPT:\n{prompt}")
    
    answer = generate_answer(prompt)
    print("\n🤖 MODEL ANSWER:\n", answer)
    
    print("\n🔍 Running AFVE verification...")
    result = verify_with_afve(prompt, answer)
    
    print("\n📊 AFVE RESULT:")
    print(f"Overall Score: {result.get('overall_score')}\n")
    
    if result.get("claims"):
        for claim in result["claims"]:
            print(f"• Claim: {claim['claim']}")
            print(f"  Score: {claim['score']} ({claim['label']})")
            print(f"  Explanation: {claim['explanation']}")
            print("-"*40)
    else:
        print("No claims detected or invalid input.")

# ============================
# TEST CASES
# ============================
if __name__ == "__main__":
    
    test_prompts = [
        "Who developed the theory of relativity?",
        "Tell me about Einstein's Nobel Prize",
        "What are the benefits of drinking water?",
        "Who discovered gravity and when?",
        "Explain quantum mechanics in simple terms",
        "Random unknown prompt"
    ]
    
    for prompt in test_prompts:
        run_test(prompt)