import requests
import uuid

# ============================
# CONFIG
# ============================
AFVE_URL = "http://127.0.0.1:8000/verify"

# ============================
# FAKE GENERATION FUNCTION
# ============================
def generate_answer(prompt):
    fake_db = {
        # --- Original Prompts ---
        "Who developed the theory of relativity?":
            "Albert Einstein developed the theory of relativity in 1905.",
        
        "Tell me about Einstein's Nobel Prize":
            "Einstein won the Nobel Prize for relativity in 1921.", # AFVE should flag: Won for photoelectric effect
        
        "What are the benefits of drinking water?":
            "Drinking water helps maintain hydration, improves digestion, and regulates body temperature.",
        
        "Who discovered gravity and when?":
            "Isaac Newton discovered gravity in 1687 when an apple fell on his head.", # AFVE should flag: Apple myth
        
        "Explain quantum mechanics in simple terms":
            "Quantum mechanics explains how particles behave at very small scales, where they can exist in multiple states at once.",

        # --- New Prompts (Mix of True, False, and Myths) ---
        "What is the capital of Australia?":
            "The capital of Australia is Sydney, which was established in 1788.", # False: It's Canberra
            
        "How many bones are in the adult human body?":
            "The adult human body typically has 206 bones, though babies are born with over 300.", # True
            
        "Who wrote Romeo and Juliet?":
            "William Shakespeare wrote Romeo and Juliet in the late 1500s.", # True
            
        "Tell me about the Great Wall of China.":
            "The Great Wall of China is a massive structure built to protect against invasions, and it is the only man-made object visible from space with the naked eye.", # Myth: Not visible from space
            
        "When did the Titanic sink?":
            "The RMS Titanic sank on April 15, 1912, after hitting an iceberg in the Atlantic Ocean.", # True
            
        "Who was the first person to walk on the moon?":
            "Lance Armstrong was the first person to walk on the moon in 1969.", # False: Neil Armstrong
            
        "What is the chemical formula for table salt?":
            "The chemical formula for table salt is NaCl, which stands for sodium chloride.", # True
            
        "How many states are in the United States?":
            "There are 52 states in the United States of America, including Puerto Rico and Washington D.C.", # False: 50 states
            
        "What causes the seasons on Earth?":
            "Earth's seasons are caused by its varying distance from the Sun as it orbits in an elliptical path.", # False/Myth: Caused by axial tilt
            
        "Who painted the Mona Lisa?":
            "Leonardo da Vinci painted the Mona Lisa during the Italian Renaissance.", # True
            
        "What is the tallest mountain in the world?":
            "Mount Everest is the tallest mountain in the world above sea level, located in the Himalayas.", # True
            
        "Did Thomas Edison invent the lightbulb?":
            "Yes, Thomas Edison was the sole inventor of the incandescent light bulb in 1879.", # False/Nuanced: He improved it, didn't solely invent it
            
        "What is the speed of light?":
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.", # True
            
        "Who is the author of Harry Potter?":
            "J.R.R. Tolkien is the author of the Harry Potter fantasy series.", # False: J.K. Rowling
            
        "What is the main gas in Earth's atmosphere?":
            "Oxygen makes up about 78% of the Earth's atmosphere, making it the most abundant gas.", # False: Nitrogen is 78%
            
        "When did World War II end?":
            "World War II ended in 1945 after the surrender of the Axis powers.", # True
            
        "How do vaccines work?":
            "Vaccines work by introducing a weakened or dead pathogen to stimulate the immune system to produce antibodies.", # True
            
        "Who was the first President of the United States?":
            "George Washington served as the first President of the United States from 1789 to 1797.", # True
            
        "What is the boiling point of water?":
            "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at sea level.", # True
            
        "Tell me a fact about bats.":
            "Bats are the only mammals capable of sustained flight, and they are completely blind.", # Myth: Bats are not blind
            
        "What is the currency of Japan?":
            "The currency of Japan is the Yuan.", # False: Yen
            
        "How does photosynthesis work?":
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into oxygen and glucose.", # True
            
        "Who discovered penicillin?":
            "Alexander Fleming discovered the antibiotic penicillin in 1928.", # True
            
        "What is the powerhouse of the cell?":
            "The mitochondria is known as the powerhouse of the cell because it generates most of the cell's supply of ATP.", # True
            
        "Are humans descended from chimpanzees?":
            "Yes, humans evolved directly from modern chimpanzees millions of years ago." # False: We share a common ancestor, but didn't evolve from them
    }
    
    # Fallback for anything not in the dict
    return fake_db.get(
        prompt,
        "Einstein discovered gravity and invented electricity in 1800."
    )

# ============================
# AFVE CALL
# ============================
def verify_with_afve(prompt, answer):
    payload = {
        "job_id": str(uuid.uuid4()),
        "prompt": prompt,
        "model_output": answer
    }
    
    try:
        response = requests.post(AFVE_URL, json=payload)
        
        print("DEBUG STATUS:", response.status_code)
        # Uncomment the line below if you want raw JSON output in the console
        # print("DEBUG RESPONSE:", response.text) 
        
        return response.json()
    except Exception as e:
        print(f"Error calling AFVE: {e}")
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
    print(f"Overall Score: {result.get('overall_score', 'N/A')}\n")
    
    if result.get("claims"):
        for claim in result["claims"]:
            print(f"• Claim: {claim.get('claim', 'Unknown')}")
            print(f"  Score: {claim.get('score', 'N/A')} ({claim.get('label', 'N/A')})")
            print(f"  Explanation: {claim.get('explanation', 'None provided')}")
            print("-" * 40)
    else:
        print("No claims detected or invalid input.")

# ============================
# TEST CASES
# ============================
if __name__ == "__main__":
    
    test_prompts = [
        # The Original 5
        "Who developed the theory of relativity?",
        "Tell me about Einstein's Nobel Prize",
        "What are the benefits of drinking water?",
        "Who discovered gravity and when?",
        "Explain quantum mechanics in simple terms",
        
        # The 25 New Ones
        "What is the capital of Australia?",
        "How many bones are in the adult human body?",
        "Who wrote Romeo and Juliet?",
        "Tell me about the Great Wall of China.",
        "When did the Titanic sink?",
        "Who was the first person to walk on the moon?",
        "What is the chemical formula for table salt?",
        "How many states are in the United States?",
        "What causes the seasons on Earth?",
        "Who painted the Mona Lisa?",
        "What is the tallest mountain in the world?",
        "Did Thomas Edison invent the lightbulb?",
        "What is the speed of light?",
        "Who is the author of Harry Potter?",
        "What is the main gas in Earth's atmosphere?",
        "When did World War II end?",
        "How do vaccines work?",
        "Who was the first President of the United States?",
        "What is the boiling point of water?",
        "Tell me a fact about bats.",
        "What is the currency of Japan?",
        "How does photosynthesis work?",
        "Who discovered penicillin?",
        "What is the powerhouse of the cell?",
        "Are humans descended from chimpanzees?",
        
        # Trigger the fallback
        
        "Who is the current ruler of Mars?", 
        "Random unknown prompt"
    ]
    
    for prompt in test_prompts:
        run_test(prompt)