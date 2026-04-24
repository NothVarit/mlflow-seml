import requests

GOLDEN_SET = [
    {
        "text": "I built a machine learning model using Python and deployed it as a startup product for healthcare data science.",
        "expected_tags": ["Python", "Machine Learning", "Data Science", "Startup", "Health"]
    },
    {
        "text": "Investing in Bitcoin and Ethereum blockchain technology is changing the future of business and entrepreneurship.",
        "expected_tags": ["Cryptocurrency", "Blockchain", "Business", "Entrepreneurship"]
    },
    {
        "text": "My mental health journey taught me life lessons about self improvement and personal development through writing.",
        "expected_tags": ["Mental Health", "Life", "Self Improvement", "Personal Development", "Writing"]
    },
    {
        "text": "JavaScript and Python are essential programming skills for software development and artificial intelligence.",
        "expected_tags": ["JavaScript", "Python", "Programming", "Software Development", "Artificial Intelligence"]
    },
    {
        "text": "Marketing strategies for growing your startup business through design and entrepreneurship education.",
        "expected_tags": ["Marketing", "Startup", "Business", "Design", "Entrepreneurship", "Education"]
    },
]

def run_golden_test():
    passed = 0
    failed = []

    for i, case in enumerate(GOLDEN_SET):
        response = requests.post(
            "http://localhost:8000/predict",
            json={"inputs": case["text"]}
        )
        predicted_tags = response.json()["predictions"][0]

        hits = [tag for tag in case["expected_tags"] if tag in predicted_tags]
        missed = [tag for tag in case["expected_tags"] if tag not in predicted_tags]
        match_count = len(hits)
        total = len(case["expected_tags"])

        print(f"\nCase {i+1}: match {match_count}/{total}")
        print(f"  predicted : {predicted_tags}")
        print(f"  matched   : {hits}")
        print(f"  missed    : {missed}")

        if match_count > 0:
            passed += 1
        else:
            failed.append(case)

    print(f"\nGolden Test: {passed}/{len(GOLDEN_SET)} passed")

    if failed:
        print("\n❌ Failed cases:")
        for f in failed:
            print(f"  text: {f['text'][:60]}")
        return False

    print("✅ All passed → safe to promote")
    return True

if __name__ == "__main__":
    run_golden_test()