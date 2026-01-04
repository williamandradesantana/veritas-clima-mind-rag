from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

prompts = [
    "Extreme weather conditions may lead to insomnia and nightmares which could potentially contribute to depression and PTSD, according to scientific evidence related to mental health and climate change effects on individuals.",

    "Severe weather-related events can result in a significant need for enhanced long-term mental health services due to the profound psychological distress they cause, as suggested by discussions surrounding climate change and its effects.",

    "A person’s comprehension of terms like climate context is shaped largely based on climatic variables they frequently encounter in their locality. Their interpretation of how these elements influence their psychological state relies heavily on personal experiences and observations, as indicated by discussions among various individuals across different cultures.",
    
    "Recent research has taken into account daily or weekly variations in factors like day-time temperature, night-time temperature, rainfall, sunshine hours and cloud cover when examining their potential effects on psychological wellbeing.",
]

prompt_base = "How are mental health and climate change related?"

embeddings = model.encode(prompts, convert_to_tensor=True)
embedding_base = model.encode(prompt_base, convert_to_tensor=True)

similarities = util.cos_sim(embedding_base, embeddings)[0]

result = list(zip(prompts, similarities.tolist()))
sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

for text, score in sorted_result:
    print(f"Score: {score:.4f} | Prompt: {text}")
