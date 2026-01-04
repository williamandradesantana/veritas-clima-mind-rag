from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

prompts = [
    "Seasonal changes have been known to influence moods, but specific effects on those with anxiety disorders can vary greatly depending on personal sensitivities and individual circumstances. Some may find warmer temperatures calming while others might feel more anxious due to the heat or longer daylight hours associated with summer seasons leading to potential social pressure for outdoor activities.",

    "Fluctuating night time temperatures can lead to disturbed sleep patterns and exacerbate existing conditions in individuals suffering from insomnia or other sleep disorders, although responses may differ among those sensitive to cold weather. Cooler temperatures are generally more conducive for good quality sleep.",

    "Different people respond differently; some might find ample daylight uplifting and beneficial, while others may feel overwhelmed by too much exposure leading to increased lethargy or even seasonal affective disorder. The balance of sunlight is crucial for maintaining mood stability in individuals with depression.",
    
    "Heavy and prolonged periods of rain can lead to feelings of melancholy or seasonal affective disorder due to lack of sunlight, but effects may differ among individuals sensitive to changes in weather patterns.",
]

prompt_base = "How are mental health and climate change related?"

embeddings = model.encode(prompts, convert_to_tensor=True)
embedding_base = model.encode(prompt_base, convert_to_tensor=True)

similarities = util.cos_sim(embedding_base, embeddings)[0]

result = list(zip(prompts, similarities.tolist()))
sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

for text, score in sorted_result:
    print(f"Score: {score:.4f} | Prompt: {text}")
