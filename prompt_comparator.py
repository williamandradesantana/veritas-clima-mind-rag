from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

prompts = [
    "While this query is not directly tied to data from WeatherBench or weather forecasting, it's known that changes in atmospheric pressure can impact physical well-being and may indirectly affect mental health. Further research would be needed to establish a direct link between these factors using meteorological observations.",

    "This question involves understanding how changes in weather patterns, which can influence atmospheric conditions such as sunlight exposure and temperature, may impact mental health conditions that have a temporal pattern.",

    "The ability to predict extreme weather accurately could potentially reduce anxiety or stress related to upcoming severe climatic changes by allowing individuals more time to prepare and mitigate potential impacts, thereby improving overall mental health during such times.",
    
    "Changes in weather due to global warming can disrupt the dispersal of pollen and other particles, potentially impacting individuals with respiratory or allergy conditions. Research into this would involve examining correlations between long-term meteorological patterns and health records related to allergic reactions and asthma attacks within a given region over time.",
]

prompt_base = "How are mental health and climate change related?"

embeddings = model.encode(prompts, convert_to_tensor=True)
embedding_base = model.encode(prompt_base, convert_to_tensor=True)

similarities = util.cos_sim(embedding_base, embeddings)[0]

result = list(zip(prompts, similarities.tolist()))
sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

for text, score in sorted_result:
    print(f"Score: {score:.4f} | Prompt: {text}")
