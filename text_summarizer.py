
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Input text
text = """
  Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines, especially computer systems.
These processes include learning, reasoning, problem-solving, perception, and language understanding.
AI systems are typically powered by algorithms and models, such as machine learning, which allow 
computers to learn from data and improve their performance over time without being explicitly programmed.
The foundation of AI lies in the concept of creating systems that can replicate or even exceed human cognitive
abilities, enabling them to perform complex tasks like decision-making, speech recognition, image processing, and more. 
Machine learning, a subfield of AI, involves training models on vast amounts of data to recognize patterns and make 
predictions. Deep learning, a subset of machine learning, utilizes neural networks to model intricate data relationships,
contributing to breakthroughs in areas like computer vision and natural language processing. AI has made remarkable 
advancements in a variety of industries, ranging from healthcare, where it helps diagnose diseases, to autonomous driving,
where self-driving cars leverage AI to navigate the roads. The ability of AI to analyze massive datasets quickly and 
efficiently is transforming sectors like finance, manufacturing, and entertainment. However, the rapid development of AI 
has also sparked debates about its ethical implications, such as concerns over job displacement, privacy, security, and the
potential for AI systems to be biased or uncontrollable. While AI holds great promise in addressing global challenges, ensuring 
its responsible and transparent development is crucial to avoid unintended consequences. As AI continues to evolve, its integration
into daily life is expected to deepen, offering new opportunities for innovation and enhancing human capabilities. However, the question
of how society will adapt to this technological transformation remains a key topic for discussion, as AI has the potential to 
revolutionize everything from education to governance.
"""

# Generate the summary
summary = summarizer(text, max_length=60, min_length=50, do_sample=False)

# Display the summary
print("Original Text:")
print(text)
print("\nSummarized Text:")
print(summary[0]['summary_text'])
