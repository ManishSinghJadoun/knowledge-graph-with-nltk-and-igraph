import nltk
import igraph as ig
import matplotlib.pyplot as plt
from langchain.document_loaders import PyPDFLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# Ensure necessary resources are downloaded
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")


# Function to extract named entities using NLTK
def extract_named_entities(text):
    sentences = sent_tokenize(text)
    entities = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        chunked_tree = ne_chunk(tagged_words)

        for subtree in chunked_tree:
            if isinstance(subtree, Tree):
                entity_name = " ".join([token for token, pos in subtree.leaves()])
                entity_type = subtree.label()
                entities.append((entity_name, entity_type))
    
    return entities


# Load document using LangChain
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Extract text from all pages
    full_text = " ".join([page.page_content for page in pages])
    return full_text


# Construct a knowledge graph using igraph
def build_knowledge_graph(entities):
    g = ig.Graph(directed=True)
    
    # Add unique entities as nodes
    unique_entities = list(set(entities))
    g.add_vertices(len(unique_entities))

    # Assign labels to vertices
    labels = [entity[0] for entity in unique_entities]
    g.vs["label"] = labels
    g.vs["type"] = [entity[1] for entity in unique_entities]

    # Create relationships (basic co-occurrence in the same document)
    edges = []
    for i in range(len(unique_entities)):
        for j in range(i + 1, len(unique_entities)):
            edges.append((i, j))
    
    g.add_edges(edges)

    return g


# Visualize the knowledge graph
def visualize_graph(g):
    fig, ax = plt.subplots(figsize=(40, 30))

    layout = g.layout("fr")  # Force-directed layout
    ig.plot(
        g, 
        target=ax,
        layout=layout,
        vertex_label=g.vs["label"],
        vertex_color=["lightblue" if t == "PERSON" else "lightgreen" for t in g.vs["type"]],
        edge_color="gray"
    )
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Provide the path to your PDF document
    pdf_path = "D:\\pcbprojecta\\An_AOI_algorithm_for_PCB_based_on_feature_extraction.pdf"

    # Load and process document
    document_text = load_document(pdf_path)

    # Extract named entities
    named_entities = extract_named_entities(document_text)

    # Build knowledge graph
    graph = build_knowledge_graph(named_entities)

    # Visualize the graph
    visualize_graph(graph)
