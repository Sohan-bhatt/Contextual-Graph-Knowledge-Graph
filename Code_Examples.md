# Knowledge Graph - Code Examples

## Example 1: Simple Knowledge Graph with LlamaIndex (Easiest)

```python
# install: pip install llama-index llama-index-llms-openai

from llama_index import SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index import KnowledgeGraphIndex

# 1. Load documents
documents = SimpleDirectoryReader("./docs").load_data()

# 2. Setup LLM
llm = OpenAI(model="gpt-4")

# 3. Build Knowledge Graph
# Note: Requires Neo4j or use in-memory for testing
# For in-memory: skip graph_store parameter

index = KnowledgeGraphIndex.from_documents(
    documents,
    llm=llm,
    max_triplets_per_chunk=5
)

# 4. Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the company's vacation policy?")

print(response)
```

---

## Example 2: Microsoft GraphRAG (Most Powerful)

```bash
# Step 1: Install
pip install graphrag

# Step 2: Initialize
mkdir my-rag && cd my-rag
python -m graphrag.index --init --root .

# Step 3: Add your documents to ./input folder
echo "Your policy text here" > input/policy.txt

# Step 4: Run indexing
python -m graphrag.index --root .

# Step 5: Query
python -m graphrag.query --root . --query "What is the vacation policy?"
```

---

## Example 3: Neo4j + LangChain (Custom Graphs)

```python
# install: pip install langchain langchain-community neo4j

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

# 1. Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# 2. Setup LLM
llm = ChatOpenAI(model="gpt-4")

# 3. Transform documents to graph
graph_transformer = LLMGraphTransformer(llm=llm)

# 4. Load and convert documents
from langchain.document_loaders import TextLoader
loader = TextLoader("policy.txt")
docs = loader.load()

graph_documents = graph_transformer.convert_to_graph_documents(docs)

# 5. Store in Neo4j
graph.add_graph_documents(graph_documents)

# 6. Query
result = graph.query("""
MATCH (p:Policy)
WHERE p.name CONTAINS 'vacation'
RETURN p
""")
```

---

## Example 4: Contextual Retrieval (Anthropic-style)

```python
# install: pip install llama-index llama-index-llms-anthropic

from llama_index import SimpleDirectoryReader
_parser import SentenceSplfrom llama_index.nodeitter
from llama_index.llms import Anthropic
from llama_index import VectorStoreIndex

# 1. Setup
llm = Anthropic(model="claude-3-5-sonnet")

# 2. Load and split documents
docs = SimpleDirectoryReader("./docs").load_data()
splitter = SentenceSplitter(chunk_size=500)
nodes = splitter.get_nodes_from_documents(docs)

# 3. Add context to each chunk
context_prompt = """Add brief context for this chunk within the document:
{chunk}"""

for node in nodes:
    prompt = context_prompt.format(chunk=node.text[:200])
    context = llm.complete(prompt)
    node.metadata["context"] = context.text
    node.text = f"[Context: {context.text}] {node.text}"

# 4. Build index
index = VectorStoreIndex.from_nodes(nodes)

# 5. Query
query_engine = index.as_query_engine()
response = query_engine.query("Vacation policy details?")
```

---

## Example 5: PageIndex (Tree Index - Highest Accuracy)

```python
# install: pip install pageindex

from pageindex import page_index

# 1. Generate tree index from PDF
result = page_index(
    "long_document.pdf",
    model="gpt-4o",
    if_add_node_id="yes",
    if_add_node_summary="yes"
)

tree = result['structure']

# 2. Tree search function
def tree_search(query, tree):
    prompt = f"""
    Given query: {query}
    And document tree: {tree}
    
    Find relevant nodes and return their IDs.
    Return JSON: {{"thinking": "...", "node_list": ["001", "002"]}}
    """
    # Use LLM to find relevant nodes
    # Then retrieve content from those pages
    return relevant_nodes

# 3. Use for retrieval
relevant = tree_search("What was Q3 revenue?", tree)
```

---

## Example 6: Hybrid Search (Vector + Graph)

```python
from llama_index import VectorStoreIndex
from llama_index.retrievers import KnowledgeGraphRAGRetriever
from llama_index import QueryEngine

# 1. Vector index
vector_index = VectorStoreIndex.from_documents(documents)

# 2. Knowledge graph retriever
kg_retriever = KnowledgeGraphRAGRetriever(
    graph_store=graph_store,
    llm=llm,
    retriever_mode="keyword"  # or "embedding"
)

# 3. Combined query engine
query_engine = QueryEngine.from_args(
    retriever=kg_retriever,
    llm=llm,
    vector_index=vector_index  # Optional hybrid
)

# 4. Query - automatically uses both
response = query_engine.query("What is the approval process?")
```

---

## Example 7: Your Current Architecture Enhancement

```python
# How to add KG to your existing Mr. Elite system

from llama_index import KnowledgeGraphIndex

# In your Phase 1 classification:
async def classify_intent(message, context):
    classification = await run_classifier(message)
    
    if classification.intent == "QUERY":
        # Use Knowledge Graph for RAG-style queries
        kg_response = kg_index.query(message)
        return {"type": "RAG_RESPONSE", "content": kg_response}
    
    elif classification.intent in ["TASK", "GOAL"]:
        # Use your existing orchestration
        result = await run_mr_elite(message, context)
        return {"type": "ORCHESTRATION", "content": result}
    
    else:
        # Chat intent
        return {"type": "CHAT", "content": "Hello!"}
```

---

## Minimum Working Examples

### Minimal LlamaIndex KG (3 lines to index):

```python
from llama_index import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

docs = SimpleDirectoryReader("docs").load_data()
index = KnowledgeGraphIndex.from_documents(docs, llm=OpenAI())
print(index.as_query_engine().query("Your question"))
```

### Minimal GraphRAG:

```bash
pip install graphrag
mkdir input && echo "text" > input/x.txt
python -m graphrag.index --root .
python -m graphrag.query --root . --query "question"
```

---

## Quick Setup for Testing

```bash
# Create test environment
python -m venv kg-test
source kg-test/bin/activate

# Install options:

# Option A: LlamaIndex (easiest)
pip install llama-index llama-index-llms-openai

# Option B: GraphRAG
pip install graphrag

# Option C: Neo4j
pip install neo4j langchain

# Option D: PageIndex
pip install pageindex
```
