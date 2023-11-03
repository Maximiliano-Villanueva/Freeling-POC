from llama_index.graph_stores import Neo4jGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index import ServiceContext
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.llms import OpenAI

# Neo4j database credentials
username = "neo4j"
password = "password"
url = "bolt://localhost:7687"
database = "neo4j"

# Initialize Neo4j graph store
graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

# Initialize storage context
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Initialize LLM (Language Model)
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.8, p=1)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=256)

# Initialize query engine
query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
    refresh_schema=True
)

# Perform the query
response = query_engine.query("donde esta la manzana?")
print(response)
