import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear, ReLU, Softmax, BatchNorm1d, Dropout

class Retriever:
    def __init__(self, data_store, embedding_dim=200, hidden_dim=128, num_layers=2, dropout=0.2):
        self.data_store = data_store
        self.vocab = self._build_vocab(data_store)
        self.embedding = Embedding(len(self.vocab), embedding_dim)
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(Linear(embedding_dim, hidden_dim))
            self.encoder_layers.append(BatchNorm1d(hidden_dim))
            self.encoder_layers.append(ReLU())
            self.encoder_layers.append(Dropout(dropout))
            embedding_dim = hidden_dim
        self.attention = Linear(hidden_dim, 1)
        self.softmax = Softmax()
    
    def _build_vocab(self, data_store):
        vocab = {}
        for doc in data_store:
            for word in doc.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def _encode(self, text):
        indices = [self.vocab.get(word, 0) for word in text.lower().split()]
        embeddings = self.embedding(Tensor(indices))
        hidden = embeddings
        for layer in self.encoder_layers:
            hidden = layer(hidden)
        attention_scores = self.softmax(self.attention(hidden))
        context_vector = (attention_scores * hidden).sum(axis=0)
        return context_vector
    
    def search(self, query, k=5):
        query_vector = self._encode(query)
        doc_scores = []
        for doc in self.data_store:
            doc_vector = self._encode(doc)
            score = (query_vector * doc_vector).sum()
            doc_scores.append(score.data)
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        top_k_documents = [self.data_store[i] for i in top_k_indices]
        return top_k_documents

# Example usage
data_store = [
    "Retrieval-Augmented Generation (RAG) systems combine retrieval and generation components for enhanced performance.",
    "The retrieval component in RAG systems searches for relevant information in a given data store based on the input query.",
    "RAG systems can be applied to various domains, including text generation, image captioning, and code completion.",
    "In query-based RAG, the retrieved documents are concatenated with the original query before being fed to the generator.",
    "Latent-based RAG approaches combine the latent representations of the query and retrieved documents for generation.",
    "Logit-based RAG systems combine the likelihood scores from the generator and the retrieved documents during decoding.",
    "Advanced retrieval techniques in RAG systems include dense vector representations, attention mechanisms, and neural networks.",
    "The generator in RAG systems produces the final output by conditioning on the retrieved information and the original query.",
    "RAG systems have shown promising results in improving the quality and informativeness of generated content.",
    "Retrieval-augmented generation is an active area of research, with ongoing efforts to enhance retrieval and generation methods."
]

retriever = Retriever(data_store, embedding_dim=200, hidden_dim=128, num_layers=2, dropout=0.2)

query = "How does the retriever in RAG systems work?"
relevant_documents = retriever.search(query, k=3)

print("Query:", query)
print("Relevant documents:")
for doc in relevant_documents:
    print("- ", doc)