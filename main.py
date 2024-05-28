import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear, Conv1d, BatchNorm2d
import tinygrad.optim as optim

# Custom ReLU function
def ReLU(x):
    return x.maximum(Tensor(0.0))

# Custom Softmax function
def Softmax(x):
    exp_x = x.exp()
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Custom MaxPool1d class
class MaxPool1d:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def __call__(self, x):
        n, c, w = x.shape
        out_width = (w - self.kernel_size) // self.stride + 1
        out = np.zeros((n, c, out_width))

        for i in range(out_width):
            start = i * self.stride
            end = start + self.kernel_size
            out[:, :, i] = x[:, :, start:end].max(axis=2)
        
        return Tensor(out)

class Retriever:
    def __init__(self, data_store, embedding_dim=300, hidden_dim=256, num_layers=3, conv_channels=128, conv_kernel_size=3, pool_kernel_size=2):
        self.data_store = data_store
        self.vocab = self._build_vocab(data_store)
        self.embedding = Embedding(len(self.vocab), embedding_dim)
        self.conv_layers = []
        for _ in range(num_layers):
            self.conv_layers.append(Conv1d(embedding_dim, conv_channels, conv_kernel_size, stride=1, padding=conv_kernel_size//2))
            self.conv_layers.append(ReLU)
            self.conv_layers.append(MaxPool1d(pool_kernel_size))
            self.conv_layers.append(BatchNorm2d(conv_channels))
            embedding_dim = conv_channels
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(Linear(embedding_dim, hidden_dim))
            self.encoder_layers.append(BatchNorm2d(hidden_dim))
            self.encoder_layers.append(ReLU)
            embedding_dim = hidden_dim
        self.attention = Linear(hidden_dim, 1)
        self.softmax = Softmax
    
    def _build_vocab(self, data_store):
        vocab = {}
        for doc in data_store:
            for word in doc.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def _encode(self, text):
        indices = [self.vocab.get(word, 0) for word in text.lower().split()]
        embeddings = self.embedding(Tensor(indices)).unsqueeze(0).unsqueeze(0)
        hidden = embeddings
        for layer in self.conv_layers:
            hidden = layer(hidden)
        hidden = hidden.squeeze(2)
        for layer in self.encoder_layers:
            hidden = layer(hidden)
        attention_scores = self.softmax(self.attention(hidden))
        context_vector = (attention_scores * hidden).sum(axis=1)
        return context_vector

    def parameters(self):
        params = []
        for layer in self.conv_layers + self.encoder_layers + [self.attention, self.embedding]:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
    
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
    
    def train(self, queries, relevant_docs, num_epochs=10, batch_size=32, learning_rate=0.001):
        optimizer = tinygrad.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = tinygrad.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_indices = np.arange(len(queries))
            np.random.shuffle(batch_indices)
            
            for start_idx in range(0, len(queries), batch_size):
                batch_indices_subset = batch_indices[start_idx:start_idx+batch_size]
                batch_queries = [queries[i] for i in batch_indices_subset]
                batch_docs = [relevant_docs[i] for i in batch_indices_subset]
                
                query_vectors = [self._encode(query) for query in batch_queries]
                doc_vectors = [self._encode(doc) for doc in batch_docs]
                
                scores = Tensor([])
                for query_vector, doc_vector in zip(query_vectors, doc_vectors):
                    score = (query_vector * doc_vector).sum()
                    scores = scores.cat((scores, score.unsqueeze(0)), dim=0)
                
                target = Tensor([0] * len(scores)).long()
                loss = criterion(scores, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.data
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(queries):.4f}")
    
    def evaluate(self, queries, relevant_docs, k=5):
        num_correct = 0
        for query, doc in zip(queries, relevant_docs):
            retrieved_docs = self.search(query, k)
            if doc in retrieved_docs:
                num_correct += 1
        accuracy = num_correct / len(queries)
        return accuracy

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

retriever = Retriever(data_store, embedding_dim=300, hidden_dim=256, num_layers=3, conv_channels=128, conv_kernel_size=3, pool_kernel_size=2)

# Training
train_queries = [
    "What is Retrieval-Augmented Generation?",
    "How does the retrieval component work in RAG?",
    "What are some applications of RAG systems?",
    "Explain query-based RAG.",
    "What is latent-based RAG?",
    "How does logit-based RAG work?",
    "What are some advanced retrieval techniques used in RAG?",
    "What is the role of the generator in RAG systems?",
    "How effective are RAG systems in improving generated content?",
    "Is retrieval-augmented generation an active research area?"
]
train_docs = [
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
retriever.train(train_queries, train_docs, num_epochs=10, batch_size=4, learning_rate=0.001)

# Evaluation
eval_queries = [
    "What are the key components of RAG systems?",
    "How does retrieval enhance the performance of generative models?",
    "Can RAG be used for code generation?",
    "What happens to the retrieved documents in query-based RAG?",
    "How are the representations combined in latent-based RAG?"
]
eval_docs = [
    "Retrieval-Augmented Generation (RAG) systems combine retrieval and generation components for enhanced performance.",
    "The retrieval component in RAG systems searches for relevant information in a given data store based on the input query.",
    "RAG systems can be applied to various domains, including text generation, image captioning, and code completion.",
    "In query-based RAG, the retrieved documents are concatenated with the original query before being fed to the generator.",
    "Latent-based RAG approaches combine the latent representations of the query and retrieved documents for generation."
]
accuracy = retriever.evaluate(eval_queries, eval_docs, k=3)
print(f"Retrieval Accuracy: {accuracy:.4f}")

# Inference
query = "How does the retriever in RAG systems work?"
relevant_documents = retriever.search(query, k=3)

print("Query:", query)
print("Relevant documents:")
for doc in relevant_documents:
    print("- ", doc)
