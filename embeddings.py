
from transformers import AutoModel, AutoTokenizer
import torch

class PhoBertEmbeddings:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_embeddings(self, sentences):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.tolist()
    
    def embed_documents(self, documents):
        return self.get_embeddings(documents)
    
    def embed_query(self, query):
        return self.get_embeddings([query])[0]
    
# # Example
# embedding_function = PhoBertEmbeddings(model_name="vinai/phobert-base")

# # Example usage
# sentences = ["Đây là một câu ví dụ", "Đây là một cái khác"]
# print(sentences)
# embeddings = embedding_function.get_embeddings(sentences)
# print(embeddings)