import requests
import time
import json

API_URL = "http://127.0.0.1:8000/api/v1/retrieval"

questions = [
    "What is supervised learning?",
    "Explain unsupervised learning.",
    "What is reinforcement learning?",
    "How does a decision tree work?",
    "What is overfitting in machine learning?",
    "How can overfitting be prevented?",
    "What is underfitting?",
    "Explain the bias-variance tradeoff.",
    "What are activation functions?",
    "Describe the sigmoid activation function.",
    "What is the ReLU activation function?",
    "What is a neural network?",
    "How does backpropagation work?",
    "What is gradient descent?",
    "Explain stochastic gradient descent.",
    "What are epochs in training?",
    "What is a loss function?",
    "Define cross-entropy loss.",
    "What is mean squared error?",
    "What is a support vector machine (SVM)?",
    "How do kernels work in SVM?",
    "What is clustering?",
    "Explain k-means clustering.",
    "What is hierarchical clustering?",
    "What is dimensionality reduction?",
    "Describe principal component analysis (PCA).",
    "What is t-SNE?",
    "What is feature scaling?",
    "Explain normalization vs standardization.",
    "What is regularization?",
    "Describe L1 regularization (Lasso).",
    "Describe L2 regularization (Ridge).",
    "What is dropout in neural networks?",
    "What is batch normalization?",
    "Explain convolutional neural networks (CNNs).",
    "What is pooling in CNNs?",
    "What is a recurrent neural network (RNN)?",
    "What is an LSTM?",
    "Explain the vanishing gradient problem.",
    "What is an embedding in NLP?",
    "Describe word2vec.",
    "What is transfer learning?",
    "What is fine-tuning?",
    "Explain attention mechanisms.",
    "What is the Transformer architecture?",
    "Describe BERT model.",
    "What is GPT?",
    "What is the difference between classification and regression?",
    "What is the confusion matrix?",
    "Define precision, recall, and F1 score.",
    "What is ROC curve?",
    "Explain AUC.",
    "What is cross-validation?",
    "What is the curse of dimensionality?",
    "What is ensemble learning?",
    "Describe bagging.",
    "Describe boosting.",
    "What is random forest?",
    "What is XGBoost?",
    "Explain the difference between parametric and non-parametric models.",
    "What is a generative adversarial network (GAN)?",
    "What is reinforcement learning policy?",
    "Explain Q-learning.",
    "What is the exploration vs exploitation dilemma?",
    "What is the Markov decision process?",
    "What are autoencoders?",
    "What is unsupervised pretraining?",
    "What is the difference between AI, ML, and Deep Learning?",
    "Explain the concept of feature engineering.",
    "What is a confusion matrix?",
    "What is data augmentation?",
    "What is early stopping?",
    "What are embeddings used for?",
    "Explain how k-nearest neighbors (KNN) works.",
    "What is the role of a validation set?",
    "What is data leakage?",
    "What is a hyperparameter?",
    "Explain grid search.",
    "Explain random search.",
    "What is batch size?",
    "What is learning rate?",
    "What is an optimizer?",
    "Describe the Adam optimizer.",
    "What is the difference between classification and clustering?",
    "What is the elbow method in clustering?",
    "What is a confusion matrix?",
    "What is meant by model interpretability?",
    "What is explainable AI (XAI)?",
    "What is a time series forecasting?",
    "Explain the difference between stationary and non-stationary time series.",
    "What is a Markov chain?",
    "What is a hidden Markov model?",
    "What is a probabilistic graphical model?",
    "What is reinforcement learning reward function?",
    "Explain the difference between batch and online learning.",
    "What is a precision-recall tradeoff?",
    "What is the difference between sample and population?",
    "What is hypothesis testing?",
    "What is p-value?",
    "What is bootstrapping?",
    "What is a confusion matrix?"
]

results = []          # For timing, errors, length etc.
outputs = []          # For detailed outputs: question, params, and full API response content

for query in questions:
    params = {
        "query": query,
        "top_k": 2,
        "max_hop": 2
    }
    try:
        start_time = time.time()
        response = requests.post(API_URL, params=params, data="")
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            results.append({
                "query": query,
                "retrieval_time_seconds": round(elapsed, 4),
                "response_length": len(response.content)
            })
            outputs.append({
                "query": query,
                "top_k": params["top_k"],
                "max_hop": params["max_hop"],
                "response": response.json()  # save JSON response here
            })
        else:
            results.append({
                "query": query,
                "retrieval_time_seconds": round(elapsed, 4),
                "error": f"HTTP {response.status_code}"
            })
            outputs.append({
                "query": query,
                "top_k": params["top_k"],
                "max_hop": params["max_hop"],
                "error": f"HTTP {response.status_code}"
            })
    except Exception as e:
        results.append({
            "query": query,
            "retrieval_time_seconds": None,
            "error": str(e)
        })
        outputs.append({
            "query": query,
            "top_k": params["top_k"],
            "max_hop": params["max_hop"],
            "error": str(e)
        })

with open("retrieval_times.json", "w") as f:
    json.dump(results, f, indent=2)

with open("retrieval_outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Done! Results saved to retrieval_times.json and retrieval_outputs.json")

# import json

# with open("retrieval_times.json", "r") as f:
#     results = json.load(f)

# # Filter out any results that have errors or no time recorded
# times = [r["retrieval_time_seconds"] for r in results if r.get("retrieval_time_seconds") is not None]

# if times:
#     average_time = sum(times) / len(times)
#     print(f"Average retrieval time: {average_time:.4f} seconds")
# else:
#     print("No valid retrieval times found.")
