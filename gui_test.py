from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import time

# Load GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"  # Ensure you have the correct model files
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load Sentence Transformer for document embedding and search
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents for testing
docs = [
    "Albert Einstein developed the theory of relativity.",
    "World War II started in 1939 and ended in 1945.",
    "The Pythagorean theorem states that a² + b² = c² in a right triangle.",
    "Atoms are the basic units of matter and the defining structure of elements.",
    "Water can often be found near vegetation or in depressions in the ground."
]

# Encode documents for FAISS search
embeddings = st_model.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to search documents
def search_docs(query):
    query_embedding = st_model.encode([query])
    distances, indices = index.search(query_embedding, k=1)
    return docs[indices[0][0]]

# Function to generate a response using GPT-J
def generate_response(context, query, timeout=10):
    try:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Start a timer for the timeout
        start_time = time.time()
        output = model.generate(input_ids, max_length=100, temperature=0.7, do_sample=True)
        elapsed_time = time.time() - start_time
        
        # Check if response took too long
        if elapsed_time > timeout:
            return "Response generation timed out. Please try again."
        
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating response: {e}"

# Function to update the status label in the GUI
def update_status(label, text, delay=0):
    label.config(text=text)
    label.update()
    time.sleep(delay)

# Function to handle user queries in the GUI
def ask_question(status_label):
    query = simpledialog.askstring("Ask a Question", "What would you like to know?")
    if query:
        try:
            # Update the status to indicate processing
            update_status(status_label, "Searching for relevant documents...", 0)
            
            # Step 1: Search for the most relevant document
            best_match = search_docs(query)
            update_status(status_label, "Generating response with GPT-J...", 0)
            
            # Step 2: Generate a response using GPT-J
            response = generate_response(best_match, query, timeout=10)
            
            # Update the status to done
            update_status(status_label, "Done!", 1)
            
            # Display the response in a messagebox
            messagebox.showinfo("AI Response", f"Context: {best_match}\n\nResponse: {response}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            # Reset the status label
            status_label.config(text="Ready")

# Create the GUI
root = tk.Tk()
root.title("Survival Guide AI")
root.geometry("400x250")

# Status label to show processing updates
status_label = tk.Label(root, text="Ready", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)

# Create a button for asking questions
ask_button = tk.Button(root, text="Ask a Question", command=lambda: threading.Thread(target=ask_question, args=(status_label,)).start(), width=20, height=2)
ask_button.pack(pady=50)

# Run the GUI loop
root.mainloop()
