from flask import Flask, render_template, request, jsonify
import os
import time
from werkzeug.utils import secure_filename
from utils import process_document, chunk_text
from vector_db import VectorDB
from reranker import Reranker
from llm import LLM

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
vector_db = VectorDB()
reranker = Reranker()
llm = LLM()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and text input"""
    start_time = time.time()
    
    try:
        chunks_added = 0
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process document
                chunks = process_document(filepath, filename)
                chunks_added += vector_db.add_documents(chunks)
                
                # Clean up
                os.remove(filepath)
        
        # Handle text input
        if 'text' in request.form:
            text = request.form['text']
            if text.strip():
                # Chunk the text
                text_chunks = chunk_text(text, chunk_size=3000, overlap=450)
                
                # Format for vector DB
                chunks = []
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": "user_input",
                            "chunk_id": i,
                            "total_chunks": len(text_chunks)
                        }
                    })
                
                chunks_added += vector_db.add_documents(chunks)
        
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'chunks_added': chunks_added,
            'time': f"{elapsed:.2f}s",
            'message': f'Successfully added {chunks_added} chunks to database'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query():
    """Handle query and return answer with citations"""
    start_time = time.time()
    
    try:
        user_query = request.json.get('query', '')
        
        if not user_query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Step 1: Retrieve from vector DB (get top 10)
        retrieval_start = time.time()
        initial_results = vector_db.search(user_query, top_k=10)
        retrieval_time = time.time() - retrieval_start
        
        if not initial_results:
            return jsonify({
                'answer': 'No relevant documents found in the database. Please upload documents first.',
                'sources': [],
                'time': f"{time.time() - start_time:.2f}s",
                'retrieval_time': f"{retrieval_time:.2f}s",
                'rerank_time': '0.00s',
                'llm_time': '0.00s'
            })
        
        # Step 2: Rerank (get top 3)
        rerank_start = time.time()
        reranked_results = reranker.rerank(user_query, initial_results, top_k=3)
        rerank_time = time.time() - rerank_start
        
        # Step 3: Generate answer with LLM
        llm_start = time.time()
        result = llm.generate_answer(user_query, reranked_results)
        llm_time = time.time() - llm_start
        
        # Calculate costs
        cost_info = llm.estimate_cost(result['usage'])
        
        total_time = time.time() - start_time
        
        return jsonify({
            'answer': result['answer'],
            'sources': reranked_results,
            'time': f"{total_time:.2f}s",
            'retrieval_time': f"{retrieval_time:.2f}s",
            'rerank_time': f"{rerank_time:.2f}s",
            'llm_time': f"{llm_time:.2f}s",
            'tokens': result['usage'],
            'cost': cost_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get database statistics"""
    try:
        stats = vector_db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Clear all documents from database"""
    try:
        vector_db.clear_index()
        return jsonify({'success': True, 'message': 'Database cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)