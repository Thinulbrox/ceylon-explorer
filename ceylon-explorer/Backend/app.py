import os
import json
import logging
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import time
from functools import wraps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlalchemy
from sqlalchemy import create_engine, text, exc as sa_exc
from sqlalchemy.engine import Connection
from mysql.connector import Error as MySQLError
import requests
import ollama

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ceylon_explorer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class Config:

    def __init__(self):

        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = int(os.getenv('DB_PORT', 3306))
        self.DB_USER = os.getenv('DB_USER', 'root')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
        self.DB_DATABASE = os.getenv('DB_DATABASE', 'tourism')

        self.DB_POOL_SIZE = self._get_env_int('DB_POOL_SIZE', 10)
        self.DB_POOL_TIMEOUT = self._get_env_int('DB_POOL_TIMEOUT', 30)
        self.DB_POOL_RECYCLE = self._get_env_int('DB_POOL_RECYCLE', 3600)

        self.OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.LLM_MODEL = os.getenv('LLM_MODEL', 'llama3')

        self.MAX_QUERY_LENGTH = self._get_env_int('MAX_QUERY_LENGTH', 500)
        self.CACHE_EXPIRATION_TIME = self._get_env_int('CACHE_EXPIRATION_TIME', 3600)
        self.RATE_LIMIT_PER_MINUTE = self._get_env_int('RATE_LIMIT_PER_MINUTE', 30)

        self.API_KEY_SECRET = os.getenv('API_KEY_SECRET', 'your_super_secret_api_key_here')

        self.validate_config()

    def _get_env_int(self, var_name: str, default: int) -> int:
        try:
            val = os.getenv(var_name)
            if val is not None and val.strip():
                return int(val)
            return default
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for '{var_name}', using default '{default}'.")
            return default

    def validate_config(self):
        required_vars = {
            'DB_HOST': self.DB_HOST,
            'DB_USER': self.DB_USER,
            'DB_PASSWORD': self.DB_PASSWORD,
            'DB_DATABASE': self.DB_DATABASE,
            'OLLAMA_HOST': self.OLLAMA_HOST
        }
        for name, value in required_vars.items():
            if not value:
                logger.critical(f"Missing critical environment variable: {name}")
                raise ValueError(f"Incomplete configuration: {name} is not set.")
        
        if self.DB_PORT <= 0:
            logger.critical("Invalid database port specified.")
            raise ValueError("Invalid database port.")

config = Config()

class CacheService:
    def __init__(self, expiration_time: int):
        self.cache = {}
        self.expiration_time = expiration_time

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.expiration_time:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: Dict[str, Any]):
        self.cache[key] = (data, time.time())

    def clear(self):
        self.cache.clear()
        
cache_service = CacheService(config.CACHE_EXPIRATION_TIME)

class RateLimiter:

    def __init__(self, limit: int):
        self.limit = limit
        self.request_timestamps = {}
        
    def limit(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ip_address = request.remote_addr
            current_time = time.time()
            
            if ip_address not in self.request_timestamps:
                self.request_timestamps[ip_address] = []
            self.request_timestamps[ip_address] = [t for t in self.request_timestamps[ip_address] if current_time - t < 60]
            
            if len(self.request_timestamps[ip_address]) >= self.limit:
                logger.warning(f"Rate limit exceeded for IP: {ip_address}")
                return jsonify({"error": "Rate limit exceeded. Please try again in a minute."}), 429
            
            self.request_timestamps[ip_address].append(current_time)
            return f(*args, **kwargs)
        return wrapper
        
rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)

class LLMService:
    def __init__(self, ollama_host: str, model_name: str):
        self.ollama_host = ollama_host
        self.model_name = model_name

    def is_reachable(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama server is reachable.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama server not reachable: {e}")
            return False

    def generate_sql_query(self, user_query: str, schema: Dict[str, List[str]]) -> str:
    
        full_prompt = f"""
            
            The user wants to know about: {user_query}
            
            Database Schema:
            {json.dumps(schema, indent=2)}
            """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=False,
                options={'temperature': 0}
            )
            sql_query = response['response'].strip()
            sql_query = re.sub(r';$', '', sql_query)
            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query from LLM: {e}")
            return ""

    def format_response(self, user_query: str, db_results: List[Dict[str, Any]]) -> str:

        prompt = f"""
            Based on the user's query and the database results, provide a friendly and informative response.
            User Query: {user_query}
            Database Results: {json.dumps(db_results, default=str)}
            
            """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={'temperature': 0.7}
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error formatting response with LLM: {e}")
            return "I am sorry, I couldn't generate a response at this time."

    def get_fallback_response(self, user_query: str) -> str:
        logger.warning(f"Using fallback response for user query: {user_query}")
        return "I am sorry, but I cannot process that query for security reasons. Please ask a different question about Sri Lanka tourism."

class DBManager:
    def __init__(self, host: str, user: str, password: str, database: str,
                 port: int, pool_size: int, pool_recycle: int):
        self.engine = create_engine(
            f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}",
            pool_size=pool_size,
            pool_recycle=pool_recycle
        )
        self.schema_cache: Dict[str, List[str]] = {}

    def get_table_schema(self, table_name: str) -> List[str]:
        if table_name in self.schema_cache:
            return self.schema_cache[table_name]
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SHOW COLUMNS FROM {table_name}")).fetchall()
                columns = [row[0] for row in result]
                self.schema_cache[table_name] = columns
                logger.debug(f"Fetched schema for table '{table_name}': {columns}")
                return columns
        except sa_exc.NoSuchTableError:
            logger.error(f"Table '{table_name}' does not exist.")
            return []
        except Exception as e:
            logger.error(f"Error getting schema for table '{table_name}': {e}")
            return []

    def get_all_tables(self) -> List[str]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SHOW TABLES")).fetchall()
                tables = [row[0] for row in result]
                logger.info(f"Successfully retrieved {len(tables)} tables.")
                return tables
        except Exception as e:
            logger.error(f"Error getting all tables: {e}")
            return []

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchall()
                return [row._asdict() for row in result]
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error executing query: {e}")
            return []
        except Exception as e:
            logger.error(f"General error executing query: {e}")
            return []

class QueryValidator:
    DISALLOWED_PATTERNS = [
        r'\b(DELETE|DROP|UPDATE|INSERT|ALTER|TRUNCATE|UNION|EXEC|PROCEDURE|CREATE)\b',
        r'--' 
    ]
    
    ALLOWED_TABLES = [
        'attractions', 'hotels', 'restaurants', 'activities',
        'wildlife_parks', 'beaches', 'cultural_sites',
        'historical_periods', 'tourism_categories'
    ]

    def validate(self, query: str) -> bool:

        upper_query = query.upper()

        if any(re.search(pattern, upper_query, re.IGNORECASE) for pattern in self.DISALLOWED_PATTERNS):
            logger.warning(f"Blocked query due to disallowed keywords: {query}")
            return False

        if ';' in query:
            logger.warning(f"Blocked query due to multiple statements: {query}")
            return False

        from_matches = re.findall(r'FROM\s+`?(\w+)`?', upper_query)
        join_matches = re.findall(r'(?:JOIN|FROM)\s+`?(\w+)`?', upper_query)
        all_tables = set(from_matches + join_matches)
        
        if not all(table in self.ALLOWED_TABLES for table in all_tables):
            logger.warning(f"Blocked query due to unauthorized table access: {all_tables}")
            return False
            
        logger.info(f"Query validation successful for: {query}")
        return True

class AnalyticsService:
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'cached_requests': 0,
            'llm_requests': 0,
            'db_queries': 0,
            'security_failures': 0,
            'errors': 0,
            'avg_processing_time': 0.0,
            'avg_llm_response_time': 0.0,
            'avg_db_query_time': 0.0,
            'last_updated': datetime.utcnow().isoformat()
        }
        self.processing_times = []
        self.llm_response_times = []
        self.db_query_times = []
        
    def _update_average(self, metric_list: List[float], new_value: float, key: str):
        metric_list.append(new_value)
        if len(metric_list) > 100:
            metric_list.pop(0)
        self.metrics[key] = sum(metric_list) / len(metric_list) if metric_list else 0.0
        
    def log_request_start(self):
        self.metrics['total_requests'] += 1
        
    def log_cached_request(self):
        self.metrics['cached_requests'] += 1
        
    def log_llm_request(self, response_time: float):
        self.metrics['llm_requests'] += 1
        self._update_average(self.llm_response_times, response_time, 'avg_llm_response_time')
        
    def log_db_query(self, query_time: float):
        self.metrics['db_queries'] += 1
        self._update_average(self.db_query_times, query_time, 'avg_db_query_time')

    def log_security_failure(self):
        self.metrics['security_failures'] += 1
        
    def log_error(self):
        self.metrics['errors'] += 1
        
    def log_processing_time(self, processing_time: float):
        self._update_average(self.processing_times, processing_time, 'avg_processing_time')
        
    def get_metrics(self) -> Dict[str, Any]:
        self.metrics['last_updated'] = datetime.utcnow().isoformat()
        return self.metrics.copy()
        
    def reset(self):
        self.metrics = {
            'total_requests': 0,
            'cached_requests': 0,
            'llm_requests': 0,
            'db_queries': 0,
            'security_failures': 0,
            'errors': 0,
            'avg_processing_time': 0.0,
            'avg_llm_response_time': 0.0,
            'avg_db_query_time': 0.0,
            'last_updated': datetime.utcnow().isoformat()
        }
        self.processing_times.clear()
        self.llm_response_times.clear()
        self.db_query_times.clear()
        logger.info("Analytics metrics have been reset.")
        
analytics_service = AnalyticsService()
class FeedbackService:
    def __init__(self):
        self.feedback_list = []

    def submit_feedback(self, user_query: str, rating: int, comment: Optional[str] = None) -> bool:
        if not isinstance(rating, int) or not (1 <= rating <= 5):
            logger.warning(f"Invalid rating received: {rating}")
            return False
            
        feedback_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_query': user_query,
            'rating': rating,
            'comment': comment
        }
        self.feedback_list.append(feedback_entry)
        logger.info(f"Received feedback: {feedback_entry}")
        return True

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        return self.feedback_list.copy()

feedback_service = FeedbackService()

class Chatbot:
    def __init__(self, llm_service: LLMService, db_manager: DBManager, query_validator: QueryValidator):
        self.llm = llm_service
        self.db = db_manager
        self.validator = query_validator
        self.schema_cache = {}
        self._build_schema_cache()

    def _build_schema_cache(self):
        try:
            tables = self.db.get_all_tables()
            for table in tables:
                self.schema_cache[table] = self.db.get_table_schema(table)
            logger.info("Database schema cache built successfully.")
        except Exception as e:
            logger.error(f"Error building schema cache: {e}")
            self.schema_cache = {}

    def process_query(self, user_query: str) -> Dict[str, Any]:

        start_time = time.time()
        analytics_service.log_request_start()
        
        try:
            if not user_query or len(user_query) > config.MAX_QUERY_LENGTH:
                analytics_service.log_error()
                return {
                    "error": f"Query is too long or empty. Maximum {config.MAX_QUERY_LENGTH} characters allowed."
                }


            cached_response = cache_service.get(user_query)
            if cached_response:
                analytics_service.log_cached_request()
                logger.info(f"Serving query '{user_query}' from cache.")

                cached_response['processing_time'] = round(time.time() - start_time, 3)
                return cached_response

            llm_start_time = time.time()
            sql_query = self.llm.generate_sql_query(user_query, self.schema_cache)
            llm_end_time = time.time()
            analytics_service.log_llm_request(llm_end_time - llm_start_time)

            if not self.validator.validate(sql_query):
                analytics_service.log_security_failure()
                return {
                    "response": self.llm.get_fallback_response(user_query),
                    "query_used": "Invalid query blocked.",
                    "results_count": 0,
                    "processing_time": round(time.time() - start_time, 3)
                }

            db_start_time = time.time()
            db_results = self.db.execute_query(sql_query)
            db_end_time = time.time()
            analytics_service.log_db_query(db_end_time - db_start_time)

            formatted_response = self.llm.format_response(user_query, db_results)

            processing_time = time.time() - start_time
            analytics_service.log_processing_time(processing_time)
            
            response_data = {
                "response": formatted_response,
                "query_used": sql_query,
                "results_count": len(db_results),
                "processing_time": round(processing_time, 3)
            }
            
            cache_service.set(user_query, response_data)
            
            return response_data
            
        except Exception as e:
            analytics_service.log_error()
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return {"error": "An internal error occurred. Please check the logs for details."}

# --- API Endpoints ---
@app.route('/')
def home():
    return jsonify({"message": "Ceylon Explorer Backend is running."})

@app.route('/chat', methods=['POST'])
@rate_limiter.limit
def chat():
    data = request.json
    user_query = data.get('message', '').strip()
    
    if not user_query:
        return jsonify({"error": "No message provided."}), 400
    
    response_data = chatbot.process_query(user_query)
    return jsonify(response_data)

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'app_status': 'ok',
        'database_status': 'ok',
        'llm_status': 'ok'
    }

    try:
        with db_manager.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        status['database_status'] = 'error'
        logger.error(f"Health check failed for database: {e}")

    if not llm_service.is_reachable():
        status['llm_status'] = 'error'
        
    return jsonify(status)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(analytics_service.get_metrics())

@app.route('/cache/clear', methods=['POST'])
def clear_cache_endpoint():
    cache_service.clear()
    logger.info("Query cache has been cleared via API endpoint.")
    return jsonify({"message": "Query cache cleared."})

@app.route('/feedback', methods=['POST'])
def submit_feedback_endpoint():
    data = request.json
    query = data.get('query')
    rating = data.get('rating')
    comment = data.get('comment')
    
    if not query or rating is None:
        return jsonify({"error": "Query and rating are required."}), 400
        
    if feedback_service.submit_feedback(query, rating, comment):
        return jsonify({"message": "Feedback submitted successfully."})
    else:
        return jsonify({"error": "Failed to submit feedback. Invalid rating."}), 500

@app.route('/feedback/all', methods=['GET'])
def get_all_feedback_endpoint():
    return jsonify({"feedback": feedback_service.get_all_feedback()})

@app.route('/debug/sql', methods=['POST'])
def debug_sql_endpoint():
    data = request.json
    api_key = data.get('api_key')
    query = data.get('query')
    
    if api_key != config.API_KEY_SECRET:
        return jsonify({"error": "Invalid API Key"}), 401
    
    if not query:
        return jsonify({"error": "No query provided."}), 400
        
    if not query_validator.validate(query):
        return jsonify({"error": "Query validation failed."}), 403
    
    try:
        results = db_manager.execute_query(query)
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Failed to execute debug query: {e}")
        return jsonify({"error": "Failed to execute query."}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Ceylon Explorer Backend...")

        llm_service = LLMService(config.OLLAMA_HOST, config.LLM_MODEL)
        if not llm_service.is_reachable():
            logger.critical("Ollama server is not reachable. Please check if the server is running.")
            exit(1)
            
        try:
            db_manager = DBManager(
                config.DB_HOST, config.DB_USER, config.DB_PASSWORD, 
                config.DB_DATABASE, config.DB_PORT, 
                config.DB_POOL_SIZE, config.DB_POOL_RECYCLE
            )
            db_manager.get_all_tables()
            logger.info("Successfully connected to the database and tested connection.")
        except Exception as db_e:
            logger.critical(f"Database connection failed: {db_e}")
            logger.error(traceback.format_exc())
            exit(1)

        query_validator = QueryValidator()
        chatbot = Chatbot(llm_service, db_manager, query_validator)

        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Failed to start the application: {e}")
        logger.error(traceback.format_exc())
        exit(1)
