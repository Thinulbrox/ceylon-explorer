import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:

    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'ceylon_tourism')
    DB_CHARSET = 'utf8mb4'

    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 10))
    DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', 20))
    DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', 30))
    DB_POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', 3600))

    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 30))

    OLLAMA_FALLBACK_MODELS = [
        'llama3.2:1b',
        'mistral:7b', 
        'qwen2.5:3b'
    ]
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    MAX_QUERY_LENGTH = int(os.getenv('MAX_QUERY_LENGTH', 500))
    MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', 2000))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))
 
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 30))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 60))

    ALLOWED_TABLES = [
        'attractions', 'hotels', 'restaurants', 'transport', 
        'cultural_sites', 'beaches', 'wildlife_parks', 'tea_estates',
        'activities', 'events', 'accommodation', 'food_places'
    ]
    
    ALLOWED_COLUMNS = [
        'id', 'name', 'description', 'location', 'region', 'province',
        'category', 'rating', 'price_range', 'contact', 'facilities',
        'opening_hours', 'best_time_to_visit', 'entrance_fee',
        'coordinates', 'image_url', 'website', 'type', 'cuisine',
        'accommodation_type', 'amenities', 'distance_from_colombo',
        'activities', 'specialties', 'tea_types', 'animals_seen'
    ]

    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'ceylon_explorer.log')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))

    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'True').lower() == 'true'
    CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # 5 minutes

    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    ASYNC_MODE = os.getenv('ASYNC_MODE', 'False').lower() == 'true'

    ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'True').lower() == 'true'
    METRICS_PORT = int(os.getenv('METRICS_PORT', 9090))
    
    @property
    def DATABASE_URL(self):
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset={self.DB_CHARSET}"
    
    @classmethod
    def validate_config(cls):
        errors = []

        required_vars = ['DB_PASSWORD']
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Missing required environment variable: {var}")

        if cls.DB_PORT < 1 or cls.DB_PORT > 65535:
            errors.append("DB_PORT must be between 1 and 65535")

        if cls.RATE_LIMIT_REQUESTS < 1:
            errors.append("RATE_LIMIT_REQUESTS must be positive")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))
        
        return True

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'
    RATE_LIMIT_REQUESTS = 60
    RATE_LIMIT_WINDOW = 60

class TestingConfig(Config):
    TESTING = True
    DB_NAME = 'ceylon_tourism_test'
    OLLAMA_MODEL = 'llama3.2:1b'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])

import sys
import time
import subprocess
import mysql.connector
import requests
from pathlib import Path

class SystemChecker:
    
    def __init__(self, config):
        self.config = config
        self.checks_passed = 0
        self.total_checks = 7
    
    def check_python_version(self):
        print("🐍 Checking Python version...")
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher required")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def check_mysql_connection(self):
        print("🗄️ Checking MySQL connection...")
        try:
            conn = mysql.connector.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME
            )
            conn.close()
            print(f"✅ Connected to MySQL at {self.config.DB_HOST}:{self.config.DB_PORT}")
            return True
        except Exception as e:
            print(f"❌ MySQL connection failed: {e}")
            return False
    
    def check_database_tables(self):
        print("📋 Checking database tables...")
        try:
            conn = mysql.connector.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME
            )
            cursor = conn.cursor()
            
            missing_tables = []
            for table in self.config.ALLOWED_TABLES:
                cursor.execute(f"SHOW TABLES LIKE '{table}'")
                if not cursor.fetchone():
                    missing_tables.append(table)
            
            cursor.close()
            conn.close()
            
            if missing_tables:
                print(f"❌ Missing tables: {', '.join(missing_tables)}")
                print("💡 Run: mysql -u user -p database < database_schema.sql")
                return False
            
            print(f"✅ All {len(self.config.ALLOWED_TABLES)} tables found")
            return True
            
        except Exception as e:
            print(f"❌ Error checking tables: {e}")
            return False
    
    def check_ollama_service(self):
        print("🤖 Checking Ollama service...")
        try:
            response = requests.get(f"{self.config.OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama service running at {self.config.OLLAMA_HOST}")
                return True
            else:
                print(f"❌ Ollama service not responding (status: {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Start Ollama with: ollama serve")
            return False
    
    def check_ollama_model(self):
        print(f"🧠 Checking Ollama model: {self.config.OLLAMA_MODEL}")
        try:
            response = requests.get(f"{self.config.OLLAMA_HOST}/api/tags", timeout=5)
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.config.OLLAMA_MODEL in model_names:
                print(f"✅ Model {self.config.OLLAMA_MODEL} is available")
                return True
            else:
                print(f"❌ Model {self.config.OLLAMA_MODEL} not found")
                print(f"💡 Install with: ollama pull {self.config.OLLAMA_MODEL}")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking model: {e}")
            return False
    
    def check_required_packages(self):
        print("📦 Checking Python packages...")
        required_packages = [
            'flask', 'mysql.connector', 'requests', 
            'ollama', 'sqlalchemy', 'flask_cors'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("💡 Install with: pip install -r requirements.txt")
            return False
        
        print("✅ All required packages installed")
        return True
    
    def check_ports(self):
        print("🔌 Checking port availability...")
        import socket
        
        ports_to_check = [5000] 
        busy_ports = []
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                busy_ports.append(port)
        
        if busy_ports:
            print(f"❌ Ports in use: {', '.join(map(str, busy_ports))}")
            print("💡 Stop other services or change port in app.py")
            return False
        
        print("✅ Required ports available")
        return True
    
    def run_all_checks(self):
        print("🚀 Ceylon Explorer System Check")
        print("=" * 50)
        
        checks = [
            self.check_python_version,
            self.check_required_packages,
            self.check_mysql_connection,
            self.check_database_tables,
            self.check_ollama_service,
            self.check_ollama_model,
            self.check_ports
        ]
        
        for check in checks:
            if check():
                self.checks_passed += 1
            print()
        
        print("=" * 50)
        print(f"✅ Checks passed: {self.checks_passed}/{self.total_checks}")
        
        if self.checks_passed == self.total_checks:
            print("🎉 System ready to start!")
            return True
        else:
            print("❌ Please fix the issues above before starting")
            return False

class HealthMonitor:
    """Application health monitor"""
    
    def __init__(self, config):
        self.config = config
    
    def check_database_health(self):
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            conn = mysql.connector.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM attractions")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time * 1000, 2),
                'attraction_count': result[0] if result else 0
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_ollama_health(self):
        """Check Ollama service health"""
        try:
            start_time = time.time()

            response = requests.get(f"{self.config.OLLAMA_HOST}/api/tags", timeout=5)
            models = response.json().get('models', [])

            test_response = requests.post(
                f"{self.config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.OLLAMA_MODEL,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy' if test_response.status_code == 200 else 'unhealthy',
                'response_time_ms': round(response_time * 1000, 2),
                'model': self.config.OLLAMA_MODEL,
                'available_models': len(models)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        print("🏥 Ceylon Explorer Health Check")
        print("=" * 40)

        db_health = self.check_database_health()
        print(f"Database: {db_health['status'].upper()}")
        if db_health['status'] == 'healthy':
            print(f"  Response time: {db_health['response_time_ms']}ms")
            print(f"  Attractions: {db_health['attraction_count']}")
        else:
            print(f"  Error: {db_health['error']}")

        ollama_health = self.check_ollama_health()
        print(f"Ollama: {ollama_health['status'].upper()}")
        if ollama_health['status'] == 'healthy':
            print(f"  Response time: {ollama_health['response_time_ms']}ms")
            print(f"  Model: {ollama_health['model']}")
            print(f"  Available models: {ollama_health['available_models']}")
        else:
            print(f"  Error: {ollama_health['error']}")
        
        return {
            'database': db_health,
            'ollama': ollama_health,
            'overall_status': 'healthy' if db_health['status'] == 'healthy' and ollama_health['status'] == 'healthy' else 'unhealthy'
        }

if __name__ == "__main__":
    config_class = get_config()
    
    try:
        config_class.validate_config()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    checker = SystemChecker(config_class)
    if not checker.run_all_checks():
        sys.exit(1)

    monitor = HealthMonitor(config_class)
    health_report = monitor.generate_health_report()
    
    if health_report['overall_status'] == 'healthy':
        print("\n🚀 Ready to launch Ceylon Explorer!")
        print("Run: python app.py")
    else:
        print("\n❌ System not ready. Please check the issues above.")
        sys.exit(1)