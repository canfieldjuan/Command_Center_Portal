"""
Configuration management for AI Portal application
"""

import os
import yaml
import urllib.parse
import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger()

class ConfigManager:
    """Centralized configuration management with environment variable support"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file and environment variables"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully", file=self.config_file)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults", file=self.config_file)
            self.config = {}
        except yaml.YAMLError as e:
            logger.error("Failed to parse config file", error=str(e), file=self.config_file)
            self.config = {}
        except Exception as e:
            logger.error("Unexpected error loading config file", error=str(e), file=self.config_file)
            self.config = {}

        # Load environment variables with validation
        self.config["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY")
        self.config["serper_api_key"] = os.environ.get("SERPER_API_KEY")
        self.config["copyshark_api_token"] = os.environ.get("COPYSHARK_API_TOKEN")
        
        # Database configuration with validation
        db_password = os.environ.get("SUPABASE_PASSWORD")
        if not db_password:
            raise ValueError("SUPABASE_PASSWORD environment variable not found. Please set it in your .env file.")
        
        encoded_password = urllib.parse.quote(db_password, safe='')
        self.config["database_url"] = f"postgresql://postgres.jacjorrzxilmrfxbdyse:{encoded_password}@aws-0-us-east-2.pooler.supabase.com:6543/postgres"

        # Set comprehensive default config values if not in yaml
        self._set_default_configurations()
        
        logger.info("Configuration initialization complete", 
                   config_keys=list(self.config.keys()),
                   model_tiers=len(self.config.get('model_tiers', {})),
                   available_tools=len(self.config.get('available_tools', [])))

    def _set_default_configurations(self):
        """Set default configuration values"""
        
        if 'model_tiers' not in self.config:
            self.config['model_tiers'] = {
                'economy': [
                    'gpt-3.5-turbo',
                    'anthropic/claude-3-haiku',
                    'google/gemini-pro'
                ],
                'standard': [
                    'anthropic/claude-3-sonnet',
                    'openai/gpt-4',
                    'google/gemini-pro-vision'
                ],
                'premium': [
                    'anthropic/claude-3-opus',
                    'openai/gpt-4-turbo',
                    'openai/gpt-4o'
                ]
            }
        
        if 'task_tier_map' not in self.config:
            self.config['task_tier_map'] = {
                'simple_qa': 'economy',
                'code_generation': 'standard',
                'image_generation': 'standard',
                'function_routing': 'economy',
                'complex_reasoning': 'premium'
            }
        
        if 'task_service_map' not in self.config:
            self.config['task_service_map'] = {
                'simple_qa': 'openrouter',
                'code_generation': 'openrouter',
                'image_generation': 'openrouter',
                'function_routing': 'openrouter',
                'complex_reasoning': 'openrouter'
            }
        
        if 'service_model_map' not in self.config:
            self.config['service_model_map'] = {
                'openrouter': [
                    'gpt-3.5-turbo',
                    'anthropic/claude-3-haiku',
                    'anthropic/claude-3-sonnet',
                    'anthropic/claude-3-opus',
                    'openai/gpt-4',
                    'openai/gpt-4-turbo',
                    'openai/gpt-4o',
                    'stable-diffusion-xl',
                    'dall-e-3'
                ],
                'google': [
                    'gemini-pro',
                    'gemini-pro-vision'
                ]
            }
        
        if 'available_tools' not in self.config:
            self.config['available_tools'] = [
                {
                    'name': 'web_search',
                    'description': 'Search the web for current information using Google',
                    'parameters': {
                        'query': {'type': 'string', 'description': 'The search query', 'required': True}
                    }
                },
                {
                    'name': 'browse_website',
                    'description': 'Visit and extract content from a website',
                    'parameters': {
                        'url': {'type': 'string', 'description': 'The URL to visit', 'required': True}
                    }
                },
                {
                    'name': 'save_to_file',
                    'description': 'Save content to a file in the workspace',
                    'parameters': {
                        'filename': {'type': 'string', 'description': 'The name of the file to save', 'required': True},
                        'content': {'type': 'string', 'description': 'The content to save', 'required': True}
                    }
                },
                {
                    'name': 'generateAdCopy',
                    'description': 'Generate advertising copy for products',
                    'parameters': {
                        'productName': {'type': 'string', 'description': 'The name of the product', 'required': True},
                        'audience': {'type': 'string', 'description': 'The target audience', 'required': True},
                        'niche': {'type': 'string', 'description': 'The product niche (optional)', 'required': False}
                    }
                }
            ]

        # Memory system configuration
        if 'memory_dir' not in self.config:
            self.config['memory_dir'] = './agent_memory'
        if 'embedding_model' not in self.config:
            self.config['embedding_model'] = 'all-MiniLM-L6-v2'
        if 'similarity_threshold' not in self.config:
            self.config['similarity_threshold'] = 0.7
        if 'max_memory_results' not in self.config:
            self.config['max_memory_results'] = 10

        # Google AI scopes configuration
        if 'google_ai_scopes' not in self.config:
            self.config['google_ai_scopes'] = [
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/generative-language'
            ]

        # Tool security configuration
        if 'max_file_size' not in self.config:
            self.config['max_file_size'] = 10 * 1024 * 1024  # 10MB
        if 'max_content_length' not in self.config:
            self.config['max_content_length'] = 1000000  # 1MB
        if 'allowed_file_extensions' not in self.config:
            self.config['allowed_file_extensions'] = [
                '.txt', '.md', '.json', '.yaml', '.yml', '.py', '.js', 
                '.html', '.css', '.xml', '.csv', '.log', '.sql'
            ]

        # CopyShark service configuration
        if 'copyshark_service' not in self.config:
            self.config['copyshark_service'] = {
                'base_url': 'https://your-copyshark-api.com'
            }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        value = self.config.get(key, default)
        if value is None and default is not None:
            logger.warning("Configuration key not found, using default", 
                         key=key, default=default)
        return value

    def reload_config(self):
        """Reload configuration from file"""
        logger.info("Reloading configuration", file=self.config_file)
        self.load_config()

    def validate_required_keys(self, required_keys: list) -> bool:
        """Validate that required configuration keys are present"""
        missing_keys = [key for key in required_keys if not self.config.get(key)]
        if missing_keys:
            logger.error("Missing required configuration keys", missing=missing_keys)
            return False
        return True