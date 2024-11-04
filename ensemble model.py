# API's
from openai import OpenAI
from anthropic import Anthropic
from llamaapi import LlamaAPI
import google.generativeai as genai
# Config
from config import APIConfig
# New Classes
from variables import BoundedFloat
# Data Handling
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import Counter

@dataclass
class AnalysisResult:
    """Standardized analysis result across all models"""
    # Model Used
    model_name: str
    # Analysis
    affects_cotton: bool # Most important field
    price_impact: float # [-1,1] score
    cotton_focus: float  # 0-1 score
    confidence: float  # 0-1 score
    # Validation to make sure values are within expected range
    def __post_init__(self):
        # Validate price_impact is between -1 and 1
        if not -1 <= self.price_impact <= 1:
            raise ValueError(f"price_impact must be between -1 and 1, got {self.price_impact}")
        
        # Validate cotton_focus is between 0 and 1
        if not 0 <= self.cotton_focus <= 1:
            raise ValueError(f"cotton_focus must be between 0 and 1, got {self.cotton_focus}")
            
        # Validate confidence is between 0 and 1
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
    
config = APIConfig()

class BaseModelAnalyzer():
    """Abstract base class for model analyzers"""
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        self.api_key = api_key
        self.debug = debug
        self.last_request_times = []
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Setup model-specific configuration"""
        pass
    
    @abstractmethod
    async def analyze_article(self, article_text: str, title: str) -> AnalysisResult:
        """Analyze article and return standardized result"""
        pass

    
    async def create_analysis_prompt(self, article: str, title: str) -> str:
        """Create a structured prompt for the model to analyze the article"""
        
        return f"""You are a specialized financial and commodity markets analyst. Analyze this news article's title and content, focusing on cotton market implications. Provide a detailed analysis in JSON format focusing on cotton market implications.

Your response must be a valid JSON object exactly matching this structure:

{{
    "content_analysis": {{
        "affects_cotton": boolean,
        "price_impact": float (-1 to 1),
        "confidence": float (0-1),
        "cotton_focus": float (0-1)
    }},
}}

Title to analyze: {title}

Article content to analyze: {article}

Provide only with the JSON object, no other text."""

    def format_result(self, model_name: str, json_data: Dict[str, Any]) -> AnalysisResult:
        """
        Parse JSON analysis data into an AnalysisResult object.
        
        Args:
            model_name: Name of the model that produced the analysis
            json_data: Dictionary containing the analysis results
            
        Returns:
            AnalysisResult object with validated data
            
        Raises:
            ValueError: If data is missing or invalid
            KeyError: If required fields are missing from JSON
        """
        try:
            content_analysis = json_data["content_analysis"]
            
            return AnalysisResult(
                model_name=model_name,
                affects_cotton=bool(content_analysis["affects_cotton"]),
                price_impact=float(content_analysis["price_impact"]),
                cotton_focus=float(content_analysis["cotton_focus"]),
                confidence=float(content_analysis["confidence"])
            )
        except KeyError as e:
            raise KeyError(f"Missing required field in JSON: {str(e)}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data in JSON: {str(e)}")

class OpenAIModelAnalyzer(BaseModelAnalyzer):
    """OpenAI APIspecific analyzer using latest model"""
    
    def _setup(self):
        # Get API key
        self.api_key = self.api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        # Configure API
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4-turbo-preview"  # Latest GPT-4 model
        self.requests_per_minute = 500
            
    async def analyze_article(self, article: str, title: str) -> AnalysisResult:
        try:
            # Create the analysis prompt
            prompt = await self.create_analysis_prompt(title, article)
            # Get GPT's analysis
            response = self.client.chat.completions.create(
                model=self.model,  # or "gpt-4" or "gpt-3.5-turbo" depending on needs
                response_format={ "type": "json_object" },  # Ensure JSON response
                messages=[
                    {"role": "system", "content": "You are a financial analyst specialized in commodity markets. Provide analysis in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # Use 0 temperature for more consistent analysis
            )
            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content)
            # Format and validate the result
            return self.format_result("OpenAI", analysis)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse GPT's response as JSON")
        except Exception as e:
            raise ValueError(f"Error creating analysis prompt: {e}")
    

class AnthropicModelAnalyzer(BaseModelAnalyzer):
    """Claude 3.5 Sonnet specific analyzer"""
    
    def _setup(self):
        # Get API key
        self.api_key = self.api_key or config.CLAUDE_API_KEY
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        # Configure API
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-3-sonnet-20240229"
        self.requests_per_minute = 500
            
    async def analyze_article(self, article: str, title: str) -> AnalysisResult:
        try:
            # Create the analysis prompt
            prompt = await self.create_analysis_prompt(title, article)
            # Send request to API and get Claude analysis
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.0,  # Use 0 temperature for more consistent analysis
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            # Parse the JSON response
            analysis = json.loads(response.content[0].text)
            # Format and validate the result
            return self.format_result("Anthropic", analysis)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse Claude's response as JSON")   
        except Exception as e:
            raise ValueError(f"Error creating analysis prompt: {e}")
    

class MetaModelAnalyzer(BaseModelAnalyzer):
    """Meta model analyzer using Together.ai API"""
    
    def _setup(self):
        # Get API key
        self.api_key = self.api_key or config.LLAMA_API_KEY
        if not self.api_key:
            raise ValueError("Llama API key required")
        # Configure API
        self.client = LlamaAPI(self.api_key)
        self.model = "llama3.2-90b-vision"
        self.requests_per_minute = 300
    
    async def analyze_article(self, article: str, title: str) -> AnalysisResult:
        try:
            # Create the analysis prompt
            prompt = await self.create_analysis_prompt(title, article)
            # Send request to API
            api_request = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a specialized financial and commodity markets analyst. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                    ],
                "temperature": 0.0,
                "max_tokens": 2048
            }
            response = self.client.run(api_request)
            analysis = self.parse_llama_response(response.json())
            return self.format_result("Meta", analysis)
        except Exception as e:
            raise ValueError(f"Error creating analysis prompt: {e}")
        
    def parse_llama_response(self, response):
        """
        Parse and format the Llama API response to extract the analysis JSON.
        
        Args:
            response: Raw response from Llama API
        Returns:
            dict: Parsed JSON analysis
        """
        try:
            # Get the content from the response
            content = response['choices'][0]['message']['content']
            
            # Remove markdown code blocks
            content = content.replace('```', '').strip()
            
            # If content starts with 'json' or 'JSON', remove that
            if content.lower().startswith('json'):
                content = content[4:].strip()
                
            # Parse the JSON
            analysis = json.loads(content)
            return analysis
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return None

class GeminiAnalyzer(BaseModelAnalyzer):
    """Google's Gemini Pro analyzer"""
    
    def _setup(self):
        # Get API key
        self.api_key = self.api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key required")
        # Configure API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-001')
        self.requests_per_minute = 600
        
    async def analyze_article(self, article: str, title: str) -> AnalysisResult:
        try:
            # Create the analysis prompt
            prompt = await self.create_analysis_prompt(title, article)
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                max_output_tokens=2048,
            )
            # Get Gemini's analysis
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            # Extract the content and parse JSON
            try:
                # Clean the response text
                response_text = response.text
                # Find JSON content, get the first and last characters (brackets)
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")
                # Get the JSON string
                json_str = response_text[start_idx:end_idx]
                
                # Parse the JSON response
                analysis = json.loads(json_str)
                # Format and validate the result
                return self.format_result("Gemini", analysis)
            # If JSON parsing fails
            except json.JSONDecodeError as e:
                print(f"Raw response: {response_text}")
                raise ValueError(f"Failed to parse response as JSON: {str(e)}")
        # If any other error
        except Exception as e:
            raise ValueError(f"Error creating analysis prompt: {e}")
    

class EnsembleAnalyzer:
    """Manages ensemble of the latest AI models"""
    
    def __init__(self, config: Dict[str, Dict[str, Any]], debug: bool = False):
        """
        Initialize ensemble with configuration for each model.
        
        Example config:
        {
            'gpt4': {'api_key': 'key1', 'weight': 2.0},
            'claude': {'api_key': 'key2', 'weight': 2.0},
            'llama2': {'api_key': 'key3', 'weight': 1.5},
            'gemini': {'api_key': 'key4', 'weight': 1.5}
        }
        """
        self.debug = debug
        self.analyzers = []
        self.weights = []
        
        model_classes = {
            'OpenAI': OpenAIModelAnalyzer,
            'Anthropic': AnthropicModelAnalyzer,
            'Meta': MetaModelAnalyzer,
            'Google': GeminiAnalyzer
        }
        
        for model_type, settings in config.items():
            if model_type not in model_classes:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            analyzer = model_classes[model_type](
                api_key=settings.get('api_key'),
                debug=debug
            )
            self.analyzers.append(analyzer)
            self.weights.append(1.0)
    
    async def analyze(self, title: str, article: str) -> AnalysisResult:
        """Analyze article and return standardized result"""
        results = []
        for analyzer in self.analyzers:
            analysis = await analyzer.analyze_article(title, article)
            results.append(analysis)
        
        return results
    
def aggregated_analysis(analyses: List[AnalysisResult]) -> Dict:
    """
    Aggregate analyses from multiple models to find consensus and averages.
    
    Args:
        analyses: List of AnalysisResult objects
        
    Returns:
        dict: Aggregated results including consensus and weighted averages
    """
    # Skip analyses with very low confidence
    valid_analyses = [a for a in analyses if a.confidence > 0.2]
    
    if not valid_analyses:
        return {
            "consensus_affects_cotton": False,
            "weighted_price_impact": 0.0,
            "weighted_cotton_focus": 0.0,
            "confidence_level": 0.0,
            "agreement_level": 0.0
        }
    
    # Calculate consensus for boolean affects_cotton
    affects_cotton_votes = Counter(a.affects_cotton for a in valid_analyses)
    total_votes = len(valid_analyses)
    consensus_affects_cotton = affects_cotton_votes[True] > total_votes / 2
    
    # Calculate agreement level (what percentage of models agree with consensus)
    majority_vote_count = max(affects_cotton_votes.values())
    agreement_level = majority_vote_count / total_votes
    
    # Calculate weighted averages using confidence as weights
    total_confidence = sum(a.confidence for a in valid_analyses)
    
    weighted_price_impact = sum(
        a.price_impact * a.confidence for a in valid_analyses
    ) / total_confidence if total_confidence > 0 else 0.0
    
    weighted_cotton_focus = sum(
        a.cotton_focus * a.confidence for a in valid_analyses
    ) / total_confidence if total_confidence > 0 else 0.0
    
    # Average confidence across all models
    average_confidence = total_confidence / len(valid_analyses)
    
    return {
        "consensus_affects_cotton": consensus_affects_cotton,
        "weighted_price_impact": round(weighted_price_impact, 3),
        "weighted_cotton_focus": round(weighted_cotton_focus, 3),
        "confidence_level": round(average_confidence, 3),
        "agreement_level": round(agreement_level, 3)
    }

            
            
