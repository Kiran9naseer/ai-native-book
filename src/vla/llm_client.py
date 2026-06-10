#!/usr/bin/env python3
"""
LLM Client for VLA Pipeline

Provides a unified interface for LLM-based action planning using OpenAI GPT-4
or compatible models (with Ollama fallback).
"""

import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class ActionPrimitive:
    """Represents a single action primitive in a plan"""
    action_type: str
    parameters: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    timeout_s: float = 30.0
    retry_count: int = 3


@dataclass
class ActionPlan:
    """Represents a complete action plan"""
    plan_id: str
    explanation: str
    primitives: List[ActionPrimitive]
    estimated_duration_s: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "explanation": self.explanation,
            "primitives": [asdict(p) for p in self.primitives],
            "estimated_duration_s": self.estimated_duration_s
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class LLMClient:
    """
    Client for LLM-based action planning.
    
    Supports OpenAI GPT-4 as primary provider with Ollama fallback.
    Uses function calling for structured output.
    """
    
    # Action plan function schema for OpenAI function calling
    FUNCTION_SCHEMA = {
        "type": "function",
        "function": {
            "name": "create_action_plan",
            "description": "Create a sequence of robot actions to accomplish a task",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of the plan"
                    },
                    "primitives": {
                        "type": "array",
                        "description": "Ordered list of action primitives",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_type": {
                                    "type": "string",
                                    "enum": [
                                        "navigate_to", "look_at", "scan_environment",
                                        "identify_object", "pick_up", "place",
                                        "say", "wait", "cancel"
                                    ]
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Action-specific parameters"
                                },
                                "preconditions": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "expected_outcomes": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["action_type", "parameters"]
                        }
                    }
                },
                "required": ["explanation", "primitives"]
            }
        }
    }
    
    # Default system prompt
    DEFAULT_SYSTEM_PROMPT = """You are an intelligent robot action planner for a humanoid robot. 
Your role is to translate natural language commands into executable action sequences.

The robot can perform these actions:
- navigate_to: Move to a location (by name or coordinates)
- look_at: Orient sensors toward a target
- scan_environment: Survey the surroundings
- identify_object: Detect and localize a specific object
- pick_up: Grasp and lift an object
- place: Put down a held object
- say: Speak to the user
- wait: Pause for a duration
- cancel: Stop current action

When planning:
1. Always validate that preconditions can be met
2. Include error handling for potential failures
3. Provide clear feedback to the user via 'say' actions
4. Consider the robot's current state and capabilities
5. Generate the minimum necessary steps

If the command cannot be executed, explain why and suggest alternatives.
Always respond with a valid action plan using the create_action_plan function."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_s: float = 15.0,
        max_retries: int = 3,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        fallback_base_url: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider (openai, anthropic, ollama)
            model: Model name
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
            base_url: Custom API base URL
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout_s: Request timeout
            max_retries: Maximum retry attempts
            fallback_provider: Fallback provider on primary failure
            fallback_model: Fallback model name
            fallback_base_url: Fallback API URL
        """
        self.provider = LLMProvider(provider.lower())
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        
        # Initialize primary client
        self.client = None
        if self.provider == LLMProvider.OPENAI:
            if OPENAI_AVAILABLE:
                import os
                self.client = OpenAI(
                    api_key=api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=base_url,
                    timeout=timeout_s
                )
            else:
                logger.warning("OpenAI package not available")
        
        # Fallback configuration
        self.fallback_provider = None
        self.fallback_client = None
        if fallback_provider:
            self.fallback_provider = LLMProvider(fallback_provider.lower())
            self.fallback_model = fallback_model
            self.fallback_base_url = fallback_base_url or "http://localhost:11434"
        
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
    
    def set_system_prompt(self, prompt: str):
        """Set custom system prompt"""
        self.system_prompt = prompt
    
    def generate_plan(
        self,
        command: str,
        intent_type: str,
        target_object: Optional[str] = None,
        target_location: Optional[str] = None,
        modifiers: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ActionPlan]:
        """
        Generate an action plan for the given command.
        
        Args:
            command: The user's natural language command
            intent_type: Classified intent type
            target_object: Target object name
            target_location: Target location name
            modifiers: Command modifiers (colors, sizes, etc.)
            context: Environmental context (robot state, objects, etc.)
            
        Returns:
            ActionPlan or None if generation failed
        """
        # Build user prompt with context
        user_prompt = self._build_user_prompt(
            command, intent_type, target_object, target_location,
            modifiers, context
        )
        
        # Try primary provider
        for attempt in range(self.max_retries):
            try:
                result = self._call_llm(user_prompt)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))  # Exponential backoff
        
        # Try fallback
        if self.fallback_provider:
            logger.info("Attempting fallback LLM provider")
            try:
                return self._call_fallback_llm(user_prompt)
            except Exception as e:
                logger.error(f"Fallback LLM failed: {e}")
        
        return None
    
    def _build_user_prompt(
        self,
        command: str,
        intent_type: str,
        target_object: Optional[str],
        target_location: Optional[str],
        modifiers: Optional[List[str]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the user prompt with context"""
        context = context or {}
        
        prompt_parts = ["Current Context:"]
        
        # Robot state
        robot_pose = context.get("robot_pose", "unknown")
        prompt_parts.append(f"- Robot Location: {robot_pose}")
        
        gripper_state = context.get("gripper_state", "unknown")
        prompt_parts.append(f"- Gripper State: {gripper_state}")
        
        held_object = context.get("held_object", "none")
        prompt_parts.append(f"- Held Object: {held_object}")
        
        # Detected objects
        objects = context.get("detected_objects", [])
        if objects:
            obj_str = ", ".join([f"{o.get('name', 'unknown')} (id: {o.get('id', '?')})" 
                                for o in objects[:5]])
            prompt_parts.append(f"- Detected Objects: {obj_str}")
        else:
            prompt_parts.append("- Detected Objects: none visible")
        
        # Known locations
        locations = context.get("known_locations", ["home"])
        prompt_parts.append(f"- Known Locations: {', '.join(locations)}")
        
        # Recent actions
        history = context.get("recent_actions", [])
        if history:
            recent = history[-3:] if len(history) > 3 else history
            prompt_parts.append(f"- Recent Actions: {', '.join(recent)}")
        
        # Command details
        prompt_parts.append("")
        prompt_parts.append(f'User Command: "{command}"')
        prompt_parts.append(f"Intent Type: {intent_type}")
        
        if target_object:
            prompt_parts.append(f"Target Object: {target_object}")
        if target_location:
            prompt_parts.append(f"Target Location: {target_location}")
        if modifiers:
            prompt_parts.append(f"Modifiers: {', '.join(modifiers)}")
        
        prompt_parts.append("")
        prompt_parts.append("Generate an action plan to accomplish this command.")
        
        return "\n".join(prompt_parts)
    
    def _call_llm(self, user_prompt: str) -> Optional[ActionPlan]:
        """Call the primary LLM provider"""
        if self.provider == LLMProvider.OPENAI:
            return self._call_openai(user_prompt)
        elif self.provider == LLMProvider.OLLAMA:
            return self._call_ollama(user_prompt, self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_openai(self, user_prompt: str) -> Optional[ActionPlan]:
        """Call OpenAI API with function calling"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[self.FUNCTION_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "create_action_plan"}},
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract function call result
        message = response.choices[0].message
        if message.tool_calls:
            function_args = message.tool_calls[0].function.arguments
            return self._parse_plan_response(function_args)
        
        return None
    
    def _call_ollama(
        self,
        user_prompt: str,
        model: str,
        base_url: str = "http://localhost:11434"
    ) -> Optional[ActionPlan]:
        """Call Ollama API"""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not available for Ollama calls")
        
        # Ollama doesn't support function calling, so we use a structured prompt
        structured_prompt = f"""{self.system_prompt}

IMPORTANT: Respond ONLY with a JSON object in this exact format:
{{
  "explanation": "brief explanation of the plan",
  "primitives": [
    {{
      "action_type": "navigate_to|look_at|scan_environment|identify_object|pick_up|place|say|wait|cancel",
      "parameters": {{}},
      "preconditions": [],
      "expected_outcomes": []
    }}
  ]
}}

{user_prompt}"""
        
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": structured_prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "format": "json"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                return self._parse_plan_response(result["response"])
        
        return None
    
    def _call_fallback_llm(self, user_prompt: str) -> Optional[ActionPlan]:
        """Call fallback LLM provider"""
        if self.fallback_provider == LLMProvider.OLLAMA:
            return self._call_ollama(
                user_prompt,
                self.fallback_model,
                self.fallback_base_url
            )
        return None
    
    def _parse_plan_response(self, response: str) -> Optional[ActionPlan]:
        """Parse LLM response into ActionPlan"""
        try:
            # Parse JSON
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            # Extract primitives
            primitives = []
            for p in data.get("primitives", []):
                primitive = ActionPrimitive(
                    action_type=p.get("action_type", ""),
                    parameters=p.get("parameters", {}),
                    preconditions=p.get("preconditions", []),
                    expected_outcomes=p.get("expected_outcomes", [])
                )
                primitives.append(primitive)
            
            if not primitives:
                logger.warning("No primitives in LLM response")
                return None
            
            # Create plan
            import uuid
            plan = ActionPlan(
                plan_id=str(uuid.uuid4()),
                explanation=data.get("explanation", ""),
                primitives=primitives
            )
            
            # Estimate duration
            plan.estimated_duration_s = sum(p.timeout_s for p in primitives)
            
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing plan response: {e}")
            return None
    
    def validate_plan(self, plan: ActionPlan) -> tuple[bool, List[str]]:
        """
        Validate an action plan.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        valid_actions = {
            "navigate_to", "look_at", "scan_environment",
            "identify_object", "pick_up", "place",
            "say", "wait", "cancel"
        }
        
        if not plan.primitives:
            errors.append("Plan has no primitives")
            return False, errors
        
        for i, primitive in enumerate(plan.primitives):
            if primitive.action_type not in valid_actions:
                errors.append(f"Step {i+1}: Invalid action type '{primitive.action_type}'")
            
            # Check required parameters
            if primitive.action_type == "navigate_to":
                if not primitive.parameters.get("location") and not primitive.parameters.get("pose"):
                    errors.append(f"Step {i+1}: navigate_to requires 'location' or 'pose'")
            
            elif primitive.action_type == "pick_up":
                if not primitive.parameters.get("object_id"):
                    errors.append(f"Step {i+1}: pick_up requires 'object_id'")
            
            elif primitive.action_type == "say":
                if not primitive.parameters.get("text"):
                    errors.append(f"Step {i+1}: say requires 'text'")
        
        return len(errors) == 0, errors


# Convenience function for quick usage
def create_llm_client(config: Optional[Dict] = None) -> LLMClient:
    """Create an LLM client with optional configuration"""
    config = config or {}
    
    return LLMClient(
        provider=config.get("provider", "openai"),
        model=config.get("model", "gpt-4"),
        temperature=config.get("temperature", 0.2),
        max_tokens=config.get("max_tokens", 1024),
        timeout_s=config.get("timeout_s", 15.0),
        max_retries=config.get("max_retries", 3),
        fallback_provider=config.get("fallback", {}).get("provider"),
        fallback_model=config.get("fallback", {}).get("model"),
        fallback_base_url=config.get("fallback", {}).get("base_url")
    )
