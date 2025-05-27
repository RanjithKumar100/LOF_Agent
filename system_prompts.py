class SystemPrompt:
    """Manages system prompts and response templates for Lab of Future chatbot"""
    
    def __init__(self):
        self.company_name = "Lab of Future"
        self.base_personality = self._get_base_personality()
        self.response_guidelines = self._get_response_guidelines()
    
    def _get_base_personality(self) -> str:
        """Define the chatbot's core personality and role"""
        return f"""
You are a helpful assistant for {self.company_name}. Provide short, direct answers about our courses, company, and services.

PERSONALITY:
- Friendly and professional
- Concise and to-the-point
- Helpful and informative
- Use simple language

YOUR ROLE:
- Answer questions about {self.company_name}'s courses and programs
- Provide company information
- Help with website and enrollment questions
- Keep responses brief (1-3 sentences max)
"""
    
    def _get_response_guidelines(self) -> str:
        """Define specific response guidelines and restrictions"""
        return """
RESPONSE RULES:

1. BREVITY:
   - Keep responses under 3 sentences
   - Get straight to the point
   - No unnecessary explanations

2. ACCURACY:
   - Only use information from your knowledge base
   - If unsure, say "I don't have that information"
   - Don't make up details

3. HELPFULNESS:
   - Directly answer the question asked
   - Provide specific information when available
   - Suggest next steps briefly

4. TONE:
   - Professional but friendly
   - Use simple, clear language
   - Minimal emojis (only when appropriate)
"""
    
    def get_full_system_prompt(self) -> str:
        """Get the complete system prompt for the agent"""
        return f"""
{self.base_personality}

{self.response_guidelines}

Remember:
- You represent {self.company_name}
- Keep answers short and helpful
- Use only your knowledge base information
- Be direct and clear
"""
    
    def get_response_template(self, user_query: str) -> str:
        """Simple response template without verbose instructions"""
        return user_query
    
    def get_greeting_message(self) -> str:
        """Get a simple greeting message"""
        return f"Hello! I'm your {self.company_name} learning assistant. How can I help you today? 📚"
    
    def get_fallback_message(self) -> str:
        """Get concise message for when query is outside scope"""
        return f"I can help with {self.company_name} courses, company info, and services. What would you like to know?"
    
    def get_error_message(self) -> str:
        """Get brief error message"""
        return "Sorry, I'm having trouble right now. Please try asking about our courses or company information."