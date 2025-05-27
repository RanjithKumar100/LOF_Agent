import re
from typing import Tuple, List

class FallbackHandler:
    """Handles fallback responses and ensures chatbot stays within scope"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.company_name = "Lab of Future"
        
        # Define allowed topics and keywords
        self.allowed_topics = self._get_allowed_topics()
        self.restricted_topics = self._get_restricted_topics()
        self.company_keywords = self._get_company_keywords()
        
    def _get_allowed_topics(self) -> List[str]:
        """Define topics that are within the chatbot's scope"""
        return [
            # Company related
            "lab of future", "laboffuture", "company", "about us", "mission", "vision",
            "team", "contact", "location", "office", "history", "background",
            
            # Courses and education
            "course", "courses", "program", "programs", "training", "education",
            "learning", "class", "classes", "curriculum", "syllabus", "module",
            "lesson", "tutorial", "workshop", "seminar", "bootcamp",
            
            # Services
            "service", "services", "offering", "offerings", "enrollment", "admission",
            "registration", "fees", "pricing", "cost", "payment", "schedule",
            "duration", "certificate", "certification", "diploma",
            
            # Website and platform
            "website", "platform", "portal", "login", "account", "dashboard",
            "navigation", "help", "support", "faq", "documentation",
            
            # General inquiries
            "how", "what", "when", "where", "why", "who", "can", "do", "is",
            "are", "will", "would", "should", "could"
        ]
    
    def _get_restricted_topics(self) -> List[str]:
        """Define topics that are outside the chatbot's scope"""
        return [
            # General knowledge
            "weather", "news", "politics", "sports", "entertainment", "celebrity",
            "movie", "music", "book", "recipe", "cooking", "health", "medical",
            "financial advice", "investment", "stock", "cryptocurrency",
            
            # Technical help (non-company)
            "programming", "coding", "debug", "error", "bug", "software",
            "hardware", "computer", "troubleshoot", "fix", "install",
            
            # Personal advice
            "relationship", "dating", "marriage", "family", "personal",
            "psychological", "therapy", "counseling", "advice",
            
            # Other services
            "job", "career", "resume", "interview", "hiring", "recruitment",
            "travel", "hotel", "flight", "vacation", "tourism"
        ]
    
    def _get_company_keywords(self) -> List[str]:
        """Keywords that clearly indicate company-related queries"""
        return [
            "lab of future", "laboffuture", "lof", "your company", "this company",
            "your courses", "your programs", "your services", "your website",
            "enroll", "register", "sign up", "join", "apply"
        ]
    
    def is_educational_query(self, query: str) -> bool:
        """
        Determine if a query is within the educational/company scope
        Returns True if query should be processed, False if it should be rejected
        """
        query_lower = query.lower().strip()
        
        # Empty or very short queries
        if len(query_lower) < 3:
            return False
        
        # Check for company keywords (high priority)
        for keyword in self.company_keywords:
            if keyword in query_lower:
                return True
        
        # Check for restricted topics (immediate rejection)
        for restricted in self.restricted_topics:
            if restricted in query_lower:
                # Double-check if it's company-related despite restricted keyword
                company_context = any(keyword in query_lower for keyword in self.company_keywords[:5])
                if not company_context:
                    return False
        
        # Check for allowed topics
        allowed_count = sum(1 for topic in self.allowed_topics if topic in query_lower)
        
        # If query contains multiple allowed topics, it's likely relevant
        if allowed_count >= 2:
            return True
        
        # If query contains at least one allowed topic and isn't restricted, allow it
        if allowed_count >= 1:
            return True
        
        # Default to allowing queries that might be relevant
        # This ensures we don't accidentally block legitimate questions
        return True
    
    def get_fallback_response(self, query: str) -> str:
        """Get appropriate fallback response for out-of-scope queries"""
        query_lower = query.lower().strip()
        
        # Detect specific types of restricted queries for tailored responses
        if any(topic in query_lower for topic in ["weather", "news", "politics"]):
            return self._get_general_knowledge_fallback()
        elif any(topic in query_lower for topic in ["health", "medical", "advice"]):
            return self._get_advice_fallback()
        elif any(topic in query_lower for topic in ["programming", "coding", "technical"]):
            return self._get_technical_fallback()
        else:
            return self._get_standard_fallback()
    
    def _get_standard_fallback(self) -> str:
        """Standard fallback response"""
        return f"""
I'm specifically designed to help with questions about {self.company_name}. While I'd love to chat about everything, I focus on providing accurate information about our company, courses, and services.

Here's what I can help you with:
• Information about our courses and programs
• Company details and background
• Enrollment and registration process  
• Website navigation and support
• Pricing and schedule information

What would you like to know about {self.company_name}? 🎓
"""
    
    def _get_general_knowledge_fallback(self) -> str:
        """Fallback for general knowledge queries"""
        return f"""
I appreciate your curiosity! However, I'm focused on helping with {self.company_name} related questions rather than general information.

I'm your go-to resource for:
• Our educational programs and courses
• Company information and services
• Enrollment guidance and support

Is there anything about {self.company_name} that I can help you explore today?
"""
    
    def _get_advice_fallback(self) -> str:
        """Fallback for advice-seeking queries"""
        return f"""
While I can't provide personal advice, I'm here to help you with all things {self.company_name}!

Let me assist you with:
• Finding the right course for your goals
• Understanding our programs and offerings
• Getting started with enrollment
• Navigating our learning platform

What educational opportunities at {self.company_name} interest you most?
"""
    
    def _get_technical_fallback(self) -> str:
        """Fallback for technical queries not related to company"""
        return f"""
For technical questions, I focus specifically on {self.company_name}'s platform and services rather than general technical support.

I can help you with:
• Our learning platform navigation
• Course access and technical requirements
• Account setup and login issues
• {self.company_name} website functionality

Do you need help with any technical aspects of our platform or courses?
"""
    
    def process_response(self, response: str, original_query: str) -> Tuple[str, bool]:
        """
        Process the agent's response to ensure it stays within scope
        Returns: (processed_response, used_fallback)
        """
        if not response or len(response.strip()) < 10:
            return self.get_fallback_response(original_query), True
        
        # Check if response seems to contain information outside our scope
        if self._contains_external_info(response):
            return self.get_fallback_response(original_query), True
        
        # If response is good, return as-is
        return response, False
    
    def _contains_external_info(self, response: str) -> bool:
        """Check if response contains information outside company scope"""
        response_lower = response.lower()
        
        # Indicators of external information
        external_indicators = [
            "according to", "research shows", "studies indicate", "experts say",
            "it is known that", "generally speaking", "in the real world",
            "outside of", "beyond", "external", "third party"
        ]
        
        # Check for external information indicators
        external_count = sum(1 for indicator in external_indicators if indicator in response_lower)
        
        # If response has multiple external indicators, it might be off-topic
        return external_count >= 2
    
    def enhance_response(self, response: str, original_query: str) -> str:
        """
        Enhance response with company context and call-to-action
        """
        if not response:
            return self.get_fallback_response(original_query)
        
        # Don't enhance fallback responses (they're already complete)
        if "specifically designed to help" in response or "focus on providing" in response:
            return response
        
        # Add helpful closing for company responses
        enhanced_response = response.strip()
        
        # Add call-to-action if response doesn't already have one
        if not any(phrase in enhanced_response.lower() for phrase in ["let me know", "feel free", "contact", "help you", "questions"]):
            enhanced_response += f"\n\nIs there anything else about {self.company_name} that I can help you with today?"
        
        return enhanced_response