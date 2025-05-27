from pathlib import Path
from agno.agent import Agent
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
from system_prompts import SystemPrompt
from fallback_handler import FallbackHandler
import logging
import io
import contextlib
import re
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_line(line: str) -> str:
    # Lowercase, strip whitespace and trailing punctuation (.,!?)
    line = line.lower().strip()
    return line.rstrip(string.punctuation)

def clean_agent_output(text: str, user_query: str = "") -> str:
    """
    Clean the agent output by removing ANSI escape sequences, box drawing chars,
    log lines, 'Thinking...' prefix, and repeated user query echoes at the start.
    """
    if not text:
        return ""

    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)

    # Remove box drawing unicode characters
    text = re.sub(r'[\u2500-\u257F]', '', text)

    # Remove other control chars except newline and tab
    text = re.sub(r'[^\x09\x0A\x20-\x7E]', '', text)

    lines = text.splitlines()

    # Remove log lines and empty lines
    filtered_lines = [
        line.strip() for line in lines
        if line.strip() and not re.match(
            r'^\s*(INFO|DEBUG|WARNING|ERROR|TRACE|Setting default model).*', line, re.I)
    ]

    user_query_norm = normalize_line(user_query)
    i = 0
    while i < len(filtered_lines):
        line_norm = normalize_line(filtered_lines[i])

        # Remove 'Thinking...' lines
        if line_norm.startswith("thinking"):
            i += 1
            continue

        # Remove lines exactly equal to user query
        if line_norm == user_query_norm:
            i += 1
            continue

        # Remove lines with user query repeated twice or more
        repeated = f"{user_query_norm} {user_query_norm}"
        if line_norm.startswith(repeated):
            i += 1
            continue

        # Remove lines that are user query repeated multiple times separated by spaces
        if all(word == user_query_norm for word in line_norm.split()):
            i += 1
            continue

        # If line doesn't match any above, stop removing
        break

    filtered_lines = filtered_lines[i:]

    cleaned_text = '\n'.join(filtered_lines).strip()

    return cleaned_text

class ChatbotManager:
    """Main chatbot manager that handles initialization and response generation"""

    def __init__(self):
        self.db_url = "postgresql+psycopg://postgres:12345@localhost:5432/ai"
        self.csv_path = Path(r"C:\Users\TRG-LOF-131-050\Desktop\CHATBOT-Lof\Sample_Chatbot1\Laboffuturebot\laboffuture_chunks.csv")
        self.similarity_threshold = 0.7

        print("🚀 Initializing Lab of Future Chatbot...")
        print(f"📁 Using CSV file: {self.csv_path}")
        print(f"🗄️ Using database: {self.db_url}")

        self.system_prompt = SystemPrompt()
        self.fallback_handler = FallbackHandler(similarity_threshold=self.similarity_threshold)
        self.agent = None
        self.knowledge_base = None

        self._initialize_chatbot()

    def _initialize_chatbot(self):
        try:
            print("📚 Loading knowledge base...")
            self.knowledge_base = CSVKnowledgeBase(
                path=self.csv_path,
                vector_db=PgVector(
                    table_name="csv_documents",
                    db_url=self.db_url,
                ),
                num_documents=5,
            )

            self.knowledge_base.load(recreate=False)

            enhanced_system_prompt = f"""
{self.system_prompt.get_full_system_prompt()}

Additional instructions:
- When listing courses or programs, provide only the course titles as a numbered list.
- Do NOT include any descriptions or extra details after the titles.
- Each course title should be on a separate line starting with its number.
- Avoid repeating the user's question in the answer.
"""

            print("🤖 Initializing AI agent...")
            self.agent = Agent(
                knowledge=self.knowledge_base,
                search_knowledge=True,
                instructions=enhanced_system_prompt,
                show_tool_calls=False,
                markdown=False,
                add_history_to_messages=True
            )

            print("✅ Lab of Future Chatbot initialized successfully!")
            print("🎓 Ready to help with courses, company info, and website guidance!")

        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            print(f"❌ Error initializing chatbot: {str(e)}")
            raise e

    def extract_course_headings(self, response_text: str) -> str:
        """Extract only numbered course titles from AI response, removing descriptions."""
        lines = response_text.split('\n')
        titles = []
        for line in lines:
            line = line.strip()
            # Match lines starting with number and dot and bold markdown title, e.g. "1. **Celestial Voyages (Space & Astronomy):** ..."
            m = re.match(r'^\d+\.\s*\*\*(.*?)\*\*[:.]?', line)
            if m:
                titles.append(m.group(1).strip())
            else:
                # Alternative: for lines without bold markdown, split on colon or just trim after number-dot prefix
                if ':' in line:
                    title = line.split(':')[0].strip()
                    # Remove leading numbering if present
                    title = re.sub(r'^\d+\.\s*', '', title)
                    if title:
                        titles.append(title)
                else:
                    # If line starts with number-dot, take rest as title
                    if re.match(r'^\d+\.\s*.+', line):
                        title = re.sub(r'^\d+\.\s*', '', line)
                        titles.append(title)

        return '\n'.join(titles) if titles else response_text

    def get_response(self, user_query: str) -> str:
        try:
            if not self._is_query_acceptable(user_query):
                logger.info(f"Query outside scope: {user_query}")
                return self.fallback_handler.get_fallback_response(user_query)

            logger.info(f"Processing query with agent: {user_query}")

            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                self.agent.print_response(user_query, markdown=False)
            raw_response = captured_output.getvalue()

            cleaned_response = clean_agent_output(raw_response, user_query=user_query)
            final_response = self._clean_response(cleaned_response, user_query=user_query)

            # Post-process to extract only course titles if the user query is about courses or programs
            if any(keyword in user_query.lower() for keyword in ["course", "program", "lab program"]):
                final_response = self.extract_course_headings(final_response)

            if not self._is_answer_relevant(final_response):
                logger.info("Response not relevant, using fallback")
                return self.fallback_handler.get_fallback_response(user_query)

            processed_response, used_fallback = self.fallback_handler.process_response(final_response, user_query)
            if used_fallback:
                logger.info("Used fallback response")
                return processed_response
            else:
                enhanced_response = self.fallback_handler.enhance_response(processed_response, user_query)
                logger.info("Response processed successfully")
                return enhanced_response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self.system_prompt.get_error_message()

    def _clean_response(self, response: str, user_query: str = "") -> str:
        """
        Remove extra lines and metadata from the response and strip repeated query substrings.
        """
        if not response:
            return "I couldn't generate a proper response."

        lines = response.split('\n')
        cleaned_lines = [
            line.strip()
            for line in lines
            if line.strip() and
               not any(skip_word in line for skip_word in ["Message", "Response", "┌", "│", "└", "├", "─"])
        ]

        # Remove repeated user query substrings at start of first line
        if user_query and cleaned_lines:
            query = user_query.lower().strip()
            first_line_lower = cleaned_lines[0].lower()

            while first_line_lower.startswith(query):
                cleaned_lines[0] = cleaned_lines[0][len(query):].strip()
                if not cleaned_lines[0]:
                    cleaned_lines.pop(0)
                    if not cleaned_lines:
                        break
                    first_line_lower = cleaned_lines[0].lower()
                else:
                    first_line_lower = cleaned_lines[0].lower()

        final_response = '\n'.join(cleaned_lines).strip()

        if not final_response or len(final_response) < 5:
            return "I'm here to help you with Lab of Future information. What would you like to know?"

        return final_response

    def _is_query_acceptable(self, query: str) -> bool:
        query_lower = query.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in query_lower for greeting in greetings):
            return True
        return self.fallback_handler.is_educational_query(query)

    def _is_answer_relevant(self, response_text: str) -> bool:
        if not response_text or response_text.strip() == "":
            return False

        dont_know_patterns = [
            "i don't know",
            "i'm not sure",
            "i cannot provide",
            "i can't provide",
            "sorry, i don't have",
            "i don't have information",
            "i'm unable to",
            "i apologize, but i don't know",
            "couldn't generate a proper response",
            "i couldn't find"
        ]

        response_lower = response_text.lower()
        for pattern in dont_know_patterns:
            if pattern in response_lower:
                return False

        if len(response_text.strip()) < 10:
            return False

        return True

    def get_greeting(self) -> str:
        return "Hello! I'm your Lab of Future learning assistant. How can I help you today? 📚"


if __name__ == "__main__":
    try:
        chatbot = ChatbotManager()

        print("============================================================")
        print("🎓 Lab of Future Chatbot is ready!")
        print("🤖 I'm here to help with courses, company info, and website guidance")
        print("💬 Type 'exit', 'quit', or 'bye' to stop")
        print("🔄 Type 'help' for available commands")
        print("============================================================")

        print("Chatbot:", chatbot.get_greeting())

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Thank you for using Lab of Future chatbot. Goodbye! 👋")
                break

            if user_input.lower() == 'help':
                print("Chatbot: I can help you with:")
                print("• Information about Lab of Future courses and programs")
                print("• Company details and background")
                print("• Enrollment and registration information")
                print("• Website navigation and support")
                print("• Pricing and schedule details")
                continue

            if not user_input:
                print("Chatbot: Please ask me something about Lab of Future!")
                continue

            response = chatbot.get_response(user_input)
            print("Chatbot:", response)

    except KeyboardInterrupt:
        print("\n\nChatbot: Goodbye! 👋")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {str(e)}")
        print("Please check your database connection and CSV file path.")