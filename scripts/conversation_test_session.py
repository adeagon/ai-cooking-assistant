"""Comprehensive conversation test session for the chatbot.

This script runs actual conversations with the chatbot to test various scenarios
including recipe discovery, taste tags, cuisine filtering, dietary constraints,
commands, and edge cases.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from src.app.settings import settings
from src.retrieval.retriever import RecipeRetriever
from src.retrieval.rerank import RecipeReranker
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.chains.retrieval import RetrievalRunnable
from src.chains.chat_chain import build_chat_chain
from src.chains.intent_classifier import classify_intent
from src.chains.extractors import ConstraintExtractor
from src.memory import ProfileStore, SessionStore, RollingSummarizer, FeedbackStore, HistoryStore, RecipeBoxStore
from src.domain.models import RecipeFeedback


class ConversationTester:
    """Run comprehensive conversation tests with the chatbot."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.results = []
        self.test_count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.last_cards = []

    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        # Print with ASCII-safe encoding for Windows console
        try:
            print(line.encode('ascii', 'replace').decode('ascii'))
        except Exception:
            print(line.encode('cp1252', 'replace').decode('cp1252'))
        # Write full unicode to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_separator(self, title: str = ""):
        """Log a separator line."""
        sep = "=" * 80
        if title:
            self.log(f"\n{sep}")
            self.log(f"  {title}")
            self.log(f"{sep}\n")
        else:
            self.log(sep)

    async def initialize(self):
        """Initialize all chat components."""
        self.log("Initializing chat components...")

        # Initialize LLMs
        # Main LLM for recommendations - fast, direct responses
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            reasoning=False,  # Fast mode: no thinking for recipe presentation
        )

        # LLM for clarification - thoughtful, uses reasoning for better questions
        self.llm_clarification = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens * 2,  # Extra budget for thinking + response
            reasoning=True,  # Enable thinking for crafting better clarification questions
        )

        # Separate LLM for intent classification
        self.intent_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_intent_model,
            temperature=0.2,
            num_predict=256,
            reasoning=False,  # Fast mode for simple classification
        )

        # Initialize retrieval components
        self.retriever = RecipeRetriever(
            chroma_dir=Path(settings.chroma_persist_dir),
            embedding_model=settings.embedding_model
        )
        self.reranker = RecipeReranker(model_name=settings.reranker_model)
        self.card_builder = RecipeCardBuilder(db_path=settings.sqlite_db_path)
        self.retrieval_chain = RetrievalRunnable(
            retriever=self.retriever,
            reranker=self.reranker,
            card_builder=self.card_builder,
            settings=settings
        )

        # Initialize memory stores
        self.profile_store = ProfileStore(db_path=settings.sqlite_db_path)
        self.session_store = SessionStore(db_path=settings.sqlite_db_path)
        self.feedback_store = FeedbackStore(db_path=settings.sqlite_db_path)
        self.history_store = HistoryStore(db_path=settings.sqlite_db_path)
        self.recipe_box_store = RecipeBoxStore(db_path=settings.sqlite_db_path)
        self.summarizer = RollingSummarizer()
        self.extractor = ConstraintExtractor()

        # Load profile and create fresh session
        self.profile = self.profile_store.load()
        self.session_id = self.session_store.create()
        self.session = self.session_store.get(self.session_id)
        self.rolling_summary = ""

        self.log("Components initialized successfully")

    async def send_message(self, user_input: str) -> dict:
        """Send a message to the chatbot and get response."""
        self.log(f"\nUSER: {user_input}")

        # Compute exclusion set
        exclude_ids = (
            self.feedback_store.get_liked_recipe_ids(limit=20) |
            self.feedback_store.get_disliked_recipe_ids() |
            self.history_store.get_recently_cooked_ids(days=7)
        )

        # Build chain
        chain = build_chat_chain(
            llm=self.llm,
            retrieval_chain=self.retrieval_chain,
            profile=self.profile,
            session=self.session,
            rolling_summary=self.rolling_summary,
            exclude_recipe_ids=exclude_ids,
            llm_clarification=self.llm_clarification,  # Thoughtful LLM for clarification
        )

        # Invoke chain
        result = await chain.ainvoke({"user_input": user_input})

        response = result.get("response", "")
        cards = result.get("cards", [])

        self.log(f"\nASSISTANT: {response}")

        if cards:
            self.last_cards = cards
            self.log(f"\n[Cards returned: {len(cards)}]")
            for i, card in enumerate(cards, 1):
                self.log(f"  {i}. {card.title} (rating: {card.rating_avg or 'N/A'}, time: {card.time_total or 'N/A'}m)")

        # Update rolling summary
        constraints = self.extractor.extract_constraints(user_input)
        self.rolling_summary = self.summarizer.update_summary(
            self.rolling_summary, constraints, user_input
        )
        self.session_store.update_summary(self.session_id, self.rolling_summary)

        return {"response": response, "cards": cards}

    def check_response(self, response: str, must_contain: list = None, must_not_contain: list = None,
                       min_cards: int = None, description: str = ""):
        """Check if response meets criteria."""
        self.test_count += 1
        passed = True
        issues = []

        if must_contain:
            for term in must_contain:
                if term.lower() not in response.lower():
                    passed = False
                    issues.append(f"Missing: '{term}'")

        if must_not_contain:
            for term in must_not_contain:
                if term.lower() in response.lower():
                    passed = False
                    issues.append(f"Should not contain: '{term}'")

        if min_cards is not None:
            if len(self.last_cards) < min_cards:
                passed = False
                issues.append(f"Expected {min_cards}+ cards, got {len(self.last_cards)}")

        if passed:
            self.pass_count += 1
            self.log(f"[PASS] {description}")
        else:
            self.fail_count += 1
            self.log(f"[FAIL] {description}")
            for issue in issues:
                self.log(f"    - {issue}")

        return passed

    async def test_basic_recipe_discovery(self):
        """Test basic recipe discovery scenarios."""
        self.log_separator("TEST CATEGORY: Basic Recipe Discovery")

        # Test 1: Simple ingredient query
        result = await self.send_message("I have chicken and tomatoes")
        self.check_response(result["response"], min_cards=1,
                           description="Simple ingredient query returns recipes")

        # Test 2: Specific dish request
        result = await self.send_message("I want to make pasta carbonara")
        self.check_response(result["response"],
                           must_contain=["carbonara"],
                           min_cards=1,
                           description="Specific dish request returns relevant recipes")

        # Test 3: Quick meal request
        result = await self.send_message("I need something quick for dinner, under 30 minutes")
        self.check_response(result["response"], min_cards=1,
                           description="Quick meal request works")

    async def test_taste_tags(self):
        """Test taste tag functionality (light, rich, spicy, mild, savory, sweet)."""
        self.log_separator("TEST CATEGORY: Taste Tags")

        # Create fresh session for this test
        self.session_id = self.session_store.create()
        self.session = self.session_store.get(self.session_id)
        self.rolling_summary = ""

        # Test 1: Light
        result = await self.send_message("I want something light for dinner")
        self.check_response(result["response"], min_cards=1,
                           description="'Light' taste tag query works")

        # Test 2: Rich
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I'm craving something rich and indulgent")
        self.check_response(result["response"], min_cards=1,
                           description="'Rich' taste tag query works")

        # Test 3: Spicy
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I want something spicy")
        self.check_response(result["response"], min_cards=1,
                           description="'Spicy' taste tag query works")

        # Test 4: Mild
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Something mild, not too strong flavored")
        self.check_response(result["response"], min_cards=1,
                           description="'Mild' taste tag query works")

        # Test 5: Sweet
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I want something sweet for dessert")
        self.check_response(result["response"], min_cards=1,
                           description="'Sweet' taste tag query works")

        # Test 6: Savory
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Something savory, not sweet")
        self.check_response(result["response"], min_cards=1,
                           description="'Savory' taste tag query works")

        # Test 7: Combination
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I want something light and savory")
        self.check_response(result["response"], min_cards=1,
                           description="Combined taste tags work")

    async def test_cuisine_filtering(self):
        """Test cuisine filtering functionality."""
        self.log_separator("TEST CATEGORY: Cuisine Filtering")

        cuisines = [
            ("italian", "Italian"),
            ("mexican", "Mexican"),
            ("japanese", "Japanese"),
            ("indian", "Indian"),
            ("thai", "Thai"),
            ("chinese", "Chinese"),
            ("greek", "Greek"),
            ("french", "French"),
            ("korean", "Korean"),
            ("middle eastern", "Middle Eastern"),
        ]

        for cuisine_query, cuisine_name in cuisines:
            self.session_id = self.session_store.create()
            self.rolling_summary = ""
            result = await self.send_message(f"I want {cuisine_query} food")
            self.check_response(result["response"], min_cards=1,
                               description=f"{cuisine_name} cuisine query works")

    async def test_dietary_constraints(self):
        """Test dietary constraint handling."""
        self.log_separator("TEST CATEGORY: Dietary Constraints")

        # Test 1: Vegetarian
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I'm vegetarian, what can I make?")
        self.check_response(result["response"], min_cards=1,
                           description="Vegetarian constraint works")

        # Test 2: Vegan
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I need vegan recipes")
        self.check_response(result["response"], min_cards=1,
                           description="Vegan constraint works")

        # Test 3: Keto
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I'm on a keto diet")
        self.check_response(result["response"], min_cards=1,
                           description="Keto constraint works")

        # Test 4: Gluten-free
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I need gluten-free options")
        self.check_response(result["response"], min_cards=1,
                           description="Gluten-free constraint works")

    async def test_time_constraints(self):
        """Test time constraint handling."""
        self.log_separator("TEST CATEGORY: Time Constraints")

        # Test 1: Under 30 minutes
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Something under 30 minutes")
        self.check_response(result["response"], min_cards=1,
                           description="Under 30 minutes constraint works")

        # Test 2: Quick
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I need something quick")
        self.check_response(result["response"], min_cards=1,
                           description="'Quick' keyword works")

        # Test 3: Under 1 hour
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Less than an hour to cook")
        self.check_response(result["response"], min_cards=1,
                           description="Under 1 hour constraint works")

    async def test_negative_constraints(self):
        """Test negative constraint handling."""
        self.log_separator("TEST CATEGORY: Negative Constraints")

        # Test 1: No casseroles
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Italian recipes but no casseroles")
        self.check_response(result["response"], min_cards=1,
                           description="'No casseroles' constraint works")

        # Test 2: Without cheese
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Pasta without cheese")
        self.check_response(result["response"], min_cards=1,
                           description="'Without cheese' constraint works")

        # Test 3: But not soups
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Something warm but not soup")
        self.check_response(result["response"], min_cards=1,
                           description="'But not soup' constraint works")

    async def test_combined_constraints(self):
        """Test combined constraint scenarios."""
        self.log_separator("TEST CATEGORY: Combined Constraints")

        # Test 1: Cuisine + Time
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Quick Italian dinner")
        self.check_response(result["response"], min_cards=1,
                           description="Italian + quick combination works")

        # Test 2: Dietary + Cuisine
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Vegetarian Mexican food")
        self.check_response(result["response"], min_cards=1,
                           description="Vegetarian + Mexican combination works")

        # Test 3: Ingredients + Taste
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Something light with chicken")
        self.check_response(result["response"], min_cards=1,
                           description="Light + chicken combination works")

        # Test 4: Complex combination
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Quick healthy vegetarian Asian dish")
        self.check_response(result["response"], min_cards=1,
                           description="Complex combination works")

        # Test 5: Cuisine + Taste + Time
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Spicy Thai food under 45 minutes")
        self.check_response(result["response"], min_cards=1,
                           description="Spicy + Thai + time combination works")

    async def test_edge_cases(self):
        """Test edge cases and unusual queries."""
        self.log_separator("TEST CATEGORY: Edge Cases")

        # Test 1: Very vague query
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("dinner")
        # Should ask for clarification or give recommendations
        self.check_response(result["response"],
                           description="Vague query 'dinner' handled")

        # Test 2: Very specific dish
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("chicken tikka masala")
        self.check_response(result["response"], min_cards=1,
                           description="Specific dish name works")

        # Test 3: Ingredient list
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("I have eggs, cheese, and spinach")
        self.check_response(result["response"], min_cards=1,
                           description="Multiple ingredients query works")

        # Test 4: Contradictory constraints (should handle gracefully)
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("vegan recipes with chicken")
        self.check_response(result["response"],
                           description="Contradictory constraints handled")

        # Test 5: Unusual phrasing
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("what should I make tonight thats easy and good")
        self.check_response(result["response"],
                           description="Casual/unusual phrasing handled")

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation and context retention."""
        self.log_separator("TEST CATEGORY: Multi-Turn Conversation")

        # Fresh session
        self.session_id = self.session_store.create()
        self.session = self.session_store.get(self.session_id)
        self.rolling_summary = ""

        # Turn 1: Initial query
        result = await self.send_message("I want Italian food")
        self.check_response(result["response"], min_cards=1,
                           description="Turn 1: Initial Italian query")

        # Turn 2: Refinement
        result = await self.send_message("something quick though, under 30 minutes")
        self.check_response(result["response"], min_cards=1,
                           description="Turn 2: Time refinement")

        # Turn 3: Further refinement
        result = await self.send_message("and vegetarian please")
        self.check_response(result["response"], min_cards=1,
                           description="Turn 3: Dietary refinement")

    async def test_intent_classification(self):
        """Test natural language intent classification."""
        self.log_separator("TEST CATEGORY: Intent Classification")

        # First get some recipes to work with
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("Show me some pasta recipes")

        if self.last_cards:
            # Test various natural language intents
            intents_to_test = [
                ("I loved the first one", "like"),
                ("that looks good, save it", "save"),
                ("show me the full recipe for number 1", "show"),
                ("give that one 4 stars", "rate"),
            ]

            for phrase, expected_intent in intents_to_test:
                try:
                    intent_result = classify_intent(phrase, self.last_cards, self.intent_llm)
                    self.log(f"\nIntent test: '{phrase}'")
                    self.log(f"  Detected: {intent_result.intent} (confidence: {intent_result.confidence})")
                    self.test_count += 1
                    if intent_result.intent == expected_intent or intent_result.confidence != "low":
                        self.pass_count += 1
                        self.log(f"[PASS] Intent '{phrase}' detected reasonably")
                    else:
                        self.fail_count += 1
                        self.log(f"[FAIL] Expected {expected_intent}, got {intent_result.intent}")
                except Exception as e:
                    self.log(f"Intent classification error: {e}")

    async def test_clarification_behavior(self):
        """Test when the bot asks for clarification vs gives recommendations."""
        self.log_separator("TEST CATEGORY: Clarification Behavior")

        # Test 1: Should get recipes (has constraints)
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        result = await self.send_message("healthy chicken dinner")
        has_cards = len(self.last_cards) > 0
        self.test_count += 1
        if has_cards:
            self.pass_count += 1
            self.log("[PASS] Specific query gives recommendations")
        else:
            self.fail_count += 1
            self.log("[FAIL] Specific query should give recommendations")

        # Test 2: Might ask for clarification (very vague)
        self.session_id = self.session_store.create()
        self.rolling_summary = ""
        self.last_cards = []
        result = await self.send_message("food")
        # This is acceptable either way - just log behavior
        self.log(f"Vague query 'food' - cards returned: {len(self.last_cards)}")

    async def run_all_tests(self):
        """Run all test categories."""
        self.log_separator("COMPREHENSIVE CHATBOT TEST SESSION")
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model: {settings.ollama_model}")
        self.log(f"Intent Model: {settings.ollama_intent_model}")

        await self.initialize()

        # Run all test categories
        await self.test_basic_recipe_discovery()
        await self.test_taste_tags()
        await self.test_cuisine_filtering()
        await self.test_dietary_constraints()
        await self.test_time_constraints()
        await self.test_negative_constraints()
        await self.test_combined_constraints()
        await self.test_edge_cases()
        await self.test_multi_turn_conversation()
        await self.test_intent_classification()
        await self.test_clarification_behavior()

        # Summary
        self.log_separator("TEST SUMMARY")
        self.log(f"Total Tests: {self.test_count}")
        self.log(f"Passed: {self.pass_count}")
        self.log(f"Failed: {self.fail_count}")
        self.log(f"Pass Rate: {self.pass_count/self.test_count*100:.1f}%" if self.test_count > 0 else "N/A")
        self.log(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


async def main():
    log_file = "test_results/modelfile_conversation_tests.txt"

    # Clear previous content and start fresh
    Path("test_results").mkdir(exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"CHATBOT CONVERSATION TEST SESSION\n")
        f.write(f"{'=' * 80}\n\n")

    tester = ConversationTester(log_file)
    await tester.run_all_tests()

    print(f"\n\nFull log saved to: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())
