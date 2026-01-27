# CHUNK_META:
#   Purpose: Separate CLI for Papa (creator) - special interface with trust
#   Dependencies: pfc, train, hippocampus, cortex
#   API: main(), PapaSession

"""
Papa CLI — Dedicated interface for the creator/parent.

BIOLOGY:
- Creator has special relationship with system (like parent to child)
- Information from papa is trusted and stored long-term
- Papa can resolve contradictions and update knowledge
- Separate interface ensures only papa uses this mode

This is NOT hardcoded importance - it's a special RELATIONSHIP
established through the interface, like how a child knows
who their parent is through consistent interaction.
"""

from __future__ import annotations

import sys
from typing import Optional, Dict, Any

from pfc import PFC, MemoryRouter, SourceType, AttentionGate


# ANCHOR: PAPA_SESSION - session management for papa
class PapaSession:
    """
    Session for papa's interaction with Brain.
    
    BIOLOGY: Parent-child relationship is special:
    - Consistent interaction builds trust
    - Parent can teach and correct
    - Information about parent is important
    
    Intent: Provide dedicated interface where papa's input
            is processed with appropriate trust level.
    """
    
    def __init__(self):
        """Initialize papa session."""
        self.pfc = PFC()
        self.router = MemoryRouter(self.pfc)
        self.attention = AttentionGate(self.pfc)
        self._pending_contradiction: Optional[Dict[str, Any]] = None
        self._history: list = []
        
        # Papa's known information (builds over time)
        self._papa_info: Dict[str, str] = {}
    
    # API_PUBLIC
    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Process input from papa.
        
        Args:
            text: Papa's input text
        
        Returns:
            Response with action taken
        """
        text = text.strip()
        if not text:
            return {"action": "empty", "response": ""}
        
        self._history.append({"role": "papa", "text": text})
        
        # Check if resolving pending contradiction
        if self._pending_contradiction:
            return self._resolve_contradiction(text)
        
        # Process through router as PAPA source
        result = self.router.process(text, SourceType.PAPA)
        
        if result["action"] == "ask_clarification":
            # Store pending contradiction for resolution
            self._pending_contradiction = {
                "tokens": result["tokens"],
                "contradiction": result["contradiction"],
            }
            response = result.get("message", "I'm confused. Which is correct?")
            return {"action": "ask", "response": response}
        
        elif result["action"] == "consolidate_important":
            # Store papa's information
            self._store_papa_info(text, result["tokens"])
            return {
                "action": "stored",
                "response": f"I'll remember that.",
            }
        
        return {"action": result["action"], "response": "OK"}
    
    # API_PUBLIC
    def ask(self, question: str) -> str:
        """
        Papa asks a question.
        
        Args:
            question: Question text
        
        Returns:
            Answer from Brain
        """
        self._history.append({"role": "papa", "text": question})
        
        # Set goal in PFC
        tokens = set(question.lower().split())
        self.pfc.set_goal(tokens, metadata={"question": question})
        
        # Try to answer from papa_info first
        for key, value in self._papa_info.items():
            if key in question.lower():
                return value
        
        # Would integrate with train.ask() here
        return "I don't know yet."
    
    # API_PRIVATE
    def _resolve_contradiction(self, response: str) -> Dict[str, Any]:
        """Resolve pending contradiction based on papa's answer."""
        response_lower = response.lower()
        
        if self._pending_contradiction:
            tokens = self._pending_contradiction["tokens"]
            
            # Check papa's response
            if "new" in response_lower or "correct" in response_lower or "yes" in response_lower:
                # Papa confirms new info is correct
                self.router.resolve_contradiction(keep_new=True, tokens=tokens)
                self._pending_contradiction = None
                return {
                    "action": "updated",
                    "response": "I've updated my knowledge.",
                }
            elif "old" in response_lower or "no" in response_lower or "keep" in response_lower:
                # Papa says keep old knowledge
                self._pending_contradiction = None
                return {
                    "action": "kept",
                    "response": "I'll keep what I knew before.",
                }
            else:
                # Unclear response
                return {
                    "action": "ask",
                    "response": "I don't understand. Should I update my knowledge? (yes/no)",
                }
        
        return {"action": "error", "response": "No pending contradiction."}
    
    # API_PRIVATE
    def _store_papa_info(self, text: str, tokens: set) -> None:
        """Store information about/from papa."""
        text_lower = text.lower()
        
        # Extract key-value patterns
        if "my name is" in text_lower:
            name = text_lower.split("my name is")[-1].strip().split()[0]
            self._papa_info["name"] = name
            self._papa_info["papa_name"] = name
        
        if "call me" in text_lower:
            nickname = text_lower.split("call me")[-1].strip().split()[0]
            self._papa_info["nickname"] = nickname
        
        if "i like" in text_lower or "i love" in text_lower:
            thing = text_lower.split("like" if "like" in text_lower else "love")[-1].strip()
            self._papa_info["likes"] = thing
    
    # API_PUBLIC
    def get_papa_info(self) -> Dict[str, str]:
        """Get stored information about papa."""
        return self._papa_info.copy()


# ANCHOR: CLI_MAIN - command line interface
def main() -> None:
    """
    Main CLI loop for papa.
    
    Commands:
        /info - show what Brain knows about papa
        /clear - clear PFC
        /quit - exit
    """
    print("=" * 60)
    print("PAPA CLI - Brain's Creator Interface")
    print("=" * 60)
    print("This is a special interface for papa (creator).")
    print("Information you provide will be stored as important.")
    print("")
    print("Commands: /info, /clear, /quit")
    print("=" * 60)
    print("")
    
    session = PapaSession()
    
    while True:
        try:
            user_input = input("Papa> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye, papa!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye, papa!")
                break
            elif cmd == "/info":
                info = session.get_papa_info()
                if info:
                    print("What I know about you:")
                    for k, v in info.items():
                        print(f"  {k}: {v}")
                else:
                    print("I don't know anything about you yet.")
                continue
            elif cmd == "/clear":
                session.pfc.clear()
                print("PFC cleared.")
                continue
            elif cmd == "/pfc":
                print(f"PFC: {session.pfc}")
                print(f"Active: {session.pfc.get_active_tokens()}")
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Check if it's a question
        if user_input.endswith("?"):
            answer = session.ask(user_input)
            print(f"Brain: {answer}")
        else:
            # Process as statement
            result = session.process_input(user_input)
            print(f"Brain: {result['response']}")


if __name__ == "__main__":
    main()
