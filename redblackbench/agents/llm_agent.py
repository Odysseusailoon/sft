"""LLM-powered agent implementation for RedBlackBench."""

import re
from typing import List, Optional, TYPE_CHECKING, Tuple

from redblackbench.agents.base import BaseAgent, AgentResponse
from redblackbench.agents.prompts import (
    build_system_prompt,
    build_initial_opinion_prompt,
    build_final_vote_prompt,
    build_willingness_prompt,
    PromptTemplate,
    DEFAULT_PROMPTS,
)
from redblackbench.game.scoring import Choice

if TYPE_CHECKING:
    from redblackbench.providers.base import BaseLLMProvider


class LLMAgent(BaseAgent):
    """An agent powered by a Large Language Model.

    Uses an LLM provider to generate responses during deliberation and voting.
    """

    # Approximate token limit for context (conservative to leave room for response)
    # Model limit is 65535, we need ~2000 for response, so max input ~63000
    # But being more conservative to account for system prompt overhead
    MAX_CONTEXT_TOKENS = 40000
    # Rough estimate: 1 token ≈ 4 characters (conservative - some models use ~3.5)
    CHARS_PER_TOKEN = 3

    def __init__(
        self,
        agent_id: str,
        team_name: str,
        provider: "BaseLLMProvider",
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """Initialize the LLM agent.

        Args:
            agent_id: Unique identifier for this agent
            team_name: Name of the team this agent belongs to
            provider: LLM provider for generating responses
            prompt_template: Optional custom prompt template
        """
        super().__init__(agent_id, team_name)
        self.provider = provider
        self.prompt_template = prompt_template or DEFAULT_PROMPTS
        self._system_prompt = build_system_prompt(
            agent_id, team_name, self.prompt_template
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        return len(text) // self.CHARS_PER_TOKEN

    def _truncate_history_if_needed(self, messages: List[dict]) -> List[dict]:
        """Truncate conversation history to stay under context limits.

        Keeps the most recent messages, always preserving at least the last
        user message for context.

        Args:
            messages: List of conversation messages

        Returns:
            Truncated list of messages
        """
        if not messages:
            return messages

        # Estimate total tokens including system prompt
        system_tokens = self._estimate_tokens(self._system_prompt)
        max_message_tokens = self.MAX_CONTEXT_TOKENS - system_tokens

        # Calculate total message tokens
        total_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in messages)

        if total_tokens <= max_message_tokens:
            return messages

        # Need to truncate - keep removing oldest messages until under limit
        # Always keep at least the last 2 messages (last user + assistant pair or just last user)
        truncated = list(messages)
        while len(truncated) > 2:
            total_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in truncated)
            if total_tokens <= max_message_tokens:
                break
            # Remove oldest message
            truncated.pop(0)

        return truncated
    
    def _extract_private_thought(self, response: str) -> Tuple[str, Optional[str]]:
        """Extract hidden thinking content wrapped in delimiters.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (cleaned_response, private_thought)
        """
        # Look for the thinking block with our specific delimiters
        pattern = r"__THINKING_START__\n(.*?)\n__THINKING_END__\n\n?"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            private_thought = match.group(1).strip()
            # Remove the thinking block from the public response
            cleaned_response = re.sub(pattern, "", response, flags=re.DOTALL).strip()
            return cleaned_response, private_thought
            
        return response, None

    def _parse_choice(self, response: str) -> Choice:
        """Parse the choice from an LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed Choice (RED or BLACK)

        Raises:
            ValueError: If no valid choice found in response
        """
        response_upper = response.upper()

        # Look for explicit RECOMMENDATION: or VOTE: patterns with RED/BLACK
        recommendation_match = re.search(r'RECOMMENDATION:\s*(RED|BLACK)', response_upper)
        if recommendation_match:
            return Choice.RED if recommendation_match.group(1) == "RED" else Choice.BLACK

        vote_match = re.search(r'VOTE:\s*(RED|BLACK)', response_upper)
        if vote_match:
            return Choice.RED if vote_match.group(1) == "RED" else Choice.BLACK

        # Look for Option A / Option B patterns (for scenario prompts)
        # Option A = cooperate = BLACK, Option B = defect = RED
        option_rec_match = re.search(r'RECOMMENDATION:\s*OPTION\s*([AB])', response_upper)
        if option_rec_match:
            return Choice.BLACK if option_rec_match.group(1) == "A" else Choice.RED

        option_vote_match = re.search(r'VOTE:\s*OPTION\s*([AB])', response_upper)
        if option_vote_match:
            return Choice.BLACK if option_vote_match.group(1) == "A" else Choice.RED

        # Fallback: look for the last occurrence of Option A/B or RED/BLACK
        option_a_pos = max(response_upper.rfind("OPTION A"), response_upper.rfind("OPTION: A"))
        option_b_pos = max(response_upper.rfind("OPTION B"), response_upper.rfind("OPTION: B"))
        red_pos = response_upper.rfind("RED")
        black_pos = response_upper.rfind("BLACK")

        # Find the latest position among all choices
        positions = {
            Choice.BLACK: max(option_a_pos, black_pos),
            Choice.RED: max(option_b_pos, red_pos),
        }

        if positions[Choice.BLACK] == -1 and positions[Choice.RED] == -1:
            raise ValueError(f"Could not parse choice from response: {response[:200]}...")

        # Return whichever appears last (most likely the final decision)
        if positions[Choice.RED] > positions[Choice.BLACK]:
            return Choice.RED
        return Choice.BLACK
    
    def _parse_reasoning(self, response: str) -> str:
        """Parse the reasoning from an LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Extracted reasoning text
        """
        # Look for REASONING: pattern
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=\n\n|\Z)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            return reasoning_match.group(1).strip()
        
        # Fallback: return everything after the choice
        for marker in ["RECOMMENDATION:", "VOTE:"]:
            if marker in response.upper():
                idx = response.upper().find(marker)
                remaining = response[idx:].split("\n", 1)
                if len(remaining) > 1:
                    return remaining[1].strip()
        
        return response.strip()

    def _parse_willingness(self, response: str) -> int:
        response_upper = response.upper()
        m = re.search(r"WILLINGNESS:\s*([0-3])", response_upper)
        if m:
            return max(0, min(3, int(m.group(1))))
        digits = re.findall(r"\b([0-3])\b", response_upper)
        if digits:
            return max(0, min(3, int(digits[-1])))
        return 1
    
    async def get_initial_opinion(
        self,
        round_context: dict,
        team_identifier: str,
        prior_messages: Optional[List[dict]] = None,
    ) -> AgentResponse:
        """Get the agent's initial opinion, optionally seeing prior teammates' messages.

        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            prior_messages: Messages from teammates who spoke earlier this round

        Returns:
            Agent's initial response with choice and reasoning
        """
        user_prompt = build_initial_opinion_prompt(
            round_context, team_identifier, self.prompt_template
        )

        # If there are prior messages from teammates, include them
        if prior_messages:
            prior_text = "\n".join([
                f"- {msg['agent_id']}: {msg['message']}"
                for msg in prior_messages
            ])
            user_prompt = f"""{user_prompt}

## TEAMMATES WHO HAVE ALREADY SPOKEN THIS ROUND
{prior_text}

## IMPORTANT
You must respond to your teammates' points above. Explicitly reference their arguments - agree, disagree, or build on their reasoning. Do not simply repeat what they said."""
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })

        # Truncate history if needed to stay under context limits
        messages_to_send = self._truncate_history_if_needed(self.conversation_history)

        # Get LLM response
        raw_text = await self.provider.generate(
            system_prompt=self._system_prompt,
            messages=messages_to_send,
        )

        # Extract private thought if present
        public_text, private_thought = self._extract_private_thought(raw_text)
        
        # Add response to history (we store the public version to avoid confusing the model later? 
        # Actually, standard practice is to store what the model generated. 
        # But here we want to HIDE the thinking from teammates. 
        # The history is self-history. The model should know its own thoughts.
        # So we store raw_text in history.)
        self.conversation_history.append({
            "role": "assistant", 
            "content": raw_text,
        })
        
        # Parse response from PUBLIC text
        try:
            choice = self._parse_choice(public_text)
        except ValueError:
            # Default to BLACK if parsing fails (cooperative default)
            choice = Choice.BLACK
        
        reasoning = self._parse_reasoning(public_text)
        
        return AgentResponse(
            choice=choice,
            reasoning=reasoning,
            raw_response=raw_text, # Keep full response including thinking for logs
            private_thought=private_thought,
        )

    async def get_willingness_to_speak(
        self,
        round_context: dict,
        team_identifier: str,
        seen_messages: list,
    ) -> int:
        user_prompt = build_willingness_prompt(round_context, team_identifier, seen_messages, self.prompt_template)
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })

        # Truncate history if needed to stay under context limits
        messages_to_send = self._truncate_history_if_needed(self.conversation_history)

        raw_text = await self.provider.generate(
            system_prompt=self._system_prompt,
            messages=messages_to_send,
        )
        # We don't expect thinking here usually, but good to handle it
        public_text, _ = self._extract_private_thought(raw_text)
        
        self.conversation_history.append({
            "role": "assistant",
            "content": raw_text,
        })
        return self._parse_willingness(public_text)
    
    async def get_final_vote(
        self,
        round_context: dict,
        team_identifier: str,
        teammate_opinions: List[AgentResponse],
    ) -> AgentResponse:
        """Get the agent's final vote after seeing all teammates' opinions.
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            teammate_opinions: List of all teammates' initial opinions
            
        Returns:
            Agent's final vote with choice and reasoning
        """
        user_prompt = build_final_vote_prompt(
            round_context, team_identifier, teammate_opinions, self.prompt_template
        )

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })

        # Truncate history if needed to stay under context limits
        messages_to_send = self._truncate_history_if_needed(self.conversation_history)

        # Get LLM response
        raw_text = await self.provider.generate(
            system_prompt=self._system_prompt,
            messages=messages_to_send,
        )

        public_text, private_thought = self._extract_private_thought(raw_text)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": raw_text,
        })
        
        # Parse response
        try:
            choice = self._parse_choice(public_text)
        except ValueError:
            # Default to BLACK if parsing fails
            choice = Choice.BLACK
        
        reasoning = self._parse_reasoning(public_text)
        
        return AgentResponse(
            choice=choice,
            reasoning=reasoning,
            raw_response=raw_text,
            private_thought=private_thought,
        )

    async def get_followup_response(
        self,
        round_context: dict,
        team_identifier: str,
        discussion_history: List[dict],
    ) -> AgentResponse:
        """Get a followup response after seeing other agents' messages.

        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B'
            discussion_history: List of all messages so far in this discussion

        Returns:
            Agent's followup response with choice and reasoning
        """
        # Format discussion history
        discussion_text = "\n".join([
            f"- {msg['agent_id']}: {msg['message']}"
            for msg in discussion_history
        ])

        user_prompt = f"""## DISCUSSION SO FAR
{discussion_text}

## YOUR TASK
Respond to your colleagues' points. You may:
- Support or challenge specific arguments
- Raise new considerations
- Update your position if convinced

Format your response as:

RECOMMENDATION: [A or B]
REASONING: [Your response to the discussion]"""

        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })

        # Truncate history if needed to stay under context limits
        messages_to_send = self._truncate_history_if_needed(self.conversation_history)

        raw_text = await self.provider.generate(
            system_prompt=self._system_prompt,
            messages=messages_to_send,
        )

        public_text, private_thought = self._extract_private_thought(raw_text)

        self.conversation_history.append({
            "role": "assistant",
            "content": raw_text,
        })

        try:
            choice = self._parse_choice(public_text)
        except ValueError:
            choice = Choice.BLACK

        reasoning = self._parse_reasoning(public_text)

        return AgentResponse(
            choice=choice,
            reasoning=reasoning,
            raw_response=raw_text,
            private_thought=private_thought,
        )
