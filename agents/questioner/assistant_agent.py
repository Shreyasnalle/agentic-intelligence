from typing import Optional, Any
from agents.questioner.schemas import FramedInteraction
from agents.models import get_hf_llm


class AssistantAgent:
    """
    Assistant Agent that bridges user input and the Questioner Agent.
    Frames the user's answer along with the asked question into a structured
    JSON FramedInteraction component to pass back to the Questioner Agent.
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm()
        if self.llm is not None:
            try:
                self.structured_llm = self.llm.with_structured_output(FramedInteraction)
            except Exception:
                self.structured_llm = None
        else:
            self.structured_llm = None

    def frame_interaction(
        self,
        turn_index: int,
        question: str,
        raw_user_response: str,
        dimension_probed: str = "general",
    ) -> FramedInteraction:
        """
        Frames and structures the question-response pair into a strictly-typed JSON component.
        """
        if self.structured_llm is not None:
            prompt = (
                "You are an Assistant Agent responsible for framing and structuring user evaluation responses.\n"
                f"Turn: {turn_index}\n"
                f"Dimension: {dimension_probed}\n"
                f"Question Asked: {question}\n"
                f"Raw User Response: {raw_user_response}\n\n"
                "Tasks:\n"
                "1. Frame the user's response clearly into 'framed_response' summarizing their core argument, answer, and thought process.\n"
                "2. Provide concise 'key_observations' regarding their problem-solving clarity, heuristics, or cognitive cues.\n"
                "3. Preserve the exact question and verbatim raw response in the output.\n"
                "4. Return strictly adhering to the FramedInteraction structure."
            )
            try:
                result = self.structured_llm.invoke(prompt)
                if isinstance(result, FramedInteraction):
                    result.turn_index = turn_index
                    result.question = question
                    result.raw_user_response = raw_user_response
                    return result
            except Exception as e:
                print(f"[AssistantAgent] Warning: LLM invocation failed ({e}), using direct framing.")

        # Default structured framing
        return FramedInteraction(
            turn_index=turn_index,
            question=question,
            raw_user_response=raw_user_response,
            framed_response=raw_user_response.strip(),
            dimension_probed=dimension_probed,
            key_observations="Direct framing of user response without external LLM normalization.",
        )
