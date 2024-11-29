from typing import List

from openai.types.chat import ChatCompletionMessageParam
from .datamodels import GuardrailOnErrorAction, GuardrailValidationResult, GuardrailStates
from .guardrail_base import GuardrailBase


class DetectDummy(GuardrailBase):
    """
    Dummy guardrail to detect profanity in text.
    """

    PROHIBITED_WORDS = frozenset(["word_a", "word_b", "word_c"])

    def __init__(self):
        super().__init__(
            name="dummy",
            error_action=GuardrailOnErrorAction.BLOCK,
            continue_on_failure=False,
            validate_failed_output=True,
        )

    @property
    def template(self) -> str:
        return (
            "I apologize, but I cannot process this message as it contains "
            "prohibited content. Please rephrase your message appropriately."
        )

    def contains_prohibited_words(self, text: str) -> bool:
        """
        Checks if the text contains any prohibited words.

        Args:
            text: The text to check

        Returns:
            True if prohibited words are found, False otherwise
        """
        text_lower = text.lower()
        return any(word in text_lower for word in self.PROHIBITED_WORDS)

    async def validate(
        self,
        messages: List[ChatCompletionMessageParam],
        **kwargs,
    ) -> GuardrailValidationResult:
        """
        Validates the latest message against prohibited words.

        Args:
            messages: List of chat messages, with the latest message to validate

        Returns:
            GuardrailValidationResult indicating whether the message passed or failed
        """
        latest_message = messages[-1]["content"]

        if self.contains_prohibited_words(latest_message):
            return GuardrailValidationResult(
                guardrail_name=self.name,
                state=GuardrailStates.FAILED,
                message="Message contains prohibited content.",
            )

        return GuardrailValidationResult(
            guardrail_name=self.name,
            state=GuardrailStates.PASSED,
            message="Message passed content filter.",
        )
