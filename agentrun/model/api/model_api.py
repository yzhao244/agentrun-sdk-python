from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Union

from .data import BaseInfo

if TYPE_CHECKING:
    from litellm import ResponseInputParam


class ModelAPI(ABC):

    @abstractmethod
    def model_info(self) -> BaseInfo:
        ...

    def completions(
        self,
        **kwargs,
    ):
        """
        Deprecated. Use completion() instead.
        """
        import warnings

        warnings.warn(
            "completions() is deprecated, use completion() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.completion(**kwargs)

    def completion(
        self,
        messages=[],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        from litellm import completion

        info = self.model_info()
        return completion(
            **kwargs,
            api_key=info.api_key,
            base_url=info.base_url,
            model=model or info.model or "",
            custom_llm_provider=custom_llm_provider
            or info.provider
            or "openai",
            messages=messages,
        )

    async def acompletion(
        self,
        messages=[],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        from litellm import acompletion

        info = self.model_info()
        return await acompletion(
            **kwargs,
            api_key=info.api_key,
            base_url=info.base_url,
            model=model or info.model or "",
            custom_llm_provider=custom_llm_provider
            or info.provider
            or "openai",
            messages=messages,
        )

    def responses(
        self,
        input: Union[str, "ResponseInputParam"],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        from litellm import responses

        info = self.model_info()
        return responses(
            **kwargs,
            api_key=info.api_key,
            base_url=info.base_url,
            model=model or info.model or "",
            custom_llm_provider=custom_llm_provider
            or info.provider
            or "openai",
            input=input,
        )

    async def aresponses(
        self,
        input: Union[str, "ResponseInputParam"],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        from litellm import aresponses

        info = self.model_info()
        return await aresponses(
            **kwargs,
            api_key=info.api_key,
            base_url=info.base_url,
            model=model or info.model or "",
            custom_llm_provider=custom_llm_provider
            or info.provider
            or "openai",
            input=input,
        )

    def embedding(
        self,
        input=[],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        from litellm import embedding

        info = self.model_info()
        return embedding(
            **kwargs,
            api_key=info.api_key,
            api_base=info.base_url,
            model=model or info.model or "",
            custom_llm_provider=custom_llm_provider
            or info.provider
            or "openai",
            input=input,
        )

    def aembedding(
        self,
        input=[],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        from litellm import aembedding

        info = self.model_info()
        return aembedding(
            **kwargs,
            api_key=info.api_key,
            api_base=info.base_url,
            model=model or info.model or "",
            custom_llm_provider=custom_llm_provider
            or info.provider
            or "openai",
            input=input,
        )
