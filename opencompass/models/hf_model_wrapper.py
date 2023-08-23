import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module()
class HuggingFaceFEPECausalLM(BaseModel):
    """Model wrapper around HuggingFace general models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 model_path: str,
                 pe_config: dict,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        from opencompass.utils.fileio import patch_hf_auto_model
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        patch_hf_auto_model(hf_cache_dir)
        self.logger = get_logger()
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_path=model_path,
                             pe_config=pe_config,
                             model_kwargs=model_kwargs)

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # if self.tokenizer.pad_token_id is None:
        #     self.logger.warning('pad_token_id is not set for the tokenizer. '
        #                         'Using eos_token_id as pad_token_id.')
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # A patch for llama when batch_padding = True
        # if 'decapoda-research/llama' in path or \
        #         (tokenizer_path and
        #          'decapoda-research/llama' in tokenizer_path):
        #     self.logger.warning('We set new pad_token_id for LLaMA model')
        #     # keep consistent with official LLaMA repo
        #     # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
        self.tokenizer.bos_token = '<s>'
        self.tokenizer.eos_token = '</s>'
        self.tokenizer.pad_token_id = 0

    def _load_model(self,
                    path: str,
                    model_path: str,
                    pe_config: dict,
                    model_kwargs: dict):
        from opencompass.models.hf_model_with_pe import LlamaForCausalLM, LlamaConfig
        from collie.driver.io import PetrelIODriver
        
        state_dict = PetrelIODriver.load(path=model_path, mode='rb')
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=path)
        config.torch_dtype = torch.bfloat16
        self.model = LlamaForCausalLM(config=config, pe_config=pe_config)
        self.model.load_state_dict(state_dict)
        
        # model_kwargs.setdefault('torch_dtype', torch.float16)
        # self.model = LlamaForCausalLM.from_pretrained(path, pe_config, **model_kwargs)
        
        self.model.eval().cuda()
        # generate_ids = self.model.generate(input_ids=torch.tensor([[1, 1, ]]).cuda())
        # print(generate_ids)
        # import sys; sys.exit()

    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        **kwargs)
        else:
            return sum((self._single_generate(
                inputs=[input_], max_out_len=max_out_len, **kwargs)
                for input_ in inputs), [])

    def _batch_generate(self, inputs: List[str], max_out_len: int,
                        **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.max_seq_len -
                                                             max_out_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds

    def _single_generate(self, inputs: List[str], max_out_len: int,
                         **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        input_ids = self.tokenizer(inputs,
                                   truncation=True,
                                   max_length=self.max_seq_len -
                                              max_out_len)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_out_len, **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds

    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}

            outputs = self.model(input_ids)
        return outputs[0], {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous()

        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()

        self.tokenizer.pad_token_id = 0
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens']['input_ids'] !=
                0).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        loss = loss.float()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))


# @MODELS.register_module()
# class HuggingFaceFEPECausalLM(HuggingFaceFEPE):
#     """Model wrapper around HuggingFace CausalLM.

#     Args:
#         path (str): The name or path to HuggingFace's model.
#         hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
#             use the env variable HF_MODEL_HUB. Defaults to None.
#         max_seq_len (int): The maximum length of the input sequence. Defaults
#             to 2048.
#         tokenizer_path (str): The path to the tokenizer. Defaults to None.
#         tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
#             Defaults to {}.
#         peft_path (str, optional): The name or path to the HuggingFace's PEFT
#             model. If None, the original model will not be converted to PEFT.
#             Defaults to None.
#         tokenizer_only (bool): If True, only the tokenizer will be initialized.
#             Defaults to False.
#         model_kwargs (dict): Keyword arguments for the model, used in loader.
#             Defaults to dict(device_map='auto').
#         meta_template (Dict, optional): The model's meta prompt
#             template if needed, in case the requirement of injecting or
#             wrapping of any meta instructions.
#         batch_padding (bool): If False, inference with be performed in for-loop
#             without batch padding.
#     """
