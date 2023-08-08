from alpaca_farm.auto_annotations import alpaca_leaderboard
from datasets import load_dataset

from opencompass.datasets import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class AlpaceFarmDataset(BaseDataset):

    @staticmethod
    def load(path, name):
        import os
        os.environ[
            'http_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'https_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTP_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTPS_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'

        dataset = load_dataset(path, name)['eval']
        dataset = dataset.filter(lambda x: x['input'] != '')
        # for data_row in dataset:
        #     # 检查input列是否为空
        #     if data_row["input"].strip() != "":
        #         # 将input不为空的数据行添加到filtered_data中
        #         filtered_data.append(data_row)
        #
        # # 将新的filtered_data列表转换回dataset格式
        # dataset = datasets.Dataset.from_dict(filtered_data)
        print(len(dataset))
        return dataset


@LOAD_DATASET.register_module()
class AlpaceFarmDataset_no_input(BaseDataset):

    @staticmethod
    def load(path, name):
        import os
        os.environ[
            'http_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'https_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTP_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTPS_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        dataset = load_dataset(path, name)['eval']
        dataset = dataset.filter(lambda x: x['input'] == '')
        # for data_row in dataset:
        #     # 检查input列是否为空
        #     if data_row["input"].strip() == "":
        #         # 将input不为空的数据行添加到filtered_data中
        #         filtered_data.append(data_row)
        #
        # # 将新的filtered_data列表转换回dataset格式
        # dataset = datasets.Dataset.from_dict(filtered_data)
        print(len(dataset))
        return dataset


@ICL_EVALUATORS.register_module()
class alpacaEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        import os
        os.environ[
            'http_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'https_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTP_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTPS_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        dataset = load_dataset(path='tatsu-lab/alpaca_farm',
                               name='alpaca_farm_evaluation')['eval']
        instructions = dataset['instruction']
        inputs = dataset['input']
        outputs = []
        for pred, instruction, input in zip(predictions, instructions, inputs):
            outputs.append({
                'instruction': instruction,
                'input': input,
                'output': pred
            })
        df_results = alpaca_leaderboard(
            path_or_all_outputs=outputs,
            is_add_reference_methods=False,
            annotators_config='annotators/greedy_gpt4/config_gpt3.5.yaml',
        )
        # score = df_results.to_string(float_format="%.2f")
        score = df_results.iloc[0]['win_rate']
        score = round(score, 2)
        return {'score': score}
