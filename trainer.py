import copy
import json
import math
import random
from collections import Counter, defaultdict, deque
from typing import Optional

import datasets
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available


class DPRTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        dataloader = DataLoader(train_dataset, **dataloader_params)

        def __len__(self):
            length = math.ceil(len(self.sampler) / self.batch_size)
            return length

        dataloader.__len__ = __len__
        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            Warning("group_by_length는 현재 코드에서 동작하지 않음.")

        return DistributedUniqueBM25Sampler(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            tokenizer=self.tokenizer,
            indices_path="/root/clip/korquad_klue_bm25_sampler_indices.json",
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
            seed=self.args.seed,
        )


class DistributedUniqueBM25Sampler(DistributedSampler):
    r"""
    DistributedUniqueBM25Sampler
    DistributedSampler를 상속받아 'answer' 값 중복 방지 + BM25 중복 방지 로직을 추가한 사용자 정의 Sampler
    """

    def __init__(
        self,
        dataset,
        batch_size,
        tokenizer,
        indices_path,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=42,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self._init_data_structures()
        with open(indices_path, "r", encoding="utf-8") as f:
            self.indices_dict_list = json.load(f)
        self.length = len(list(self.__iter__()))

    def _init_data_structures(self):
        random.seed(self.seed + self.epoch)
        # answer가 중복되질 않길 바라며 입력할 예정
        # 전처리로 저장된 bm25 passage text들에 대한 실제 dataset의 indices를 찾기 위함
        self.answer_to_indices = {}
        self.context_to_indices = {}
        for index, item in enumerate(self.dataset):
            answer = item["answer"]
            if answer in self.answer_to_indices:
                self.answer_to_indices[answer].append(index)
            else:
                self.answer_to_indices[answer] = [index]

            ctx = item["context"]
            if ctx in self.context_to_indices:
                self.context_to_indices[ctx].append(index)
            else:
                self.context_to_indices[ctx] = [index]
        self.ori_answer_to_indices = copy.deepcopy(self.answer_to_indices)
        self.answer_set = set(self.ori_answer_to_indices.keys())
        assert (
            len(self.answer_to_indices.keys()) >= self.batch_size
        ), "데이터의 카테고리가 batch_size보다 작습니다. negative sampling 학습이 불가합니다."

    def __iter__(self):
        indices = self.indices_dict_list[self.epoch]["indices"]
        min_val = math.floor(len(indices) / self.batch_size / self.num_replicas)
        drop_last = min_val * self.batch_size * self.num_replicas
        drop_last_indices = indices[:drop_last]
        each_rank_indices = drop_last_indices[self.rank :: self.num_replicas]
        self.length = len(each_rank_indices)
        return iter(each_rank_indices)

    def __len__(self):
        return self.length

    def __make_indices__(self, start_index, end_index, outputs):
        # 중복 없이 배치 생성
        for process_idx in tqdm(range(start_index, end_index)):
            self.set_epoch(process_idx)
            self._init_data_structures()
            indices = []
            answer_keys = list(self.answer_to_indices.keys())
            # answer_keys를 섞기 때문에, 매 에포크별 최소한의 랜덤성이 부여된다.
            if self.shuffle:
                random.shuffle(answer_keys)

            q = deque(answer_keys)
            # dict의 카테고리가 num_replicas * batch_size보다 작으면 무조건 중복이 발생할 수 밖에 없다. (4*4인데 5,1,1,1,1,1,1,1,1,1,1,1 인경우, 현재 로직상 맨 마지막에 5-4=1이 겹침)
            while len(q) >= self.num_replicas * self.batch_size:
                # text answer를 하나 꺼내서
                answer_key = q.popleft()
                if len(self.answer_to_indices[answer_key]) <= self.num_replicas:
                    # num_replicas보다 작으면, 그냥 extend해도 해당 카테고리는 배치에 중복으로 들어갈 일이 없다.
                    for idx in self.answer_to_indices[answer_key]:
                        indices.append(idx)
                        if len(indices) % (self.batch_size * self.num_replicas) == 0:
                            q.append(answer_key)
                            break
                else:
                    # num_replicas보다 크면, gpu 4갠데, 카테고리 1개가 5개인 경우, 0,1,2,3,0 과 같이 무지성으로 붙히면 gpu 0번에서 중복데이터가 발생한다.
                    for _ in range(self.num_replicas):
                        indices.append(self.answer_to_indices[answer_key].pop())
                        if len(indices) % (self.batch_size * self.num_replicas) == 0:
                            break
                    # 이 경우에는 queue의 맨 끝에 붙혀서, 추후에 볼 수 있도록 만들어본다.
                    q.append(answer_key)

                if len(self.answer_to_indices[answer_key]) == 0:
                    # data를 다 뺐으니, key 삭제
                    self.answer_to_indices.pop(answer_key)

                if len(indices) % (self.batch_size * self.num_replicas) == 0:
                    # batch로 만들 것이 한바퀴 완성되었으면, 그 뒤에는 bm25 batch를 구성해줘야함
                    # 마지막 batch_size의 num_replicas 만큼 잘라서,
                    temp = copy.deepcopy(indices[-(self.batch_size * self.num_replicas) :])
                    results = list()
                    for replica in range(self.num_replicas):
                        each_replica = list()
                        answer_bm25_hist = list()
                        replicas_last_batch_indices = temp[replica :: self.num_replicas]
                        batch = self.dataset[replicas_last_batch_indices]
                        answer_bm25_hist.extend(batch["answer"])
                        batch_answer_cnt = Counter(batch["answer"])
                        bm25_visited = defaultdict(lambda: False)
                        assert batch_answer_cnt.most_common(1)[0][1] == 1, "batch내의 중복값 발생!"
                        for bm25_list in batch["bm25_hard"]:
                            # 각 batch의 각 데이터별 bm25_hard 리스트
                            for ctx in bm25_list:
                                # bm25_list는 ranking별로 내림차순 정렬되어있다.
                                try:
                                    for idx in self.context_to_indices[ctx]:
                                        # batch 대상에 answer에 포함되지 않은 context 이면서,
                                        # bm25로도 방문하지 않은 answer인 경우
                                        # batch에도, bm25에도 포함되지 않는 데이터인 경우
                                        target = self.dataset[idx]["answer"]
                                        if (
                                            batch_answer_cnt[target] == 0
                                            and not bm25_visited[target]
                                        ):
                                            each_replica.append(idx)
                                            answer_bm25_hist.append(target)
                                            bm25_visited[target] = True
                                            break
                                except KeyError:
                                    # 전처리 진행 시 bm25에서는 검색이 되었으나, 정작 자기 스스로 qustion으로 bm25을 했을때
                                    # 데이터가 없었던 경우, bm25_list에는 들어있지만, 실제 데이터에선 빠r질 여지가 있다.
                                    continue
                                # 각 배치별로 1개의 bm25_hard를 추출하면 되므로, break로 다음 batch를 보도록 하자.
                                break
                        diff_cnt = len(replicas_last_batch_indices) - len(each_replica)
                        unique_answer = list(self.answer_set - set(answer_bm25_hist))
                        random.shuffle(unique_answer)
                        while diff_cnt > 0:
                            # 어떤 batch에 한해서는, bm25 결과가 없을 수도 있다.
                            # 논문에서는 이런 데이터는 제외하였다고 하지만, 생각해보면 contrastive learning에서
                            # 배치의 데이터 구조가 달라지면 결국 새로운 데이터라고 정의할 수 있다. 따라서 그냥 중복되지 않은 놈
                            # 아무거나 하나 넣는다.
                            # random으로 bm25랑은 관련 없지만, answer가 안겹치는 애들 삽입
                            value = unique_answer.pop()
                            if value in self.ori_answer_to_indices:
                                each_replica.append(self.ori_answer_to_indices[value][0])
                                diff_cnt -= 1

                        # 여기까지 당도하면, 무조건 bm25하나는 있는 데이터만 활용된다.
                        results.append(each_replica)

                    for column in zip(*results):
                        indices.extend(column)
                        temp.extend(column)

                    for replica in range(self.num_replicas):
                        replicas_last_batch_indices = temp[replica :: self.num_replicas]
                        batch = self.dataset[replicas_last_batch_indices]
                        batch_answer_cnt = Counter(batch["answer"])
                        assert batch_answer_cnt.most_common(1)[0][1] == 1, "bm25 완성 batch내의 중복값 발생!"
                        assert len(replicas_last_batch_indices) == self.batch_size * 2, "누락값 발생!"
            outputs.append(
                {"epoch": self.epoch, "num_replicas": self.num_replicas, "indices": indices}
            )
