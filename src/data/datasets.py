"""Dataset implementations for LLM-DFL."""
import json
from pathlib import Path
from typing import Any, Optional, List
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from .base import BaseDataset, Sample


class NewsGroupsDataset(BaseDataset):
    """20 Newsgroups dataset for news topic classification."""
    
    TASK_TYPE = "classification"
    DATASET_NAME = "20newsgroups"
    
    INSTRUCTION = """Task: news topic classification.
Classify the following document into one of the 20 newsgroups.
Answer with ONLY the newsgroup name."""
    
    def _load_data(self):
        subset = "train" if self.split == "train" else "test"
        data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        self.label_names = data.target_names
        
        for i, (text, label) in enumerate(zip(data.data, data.target)):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample({"text": text, "label": label})
            if sample:
                self.samples.append(sample)
    
    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        text = raw_sample["text"].strip()
        if not text:
            return None
        label = raw_sample["label"]
        label_name = self.label_names[label]
        
        # Try to extract subject line
        lines = text.split('\n')
        title = lines[0][:100] if lines else ""
        body = '\n'.join(lines[1:])[:2000] if len(lines) > 1 else text[:2000]
        
        input_text = f"Title: {title}\nBody: {body}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=label_name,
            label=label
        )


class YahooAnswersDataset(BaseDataset):
    """Yahoo! Answers Topic Classification dataset."""
    
    TASK_TYPE = "classification"
    DATASET_NAME = "yahoo_answers"
    
    LABEL_NAMES = [
        "Society & Culture", "Science & Mathematics", "Health",
        "Education & Reference", "Computers & Internet", "Sports",
        "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government"
    ]
    
    INSTRUCTION = """Task: question topic classification.
Classify the following question into one of these 10 topics:
Society & Culture, Science & Mathematics, Health, Education & Reference,
Computers & Internet, Sports, Business & Finance, Entertainment & Music,
Family & Relationships, Politics & Government.
Answer with ONLY the topic name, nothing else."""
    
    def _load_data(self):
        self.label_names = self.LABEL_NAMES
        dataset = load_dataset("yahoo_answers_topics", split=self.split, trust_remote_code=True)
        
        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)
    
    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        title = raw_sample.get("question_title", "").strip()
        content = raw_sample.get("question_content", "").strip()
        label = raw_sample["topic"]
        
        if not title and not content:
            return None
        
        input_text = f"Question Title: {title[:500]}\nQuestion Body: {content[:1500]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=self.label_names[label],
            label=label
        )


class YahooSubsetDataset(BaseDataset):
    """Yahoo! Answers subset dataset loaded from local JSON files."""
    
    TASK_TYPE = "classification"
    DATASET_NAME = "yahoo_subset"
    
    LABEL_NAMES = [
        "Society & Culture", "Science & Mathematics", "Health",
        "Education & Reference", "Computers & Internet", "Sports",
        "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government"
    ]
    
    INSTRUCTION = """Task: question topic classification.
Classify the following question into one of these 10 topics:
Society & Culture, Science & Mathematics, Health, Education & Reference,
Computers & Internet, Sports, Business & Finance, Entertainment & Music,
Family & Relationships, Politics & Government.
Answer with ONLY the topic name, nothing else."""
    
    def _load_data(self):
        self.label_names = self.LABEL_NAMES
        # Load from local JSON file
        data_dir = Path(__file__).parent.parent.parent / "data" / "yahoo_subset"
        json_file = data_dir / f"{self.split}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Yahoo subset file not found: {json_file}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for i, item in enumerate(data):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)
    
    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        title = raw_sample.get("question_title", "").strip()
        content = raw_sample.get("question_content", "").strip()
        label = raw_sample["topic"]
        
        if not title and not content:
            return None
        
        input_text = f"Question Title: {title[:500]}\nQuestion Body: {content[:1500]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=self.label_names[label],
            label=label
        )


class DBpediaDataset(BaseDataset):
    """DBpedia Ontology Classification dataset."""
    
    TASK_TYPE = "classification"
    DATASET_NAME = "dbpedia_14"
    
    LABEL_NAMES = [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
    ]
    
    INSTRUCTION = """Task: entity type classification.
You are given a short text describing an entity.
Classify it into one of the DBpedia ontology types:
Company, EducationalInstitution, Artist, Athlete, OfficeHolder,
MeanOfTransportation, Building, NaturalPlace, Village, Animal,
Plant, Album, Film, WrittenWork.
Answer with ONLY the type name."""
    
    def _load_data(self):
        self.label_names = self.LABEL_NAMES
        dataset = load_dataset("fancyzhx/dbpedia_14", split=self.split, trust_remote_code=True)
        
        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)
    
    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        title = raw_sample.get("title", "").strip()
        content = raw_sample.get("content", "").strip()
        label = raw_sample["label"]
        
        input_text = f"Text:\n{title}\n{content[:2000]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=self.label_names[label],
            label=label
        )


class AlpacaGPT4Dataset(BaseDataset):
    """Alpaca-GPT4 instruction following dataset."""

    TASK_TYPE = "generation"
    DATASET_NAME = "alpaca_gpt4"

    INSTRUCTION = """Task: general instruction following.
Follow the user's instruction and provide a helpful and safe answer."""

    def _load_data(self):
        self.label_names = []
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", trust_remote_code=True)

        total = len(dataset)
        if self.split == "train":
            dataset = dataset.select(range(int(total * 0.8)))
        else:
            dataset = dataset.select(range(int(total * 0.8), total))

        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)

    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        instruction = raw_sample.get("instruction", "").strip()
        inp = raw_sample.get("input", "").strip()
        output = raw_sample.get("output", "").strip()

        if not instruction or not output:
            return None

        optional_input = inp if inp else "(none)"
        input_text = f"Instruction: {instruction[:1000]}\nOptional input: {optional_input[:500]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=output[:2000],
            label=None
        )


class FinGPTSentimentDataset(BaseDataset):
    """FinGPT financial sentiment classification dataset."""

    TASK_TYPE = "classification"
    DATASET_NAME = "fingpt_sentiment"

    LABEL_NAMES = ["positive", "neutral", "negative"]

    INSTRUCTION = """Task: financial sentiment classification.
Decide the market sentiment of the following news or tweet.
Possible labels: positive, neutral, negative.
Answer with ONLY one label word."""

    def _load_data(self):
        self.label_names = self.LABEL_NAMES
        dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train", trust_remote_code=True)

        total = len(dataset)
        if self.split == "train":
            dataset = dataset.select(range(int(total * 0.8)))
        else:
            dataset = dataset.select(range(int(total * 0.8), total))

        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)

    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        text = raw_sample.get("input", "").strip()
        output = raw_sample.get("output", "").strip().lower()

        if not text or output not in self.LABEL_NAMES:
            return None

        label = self.LABEL_NAMES.index(output)
        input_text = f"Text: {text[:2000]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=output,
            label=label
        )


class MedicalFlashcardsDataset(BaseDataset):
    """Medical flashcards QA dataset."""

    TASK_TYPE = "generation"
    DATASET_NAME = "medical_flashcards"

    INSTRUCTION = """Task: medical question answering.
Answer the following medical question concisely and accurately."""

    def _load_data(self):
        self.label_names = []
        dataset = load_dataset("medalpaca/medical_meadow_flashcards", split="train", trust_remote_code=True)

        total = len(dataset)
        if self.split == "train":
            dataset = dataset.select(range(int(total * 0.8)))
        else:
            dataset = dataset.select(range(int(total * 0.8), total))

        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)

    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        question = raw_sample.get("input", "").strip()
        answer = raw_sample.get("output", "").strip()

        if not question or not answer:
            return None

        input_text = f"Question: {question[:1000]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=answer[:1000],
            label=None
        )


class CodeAlpacaDataset(BaseDataset):
    """Code Alpaca code generation dataset."""

    TASK_TYPE = "generation"
    DATASET_NAME = "code_alpaca"

    INSTRUCTION = """Task: code generation.
Write code that correctly solves the user's request.
Return ONLY the code, without extra explanations."""

    def _load_data(self):
        self.label_names = []
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train", trust_remote_code=True)

        total = len(dataset)
        if self.split == "train":
            dataset = dataset.select(range(int(total * 0.8)))
        else:
            dataset = dataset.select(range(int(total * 0.8), total))

        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            sample = self._process_sample(item)
            if sample:
                self.samples.append(sample)

    def _process_sample(self, raw_sample: Any) -> Optional[Sample]:
        instruction = raw_sample.get("instruction", "").strip()
        output = raw_sample.get("output", "").strip()

        if not instruction or not output:
            return None

        input_text = f"Instruction: {instruction[:1000]}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=output[:2000],
            label=None
        )


class TOFUDataset(BaseDataset):
    """TOFU (Task of Fictitious Unlearning) dataset for QA-based unlearning.
    
    Supports two sources:
    1) Repo-local small subset (default): data/tofu/train_perturbed.json (JSON array)
       - 40 authors (800 QA), 20 QA per author
    2) Local TOFU directory (jsonl): e.g. TOFU/full.json, TOFU/retain90.json, TOFU/forget10.json
       - JSONL (one JSON object per line), typically 200 authors (4000 QA)
    
    数据结构:
    - question, answer: 问答对
    - paraphrased_answer: 改写答案（用于TR计算）
    - perturbed_answer: 扰动答案列表（用于TR计算）
    - author_id: 作者ID (0-39)
    
    Non-IID划分策略:
    - 将40个作者两两分组，形成20个"作者组"作为类别
    - 作者组 0 = 作者 0,1; 作者组 1 = 作者 2,3; ...
    - 每组40条数据，共20组，与20newsgroups类别数一致
    - 使用 Dirichlet 分布进行 non-IID 划分
    
    DFL场景:
    - Forget数据 = 目标客户端的数据
    - Retain数据 = 其他客户端的数据
    """
    
    TASK_TYPE = "tofu"
    DATASET_NAME = "tofu"
    
    INSTRUCTION = """Task: question answering about fictional authors.
Answer the following question about a fictional author based on their biography.
Provide a concise and accurate answer."""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        num_authors: int = 40,
        group_size: int = 2,  # 每组包含的作者数量
        tofu_local_dir: Optional[str] = None,
        tofu_split: str = "train_perturbed",
        qa_per_author: int = 20,
        **kwargs  # 忽略其他参数（如hf_split）
    ):
        """初始化TOFU数据集。
        
        Args:
            split: 数据集划分 ("train")，TOFU遗忘任务不需要验证集
            max_samples: 最大样本数量
            num_authors: 使用的作者数量（默认40，每作者20条QA，共800条）
            group_size: 每组包含的作者数量（默认2，即两个作者一组）
                - group_size=2: 40作者 → 20组，每组40条
                - group_size=4: 40作者 → 10组，每组80条
            tofu_local_dir: Optional local directory containing TOFU json/jsonl files (e.g. "TOFU")
            tofu_split: File stem to load from tofu_local_dir (e.g. "full", "retain90", "forget10",
                        or "forget10_perturbed"). When tofu_local_dir is None, defaults to
                        "train_perturbed" from repo-local `data/tofu/`.
            qa_per_author: Used to infer author_id when missing in local jsonl files.
        """
        # NOTE: The effective author count depends on the data source.
        # - repo-local subset has at most 40 authors
        # - local_dir splits typically have up to 200 authors
        self.num_authors = int(num_authors)
        self.qa_per_author = int(qa_per_author)
        self.group_size = int(group_size)
        self.num_groups = 0  # computed after loading (depends on effective num_authors)
        self.tofu_local_dir = tofu_local_dir
        self.tofu_split = str(tofu_split or "train_perturbed")
        super().__init__(split, max_samples)
    
    def _load_data(self):
        """从本地JSON文件加载TOFU数据。"""
        def _load_jsonl(path: Path):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
            return items

        data: List[Any]
        if self.tofu_local_dir:
            local_dir = Path(self.tofu_local_dir)
            fname = self.tofu_split
            if not fname.endswith(".json"):
                fname = f"{fname}.json"
            data_file = local_dir / fname
            if not data_file.exists():
                raise FileNotFoundError(f"TOFU local split not found: {data_file}")

            # Detect JSON array vs JSONL.
            with open(data_file, "r", encoding="utf-8") as f:
                head = ""
                while True:
                    ch = f.read(1)
                    if not ch:
                        break
                    if not ch.isspace():
                        head = ch
                        break
            if head == "[":
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = _load_jsonl(data_file)

            # When using local_dir, allow >40 authors. Keep explicit limit if caller sets it.
            inferred_authors = max(1, int((len(data) + max(1, self.qa_per_author) - 1) // max(1, self.qa_per_author)))
            if self.num_authors <= 0:
                self.num_authors = inferred_authors
            else:
                self.num_authors = min(int(self.num_authors), inferred_authors)
        else:
            data_file = Path(__file__).parent.parent.parent / "data" / "tofu" / "train_perturbed.json"
            if not data_file.exists():
                raise FileNotFoundError(
                    f"TOFU数据文件不存在: {data_file}\n"
                    f"请先运行: python scripts/download_tofu_data.py"
                )
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Repo-local subset contains at most 40 authors.
            self.num_authors = min(int(self.num_authors), 40)
            if self.num_authors <= 0:
                self.num_authors = 40
        
        # Compute group count based on the effective author count (ceiling division).
        self.num_groups = int((self.num_authors + self.group_size - 1) // self.group_size)

        # 设置label_names（按组）
        self.label_names = [f"author_group_{i}" for i in range(self.num_groups)]
        
        # 过滤并处理样本
        for i, item in enumerate(data):
            author_id = item.get("author_id")
            if author_id is None:
                author_id = int(i // max(1, self.qa_per_author))
            else:
                author_id = int(author_id)
            
            # 只加载指定数量作者的数据
            if author_id >= self.num_authors:
                continue
            
            if self.max_samples and len(self.samples) >= self.max_samples:
                break
            
            # 计算作者组ID（两个作者一组）
            group_id = author_id // self.group_size
            
            sample = self._process_sample(item, group_id, author_id)
            if sample:
                self.samples.append(sample)
    
    def _process_sample(self, raw_sample: Any, group_id: int, author_id: int = None) -> Optional[Sample]:
        """将原始QA对转换为Sample对象。
        
        Args:
            raw_sample: 原始数据
            group_id: 作者组ID（用于non-IID划分的label）
            author_id: 原始作者ID（保留用于评估）
        """
        question = raw_sample.get("question", "").strip()
        answer = raw_sample.get("answer", "").strip()
        
        if not question or not answer:
            return None
        
        # 处理perturbed_answer（可能是列表或字符串）
        perturbed = raw_sample.get("perturbed_answer", [])
        if isinstance(perturbed, str):
            perturbed = [perturbed] if perturbed else []
        
        input_text = f"Question: {question}"
        return Sample(
            instruction=self.INSTRUCTION,
            input_text=input_text,
            output_text=answer,
            label=group_id,  # 使用组ID作为label进行non-IID划分
            paraphrased_answer=raw_sample.get("paraphrased_answer", answer),
            perturbed_answers=perturbed
        )
