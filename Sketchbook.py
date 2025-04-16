def __init__(self, word2vec_dataset, vocab_size, negative_samples=50):
    self.word2vec_dataset = word2vec_dataset
    self.vocab_size = vocab_size
    self.negative_samples = negative_samples

    # More sophisticated negative sampling table
    self.negative_table = self._create_adaptive_negative_table()

    # Compute word co-occurrence matrix
    self.co_occurrence_matrix = self._compute_sampled_co_occurrence()

    # Precompute training pairs with advanced negative sampling
    self.training_pairs = self._generate_smart_training_pairs()


def _create_adaptive_negative_table(self, table_size=100_000_000):
    """Create a more nuanced negative sampling table"""
    # Hybrid frequency-based and semantic-aware sampling
    word_counts = Counter()
    semantic_scores = np.ones(self.vocab_size)

    # Count word frequencies
    with open(self.word2vec_dataset.corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building adaptive negative sampling table"):
            words = self.word2vec_dataset._preprocess_text(line.strip())
            words = [word for word in words if word in self.word2vec_dataset.word_to_idx]
            word_counts.update(words)

    # Convert counts to probabilities
    word_freqs = np.zeros(self.vocab_size)
    for word, count in word_counts.items():
        if word in self.word2vec_dataset.word_to_idx:
            idx = self.word2vec_dataset.word_to_idx[word]
            word_freqs[idx] = count

    # Apply power law with semantic awareness
    word_freqs = np.power(word_freqs, 0.75)

    # Normalize frequencies
    word_freqs /= np.sum(word_freqs)

    # Create adaptive table with dynamic sampling
    negative_table = []
    for idx, freq in enumerate(word_freqs):
        # Adjust sampling based on frequency and semantic potential
        table_entry_count = max(1, int(freq * table_size * semantic_scores[idx]))
        negative_table.extend([idx] * table_entry_count)

    # Ensure exact table size
    negative_table = negative_table[:table_size] if len(negative_table) > table_size else \
        negative_table + [np.random.randint(0, self.vocab_size)] * (table_size - len(negative_table))

    return np.array(negative_table)


def _compute_sampled_co_occurrence(self, sample_size=100000):
    """
    Compute co-occurrence using random sampling

    Args:
    - sample_size: Number of words to sample for co-occurrence
    """
    co_occurrence = defaultdict(lambda: defaultdict(float))

    with open(self.word2vec_dataset.corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sampled_lines = random.sample(lines, min(len(lines), sample_size))

        for line in tqdm(sampled_lines, desc="Sampling co-occurrences"):
            words = self.word2vec_dataset._preprocess_text(line.strip())

            for i, center_word in enumerate(words):
                window = random.randint(1, 5)
                start = max(0, i - window)
                end = min(len(words), i + window + 1)

                for j in range(start, end):
                    if i != j:
                        # Use point-wise mutual information (PMI) concept
                        co_occurrence[center_word][words[j]] += 1 / (abs(i - j) + 1)

    return co_occurrence


def _generate_smart_training_pairs(self):
    """Generate training pairs with semantic-aware negative sampling"""
    training_pairs = []

    with open(self.word2vec_dataset.corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Generating smart training pairs"):
            words = self.word2vec_dataset._preprocess_text(line.strip())
            indices = [self.word2vec_dataset.word_to_idx.get(word) for word in words]
            indices = [idx for idx in indices if idx is not None]

            for i, target_idx in enumerate(indices):
                # Dynamic windowing
                window = random.randint(1, 5)
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)

                context_indices = indices[start:i] + indices[i + 1:end]

                for context_idx in context_indices:
                    # Advanced negative sampling
                    # Find semantically dissimilar words
                    neg_candidates = self._find_negative_candidates(target_idx, context_idx)

                    training_pairs.append((
                        target_idx,
                        context_idx,
                        neg_candidates
                    ))

    return training_pairs


def _find_negative_candidates(self, target_idx, context_idx):
    """Find negative samples with semantic distance"""
    # Use co-occurrence matrix to find dissimilar words
    target_co_occurrence = self.co_occurrence_matrix[target_idx].toarray().flatten()
    context_co_occurrence = self.co_occurrence_matrix[context_idx].toarray().flatten()

    # Compute semantic distance
    semantic_distance = 1 - cosine_similarity(
        target_co_occurrence.reshape(1, -1),
        context_co_occurrence.reshape(1, -1)
    )[0][0]

    # Select negative samples based on semantic distance
    neg_candidates = np.random.choice(
        self.negative_table,
        size=self.negative_samples,
        p=semantic_distance
    )

    return neg_candidates


def __len__(self):
    return len(self.training_pairs)


def __getitem__(self, idx):
    return self.training_pairs[idx]