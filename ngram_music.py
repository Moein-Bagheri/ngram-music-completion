"""
مدل زبانی N-gram برای تکمیل قطعات موسیقی
========================================
این برنامه یک مدل زبانی مبتنی بر N-gram با هموارسازی Lidstone را
بر روی داده‌های نت‌های موسیقی پیاده‌سازی می‌کند.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# ۱. بارگذاری و پیش‌پردازش داده‌ها
# ─────────────────────────────────────────────

def load_data(filepath: str, ignore_accidentals: bool = False) -> list[list[str]]:
    """
    بارگذاری مجموعه داده CSV و تبدیل نت‌ها به لیست توکن.

    Args:
        filepath: مسیر فایل CSV
        ignore_accidentals: اگر True باشد، علائم تغییردهنده (b و k) نادیده گرفته می‌شوند

    Returns:
        لیستی از قطعات، هر قطعه لیستی از توکن‌های نت است
    """
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    corpus = []
    for _, row in df.iterrows():
        notes_str = str(row['note']).strip()
        tokens = notes_str.split()

        if ignore_accidentals:
            # حذف علائم تغییردهنده: b (بمل) و k (کرن) از انتهای هر نت
            tokens = [t.rstrip('b').rstrip('k') if t not in ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb',
                                                               'Bk', 'Ek', 'Ak', 'Dk', 'Gk', 'Ck', 'Fk']
                      else t[0]  # فقط حرف پایه را نگه‌دار
                      for t in tokens]
            # برای تمام توکن‌هایی که با حرف بزرگ شروع می‌شوند و با b یا k ختم می‌شوند
            cleaned = []
            for t in tokens:
                if len(t) >= 2 and t[-1] in ('b', 'k') and t[0].isupper():
                    cleaned.append(t[:-1])
                else:
                    cleaned.append(t)
            tokens = cleaned

        corpus.append(tokens)

    return corpus


def get_vocabulary(corpus: list[list[str]]) -> list[str]:
    """استخراج واژگان یکتا از کل مجموعه داده."""
    vocab = set()
    for piece in corpus:
        vocab.update(piece)
    return sorted(vocab)


# ─────────────────────────────────────────────
# ۲. کلاس مدل N-gram
# ─────────────────────────────────────────────

class NGramModel:
    """
    مدل زبانی N-gram با هموارسازی Lidstone

    :هموارسازی Lidstone
        P(w | context) = (count(context, w) + α) / (count(context) + α × |V|)

    که در آن:
        α  : پارامتر هموارسازی (α=1 → Laplace, α=0.5 → Jeffreys-Perks)
        |V| : اندازه واژگان
    """

    def __init__(self, n: int, alpha: float = 1.0):
        """
        Args:
            n: درجه مدل (1=unigram, 2=bigram, 3=trigram)
            alpha: ضریب هموارسازی Lidstone
        """
        self.n = n
        self.alpha = alpha
        self.ngram_counts = defaultdict(Counter)   # count(context) → Counter(next_token)
        self.context_counts = Counter()             # count(context)
        self.vocabulary = []
        self.vocab_size = 0

    def fit(self, corpus: list[list[str]], vocabulary: list[str]):
        """
        آموزش مدل بر روی مجموعه داده.

        Args:
            corpus: لیست قطعات موسیقی (هر قطعه لیستی از توکن)
            vocabulary: فهرست کامل واژگان
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)

        for piece in corpus:
            tokens = piece
            for i in range(len(tokens)):
                if self.n == 1:
                    # Unigram: زمینه خالی است
                    context = ()
                else:
                    # N-gram: زمینه n-1 توکن قبلی
                    if i < self.n - 1:
                        continue  # توکن‌های ابتدایی که زمینه کامل ندارند
                    context = tuple(tokens[i - (self.n - 1): i])

                self.ngram_counts[context][tokens[i]] += 1
                self.context_counts[context] += 1

    def log_probability(self, token: str, context: tuple) -> float:
        """
        محاسبه لگاریتم احتمال یک توکن با توجه به زمینه.

        Args:
            token: توکن مورد نظر
            context: زمینه (tuple از توکن‌های قبلی)

        Returns:
            log₂ P(token | context) با هموارسازی Lidstone
        """
        count_context_token = self.ngram_counts[context][token]
        count_context = self.context_counts[context]

        prob = (count_context_token + self.alpha) / (count_context + self.alpha * self.vocab_size)
        return math.log2(prob)

    def predict_next(self, context: tuple) -> str:
        """
        پیش‌بینی محتمل‌ترین توکن بعدی با توجه به زمینه.

        Args:
            context: زمینه جاری

        Returns:
            توکن با بیشترین احتمال
        """
        best_token = None
        best_log_prob = float('-inf')

        for token in self.vocabulary:
            lp = self.log_probability(token, context)
            if lp > best_log_prob:
                best_log_prob = lp
                best_token = token

        return best_token

    def complete_sequence(self, prefix: list[str], length: int = 10) -> list[str]:
        """
        تکمیل یک قطعه ناقص با افزودن توکن‌های جدید.

        Args:
            prefix: توکن‌های ابتدایی قطعه
            length: تعداد توکن‌هایی که باید اضافه شوند

        Returns:
            لیست توکن‌های اضافه شده
        """
        generated = list(prefix)

        for _ in range(length):
            if self.n == 1:
                context = ()
            else:
                # استفاده از آخرین n-1 توکن به عنوان زمینه
                context = tuple(generated[-(self.n - 1):])

            next_token = self.predict_next(context)
            generated.append(next_token)

        return generated[len(prefix):]  # فقط توکن‌های جدید


# ─────────────────────────────────────────────
# ۳. ارزیابی با Perplexity
# ─────────────────────────────────────────────

def perplexity(model: NGramModel, sequence: list[str]) -> float:
    """
    محاسبه Perplexity (سرگشتگی) یک دنباله با توجه به مدل.

    فرمول:
        PP = 2^(- (1/N) Σ log₂ P(wᵢ | wᵢ₋ₙ₊₁...wᵢ₋₁))

    عدد پایین‌تر نشان‌دهنده مدل بهتر است.

    Args:
        model: مدل آموزش‌دیده N-gram
        sequence: دنباله توکن‌ها

    Returns:
        مقدار Perplexity
    """
    log_prob_sum = 0.0
    count = 0

    for i in range(len(sequence)):
        if model.n == 1:
            context = ()
        else:
            if i < model.n - 1:
                continue
            context = tuple(sequence[i - (model.n - 1): i])

        log_prob_sum += model.log_probability(sequence[i], context)
        count += 1

    if count == 0:
        return float('inf')

    avg_log_prob = log_prob_sum / count
    return 2 ** (-avg_log_prob)


# ─────────────────────────────────────────────
# ۴. بهینه‌یابی آلفا با cross-validation
# ─────────────────────────────────────────────

def find_optimal_alpha(corpus: list[list[str]], vocabulary: list[str],
                        n: int, alpha_candidates: list[float] = None) -> tuple[float, dict]:
    """
    یافتن بهینه‌ترین مقدار α با استفاده از Leave-One-Out Cross-Validation.

    در این روش، هر بار یک قطعه به عنوان داده آزمون و بقیه به عنوان داده آموزش
    استفاده می‌شوند. مقدار α که کمترین میانگین Perplexity را ایجاد کند، بهینه است.

    Args:
        corpus: مجموعه داده
        vocabulary: واژگان
        n: درجه مدل
        alpha_candidates: مقادیر آلفای کاندیدا

    Returns:
        (بهینه‌ترین آلفا، دیکشنری نتایج)
    """
    if alpha_candidates is None:
        # جستجو در طیف وسیعی از مقادیر آلفا
        alpha_candidates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]

    results = {}

    for alpha in alpha_candidates:
        perplexities = []

        for i in range(len(corpus)):
            # تقسیم داده‌ها: یک قطعه آزمون، بقیه آموزش
            train_corpus = [corpus[j] for j in range(len(corpus)) if j != i]
            test_piece = corpus[i]

            if len(test_piece) < n:
                continue  # قطعات خیلی کوتاه را نادیده بگیر

            model = NGramModel(n=n, alpha=alpha)
            model.fit(train_corpus, vocabulary)

            pp = perplexity(model, test_piece)
            if not math.isinf(pp) and not math.isnan(pp):
                perplexities.append(pp)

        if perplexities:
            results[alpha] = np.mean(perplexities)

    best_alpha = min(results, key=results.get)
    return best_alpha, results


# ─────────────────────────────────────────────
# ۵. اجرای اصلی برنامه
# ─────────────────────────────────────────────

def run_experiment(filepath: str, ignore_accidentals: bool = False):
    """
    اجرای کامل آزمایش برای یک حالت (با یا بدون علائم تغییردهنده).

    Args:
        filepath: مسیر فایل CSV
        ignore_accidentals: آیا علائم تغییردهنده نادیده گرفته شوند

    Returns:
        دیکشنری نتایج کامل
    """
    mode_label = "بدون علائم تغییردهنده" if ignore_accidentals else "با علائم تغییردهنده"
    print(f"\n{'='*70}")
    print(f"  حالت: {mode_label}")
    print(f"{'='*70}")

    # ─── بارگذاری داده‌ها ───
    corpus = load_data(filepath, ignore_accidentals=ignore_accidentals)
    vocabulary = get_vocabulary(corpus)

    print(f"تعداد قطعات: {len(corpus)}")
    print(f"اندازه واژگان: {len(vocabulary)}")
    print(f"واژگان: {vocabulary}")

    # ─── قطعات ناقص برای تکمیل ───
    incomplete_pieces_raw = [
        ["S", "S", "S", "S"],
        ["G", "C", "G", "C"],
        ["C", "B", "C", "D"],
        ["A", "Bb", "C", "D"],
        ["A", "S", "B", "Db"],
    ]

    # اگر علائم تغییردهنده نادیده گرفته می‌شوند، آنها را از قطعات ناقص هم حذف کن
    if ignore_accidentals:
        incomplete_pieces = []
        for piece in incomplete_pieces_raw:
            cleaned = []
            for t in piece:
                if len(t) >= 2 and t[-1] in ('b', 'k') and t[0].isupper():
                    cleaned.append(t[:-1])
                else:
                    cleaned.append(t)
            incomplete_pieces.append(cleaned)
    else:
        incomplete_pieces = incomplete_pieces_raw

    # ─── تنظیمات مدل‌ها ───
    model_configs = [
        (1, "Unigram"),
        (2, "Bigram"),
        (3, "Trigram"),
    ]
    alphas = [1.0, 0.5]
    alpha_labels = {1.0: "Laplace (α=1)", 0.5: "Jeffreys-Perks (α=0.5)"}

    all_results = {}

    for n, model_name in model_configs:
        print(f"\n{'─'*60}")
        print(f"  مدل: {model_name} (n={n})")
        print(f"{'─'*60}")

        # ─── بهینه‌یابی آلفا ───
        print(f"\n  بهینه‌یابی α با Leave-One-Out CV...")
        best_alpha, cv_results = find_optimal_alpha(corpus, vocabulary, n)
        print(f"  بهترین α = {best_alpha:.4f}  |  "
              f"میانگین Perplexity = {cv_results[best_alpha]:.4f}")

        all_results[model_name] = {'best_alpha': best_alpha, 'cv_results': cv_results, 'completions': {}}

        # ─── آموزش و تکمیل قطعات با هر ترکیب ───
        for alpha in alphas:
            label = alpha_labels[alpha]
            print(f"\n  [{label}]")

            model = NGramModel(n=n, alpha=alpha)
            model.fit(corpus, vocabulary)

            completions_for_alpha = []

            for idx, prefix in enumerate(incomplete_pieces):
                generated = model.complete_sequence(prefix, length=10)
                full_piece = prefix + generated
                pp = perplexity(model, full_piece)

                completions_for_alpha.append({
                    'prefix': prefix,
                    'generated': generated,
                    'full': full_piece,
                    'perplexity': pp
                })

                print(f"    قطعه {idx+1}: {' '.join(prefix)} → {' '.join(generated)}")
                print(f"            کل: {' '.join(full_piece)}")
                print(f"            Perplexity: {pp:.4f}")

            all_results[model_name][label] = completions_for_alpha

    return all_results, vocabulary, corpus


# ─────────────────────────────────────────────
# ۶. نقطه ورود اصلی
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "music.csv"

    print("\n" + "█"*70)
    print("  مدل زبانی N-gram برای تکمیل قطعات موسیقی")
    print("█"*70)

    # ─── حالت اول: با علائم تغییردهنده ───
    results_with, vocab_with, corpus_with = run_experiment(DATA_PATH, ignore_accidentals=False)

    # ─── حالت دوم: بدون علائم تغییردهنده ───
    results_without, vocab_without, corpus_without = run_experiment(DATA_PATH, ignore_accidentals=True)

    print("\n\n" + "="*70)
    print("  مقایسه واژگان:")
    print(f"  با علائم: {len(vocab_with)} توکن")
    print(f"  بدون علائم: {len(vocab_without)} توکن")
    print("="*70)