import os
import re
import sqlite3
import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator, Any
from collections import defaultdict
from urllib.parse import urlparse

import torch

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from bs4 import BeautifulSoup
import feedparser
import numpy as np
from transformers import pipeline as hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Config:
    db_path: str = field(default_factory=lambda: os.getenv("INTEL_DB_PATH", "strategic_intel.db"))
    log_path: str = field(default_factory=lambda: os.getenv("INTEL_LOG_PATH", "pipeline.log"))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("INTEL_OUTPUT_DIR", "outputs")))
    log_level: str = field(default_factory=lambda: os.getenv("INTEL_LOG_LEVEL", "INFO"))
    
    cpu_threads: int = 4
    finbert_model: str = "ProsusAI/finbert"
    translation_model: str = "Helsinki-NLP/opus-mt-en-fr"
    
    max_text_length: int = 512
    max_content_length: int = 1500
    min_article_length: int = 200
    dedup_threshold: float = 0.85
    dedup_max_articles: int = 600
    dedup_text_length: int = 600
    dedup_hours: int = 48
    analyze_limit: int = 150
    top_n_articles: int = 15
    translation_batch_size: int = 4
    
    weight_recency: float = 0.25
    weight_sentiment: float = 0.25
    weight_theme: float = 0.30
    weight_source: float = 0.20
    
    def __post_init__(self):
        object.__setattr__(self, 'output_dir', Path(self.output_dir))
        self.output_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls()


def setup_logging(config: Config) -> logging.Logger:
    logger = logging.getLogger("strategic_intel")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers.clear()
    
    fmt = logging.Formatter(
        '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    
    fh = logging.FileHandler(config.log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(sh)
    
    return logger


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _strip_html(s: str) -> str:
    if not s:
        return ""
    txt = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    txt = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', txt)
    return txt[:50000]


def _parse_ts(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.replace(tzinfo=dt.tzinfo or timezone.utc).isoformat()
    except Exception:
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).isoformat()
        except Exception:
            return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE NOT NULL,
            url TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            published_ts TEXT,
            collected_date TEXT NOT NULL,
            sentiment_financial REAL,
            sentiment_pos REAL,
            sentiment_neg REAL,
            sentiment_neu REAL,
            theme TEXT,
            theme_score REAL,
            strategic_score REAL,
            finance_likeness REAL,
            stress_score REAL,
            snippet_fr TEXT,
            title_fr TEXT,
            processed INTEGER DEFAULT 0,
            is_duplicate INTEGER DEFAULT 0
        )
    """
    
    _IDX = [
        "CREATE INDEX IF NOT EXISTS idx_hash ON articles(hash)",
        "CREATE INDEX IF NOT EXISTS idx_date ON articles(collected_date)",
        "CREATE INDEX IF NOT EXISTS idx_score ON articles(strategic_score)",
        "CREATE INDEX IF NOT EXISTS idx_processed ON articles(processed, is_duplicate)",
    ]
    
    def __init__(self, db_path: str, logger: logging.Logger):
        self._path = db_path
        self._log = logger
        self._init()
    
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init(self):
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.execute(self._SCHEMA)
            self._migrate(c)
            for idx in self._IDX:
                c.execute(idx)
            c.commit()
    
    def _migrate(self, c: sqlite3.Connection):
        cur = c.execute("PRAGMA table_info(articles)")
        cols = {r[1] for r in cur.fetchall()}
        if 'is_duplicate' not in cols:
            c.execute("ALTER TABLE articles ADD COLUMN is_duplicate INTEGER DEFAULT 0")
    
    def insert(self, article: Dict) -> bool:
        try:
            with self._conn() as c:
                cur = c.execute("""
                    INSERT OR IGNORE INTO articles 
                    (hash, url, title, content, source, published_ts, collected_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    article['hash'], article.get('url'), article['title'],
                    article['content'], article['source'],
                    article.get('published_ts'), article['collected_date']
                ))
                c.commit()
                return cur.rowcount == 1
        except sqlite3.Error as e:
            self._log.error(f"insert: {e}")
            return False
    
    def batch_update(self, rows: List[Tuple]):
        if not rows:
            return
        with self._conn() as c:
            c.executemany("""
                UPDATE articles SET
                    sentiment_financial=?, sentiment_pos=?, sentiment_neg=?, sentiment_neu=?,
                    theme=?, theme_score=?, strategic_score=?, finance_likeness=?, stress_score=?,
                    snippet_fr=?, title_fr=?, processed=1
                WHERE hash=?
            """, rows)
            c.commit()
    
    def mark_dupes(self, hashes: List[str]):
        if not hashes:
            return
        with self._conn() as c:
            c.executemany("UPDATE articles SET is_duplicate=1 WHERE hash=?", [(h,) for h in hashes])
            c.commit()
    
    def unprocessed(self, limit: int) -> List[Dict]:
        with self._conn() as c:
            cur = c.execute("""
                SELECT * FROM articles 
                WHERE processed=0 AND is_duplicate=0
                ORDER BY collected_date DESC LIMIT ?
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]
    
    def recent_dedup(self, hours: int, limit: int) -> List[Dict]:
        cutoff = (_utcnow() - timedelta(hours=hours)).isoformat()
        with self._conn() as c:
            cur = c.execute("""
                SELECT hash, title, content, source FROM articles 
                WHERE collected_date>? AND is_duplicate=0
                ORDER BY collected_date DESC LIMIT ?
            """, (cutoff, limit))
            return [dict(r) for r in cur.fetchall()]
    
    def top_strategic(self, date: str, limit: int) -> List[Dict]:
        with self._conn() as c:
            cur = c.execute("""
                SELECT * FROM articles 
                WHERE date(collected_date)=date(?) AND processed=1 AND is_duplicate=0
                ORDER BY strategic_score DESC LIMIT ?
            """, (date, limit))
            return [dict(r) for r in cur.fetchall()]


class Scraper:
    _FEEDS = [
        'https://feeds.reuters.com/reuters/worldNews',
        'https://feeds.reuters.com/reuters/businessNews',
        'https://feeds.reuters.com/reuters/marketsNews',
        'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664',
        'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20409666',
    ]
    
    _ALLOWED_DOMAINS = frozenset(['reuters.com', 'cnbc.com', 'ft.com', 'bloomberg.com', 'wsj.com'])
    
    def __init__(self, config: Config, logger: logging.Logger):
        self._cfg = config
        self._log = logger
    
    def collect(self) -> List[Dict]:
        out = []
        for url in self._FEEDS:
            out.extend(self._parse(url))
        self._log.info(f"scraped {len(out)}")
        return out
    
    def _validate_url(self, url: str) -> bool:
        if not url:
            return False
        try:
            p = urlparse(url)
            if p.scheme not in ('http', 'https'):
                return False
            domain = p.netloc.lower().replace('www.', '')
            return any(d in domain for d in self._ALLOWED_DOMAINS)
        except Exception:
            return False
    
    def _parse(self, feed_url: str) -> List[Dict]:
        out = []
        try:
            feed = feedparser.parse(feed_url)
            src = feed.feed.get('title', 'Unknown')[:100]
            
            for entry in feed.entries[:50]:
                a = self._entry(entry, src)
                if a:
                    out.append(a)
        except Exception as e:
            self._log.error(f"rss {feed_url}: {e}")
        return out
    
    def _entry(self, entry: Any, src: str) -> Optional[Dict]:
        content = _strip_html(entry.get('summary', entry.get('description', '')))
        title = _strip_html(entry.get('title', ''))
        
        if len(content) < self._cfg.min_article_length or len(title) < 10:
            return None
        
        url = entry.get('link', '')
        if not self._validate_url(url):
            url = None
        
        return {
            'hash': _hash(title + content),
            'url': url,
            'title': title[:500],
            'content': content[:self._cfg.max_content_length],
            'source': src,
            'published_ts': _parse_ts(entry.get('published', entry.get('updated'))),
            'collected_date': _utcnow().isoformat()
        }


class Deduplicator:
    _PRIO = {'Reuters': 1.0, 'Financial Times': 0.95, 'Bloomberg': 0.95, 'Wall Street Journal': 0.90, 'CNBC': 0.75}
    
    def __init__(self, config: Config, logger: logging.Logger):
        self._cfg = config
        self._log = logger
        self._vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    
    def run(self, db: Database):
        articles = db.recent_dedup(self._cfg.dedup_hours, self._cfg.dedup_max_articles)
        if len(articles) < 2:
            return
        
        try:
            texts = [f"{a['title']} {a['content'][:self._cfg.dedup_text_length]}" for a in articles]
            tfidf = self._vec.fit_transform(texts)
            sims = cosine_similarity(tfidf)
            
            dupes = self._find(articles, sims)
            if dupes:
                db.mark_dupes(dupes)
                self._log.info(f"dupes {len(dupes)}")
        except Exception as e:
            self._log.error(f"dedup: {e}")
    
    def _find(self, articles: List[Dict], sims: np.ndarray) -> List[str]:
        n = len(articles)
        seen = set()
        dupes = []
        
        for i in range(n):
            if i in seen:
                continue
            cluster = [i]
            for j in range(i + 1, n):
                if sims[i, j] >= self._cfg.dedup_threshold:
                    cluster.append(j)
                    seen.add(j)
            
            if len(cluster) > 1:
                prios = [(idx, self._prio(articles[idx]['source'])) for idx in cluster]
                prios.sort(key=lambda x: x[1], reverse=True)
                dupes.extend(articles[idx]['hash'] for idx, _ in prios[1:])
        
        return dupes
    
    def _prio(self, src: str) -> float:
        sl = src.lower()
        for k, v in self._PRIO.items():
            if k.lower() in sl:
                return v
        return 0.5


class NLP:
    _FIN_KW = frozenset([
        'fed', 'monetary', 'rates', 'inflation', 'gdp', 'treasury', 'bonds',
        'stocks', 'equity', 'credit', 'banking', 'fiscal', 'ecb', 'central bank',
        'yield', 'basis points', 'liquidity', 'fomc', 'powell', 'yellen', 'boj', 'pboc'
    ])
    
    _STRESS_KW = frozenset([
        'default', 'downgrade', 'bank run', 'liquidity crunch', 'margin call',
        'contagion', 'bailout', 'insolvency', 'systemic risk', 'credit crunch',
        'financial stress', 'debt crisis', 'sovereign default', 'restructuring'
    ])
    
    _THEMES = {
        'Monetary Policy': ['fed', 'ecb', 'boj', 'pboc', 'central bank', 'interest rate', 'monetary policy', 'powell', 'lagarde', 'hike', 'cut', 'tightening', 'easing', 'fomc'],
        'Inflation': ['inflation', 'cpi', 'pce', 'price', 'deflation', 'disinflation', 'core inflation'],
        'Recession': ['recession', 'contraction', 'slowdown', 'gdp', 'growth', 'downturn'],
        'USD Liquidity': ['liquidity', 'dollar', 'usd', 'funding', 'repo', 'swap lines', 'dollar shortage', 'fx swap'],
        'Banking Stress': ['bank', 'banking', 'deposit', 'svb', 'credit suisse', 'ubs', 'capital ratio', 'tier 1', 'stress test'],
        'Credit Crisis': ['credit', 'spreads', 'default', 'cds', 'high yield', 'junk', 'downgrade', 'credit crunch'],
        'Equity Markets': ['stock', 'equity', 's&p', 'dow', 'nasdaq', 'shares', 'rally', 'selloff'],
        'Commodities': ['commodity', 'copper', 'iron', 'metals', 'agriculture'],
        'Gold': ['gold', 'silver', 'precious metal', 'bullion'],
        'Energy': ['oil', 'gas', 'energy', 'opec', 'crude', 'brent', 'wti', 'lng'],
        'Geopolitics': ['war', 'conflict', 'sanctions', 'diplomacy', 'ukraine', 'russia', 'china', 'taiwan', 'tensions'],
        'Defense': ['military', 'defense', 'weapons', 'nato', 'pentagon', 'warfare'],
        'Trade': ['trade', 'tariff', 'exports', 'imports', 'wto', 'trade war'],
        'Supply Chain': ['supply chain', 'logistics', 'shipping', 'container', 'shortage', 'bottleneck']
    }
    
    def __init__(self, config: Config, logger: logging.Logger):
        self._cfg = config
        self._log = logger
        
        logger.info("loading models")
        self._sent = hf_pipeline(
            "sentiment-analysis", model=config.finbert_model,
            device=-1, truncation=True, max_length=config.max_text_length, top_k=None
        )
        self._trans = hf_pipeline("translation", model=config.translation_model, device=-1)
        logger.info("models ready")
    
    def sentiment(self, text: str) -> Dict[str, float]:
        try:
            res = self._sent(text[:self._cfg.max_text_length])[0]
            d = {r['label'].lower(): r['score'] for r in res}
            pos, neg, neu = d.get('positive', 0), d.get('negative', 0), d.get('neutral', 0)
            return {'sentiment_financial': (pos - neg) * (1 - neu), 'sentiment_pos': pos, 'sentiment_neg': neg, 'sentiment_neu': neu}
        except Exception as e:
            self._log.error(f"sentiment: {e}")
            return {'sentiment_financial': 0, 'sentiment_pos': 0, 'sentiment_neg': 0, 'sentiment_neu': 1}
    
    def theme(self, text: str) -> Tuple[str, float]:
        tl = " " + re.sub(r"[^a-z0-9\s]", " ", text.lower()) + " "
        scores = {t: sum(1 for kw in kws if f" {kw} " in tl) for t, kws in self._THEMES.items()}
        
        if not scores or max(scores.values()) == 0:
            return "Unknown", 0.3
        
        best = max(scores, key=scores.get)
        return best, min(scores[best] / 3.0, 1.0)
    
    def finance_score(self, text: str) -> float:
        tl = text.lower()
        return min(sum(1 for kw in self._FIN_KW if kw in tl) / 5.0, 1.0)
    
    def stress_score(self, text: str) -> float:
        tl = text.lower()
        return min(sum(1 for kw in self._STRESS_KW if kw in tl) / 3.0, 1.0)
    
    def snippet(self, text: str, max_chars: int = 400) -> str:
        sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        out = ""
        for s in sents:
            if len(out) + len(s) < max_chars:
                out += s + ". "
            else:
                break
        return out.strip() or text[:max_chars]
    
    def translate(self, texts: List[str]) -> List[str]:
        out = []
        bs = self._cfg.translation_batch_size
        
        for i in range(0, len(texts), bs):
            chunk = [t[:1000] if t else "" for t in texts[i:i+bs]]
            try:
                res = self._trans(chunk, max_length=512, batch_size=bs)
                out.extend(r["translation_text"] for r in res)
            except Exception as e:
                self._log.error(f"translate: {e}")
                out.extend(chunk)
        
        return out


class Scorer:
    _SRC = {'Reuters': 1.0, 'Financial Times': 0.95, 'Bloomberg': 0.95, 'Wall Street Journal': 0.90, 'CNBC': 0.75}
    _CRIT = {
        'Monetary Policy': 1.0, 'Banking Stress': 1.0, 'Credit Crisis': 1.0,
        'Recession': 0.95, 'USD Liquidity': 0.90, 'Inflation': 0.85,
        'Geopolitics': 0.80, 'Defense': 0.80, 'Energy': 0.75,
        'Equity Markets': 0.70, 'Commodities': 0.65, 'Gold': 0.65,
        'Trade': 0.60, 'Supply Chain': 0.55
    }
    
    def __init__(self, config: Config):
        self._cfg = config
    
    def calc(self, a: Dict) -> float:
        rec = self._recency(a)
        src = self._source(a.get('source', ''))
        thm = self._theme(a.get('theme', ''), a.get('theme_score', 0))
        sent = self._sent_intensity(a.get('sentiment_financial', 0), a.get('finance_likeness', 0), a.get('stress_score', 0))
        
        return min(100.0, (
            self._cfg.weight_recency * rec +
            self._cfg.weight_source * src +
            self._cfg.weight_theme * thm +
            self._cfg.weight_sentiment * sent
        ) * 100)
    
    def _recency(self, a: Dict) -> float:
        try:
            ts = a.get('published_ts') or a.get('collected_date')
            if not ts:
                return 0.5
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hrs = (_utcnow() - dt).total_seconds() / 3600
            return max(0, 1.0 - hrs / 48)
        except Exception:
            return 0.5
    
    def _source(self, src: str) -> float:
        sl = src.lower()
        for k, v in self._SRC.items():
            if k.lower() in sl:
                return v
        return 0.5
    
    def _theme(self, theme: str, conf: float) -> float:
        c = self._CRIT.get(theme, 0.5)
        return 0.5 * c + 0.5 * c * conf
    
    def _sent_intensity(self, sent: float, fin: float, stress: float) -> float:
        base = abs(sent)
        risk = max(0, -sent) * fin
        return min(1.0, base + risk * 0.3 + stress * 0.25)


class Briefing:
    _PRIO_THEMES = ['Monetary Policy', 'Banking Stress', 'Recession', 'Inflation', 'Geopolitics', 'Energy']
    
    def __init__(self, config: Config, logger: logging.Logger):
        self._cfg = config
        self._log = logger
    
    def generate(self, date: str, articles: List[Dict]) -> str:
        ds = datetime.fromisoformat(date).strftime("%d/%m/%Y")
        
        lines = [
            f"# BRIEFING STRATEGIQUE - {ds}",
            f"\n**Articles:** {len(articles)} | **Mode:** Production\n",
            "---\n",
            "## TOP RISKS\n"
        ]
        
        for i, a in enumerate(articles[:5], 1):
            lines.append(self._fmt_article(i, a))
        
        lines.append("## SYNTHESE\n")
        lines.extend(self._synthesis(articles))
        lines.append(f"\n---\n*{_utcnow().strftime('%d/%m/%Y %H:%M UTC')}*\n")
        
        return "".join(lines)
    
    def _fmt_article(self, idx: int, a: Dict) -> str:
        sent = a.get('sentiment_financial', 0)
        stress = a.get('stress_score', 0)
        
        try:
            pts = a.get('published_ts') or a.get('collected_date')
            pstr = datetime.fromisoformat(pts).strftime("%d/%m %H:%M")
        except Exception:
            pstr = "N/A"
        
        sl = "Positif" if sent > 0.3 else ("Negatif" if sent < -0.3 else "Neutre")
        imp = self._impact(a)
        title = (a.get('title_fr') or a['title'])[:140]
        
        return f"""### {idx}. {title}

**Score:** {a.get('strategic_score', 0):.0f}/100 | **{pstr}** | {a.get('source', 'N/A')}
**Theme:** {a.get('theme', 'N/A')} | **Sentiment:** {sl} | **Stress:** {stress:.2f}

{imp}

{a.get('snippet_fr', 'N/A')}

[Source]({a.get('url') or '#'})

---

"""
    
    def _impact(self, a: Dict) -> str:
        theme = a.get('theme', '')
        stress = a.get('stress_score', 0)
        neg = a.get('sentiment_neg', 0)
        
        if stress > 0.5:
            return "[ALERT] Stress systemique eleve"
        if theme in ('Banking Stress', 'Credit Crisis', 'USD Liquidity'):
            return "[CRITICAL] Impact macro critique"
        if theme in ('Monetary Policy', 'Inflation'):
            return "[POLICY] Politique monetaire"
        if neg > 0.6:
            return "[RISK] Sentiment negatif"
        return "[INFO] Evenement majeur"
    
    def _synthesis(self, articles: List[Dict]) -> List[str]:
        themes = defaultdict(list)
        for a in articles:
            themes[a.get('theme', 'Unknown')].append(a)
        
        lines = []
        for t in self._PRIO_THEMES:
            if t in themes:
                items = themes[t]
                titles = ", ".join((a.get('title_fr') or a['title'])[:60] + "..." for a in items[:2])
                lines.append(f"**{t}** ({len(items)}): {titles}\n\n")
        
        return lines
    
    def save(self, content: str, date: str) -> Path:
        fn = f"briefing_{date.replace(':', '-').replace(' ', '_')}.md"
        fp = self._cfg.output_dir / fn
        fp.write_text(content, encoding='utf-8')
        self._log.info(f"saved {fp}")
        return fp


class Pipeline:
    def __init__(self, config: Config):
        self._cfg = config
        self._log = setup_logging(config)
        
        self._db = Database(config.db_path, self._log)
        self._scraper = Scraper(config, self._log)
        self._dedup = Deduplicator(config, self._log)
        self._nlp = NLP(config, self._log)
        self._scorer = Scorer(config)
        self._brief = Briefing(config, self._log)
    
    def collect(self) -> int:
        self._log.info("collection")
        articles = self._scraper.collect()
        inserted = sum(1 for a in articles if self._db.insert(a))
        self._log.info(f"inserted {inserted}")
        return inserted
    
    def dedup(self):
        self._log.info("dedup")
        self._dedup.run(self._db)
    
    def analyze(self):
        self._log.info("analysis")
        articles = self._db.unprocessed(self._cfg.analyze_limit)
        self._log.info(f"processing {len(articles)}")
        
        prelim = []
        for i, a in enumerate(articles, 1):
            try:
                data = self._analyze_one(a)
                prelim.append(data)
                if i % 20 == 0:
                    self._log.info(f"analyzed {i}/{len(articles)}")
            except Exception as e:
                self._log.error(f"analysis {a['hash']}: {e}")
        
        prelim.sort(key=lambda x: x['strategic_score'], reverse=True)
        
        top_n = min(self._cfg.top_n_articles * 2, len(prelim))
        self._process_top(prelim[:top_n])
        self._process_rest(prelim[top_n:])
        
        self._log.info(f"processed {len(prelim)}")
    
    def _analyze_one(self, a: Dict) -> Dict:
        content = a['content']
        sent = self._nlp.sentiment(content)
        theme, tscore = self._nlp.theme(content)
        
        data = {
            'hash': a['hash'],
            'title': a['title'],
            'content': content,
            'source': a.get('source', ''),
            'published_ts': a.get('published_ts'),
            'collected_date': a.get('collected_date'),
            'theme': theme,
            'theme_score': tscore,
            'finance_likeness': self._nlp.finance_score(content),
            'stress_score': self._nlp.stress_score(content),
            **sent
        }
        data['strategic_score'] = self._scorer.calc(data)
        return data
    
    def _process_top(self, articles: List[Dict]):
        if not articles:
            return
        
        self._log.info(f"translating {len(articles)}")
        
        titles = [a['title'] for a in articles]
        snippets = [self._nlp.snippet(a['content']) for a in articles]
        
        titles_fr = self._nlp.translate(titles)
        snippets_fr = self._nlp.translate(snippets)
        
        rows = [
            (
                a['sentiment_financial'], a['sentiment_pos'],
                a['sentiment_neg'], a['sentiment_neu'],
                a['theme'], a['theme_score'], a['strategic_score'],
                a['finance_likeness'], a['stress_score'],
                snippets_fr[i], titles_fr[i][:140], a['hash']
            )
            for i, a in enumerate(articles)
        ]
        self._db.batch_update(rows)
    
    def _process_rest(self, articles: List[Dict]):
        if not articles:
            return
        
        rows = [
            (
                a['sentiment_financial'], a['sentiment_pos'],
                a['sentiment_neg'], a['sentiment_neu'],
                a['theme'], a['theme_score'], a['strategic_score'],
                a['finance_likeness'], a['stress_score'],
                '', '', a['hash']
            )
            for a in articles
        ]
        self._db.batch_update(rows)
    
    def briefing(self, date: Optional[str] = None) -> Optional[Path]:
        self._log.info("briefing")
        
        date = date or _utcnow().date().isoformat()
        articles = self._db.top_strategic(date, self._cfg.top_n_articles)
        
        if not articles:
            self._log.warning("no articles")
            return None
        
        self._log.info(f"generating {len(articles)}")
        content = self._brief.generate(date, articles)
        return self._brief.save(content, date)
    
    def run(self) -> Optional[Path]:
        self._log.info("start")
        t0 = time.time()
        
        self.collect()
        self.dedup()
        self.analyze()
        fp = self.briefing()
        
        self._log.info(f"done {time.time()-t0:.1f}s")
        return fp


def main():
    Pipeline(Config.from_env()).run()


if __name__ == "__main__":
    main()
