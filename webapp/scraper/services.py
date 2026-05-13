import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from html import unescape
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from .models import ContentChunk, WebPage

logger = logging.getLogger(__name__)

MAIN_SITE_HOST = 'www.acibadem.edu.tr'
BOLOGNA_HOST = 'obs.acibadem.edu.tr'
ALL_BOLOGNA_UNIT_TYPES = frozenset({'myo', 'lis', 'yls', 'dok'})
DEFAULT_TIMEOUT = 30
DEFAULT_RATE_LIMIT_DELAY = 1.0
DEFAULT_RENDER_WAIT_TIMEOUT_MS = 5000
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
DEFAULT_CHUNK_SIZE = 950
DEFAULT_CHUNK_OVERLAP = 140
MIN_USEFUL_TEXT_LENGTH = 50
MAX_PAGE_TITLE_LENGTH = WebPage._meta.get_field('title').max_length or 500
DEFAULT_MAIN_SITE_SEEDS = (
    'https://www.acibadem.edu.tr/',
    'https://www.acibadem.edu.tr/akademik',
    'https://www.acibadem.edu.tr/akademik/onlisans',
    'https://www.acibadem.edu.tr/universite',
    'https://www.acibadem.edu.tr/ogrenci/ogrenci',
    'https://www.acibadem.edu.tr/arastirma',
    'https://www.acibadem.edu.tr/aday/ogrenci',
    'https://www.acibadem.edu.tr/surdurulebilir-kampus',
    'https://www.acibadem.edu.tr/akademik/lisans/tip-fakultesi/akademik-kadro',
    'https://www.acibadem.edu.tr/akademik/lisans/tip-fakultesi/tip-fakultesi-yonetimi',
    'https://www.acibadem.edu.tr/akademik/lisans/tip-fakultesi/dekanin-mesaji',
    'https://www.acibadem.edu.tr/akademik/lisans/muhendislik-ve-doga-bilimleri-fakultesi/akademik-kadro',
    'https://www.acibadem.edu.tr/akademik/lisans/saglik-bilimleri-fakultesi/akademik-kadro',
    'https://www.acibadem.edu.tr/akademik/lisans/insan-ve-toplum-bilimleri-fakultesi/akademik-kadro',
    'https://www.acibadem.edu.tr/akademik/lisans/eczacilik-fakultesi/akademik-kadro',
    'https://www.acibadem.edu.tr/akademik/lisans/muhendislik-ve-doga-bilimleri-fakultesi/bolumler/bilgisayar-muhendisligi/akademik-kadro',
)
SKIP_MAIN_SITE_PATTERNS = (
    '/en',
    '/duyurular',
    '/etkinlikler',
    '/haberler',
    '/search',
    '/api/',
)
SKIP_EXTENSIONS = (
    '.csv',
    '.pdf',
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.svg',
    '.webp',
    '.zip',
    '.doc',
    '.docx',
    '.xls',
    '.xlsx',
    '.ppt',
    '.pptx',
    '.mp4',
    '.mp3',
)
NAME_STOPWORDS = (
    'akademik kadro',
    'program hakkında',
    'program hakkinda',
    'programı bilgileri',
    'programi bilgileri',
    'program yetkilileri',
    'dersler',
    'fakültesi',
    'fakulte',
    'bölümü',
    'bolumu',
    'üniversitesi',
    'universitesi',
    'yüksekokulu',
    'yuksekokulu',
)
HEADER_ROW_TOKENS = {
    'ad',
    'adi',
    'ad soyad',
    'adı soyadı',
    'soyad',
    'unvan',
    'ünvan',
    'gorev',
    'görev',
    'rol',
    'mail',
    'eposta',
    'e posta',
    'telefon',
}
ACADEMIC_TITLE_KEYWORDS = (
    'prof',
    'doç',
    'doc',
    'dr',
    'öğr',
    'ogr',
    'ogr',
    'araş',
    'aras',
    'gör',
    'gor',
    'assistant professor',
    'associate professor',
    'professor',
    'lecturer',
    'instructor',
)
STAFF_ROLE_KEYWORDS = (
    *ACADEMIC_TITLE_KEYWORDS,
    'başkan',
    'baskan',
    'dekan',
    'yardımcı',
    'yardimci',
    'coordinator',
    'koordinat',
)
MANAGEMENT_ROLE_KEYWORDS = (
    'başkan',
    'baskan',
    'dekan',
    'müdür',
    'mudur',
    'koordinat',
    'yönetim',
    'yonetim',
)
STAFF_CARD_SELECTORS = (
    '.views-row',
    '.person-card',
    '.staff-card',
    '.academic-staff-card',
    '.akademik-kadro-item',
    '.person-image',
    '.team-member',
    '.member',
    '.list-group-item',
    '.media',
    '.card',
)
MAIN_SITE_PROGRAM_NOISE_SELECTORS = (
    '.tab-component',
    '.paragraph--type--prg-tab',
    '.paragraph--type--prg-views-reference',
    '.wrapper-view-announcements',
    '.news-slider-swiper',
    '.announcements-slider-swiper',
    '.right-cover-bg',
    '.home-slider-swipper',
    '.paragraph--type--prg-slider',
    '.pagination-wrapper',
    '.swiper-bottom',
    '.swiper-control-btn',
    '.all-list-link',
)
MAIN_SITE_UNIT_KEYWORDS = (
    'fakülte',
    'fakulte',
    'yüksekokulu',
    'yuksekokulu',
    'meslek yüksekokulu',
    'meslek yuksekokulu',
    'enstitü',
    'enstitu',
)
MAIN_SITE_GENERIC_CRUMBS = {
    'anasayfa',
    'üniversite',
    'universite',
    'akademik',
    'öğrenci',
    'ogrenci',
    'aday öğrenci',
    'aday ogrenci',
    'lisans',
    'ön lisans',
    'on lisans',
    'önlisans',
    'onlisans',
    'lisansüstü',
    'lisansustu',
    'bölümler',
    'bolumler',
    'bölüm',
    'bolum',
    'programlar',
}
CANDIDATE_TOPIC_PATTERNS = (
    (
        'admissions_scores',
        'Kontenjan ve Puan Tablosu',
        '/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
    ),
    (
        'tuition',
        'Öğrenim Ücretleri',
        '/aday/ogrenci/egitim/lisans/lisans-ogrenim-ucretleri',
    ),
    (
        'scholarships',
        'Burs Olanakları',
        '/aday/ogrenci/egitim/burs/burs-olanaklari',
    ),
    (
        'dormitory',
        'Yurt Bilgileri ve Ücretleri',
        '/ogrenci/acuda-yasam/acibadem-mehmet-ali-aydinlar-universitesi-ogrenci-yurtlari/basvurular',
    ),
    (
        'international',
        'Uluslararası Olanaklar',
        '/uluslararasi-ofis/degisim-programlari/erasmus/ogrenci-hareketliligi',
    ),
    (
        'double_major_minor',
        'Çift Anadal-Yandal Programları',
        '/ogrenci/ogrenci-isleri/cift-anadal-yandal-programlari',
    ),
)
GENERAL_TOPIC_PATTERNS = (
    (
        'library',
        'Kütüphane',
        ('kütüphane', 'kutuphane', 'library'),
    ),
    (
        'sports',
        'Spor Merkezi',
        ('spor', 'sport', 'fitness', 'havuz', 'yüzme', 'yuzme', 'basketbol'),
    ),
    (
        'student_clubs',
        'Öğrenci Kulüpleri',
        (
            'öğrenci kulüpleri',
            'ogrenci kulupleri',
            'kulübü',
            'kulubu',
            'kulüp',
            'kulup',
            'topluluğu',
            'toplulugu',
            'topluluk',
        ),
    ),
    (
        'dormitory',
        'Yurt Bilgileri ve Ücretleri',
        ('yurt', 'konaklama', 'depozito', 'dorm'),
    ),
    (
        'international',
        'Uluslararası Olanaklar',
        ('erasmus', 'uluslararası', 'uluslararasi', 'değişim', 'degisim', 'hareketlilik'),
    ),
    (
        'double_major_minor',
        'Çift Anadal-Yandal Programları',
        ('çift anadal', 'cift anadal', 'yandal', 'çap', 'cap', 'minor', 'major'),
    ),
)
DEFAULT_CANDIDATE_ROOT_URL = 'https://www.acibadem.edu.tr/aday/ogrenci'
DEFAULT_CANDIDATE_TOPIC_URLS = tuple(
    f'https://{MAIN_SITE_HOST}{path}' for _topic, _label, path in CANDIDATE_TOPIC_PATTERNS
)
PROGRAM_NOTE_KEYWORDS = (
    'kontenjanına yerleşen',
    'kontenjanina yerlesen',
    'yerleşen öğrencilere',
    'yerlesen ogrencilere',
    'bölümü için',
    'bolumu icin',
    'burs hakkında bilgi almak',
    'burs hakkinda bilgi almak',
    'tercih sırasına bakılmaksızın',
    'tercih sirasina bakilmaksizin',
)
PLACEMENT_LABEL_KEYWORDS = (
    'burslu',
    'indirimli',
    'ücretli',
    'ucretli',
)


@dataclass
class ExtractedPage:
    url: str
    title: str
    text: str
    raw_html: str
    metadata: dict


class ScraperClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        playwright_factory: Any | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                'User-Agent': (
                    'Mozilla/5.0 (compatible; ACUChatbotBot/1.0; '
                    '+https://www.acibadem.edu.tr/)'
                )
            }
        )
        self._playwright_factory = playwright_factory
        self._playwright_cm = None
        self._playwright = None
        self._browser = None
        self._context = None

    def get(self, url: str, *, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
        return self.session.get(url, timeout=timeout)

    def _load_playwright(self):
        if self._playwright_factory is not None:
            return self._playwright_factory()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError('playwright is not installed') from exc
        return sync_playwright()

    def _ensure_browser_context(self):
        if self._context is not None:
            return self._context
        self._playwright_cm = self._load_playwright()
        self._playwright = self._playwright_cm.start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=self.session.headers.get('User-Agent'),
        )
        return self._context

    def get_rendered(
        self,
        url: str,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        wait_selectors: Iterable[str] | None = None,
    ) -> str:
        page = self._ensure_browser_context().new_page()
        timeout_ms = max(int(timeout * 1000), 1)
        selector_timeout = min(timeout_ms, DEFAULT_RENDER_WAIT_TIMEOUT_MS)
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=timeout_ms)
            try:
                page.wait_for_load_state('networkidle', timeout=selector_timeout)
            except Exception:
                logger.debug('playwright networkidle timeout url=%s', url)
            for selector in wait_selectors or ():
                try:
                    page.wait_for_selector(selector, timeout=selector_timeout)
                    break
                except Exception:
                    logger.debug('playwright selector timeout url=%s selector=%s', url, selector)
            return page.content()
        finally:
            page.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._playwright_cm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def build_session() -> ScraperClient:
    return ScraperClient()


def normalize_whitespace(text: str) -> str:
    text = unescape(text or '')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize_page_title(title: str) -> str:
    normalized = normalize_whitespace(title)
    if len(normalized) <= MAX_PAGE_TITLE_LENGTH:
        return normalized
    logger.warning(
        'page_title_truncated original_length=%s max_length=%s',
        len(normalized),
        MAX_PAGE_TITLE_LENGTH,
    )
    return normalized[:MAX_PAGE_TITLE_LENGTH].rstrip()


def slugify_text(value: str) -> str:
    normalized = unicodedata.normalize('NFKD', value or '')
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', ascii_only.casefold()).strip('-')
    return slug or 'item'


def _build_structured_page_url(record_type: str, *parts: str) -> str:
    slug_parts = [slugify_text(part) for part in parts if part]
    slug = '/'.join(part for part in slug_parts if part) or 'item'
    return f'https://{MAIN_SITE_HOST}/__structured__/{record_type}/{slug}'


def _get_candidate_topic_details(url: str) -> tuple[str, str] | None:
    parsed_path = urlparse(url).path.rstrip('/')
    lowered_path = parsed_path.casefold()
    for topic, label, prefix in CANDIDATE_TOPIC_PATTERNS:
        if lowered_path.startswith(prefix.casefold()):
            return topic, label
    return None


def infer_general_topic_metadata(url: str, title: str = '') -> dict:
    parsed_path = urlparse(url or '').path
    searchable = _normalize_lookup_text(f'{title} {parsed_path}')
    if not searchable:
        return {}

    for topic, topic_label, keywords in GENERAL_TOPIC_PATTERNS:
        for keyword in keywords:
            normalized_keyword = _normalize_lookup_text(keyword)
            if re.search(rf'(^|\s){re.escape(normalized_keyword)}($|\s)', searchable):
                return {
                    'topic': topic,
                    'topic_label': topic_label,
                    'section_title': topic_label,
                }
    return {}


def canonicalize_main_site_url(url: str) -> str | None:
    if '{' in url or '}' in url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'}:
        return None
    if parsed.netloc != MAIN_SITE_HOST:
        return None
    path = parsed.path.rstrip('/') or '/'
    if path.startswith('/en') or any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return None
    lowered = path.lower()
    if any(lowered.startswith(pattern) for pattern in SKIP_MAIN_SITE_PATTERNS):
        return None
    canonical = parsed._replace(scheme='https', path=path, query='', fragment='')
    return urlunparse(canonical)


def is_allowed_main_site_url(url: str) -> bool:
    return canonicalize_main_site_url(url) is not None


def extract_main_site_links(html: str, current_url: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    links: set[str] = set()
    for anchor in soup.select('a[href]'):
        href = anchor.get('href', '').strip()
        if not href:
            continue
        absolute = urljoin(current_url, href)
        canonical = canonicalize_main_site_url(absolute)
        if canonical:
            links.add(canonical)
    return sorted(links)


def _is_main_site_staff_url(url: str) -> bool:
    return urlparse(url).path.rstrip('/').endswith('/akademik-kadro')


def _infer_bologna_staff_page_type(url: str) -> str | None:
    path = urlparse(url).path.lower()
    if path.endswith('progacademicstaff.aspx'):
        return 'academic_staff'
    if path.endswith('progofficials.aspx'):
        return 'officials'
    return None


def _should_use_rendered_fetch(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = dict(parse_qsl(parsed.query))
    if parsed.netloc == MAIN_SITE_HOST:
        return False
    if parsed.netloc != BOLOGNA_HOST:
        return False
    return bool(
        path.endswith('unitselection.aspx')
        or (path.endswith('index.aspx') and query.get('curSunit'))
        or path.endswith('progabout.aspx')
        or path.endswith('progcourses.aspx')
        or path.endswith('progacademicstaff.aspx')
        or path.endswith('progofficials.aspx')
    )


def _default_wait_selectors(url: str) -> tuple[str, ...]:
    if _is_main_site_staff_url(url):
        return ('#block-acu-content .views-row', '.views-row', '.sidebar-page-content')
    page_type = _infer_bologna_staff_page_type(url)
    if page_type == 'academic_staff':
        return ('#UpdatePanel1 table tr', '#UpdatePanel1 .list-group-item', '#UpdatePanel1')
    if page_type == 'officials':
        return ('#UpdatePanel1 table tr', '#UpdatePanel1 .panel', '#UpdatePanel1')
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    if parsed.netloc == BOLOGNA_HOST and parsed.path.lower().endswith('index.aspx') and query.get('curSunit'):
        return ('#proMenu a.nav-link', '#proMenu')
    if parsed.netloc == BOLOGNA_HOST and parsed.path.lower().endswith('unitselection.aspx'):
        return ('.panel.panel-default', '.list-group-item a[href*="curSunit="]')
    return ()


def _get_request_session(client: Any) -> Any | None:
    session = getattr(client, '__dict__', {}).get('session')
    if session is not None and hasattr(session, 'get') and hasattr(session, 'post'):
        return session
    if hasattr(client, 'get') and hasattr(client, 'post'):
        return client
    return None


def _load_drupal_settings(soup: BeautifulSoup) -> dict:
    settings_node = soup.select_one('script[data-drupal-selector="drupal-settings-json"]')
    if settings_node is None:
        return {}
    raw_settings = settings_node.get_text(strip=True)
    if not raw_settings:
        return {}
    return json.loads(raw_settings)


def _flatten_form_payload(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key, nested_value in value.items():
            flattened.update(_flatten_form_payload(f'{prefix}[{key}]', nested_value))
        return flattened
    return {prefix: '' if value is None else str(value)}


def _inject_ajax_fragment(original_html: str, selector: str, fragment_html: str) -> str:
    soup = BeautifulSoup(original_html, 'html.parser')
    target = soup.select_one(selector)
    if target is None:
        raise ValueError(f'ajax target not found selector={selector}')

    fragment = BeautifulSoup(fragment_html, 'html.parser')
    target.clear()
    for child in list(fragment.contents):
        target.append(child)
    return str(soup)


def fetch_main_site_staff_html(client: Any, url: str, *, timeout: int = DEFAULT_TIMEOUT) -> str:
    session = _get_request_session(client)
    if session is None:
        raise RuntimeError('main site staff ajax fetch requires a request session with post support')

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    placeholder = soup.select_one('[data-block-ek-id]')
    if placeholder is None:
        return html

    block_id = placeholder.get('data-block-ek-id', '').strip()
    settings = _load_drupal_settings(soup)
    block_config = settings.get('ajaxBlocks', {}).get(block_id)
    if not block_id or not block_config:
        return html

    token_response = session.get(f'https://{MAIN_SITE_HOST}/session/token', timeout=timeout)
    token_response.raise_for_status()
    csrf_token = token_response.text.strip()
    if not csrf_token:
        raise RuntimeError('empty drupal csrf token')

    payload = {
        'plugin_id': block_config.get('plugin_id', ''),
        'block_id': block_config.get('block_id', block_id),
    }
    payload.update(_flatten_form_payload('settings', block_config.get('settings', {})))

    ajax_response = session.post(
        f'https://{MAIN_SITE_HOST}/ajax/akademik-kadro-v2',
        data=payload,
        headers={'X-CSRF-Token': csrf_token},
        timeout=timeout,
    )
    ajax_response.raise_for_status()
    commands = ajax_response.json()

    selector = f'[data-block-ek-id="{block_id}"]'
    fragment_html = ''
    for command in commands:
        if command.get('selector') == selector and command.get('data'):
            fragment_html = command['data']
            break
    if not fragment_html:
        raise RuntimeError(f'ajax response did not include html for {selector}')

    return _inject_ajax_fragment(html, selector, fragment_html)


def fetch_html(
    client: Any,
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    *,
    render_mode: str = 'auto',
    wait_selectors: Iterable[str] | None = None,
) -> str | None:
    use_main_site_staff_ajax = render_mode != 'rendered' and _is_main_site_staff_url(url)
    should_render = render_mode == 'rendered' or (
        render_mode == 'auto' and _should_use_rendered_fetch(url)
    )
    selectors = tuple(wait_selectors or ()) or _default_wait_selectors(url)

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(max(rate_limit_delay, 0))
        start_time = time.perf_counter()
        try:
            if use_main_site_staff_ajax:
                html = fetch_main_site_staff_html(client, url, timeout=timeout)
            elif should_render and hasattr(client, 'get_rendered'):
                html = client.get_rendered(url, timeout=timeout, wait_selectors=selectors)
            else:
                response = client.get(url, timeout=timeout)
                response.raise_for_status()
                html = response.text
        except Exception as exc:
            if use_main_site_staff_ajax and hasattr(client, 'get_rendered'):
                try:
                    html = client.get_rendered(url, timeout=timeout, wait_selectors=selectors)
                except Exception:
                    html = None
                if html is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        'fetch_html success url=%s rendered=%s duration_ms=%.2f',
                        url,
                        True,
                        duration_ms,
                    )
                    return html
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                'fetch_html failed url=%s rendered=%s attempt=%s/%s duration_ms=%.2f error=%s',
                url,
                should_render,
                attempt,
                MAX_RETRIES,
                duration_ms,
                exc,
            )
            if attempt >= MAX_RETRIES:
                return None
            time.sleep(RETRY_BACKOFF ** (attempt - 1))
            continue

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            'fetch_html success url=%s rendered=%s duration_ms=%.2f',
            url,
            should_render,
            duration_ms,
        )
        return html
    return None


def _strip_irrelevant_nodes(node: BeautifulSoup) -> None:
    selectors = [
        'script',
        'style',
        'noscript',
        'form',
        'button',
        'svg',
        'picture',
        'figure',
        'img',
        'nav',
        'header',
        'footer',
        '.breadcrumb-wrapper',
        '.footer-menu-wrapper',
        '.social-media',
        '.sidebar-custom-video-block',
        '.sidebar-custom-content-block',
        '#block-acu-domain-menu',
        '.cookie-banner',
        '.popup-overlay',
        '.share-buttons',
        '[role="navigation"]',
    ]
    for selector in selectors:
        for match in node.select(selector):
            match.decompose()


def _ordered_unique_texts(lines: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = normalize_whitespace(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _slug_to_title(slug: str) -> str:
    words = [part for part in slug.replace('-', ' ').split() if part]
    if not words:
        return ''
    lowercase_words = {'ve', 'ile', 'veya'}
    titled = []
    for index, word in enumerate(words):
        if index and word in lowercase_words:
            titled.append(word)
        else:
            titled.append(word.capitalize())
    return ' '.join(titled)


def _extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    selectors = (
        '.breadcrumb a',
        '.breadcrumb li',
        '.breadcrumbs a',
        '.breadcrumbs li',
        'nav[aria-label="breadcrumb"] a',
        'nav[aria-label="breadcrumb"] li',
    )
    return _ordered_unique_texts(
        element.get_text(' ', strip=True)
        for selector in selectors
        for element in soup.select(selector)
    )


def _strip_main_site_program_noise(url: str, node: BeautifulSoup) -> None:
    path = urlparse(url).path.casefold()
    if '/akademik/' not in path:
        return
    for selector in MAIN_SITE_PROGRAM_NOISE_SELECTORS:
        for match in node.select(selector):
            match.decompose()


def _candidate_topic_metadata(url: str) -> dict:
    details = _get_candidate_topic_details(url)
    if not details:
        return {}
    topic, topic_label = details
    return {
        'kind': 'candidate_topic_page',
        'topic': topic,
        'topic_label': topic_label,
    }


def _infer_main_site_scope(url: str, soup: BeautifulSoup, title: str) -> dict:
    metadata = {
        'host': MAIN_SITE_HOST,
        'path': urlparse(url).path,
    }
    breadcrumbs = _extract_breadcrumbs(soup)
    faculty = ''
    program_title = ''
    meaningful_breadcrumbs: list[str] = []

    for crumb in breadcrumbs:
        normalized = normalize_whitespace(crumb)
        lowered = _normalize_lookup_text(normalized)
        if not normalized or lowered in MAIN_SITE_GENERIC_CRUMBS:
            continue
        meaningful_breadcrumbs.append(normalized)
        if not faculty and any(keyword in lowered for keyword in MAIN_SITE_UNIT_KEYWORDS):
            faculty = normalized

    for crumb in reversed(meaningful_breadcrumbs):
        lowered = _normalize_lookup_text(crumb)
        if crumb == faculty:
            continue
        if lowered in {'akademik kadro', 'program hakkında', 'program hakkinda'}:
            continue
        program_title = crumb
        break

    segments = [segment for segment in urlparse(url).path.split('/') if segment]
    if not program_title and 'bolumler' in segments:
        bolum_index = segments.index('bolumler')
        if bolum_index + 1 < len(segments):
            program_title = _slug_to_title(segments[bolum_index + 1])
    if not faculty and segments:
        if 'bolumler' in segments:
            bolum_index = segments.index('bolumler')
            if bolum_index > 0:
                faculty = _slug_to_title(segments[bolum_index - 1])
        elif len(segments) >= 3 and segments[0] == 'akademik':
            faculty = _slug_to_title(segments[2])
        elif _is_main_site_staff_url(url) and len(segments) >= 2:
            faculty = _slug_to_title(segments[-2])

    normalized_title = normalize_whitespace(title)
    lowered_title = _normalize_lookup_text(normalized_title)
    if not faculty and any(keyword in lowered_title for keyword in MAIN_SITE_UNIT_KEYWORDS):
        faculty = normalized_title
    if (
        not program_title
        and normalized_title
        and 'akademik kadro' not in lowered_title
        and lowered_title not in MAIN_SITE_GENERIC_CRUMBS
        and normalized_title != faculty
    ):
        program_title = normalized_title

    if faculty:
        metadata['faculty'] = faculty
    if program_title:
        metadata['program_title'] = program_title
    return metadata


def _normalize_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize('NFKD', normalize_whitespace(value))
    normalized = ''.join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.translate(str.maketrans({'ı': 'i', 'İ': 'I'}))
    normalized = normalized.casefold()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    return ' '.join(normalized.split())


def _looks_like_staff_role(value: str) -> bool:
    lowered = _normalize_lookup_text(value)
    return any(keyword in lowered for keyword in STAFF_ROLE_KEYWORDS)


def _looks_like_header_row(values: list[str]) -> bool:
    if not values:
        return False
    normalized_values = {_normalize_lookup_text(value) for value in values if value}
    return bool(normalized_values) and normalized_values <= HEADER_ROW_TOKENS


def _is_probable_person_name(value: str) -> bool:
    normalized = normalize_whitespace(value)
    lowered = _normalize_lookup_text(normalized)
    if not normalized or any(char.isdigit() for char in normalized):
        return False
    if any(stopword in lowered for stopword in NAME_STOPWORDS):
        return False
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü'’.-]+", normalized)
    if len(words) < 2 or len(words) > 6:
        return False
    if words[-1].casefold() in {'mühendisliği', 'muhendisligi', 'psikoloji', 'sosyoloji', 'genetik', 'rehabilitasyon'}:
        return False
    if sum(1 for word in words if word and word[0].isupper()) < 2:
        return False
    return True


def _split_prefixed_title_and_name(value: str) -> tuple[str, str]:
    tokens = [token.strip(',') for token in normalize_whitespace(value).split()]
    if len(tokens) < 3:
        return '', ''
    title_tokens: list[str] = []
    name_start = 0
    for index, token in enumerate(tokens):
        normalized = _normalize_lookup_text(token).strip('.')
        if any(normalized.startswith(keyword) for keyword in ACADEMIC_TITLE_KEYWORDS):
            title_tokens.append(token)
            name_start = index + 1
            continue
        if title_tokens and normalized in {'uyesi', 'üyesi', 'gorevlisi', 'görevlisi', 'yardimcisi', 'yardımcısı'}:
            title_tokens.append(token)
            name_start = index + 1
            continue
        break
    if not title_tokens or len(tokens[name_start:]) < 2:
        return '', ''
    name = ' '.join(tokens[name_start:])
    if not _is_probable_person_name(name):
        return '', ''
    return name, ' '.join(title_tokens)


def _extract_name_and_title(lines: list[str]) -> tuple[str, str]:
    for line in lines:
        name, title = _split_prefixed_title_and_name(line)
        if name:
            return name, title

    name = ''
    title = ''
    for line in lines:
        if not name and _is_probable_person_name(line):
            name = line
            continue
        if not title and _looks_like_staff_role(line):
            title = line
    return name, title


def _add_staff_entry(entries: list[dict], seen_keys: set[tuple[str, str]], name: str, title: str) -> None:
    normalized_name = normalize_whitespace(name)
    normalized_title = normalize_whitespace(title)
    if not _is_probable_person_name(normalized_name):
        return
    key = (_normalize_lookup_text(normalized_name), _normalize_lookup_text(normalized_title))
    if key in seen_keys:
        return
    seen_keys.add(key)
    entries.append({'name': normalized_name, 'title': normalized_title})


def _extract_staff_entries_from_tables(node: BeautifulSoup) -> list[dict]:
    entries: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in node.select('table tr'):
        cells = _ordered_unique_texts(
            cell.get_text(' ', strip=True) for cell in row.select('th, td')
        )
        if not cells or _looks_like_header_row(cells):
            continue
        name, title = _extract_name_and_title(cells)
        if not name and cells:
            probable_name_index = next(
                (index for index, value in enumerate(cells) if _is_probable_person_name(value)),
                None,
            )
            if probable_name_index is not None:
                name = cells[probable_name_index]
                title_parts = [value for index, value in enumerate(cells) if index != probable_name_index]
                title = title or ' | '.join(title_parts)
            else:
                name = cells[0]
                title = title or (' | '.join(cells[1:]) if len(cells) > 1 else '')
        _add_staff_entry(entries, seen_keys, name, title)
    return entries


def _extract_staff_entries_from_cards(node: BeautifulSoup) -> list[dict]:
    entries: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for selector in STAFF_CARD_SELECTORS:
        for candidate in node.select(selector):
            if candidate.find('table'):
                continue
            lines = _ordered_unique_texts(candidate.stripped_strings)
            if not lines or len(lines) > 12:
                continue
            name, title = _extract_name_and_title(lines)
            _add_staff_entry(entries, seen_keys, name, title)
    return entries


def _extract_staff_entries_from_text(text: str) -> list[dict]:
    entries: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    lines = _ordered_unique_texts(text.splitlines())
    for index, line in enumerate(lines):
        name, title = _split_prefixed_title_and_name(line)
        if name:
            _add_staff_entry(entries, seen_keys, name, title)
            continue
        if _looks_like_staff_role(line):
            next_line = lines[index + 1] if index + 1 < len(lines) else ''
            if _is_probable_person_name(next_line):
                _add_staff_entry(entries, seen_keys, next_line, line)
                continue
        if not _is_probable_person_name(line):
            continue
        next_lines = lines[index + 1 : index + 3]
        role = ''
        for next_line in next_lines:
            if _looks_like_staff_role(next_line) and not _is_probable_person_name(next_line):
                role = next_line
                break
        _add_staff_entry(entries, seen_keys, line, role)
    return entries


def _extract_staff_entries(node: BeautifulSoup, fallback_text: str) -> list[dict]:
    entries = _extract_staff_entries_from_tables(node)
    if entries:
        return entries
    entries = _extract_staff_entries_from_cards(node)
    if entries:
        return entries
    return _extract_staff_entries_from_text(fallback_text)


def _build_staff_page_text(metadata: dict, entries: list[dict], *, count_label: str) -> str:
    lines = []
    if metadata.get('program_title'):
        lines.append(f"Program: {metadata['program_title']}")
    if metadata.get('faculty'):
        lines.append(f"Fakulte: {metadata['faculty']}")
    lines.append(f'Toplam {count_label} sayisi: {len(entries)}')
    for entry in entries:
        if entry.get('title'):
            lines.append(f"- {entry['name']} | {entry['title']}")
        else:
            lines.append(f"- {entry['name']}")
    return '\n'.join(lines)


def _looks_like_management_role(value: str) -> bool:
    role = normalize_whitespace(value)
    lowered = _normalize_lookup_text(role)
    return bool(role and len(role) <= 120 and any(keyword in lowered for keyword in MANAGEMENT_ROLE_KEYWORDS))


def _extract_management_role_label(section: BeautifulSoup) -> str:
    for selector in (
        '.accordion-button',
        '.accordion-header button',
        '.accordion-header',
        'button',
        'h2',
        'h3',
    ):
        candidate = section.select_one(selector)
        if candidate is None:
            continue
        role = normalize_whitespace(candidate.get_text(' ', strip=True))
        if _looks_like_management_role(role):
            return role
    return ''


def _extract_management_person_name(card: BeautifulSoup) -> str:
    for selector in (
        '.board-of-directors-detail-wrapper .title',
        '.board-of-directors-wrapper .title',
        'p.title',
        '.title',
        'h3',
        'h4',
    ):
        candidate = card.select_one(selector)
        if candidate is None:
            continue
        name = normalize_whitespace(candidate.get_text(' ', strip=True))
        if _is_probable_person_name(name):
            return name

    for image in card.select('img[alt]'):
        name = normalize_whitespace(image.get('alt', ''))
        if _is_probable_person_name(name):
            return name
    return ''


def _add_role_assignment(
    assignments: list[dict],
    seen_keys: set[tuple[str, str]],
    role: str,
    name: str,
) -> None:
    normalized_role = normalize_whitespace(role)
    normalized_name = normalize_whitespace(name)
    if not _looks_like_management_role(normalized_role) or not _is_probable_person_name(normalized_name):
        return
    key = (_normalize_lookup_text(normalized_role), _normalize_lookup_text(normalized_name))
    if key in seen_keys:
        return
    seen_keys.add(key)
    assignments.append({'role': normalized_role, 'name': normalized_name})


def _management_cards(section: BeautifulSoup) -> list[BeautifulSoup]:
    cards: list[BeautifulSoup] = []
    for selector in (
        '.board-of-directors-wrapper',
        '.management-card',
        '.person-card',
        '.card',
        '.views-row',
    ):
        cards.extend(section.select(selector))
    return cards or [section]


def _extract_role_assignments_from_management_page(node: BeautifulSoup, fallback_text: str) -> list[dict]:
    assignments: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    sections = node.select('.accordion-item') or node.select('section')
    for section in sections:
        role = _extract_management_role_label(section)
        if not role:
            continue
        for card in _management_cards(section):
            name = _extract_management_person_name(card)
            _add_role_assignment(assignments, seen_keys, role, name)

    if assignments:
        return assignments

    lines = _ordered_unique_texts(fallback_text.splitlines())
    for index, line in enumerate(lines):
        if not _looks_like_management_role(line):
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ''
        _add_role_assignment(assignments, seen_keys, line, next_line)
    return assignments


def _build_role_assignment_text(metadata: dict, assignments: list[dict]) -> str:
    scope = metadata.get('faculty') or metadata.get('program_title') or ''
    scope = normalize_whitespace(re.sub(r'\s+(yönetimi|yonetimi)$', '', scope, flags=re.IGNORECASE))
    lines = []
    if scope:
        lines.append(f'Birim: {scope}')
    for assignment in assignments:
        prefix = f'{scope} yönetim' if scope else 'Yönetim'
        lines.append(f"{prefix} | rol: {assignment['role']} | isim: {assignment['name']}")
    return '\n'.join(lines)


def extract_main_site_page(url: str, html: str) -> ExtractedPage | None:
    soup = BeautifulSoup(html, 'html.parser')
    title = ''
    title_tag = soup.select_one('h1.page-title') or soup.find('h1')
    if title_tag:
        title = normalize_whitespace(title_tag.get_text(' ', strip=True))
    if not title:
        document_title = soup.find('title')
        if document_title:
            title = normalize_whitespace(document_title.get_text(' ', strip=True))
    content_node = (
        soup.select_one('#block-acu-content')
        or soup.select_one('.sidebar-page-content')
        or soup.select_one('.field--name-body')
        or soup.select_one('article')
        or soup.select_one('main')
        or soup.body
    )
    if content_node is None:
        return None

    metadata = _infer_main_site_scope(url, soup, title)
    role_content = BeautifulSoup(str(content_node), 'html.parser')
    role_text = normalize_whitespace(role_content.get_text('\n', strip=True))
    content = BeautifulSoup(str(content_node), 'html.parser')
    _strip_irrelevant_nodes(content)
    _strip_main_site_program_noise(url, content)
    for noisy in content.select('[class*="sidebar"], [class*="footer"], [class*="header"]'):
        noisy.decompose()
    text = normalize_whitespace(content.get_text('\n', strip=True))

    if _is_main_site_staff_url(url):
        entries = _extract_staff_entries(content, text)
        metadata['kind'] = 'main_site_staff_page'
        if entries:
            metadata['staff_count'] = len(entries)
            text = _build_staff_page_text(metadata, entries, count_label='hoca')
    else:
        role_assignments = _extract_role_assignments_from_management_page(role_content, role_text)
        if role_assignments:
            metadata['kind'] = 'main_site_role_page'
            metadata['record_type'] = 'staff_role_assignment'
            metadata['role_count'] = len(role_assignments)
            text = _build_role_assignment_text(metadata, role_assignments)
        else:
            metadata['kind'] = 'main_site_page'
            metadata.update(_candidate_topic_metadata(url))
            if metadata.get('kind') == 'candidate_topic_page':
                metadata.setdefault('section_title', title or metadata.get('topic_label', ''))
            else:
                topic_metadata = infer_general_topic_metadata(url, title)
                for key, value in topic_metadata.items():
                    metadata.setdefault(key, value)

    if len(text) < MIN_USEFUL_TEXT_LENGTH:
        return None
    return ExtractedPage(
        url=url,
        title=title or url,
        text=text,
        raw_html=str(content_node),
        metadata=metadata,
    )


def extract_candidate_topic_pages(
    candidate_root_html: str, current_url: str = DEFAULT_CANDIDATE_ROOT_URL
) -> list[dict]:
    soup = BeautifulSoup(candidate_root_html, 'html.parser')
    discovered: dict[str, dict] = {}
    for anchor in soup.select('a[href]'):
        href = anchor.get('href', '').strip()
        if not href:
            continue
        absolute = canonicalize_main_site_url(urljoin(current_url, href))
        if not absolute:
            continue
        details = _get_candidate_topic_details(absolute)
        if not details:
            continue
        topic, topic_label = details
        discovered[topic] = {
            'url': absolute,
            'topic': topic,
            'topic_label': topic_label,
            'title': normalize_whitespace(anchor.get_text(' ', strip=True)) or topic_label,
        }

    ordered_pages: list[dict] = []
    for topic, topic_label, path_prefix in CANDIDATE_TOPIC_PATTERNS:
        page = discovered.get(topic)
        if page is None:
            page = {
                'url': f'https://{MAIN_SITE_HOST}{path_prefix}',
                'topic': topic,
                'topic_label': topic_label,
                'title': topic_label,
            }
        ordered_pages.append(page)
    return ordered_pages


def _infer_admission_level(label: str) -> str:
    lowered = _normalize_lookup_text(label)
    if 'ön lisans' in lowered or 'on lisans' in lowered or 'onlisans' in lowered:
        return 'onlisans'
    if 'lisans' in lowered:
        return 'lisans'
    return ''


def _admission_level_label(value: str) -> str:
    return {'onlisans': 'Ön Lisans', 'lisans': 'Lisans'}.get(value, value or '-')


def _split_program_name_and_notes(label: str) -> tuple[str, str]:
    cleaned = normalize_whitespace(re.sub(r'\*+', '', label))
    if not cleaned:
        return '', ''

    def _dedupe_repeated_title(value: str) -> str:
        tokens = value.split()
        if len(tokens) % 2 == 0 and tokens[: len(tokens) // 2] == tokens[len(tokens) // 2 :]:
            return ' '.join(tokens[: len(tokens) // 2])
        return value

    note_start: int | None = None
    for keyword in PROGRAM_NOTE_KEYWORDS:
        match = re.search(re.escape(keyword), cleaned, flags=re.IGNORECASE)
        if match and match.start() > 0:
            note_start = match.start() if note_start is None else min(note_start, match.start())
    if note_start is not None:
        return _dedupe_repeated_title(cleaned[:note_start].strip(' -')), cleaned[note_start:].strip()

    for match in re.finditer(r'\(([^()]*)\)', cleaned):
        inner_text = normalize_whitespace(match.group(1))
        if any(re.search(re.escape(keyword), inner_text, flags=re.IGNORECASE) for keyword in PROGRAM_NOTE_KEYWORDS):
            title = _dedupe_repeated_title(cleaned[: match.start()].strip().rstrip('(').strip())
            notes = cleaned[match.start() :].strip()
            return title, notes

    return _dedupe_repeated_title(cleaned), ''


def _split_program_title_and_placement(label: str) -> tuple[str, str, str]:
    program_label = normalize_whitespace(label)
    if not program_label:
        return '', '', ''

    matches = list(re.finditer(r'\(([^()]*)\)', program_label))
    if matches:
        last_match = matches[-1]
        inner_text = normalize_whitespace(last_match.group(1))
        if any(re.search(re.escape(keyword), inner_text, flags=re.IGNORECASE) for keyword in PLACEMENT_LABEL_KEYWORDS):
            base_title = program_label[: last_match.start()].rstrip().strip()
            return base_title or program_label, program_label, inner_text
    return program_label, program_label, ''


def _build_program_alias_text(
    *, program_title: str, placement_label: str = '', placement_type: str = ''
) -> str:
    aliases = [program_title]
    if placement_label and placement_label != program_title:
        aliases.append(placement_label)
    if placement_type:
        aliases.append(f'{program_title} {placement_type}')
    simplified = re.sub(r'\([^)]*\)', ' ', placement_label or '')
    simplified = normalize_whitespace(simplified)
    if simplified and simplified not in aliases:
        aliases.append(simplified)
    return ' | '.join(_ordered_unique_texts(aliases))


def _extract_candidate_datatables(scores_page_html: str, source_url: str) -> list[dict]:
    soup = BeautifulSoup(scores_page_html, 'html.parser')
    datatables: list[dict] = []
    for wrapper in soup.select('.datatable-wrapper'):
        table = wrapper.select_one('table.datatable-item[data-fid]')
        if table is None:
            continue
        fid = normalize_whitespace(table.get('data-fid', ''))
        if not fid:
            continue
        title_node = wrapper.select_one('.datatable-title') or wrapper.select_one('h3')
        title = normalize_whitespace(title_node.get_text(' ', strip=True) if title_node else '')
        download_link = wrapper.select_one('a[href]')
        download_url = ''
        if download_link and download_link.get('href'):
            download_url = urljoin(source_url, download_link.get('href'))
        datatables.append(
            {
                'fid': fid,
                'title': title,
                'download_url': download_url,
                'admission_level': _infer_admission_level(title),
            }
        )
    return datatables


def _fetch_candidate_datatable_payload(
    client: Any, fid: str, *, timeout: int = DEFAULT_TIMEOUT
) -> dict | None:
    session = _get_request_session(client)
    if session is None:
        raise RuntimeError('datatable fetch requires a request session')
    response = session.get(
        f'https://{MAIN_SITE_HOST}/api/datatable-file/{fid}',
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('status') is False:
        return None
    return payload


def extract_candidate_score_records(
    client: Any,
    scores_page_html: str,
    source_url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    records: list[dict] = []
    for datatable in _extract_candidate_datatables(scores_page_html, source_url):
        payload = _fetch_candidate_datatable_payload(client, datatable['fid'], timeout=timeout)
        if not payload:
            continue
        columns = [normalize_whitespace(str(column)) for column in payload.get('columns', [])]
        normalized_columns = [_normalize_lookup_text(column) for column in columns]

        def _find_column_index(pattern: str) -> int | None:
            for index, column in enumerate(normalized_columns):
                if pattern in column:
                    return index
            return None

        rank_indices = [
            index
            for index, column in enumerate(normalized_columns)
            if 'basari sirasi' in column
        ]
        score_type_index = _find_column_index('puan turu')
        quota_index = _find_column_index('kontenjan')
        top_score_index = _find_column_index('tavan puan')
        base_score_index = _find_column_index('taban puan')
        top_rank_index = rank_indices[0] if len(rank_indices) > 1 else None
        base_rank_index = rank_indices[-1] if rank_indices else None

        current_faculty = ''
        for row in payload.get('data', []):
            cells = [normalize_whitespace(str(cell)) for cell in row]
            if not any(cells):
                continue
            if cells[0] and not any(cells[1:]):
                current_faculty = cells[0]
                continue

            def _cell(index: int | None) -> str:
                if index is None or index >= len(cells):
                    return ''
                return cells[index]

            raw_program_label = cells[0]
            if not raw_program_label:
                continue
            program_title, placement_label, placement_type = _split_program_title_and_placement(
                raw_program_label
            )
            if not program_title:
                continue
            score_type = _cell(score_type_index)
            quota = _cell(quota_index)
            top_score = _cell(top_score_index)
            top_rank = _cell(top_rank_index)
            base_score = _cell(base_score_index)
            base_rank = _cell(base_rank_index)

            title = f'{placement_label} - Kontenjan ve Puan'
            record_url = _build_structured_page_url(
                'admissions-score',
                datatable['admission_level'],
                current_faculty,
                placement_label,
                score_type,
            )
            lines = [
                f'Program: {program_title}',
                f'Yerleşim: {placement_label}',
                f'Yerleşim Türü: {placement_type or "-"}',
                f'Akademik Birim: {current_faculty or "-"}',
                f'Eğitim Düzeyi: {_admission_level_label(datatable["admission_level"])}',
                f'Puan Türü: {score_type or "-"}',
                f'Kontenjan: {quota or "-"}',
                f'Tavan Puan: {top_score or "-"}',
                f'Tavan Başarı Sırası: {top_rank or "-"}',
                f'Taban Puan: {base_score or "-"}',
                f'Taban Başarı Sırası: {base_rank or "-"}',
            ]
            if datatable.get('download_url'):
                lines.append(f'İndirme Bağlantısı: {datatable["download_url"]}')

            records.append(
                {
                    'url': record_url,
                    'title': title,
                    'text': '\n'.join(lines),
                    'raw_html': json.dumps(
                        {
                            'columns': payload.get('columns', []),
                            'row': row,
                            'table_fid': datatable['fid'],
                        },
                        ensure_ascii=False,
                    ),
                    'metadata': {
                        'kind': 'structured_admissions_score',
                        'topic': 'admissions_scores',
                        'topic_label': 'Kontenjan ve Puan Tablosu',
                        'section_title': 'Kontenjan ve Puan',
                        'program_title': program_title,
                        'program_alias_text': _build_program_alias_text(
                            program_title=program_title,
                            placement_label=placement_label,
                            placement_type=placement_type,
                        ),
                        'placement_label': placement_label,
                        'placement_type': placement_type,
                        'faculty': current_faculty,
                        'admission_level': datatable['admission_level'],
                        'score_type': score_type,
                        'quota': quota,
                        'top_score': top_score,
                        'top_rank': top_rank,
                        'base_score': base_score,
                        'base_rank': base_rank,
                        'table_fid': datatable['fid'],
                        'source_url': source_url,
                        'download_url': datatable.get('download_url', ''),
                    },
                }
            )
    return records


def _normalize_fee_column(header: str) -> str:
    lowered = _normalize_lookup_text(header)
    if 'ucretli' == lowered or lowered.endswith('ucretli'):
        return 'fee_full'
    if '25 indirimli' in lowered:
        return 'fee_25'
    if '50 indirimli' in lowered:
        return 'fee_50'
    if 'kav destek' in lowered:
        return 'fee_kav_support'
    return slugify_text(header)


def extract_candidate_fee_records(fees_page_html: str, source_url: str) -> list[dict]:
    soup = BeautifulSoup(fees_page_html, 'html.parser')
    records: list[dict] = []

    for table in soup.select('table'):
        rows = [
            [normalize_whitespace(cell.get_text(' ', strip=True)) for cell in tr.select('th, td')]
            for tr in table.select('tr')
        ]
        rows = [row for row in rows if any(row)]
        if len(rows) < 3:
            continue

        table_title = rows[0][0]
        if 'ogrenim ucretleri' not in _normalize_lookup_text(table_title):
            continue
        admission_level = _infer_admission_level(table_title)
        headers = rows[1]
        current_faculty = ''

        for row in rows[2:]:
            non_empty_cells = [cell for cell in row if cell]
            if not non_empty_cells:
                continue
            if len(non_empty_cells) == 1:
                current_faculty = non_empty_cells[0]
                continue

            raw_program_label = row[0]
            if not raw_program_label:
                continue
            program_title, notes = _split_program_name_and_notes(raw_program_label)
            if not program_title:
                continue

            fee_values: dict[str, str] = {}
            for index, header in enumerate(headers[1:], start=1):
                if index >= len(row):
                    continue
                value = row[index]
                if not value:
                    continue
                fee_values[_normalize_fee_column(header)] = value
            if not fee_values:
                continue

            title = f'{program_title} - Öğrenim Ücreti'
            lines = [
                f'Program: {program_title}',
                f'Akademik Birim: {current_faculty or "-"}',
                f'Eğitim Düzeyi: {_admission_level_label(admission_level)}',
            ]
            if fee_values.get('fee_full'):
                lines.append(f'Ücretli: {fee_values["fee_full"]}')
            if fee_values.get('fee_25'):
                lines.append(f'%25 İndirimli Ücret: {fee_values["fee_25"]}')
            if fee_values.get('fee_50'):
                lines.append(f'%50 İndirimli Ücret: {fee_values["fee_50"]}')
            if fee_values.get('fee_kav_support'):
                lines.append(f'İlave %25 KAV Destek Burslu Ücret: {fee_values["fee_kav_support"]}')
            if notes:
                lines.append(f'Notlar: {notes}')

            records.append(
                {
                    'url': _build_structured_page_url(
                        'admissions-fee',
                        admission_level,
                        current_faculty,
                        program_title,
                    ),
                    'title': title,
                    'text': '\n'.join(lines),
                    'raw_html': json.dumps(
                        {
                            'table_title': table_title,
                            'headers': headers,
                            'row': row,
                        },
                        ensure_ascii=False,
                    ),
                    'metadata': {
                        'kind': 'structured_admissions_fee',
                        'topic': 'tuition',
                        'topic_label': 'Öğrenim Ücretleri',
                        'section_title': 'Öğrenim Ücreti',
                        'program_title': program_title,
                        'program_alias_text': _build_program_alias_text(program_title=program_title),
                        'faculty': current_faculty,
                        'admission_level': admission_level,
                        'source_url': source_url,
                        'notes': notes,
                        **fee_values,
                    },
                }
            )
    return records


def crawl_candidate_data(
    *,
    client: Any | None = None,
    session: Any | None = None,
    root_url: str = DEFAULT_CANDIDATE_ROOT_URL,
    force_refresh: bool = False,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
) -> dict:
    client = client or session
    if client is None:
        raise ValueError('crawl_candidate_data requires a client or session')

    failed_urls: set[str] = set()
    raw_seen_urls: set[str] = set()
    structured_seen_urls: set[str] = set()
    saved = 0
    updated = 0
    structured_saved = 0
    structured_updated = 0

    root_html = fetch_html(client, root_url, rate_limit_delay=rate_limit_delay)
    topic_pages = extract_candidate_topic_pages(root_html or '', root_url)
    if root_html is None:
        failed_urls.add(root_url)

    for topic_page in topic_pages:
        page_url = topic_page['url']
        raw_seen_urls.add(page_url)
        logger.debug('crawl_candidate_data processing topic=%s url=%s', topic_page['topic'], page_url)
        html = fetch_html(client, page_url, rate_limit_delay=rate_limit_delay)
        if html is None:
            failed_urls.add(page_url)
            continue

        extracted = extract_main_site_page(page_url, html)
        if extracted:
            _, changed = upsert_page_content(
                source='main_site',
                url=extracted.url,
                title=extracted.title,
                text=extracted.text,
                raw_html=extracted.raw_html,
                metadata=extracted.metadata,
                force_refresh=force_refresh,
            )
            saved += 1
            updated += int(changed)

        structured_records: list[dict] = []
        if topic_page['topic'] == 'admissions_scores':
            structured_records = extract_candidate_score_records(
                client,
                html,
                page_url,
            )
        elif topic_page['topic'] == 'tuition':
            structured_records = extract_candidate_fee_records(html, page_url)

        for record in structured_records:
            structured_seen_urls.add(record['url'])
            _, changed = upsert_page_content(
                source='structured',
                url=record['url'],
                title=record['title'],
                text=record['text'],
                raw_html=record['raw_html'],
                metadata=record['metadata'],
                force_refresh=force_refresh,
            )
            structured_saved += 1
            structured_updated += int(changed)

    if failed_urls:
        logger.warning(
            'crawl_candidate_data encountered failed fetches; skipping structured deactivation failed=%s',
            len(failed_urls),
        )
        deactivated = 0
    else:
        deactivated = mark_missing_pages_inactive('structured', structured_seen_urls)

    summary = {
        'seen': len(raw_seen_urls) + len(structured_seen_urls),
        'saved': saved + structured_saved,
        'updated': updated + structured_updated,
        'deactivated': deactivated,
        'failed': len(failed_urls),
        'structured_saved': structured_saved,
        'structured_updated': structured_updated,
        'raw_seen': len(raw_seen_urls),
        'structured_seen': len(structured_seen_urls),
    }
    logger.info('crawl_candidate_data summary=%s', summary)
    return summary


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r'\n{2,}', normalized) if part.strip()]
    chunks: list[dict] = []
    buffer = ''
    for paragraph in paragraphs:
        candidate = f'{buffer}\n\n{paragraph}'.strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append({'text': buffer, 'metadata': {'char_count': len(buffer)}})
        if len(paragraph) <= chunk_size:
            buffer = paragraph
            continue
        start = 0
        while start < len(paragraph):
            stop = min(start + chunk_size, len(paragraph))
            part = paragraph[start:stop].strip()
            if part:
                chunks.append({'text': part, 'metadata': {'char_count': len(part)}})
            if stop >= len(paragraph):
                break
            start = max(0, stop - overlap)
        buffer = ''
    if buffer:
        chunks.append({'text': buffer, 'metadata': {'char_count': len(buffer)}})
    return chunks


def build_chunk_embedding_text(text: str, metadata: dict) -> str:
    lines = []
    metadata = metadata or {}
    for label, key in (
        ('Program', 'program_title'),
        ('Yerlesim', 'placement_label'),
        ('Fakulte', 'faculty'),
        ('MufredatYili', 'curriculum_year'),
        ('Donem', 'period_label'),
        ('KayitTuru', 'record_type'),
        ('Konu', 'topic_label'),
        ('Bolum', 'section_title'),
        ('Baslik', 'page_title'),
    ):
        value = normalize_whitespace(str(metadata.get(key, '')))
        if value:
            lines.append(f'{label}: {value}')

    content = normalize_whitespace(text)
    if content:
        lines.append(f'Icerik: {content}')
    return '\n'.join(lines)


def _normalize_prebuilt_chunks(chunks: list[dict] | None) -> list[dict]:
    normalized_chunks: list[dict] = []
    for index, chunk in enumerate(chunks or []):
        text = normalize_whitespace(chunk.get('text', ''))
        if not text:
            continue
        normalized_chunks.append(
            {
                'text': text,
                'metadata': dict(chunk.get('metadata') or {}),
                'chunk_index': int(chunk.get('chunk_index', index)),
            }
        )
    return normalized_chunks


def _build_page_content_hash(normalized_text: str, normalized_chunks: list[dict]) -> str:
    payload = {
        'text': normalized_text,
        'chunks': [
            {
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'],
                'metadata': chunk['metadata'],
            }
            for chunk in normalized_chunks
        ],
    }
    return hash_content(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def upsert_page_chunks(
    *,
    source: str,
    url: str,
    title: str,
    text: str,
    raw_html: str,
    metadata: dict,
    chunks: list[dict],
    language: str = 'tr',
    force_refresh: bool = False,
) -> tuple[WebPage, bool]:
    os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')
    normalized_title = normalize_page_title(title)
    normalized_text = normalize_whitespace(text)
    normalized_chunks = _normalize_prebuilt_chunks(chunks)
    chunk_page_metadata = {'page_title': normalized_title, 'source': source, **(metadata or {})}
    for chunk in normalized_chunks:
        chunk['metadata'] = {**chunk_page_metadata, **chunk['metadata']}
    content_hash = _build_page_content_hash(normalized_text, normalized_chunks)
    page, created = WebPage.objects.get_or_create(
        url=url,
        defaults={
            'source': source,
            'language': language,
            'title': normalized_title,
            'content_text': normalized_text,
            'raw_html': raw_html,
            'content_hash': content_hash,
            'metadata': metadata,
            'is_active': True,
        },
    )
    changed = created or page.content_hash != content_hash or force_refresh
    page.source = source
    page.language = language
    page.title = normalized_title
    page.content_text = normalized_text
    page.raw_html = raw_html
    page.content_hash = content_hash
    page.metadata = metadata
    page.is_active = True
    page.save()
    if changed:
        page.chunks.all().delete()
        ContentChunk.objects.bulk_create(
            [
                ContentChunk(
                    page=page,
                    chunk_index=chunk['chunk_index'],
                    text=chunk['text'],
                    metadata=chunk['metadata'],
                )
                for chunk in normalized_chunks
            ]
        )
    return page, changed


def upsert_page_content(
    *,
    source: str,
    url: str,
    title: str,
    text: str,
    raw_html: str,
    metadata: dict,
    language: str = 'tr',
    force_refresh: bool = False,
) -> tuple[WebPage, bool]:
    normalized_text = normalize_whitespace(text)
    normalized_chunks = [
        {
            'text': chunk['text'],
            'metadata': chunk['metadata'],
            'chunk_index': index,
        }
        for index, chunk in enumerate(chunk_text(normalized_text))
    ]
    return upsert_page_chunks(
        source=source,
        url=url,
        title=title,
        text=normalized_text,
        raw_html=raw_html,
        metadata=metadata,
        chunks=normalized_chunks,
        language=language,
        force_refresh=force_refresh,
    )


def mark_missing_pages_inactive(source: str, seen_urls: Iterable[str]) -> int:
    os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')
    return WebPage.objects.filter(source=source).exclude(url__in=list(seen_urls)).update(
        is_active=False
    )


def _extract_bologna_menu_targets(html: str) -> list[str]:
    return re.findall(r"menu_close\(this,'([^']+)'", html)


def _absolute_bologna_url(path: str) -> str:
    absolute = urljoin(f'https://{BOLOGNA_HOST}/oibs/bologna/', path)
    parsed = urlparse(absolute)
    query = dict(parse_qsl(parsed.query))
    query['lang'] = 'tr'
    normalized_query = urlencode(sorted(query.items()))
    return urlunparse(parsed._replace(scheme='https', query=normalized_query, fragment=''))


def extract_bologna_general_pages(index_html: str) -> list[str]:
    pages: list[str] = []
    for target in _extract_bologna_menu_targets(index_html):
        if not target.startswith('dynConPage.aspx'):
            continue
        pages.append(_absolute_bologna_url(target))
    return sorted(set(pages))


def extract_bologna_programs(unit_selection_html: str) -> list[dict]:
    soup = BeautifulSoup(unit_selection_html, 'html.parser')
    programs_by_id: dict[str, dict] = {}
    for panel in soup.select('.panel.panel-default'):
        if panel.select_one('.panel.panel-default'):
            continue
        faculty_link = panel.select_one('.panel-heading .panel-title > a')
        if not faculty_link:
            continue
        faculty = normalize_whitespace(faculty_link.get_text(' ', strip=True))
        for anchor in panel.select('.list-group-item a[href*="curSunit="]'):
            href = anchor.get('href', '').strip()
            title = normalize_whitespace(anchor.get_text(' ', strip=True))
            if not href or not title:
                continue
            query = dict(parse_qsl(urlparse(href).query))
            cur_sunit = query.get('curSunit')
            cur_unit = query.get('curUnit')
            if not cur_sunit:
                continue
            programs_by_id[cur_sunit] = {
                'title': title,
                'faculty': faculty,
                'cur_sunit': cur_sunit,
                'cur_unit': cur_unit,
                'index_url': _absolute_bologna_url(href),
            }
    return list(programs_by_id.values())


def extract_bologna_program_menu(program_index_html: str, program_id: str | None = None) -> list[dict]:
    soup = BeautifulSoup(program_index_html, 'html.parser')
    sections_by_url: dict[str, dict] = {}
    for anchor in soup.select('#proMenu a.nav-link'):
        onclick = anchor.get('onclick', '')
        match = re.search(r"menu_close\(this,'([^']+)'", onclick)
        if not match:
            continue
        title = normalize_whitespace(anchor.get_text(' ', strip=True))
        href = match.group(1)
        if not title or not href.startswith('prog'):
            continue
        section_url = _absolute_bologna_url(href)
        staff_page_type = _infer_bologna_staff_page_type(section_url)
        sections_by_url[section_url] = {
            'title': title,
            'url': section_url,
            'kind': 'bologna_staff_page' if staff_page_type else 'bologna_program_page',
            'staff_page_type': staff_page_type,
        }

    if program_id:
        synthetic_sections = (
            ('Program Yetkilileri', f'progOfficials.aspx?curSunit={program_id}&lang=tr'),
            ('Akademik Kadro', f'progAcademicStaff.aspx?curSunit={program_id}&lang=tr'),
        )
        for title, href in synthetic_sections:
            section_url = _absolute_bologna_url(href)
            if section_url in sections_by_url:
                continue
            sections_by_url[section_url] = {
                'title': title,
                'url': section_url,
                'kind': 'bologna_staff_page',
                'staff_page_type': _infer_bologna_staff_page_type(section_url),
            }
    return list(sections_by_url.values())


def extract_bologna_page(url: str, html: str, metadata: dict) -> ExtractedPage | None:
    soup = BeautifulSoup(html, 'html.parser')
    content_node = (
        soup.select_one('#UpdatePanel1')
        or soup.select_one('#lblPageContent')
        or soup.select_one('article')
        or soup.select_one('body')
    )
    if content_node is None:
        return None

    page_metadata = dict(metadata or {})
    content = BeautifulSoup(str(content_node), 'html.parser')
    _strip_irrelevant_nodes(content)
    text = normalize_whitespace(content.get_text('\n', strip=True))
    title = page_metadata.get('section_title') or page_metadata.get('program_title') or url
    header = soup.select_one('.panel-heading') or soup.select_one('#lblHeader')
    if header:
        title = normalize_whitespace(header.get_text(' ', strip=True))

    staff_page_type = page_metadata.get('staff_page_type') or _infer_bologna_staff_page_type(url)
    if staff_page_type:
        entries = _extract_staff_entries(content, text)
        page_metadata['kind'] = 'bologna_staff_page'
        page_metadata['staff_page_type'] = staff_page_type
        if entries:
            page_metadata['staff_count'] = len(entries)
            count_label = 'yetkili' if staff_page_type == 'officials' else 'hoca'
            text = _build_staff_page_text(page_metadata, entries, count_label=count_label)
    elif 'kind' not in page_metadata:
        page_metadata['kind'] = 'bologna_program_page'

    if len(text) < MIN_USEFUL_TEXT_LENGTH:
        return None
    return ExtractedPage(
        url=url,
        title=title,
        text=text,
        raw_html=str(content_node),
        metadata=page_metadata,
    )


def crawl_main_site(
    *,
    client: Any | None = None,
    session: Any | None = None,
    seeds: Iterable[str],
    max_pages: int,
    force_refresh: bool = False,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
) -> dict:
    client = client or session
    if client is None:
        raise ValueError('crawl_main_site requires a client or session')

    queue: deque[str] = deque(canonicalize_main_site_url(seed) for seed in seeds if seed)
    seen_urls: set[str] = set()
    failed_urls: set[str] = set()
    saved = 0
    updated = 0
    while queue and len(seen_urls) < max_pages:
        url = queue.popleft()
        if not url or url in seen_urls:
            continue
        logger.debug('crawl_main_site processing url=%s queue_size=%s', url, len(queue))
        seen_urls.add(url)
        html = fetch_html(client, url, rate_limit_delay=rate_limit_delay)
        if html is None:
            failed_urls.add(url)
            continue
        extracted = extract_main_site_page(url, html)
        if not extracted:
            continue
        _, changed = upsert_page_content(
            source='main_site',
            url=extracted.url,
            title=extracted.title,
            text=extracted.text,
            raw_html=extracted.raw_html,
            metadata=extracted.metadata,
            force_refresh=force_refresh,
        )
        saved += 1
        updated += int(changed)
        for link in extract_main_site_links(html, url):
            if link not in seen_urls:
                queue.append(link)
    if failed_urls:
        logger.warning(
            'crawl_main_site encountered failed fetches; skipping deactivation failed=%s',
            len(failed_urls),
        )
        deactivated = 0
    else:
        deactivated = mark_missing_pages_inactive('main_site', seen_urls)
    summary = {
        'saved': saved,
        'updated': updated,
        'deactivated': deactivated,
        'seen': len(seen_urls),
        'failed': len(failed_urls),
    }
    logger.info('crawl_main_site summary=%s', summary)
    return summary


def crawl_bologna(
    *,
    client: Any | None = None,
    session: Any | None = None,
    unit_types: Iterable[str],
    include_general_pages: bool = True,
    force_refresh: bool = False,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
) -> dict:
    client = client or session
    if client is None:
        raise ValueError('crawl_bologna requires a client or session')

    requested_unit_types = {unit_type for unit_type in unit_types}
    seen_urls: set[str] = set()
    seen_program_ids: set[str] = set()
    failed_urls: set[str] = set()
    saved = 0
    updated = 0
    base_index_url = 'https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=tr'
    base_index_html = None
    if include_general_pages:
        base_index_html = fetch_html(client, base_index_url, rate_limit_delay=rate_limit_delay)
        if base_index_html is None:
            failed_urls.add(base_index_url)
    if include_general_pages and base_index_html:
        for page_url in extract_bologna_general_pages(base_index_html):
            logger.debug('crawl_bologna processing general_page=%s', page_url)
            seen_urls.add(page_url)
            html = fetch_html(client, page_url, rate_limit_delay=rate_limit_delay)
            if html is None:
                failed_urls.add(page_url)
                continue
            extracted = extract_bologna_page(
                page_url,
                html,
                metadata={'host': BOLOGNA_HOST, 'kind': 'bologna_general_page'},
            )
            if not extracted:
                continue
            _, changed = upsert_page_content(
                source='bologna',
                url=page_url,
                title=extracted.title,
                text=extracted.text,
                raw_html=extracted.raw_html,
                metadata=extracted.metadata,
                force_refresh=force_refresh,
            )
            saved += 1
            updated += int(changed)
    for unit_type in unit_types:
        unit_url = _absolute_bologna_url(f'unitSelection.aspx?type={unit_type}&lang=tr')
        logger.debug('crawl_bologna processing unit_index=%s', unit_url)
        unit_html = fetch_html(client, unit_url, rate_limit_delay=rate_limit_delay)
        if unit_html is None:
            failed_urls.add(unit_url)
            continue
        programs = extract_bologna_programs(unit_html)
        for program in programs:
            if program['cur_sunit'] in seen_program_ids:
                logger.debug(
                    'crawl_bologna skipping duplicate program_id=%s title=%s',
                    program['cur_sunit'],
                    program['title'],
                )
                continue
            seen_program_ids.add(program['cur_sunit'])
            logger.debug(
                'crawl_bologna processing program_index=%s program=%s',
                program['index_url'],
                program['title'],
            )
            index_html = fetch_html(
                client,
                program['index_url'],
                rate_limit_delay=rate_limit_delay,
            )
            if index_html is None:
                failed_urls.add(program['index_url'])
                continue
            sections = extract_bologna_program_menu(index_html, program_id=program['cur_sunit'])
            for section in sections:
                section_url = section['url']
                if section_url in seen_urls:
                    logger.debug(
                        'crawl_bologna skipping duplicate section=%s program=%s title=%s',
                        section_url,
                        program['title'],
                        section['title'],
                    )
                    continue
                logger.debug(
                    'crawl_bologna processing section=%s program=%s title=%s',
                    section_url,
                    program['title'],
                    section['title'],
                )
                seen_urls.add(section_url)
                section_html = fetch_html(
                    client,
                    section_url,
                    rate_limit_delay=rate_limit_delay,
                )
                if section_html is None:
                    failed_urls.add(section_url)
                    continue
                extracted = extract_bologna_page(
                    section_url,
                    section_html,
                    metadata={
                        'host': BOLOGNA_HOST,
                        'kind': section.get('kind', 'bologna_program_page'),
                        'staff_page_type': section.get('staff_page_type'),
                        'unit_type': unit_type,
                        'faculty': program['faculty'],
                        'program_title': program['title'],
                        'program_id': program['cur_sunit'],
                        'program_unit_id': program['cur_unit'],
                        'section_title': section['title'],
                    },
                )
                if not extracted:
                    continue
                _, changed = upsert_page_content(
                    source='bologna',
                    url=section_url,
                    title=extracted.title,
                    text=extracted.text,
                    raw_html=extracted.raw_html,
                    metadata=extracted.metadata,
                    force_refresh=force_refresh,
                )
                saved += 1
                updated += int(changed)
    full_scope_crawl = include_general_pages and requested_unit_types == ALL_BOLOGNA_UNIT_TYPES
    if failed_urls:
        logger.warning(
            'crawl_bologna encountered failed fetches; skipping deactivation failed=%s',
            len(failed_urls),
        )
        deactivated = 0
    elif not full_scope_crawl:
        logger.warning(
            'crawl_bologna partial scope; skipping deactivation unit_types=%s include_general_pages=%s',
            sorted(requested_unit_types),
            include_general_pages,
        )
        deactivated = 0
    else:
        deactivated = mark_missing_pages_inactive('bologna', seen_urls)
    summary = {
        'saved': saved,
        'updated': updated,
        'deactivated': deactivated,
        'seen': len(seen_urls),
        'failed': len(failed_urls),
    }
    logger.info('crawl_bologna summary=%s', summary)
    return summary
